use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_xpbd_3d::{math::*, prelude::*, SubstepSchedule, SubstepSet};
use parry3d_f64 as parry3d;
use parry3d::{math::Isometry, query::*};
use rand::seq::SliceRandom;
use std::f64::consts::{FRAC_PI_3, PI, TAU};
use nalgebra::point;

#[derive(PhysicsLayer)]
enum Layer {
    Ground,
    Part,
}

fn main() {
    let mut app = App::new();

    let blue = Color::rgb_u8(27, 174, 228);
    // Add plugins and startup system
    app.add_plugins((DefaultPlugins, PhysicsPlugins::default()))
        .insert_resource(ClearColor(blue))
        .add_systems(Startup, setup)
        .add_systems(Update, bevy::window::close_on_esc)
        .add_systems(Update, oscillate_motors)
        .add_plugin(PanOrbitCameraPlugin);
    // Run the app
    app.run();
}

fn oscillate_motors(time: Res<Time>, mut joints: Query<(&mut DistanceJoint, &SpringOscillator)>) {
    let seconds = time.elapsed_seconds_f64();
    for (mut joint, oscillator) in &mut joints {
        joint.rest_length = (oscillator.max - oscillator.min)
            * ((TAU * oscillator.freq * seconds).sin() * 0.5 + 0.5)
            + oscillator.min;
    }
}

// Copied from xpbd
fn make_isometry(pos: Vector, rot: &Rotation) -> Isometry<Scalar> {
    Isometry::<Scalar>::new(pos.into(), rot.to_scaled_axis().into())
}

#[derive(Component, Debug)]
struct SpringOscillator {
    freq: Scalar,
    min: Scalar,
    max: Scalar,
}

#[derive(Clone, Debug)]
struct Part {
    extents: Vector3,
    position: Vector3,
    rotation: Quaternion,
}

impl Part {
    fn shape(&self) -> shape::Box {
        let v = self.extents.as_f32();
        shape::Box::new(v[0], v[1], v[2])
    }

    fn collider(&self) -> Collider {
        let v = self.extents;
        Collider::cuboid(v[0], v[1], v[2])
    }

    fn volume(&self) -> Scalar {
        let v = self.extents;
        v.x * v.y * v.z
    }

    fn from_local(&self, point: Vector3) -> Vector3 {

        make_isometry(self.position, &Rotation(self.rotation))
            // .inverse()
            // .absolute_transform_vector(point)
            // .transform_point(point.into())
            .transform_point(&point![ point.x, point.y, point.z ]).into()
        // Transform::from_translation(self.position.as_f32())
        //     .with_rotation(self.rotation.as_f32())
        //     // .compute_matrix()
        //     // .inverse()
        //     .transform_point(point.as_f32()).into()
    }
}

trait Stampable {
    fn position(&self) -> Vector3;
    fn rotation(&self) -> Quaternion;
    fn stamp(&mut self, onto: &impl Stampable) -> Option<(Vector3, Vector3)>;
    fn cast_to(&self, point: Vector3) -> Option<Vector3>;
    fn to_local(&self, point: Vector3) -> Vector3;
}

impl Stampable for Part {
    fn position(&self) -> Vector3 {
        self.position
    }
    fn rotation(&self) -> Quaternion {
        self.rotation
    }

    fn to_local(&self, point: Vector3) -> Vector3 {
        // FIXME: Reconsider using Isometry here.
        make_isometry(self.position, &Rotation(self.rotation))
            .inverse()
            // .absolute_transform_vector(point)
            // .transform_point(point.into())
            .transform_point(&point![ point.x, point.y, point.z ]).into()
        // Transform::from_translation(self.position.as_f32())
        //     .with_rotation(self.rotation.as_f32())
        //     .compute_matrix()
        //     .inverse()
        //     .transform_point3(point.as_f32()).into()
    }

    fn stamp(&mut self, onto: &impl Stampable) -> Option<(Vector3, Vector3)> {
        if let Some(intersect1) = onto.cast_to(self.position()) {
            if let Some(intersect2) = self.cast_to(onto.position()) {
                // We can put ourself into the right place.
                let delta = intersect2 - intersect1;
                // println!("i1 {intersect1} i2 {intersect2}");
                // let p1 = intersect1;
                // let p2 = intersect2;
                let p1 = onto.to_local(intersect1);
                let p2 = self.to_local(intersect2);
                // let p1 = Transform::from_translation(onto.position())
                //     .with_rotation(onto.rotation())
                //     .compute_matrix()
                //     .inverse()
                //     .transform_point3(intersect1);
                // let p2 = Transform::from_translation(self.position)
                //     .with_rotation(self.rotation)
                //     .compute_matrix()
                //     .inverse()
                //     .transform_point3(intersect2);
                // println!("p1 {p1} p2 {p2}");
                self.position -= delta;
                return Some((p1, p2));
            }
        }
        None
    }

    fn cast_to(&self, point: Vector3) -> Option<Vector3> {
        let r = parry3d::query::details::Ray::new(
            self.position().into(),
            (point - self.position()).into(),
        );
        let m = make_isometry(self.position(), &Rotation(self.rotation));
        self.collider()
            .cast_ray_and_get_normal(&m, &r, 100., false)
            .map(|intersect| r.point_at(intersect.toi).into())
    }
}

fn make_snake(n: u8, parent: &Part) -> Vec<(Part, (Vector3, Vector3))> {
    let mut results = Vec::new();
    let mut parent = parent.clone();
    for i in 0..n {
        let mut child: Part = parent.clone();
        child.position += 5. * Vector3::X;
        child.extents *= 0.6;
        if let Some((p1, p2)) = child.stamp(&parent) {
            // let joint = make_joint(p1, p2);
            results.push((child.clone(), (p1, p2)));
        }
        parent = child;
    }
    results
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let cube_mesh = PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        ..default()
    };

    let mut rng = rand::thread_rng();
    let ground_color = Color::rgb_u8(226, 199, 184);
    // Ground
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Box::new(10., 0.1, 10.))),
            material: materials.add(ground_color.into()),
            ..default()
        },
        CollisionLayers::new([Layer::Ground], [Layer::Part]),
        RigidBody::Static,
        Collider::cuboid(10., 0.1, 10.),
    ));

    let mut parent = Part {
        extents: Vector::new(1., 1., 1.),
        position: Vector::Y,
        rotation: Quaternion::IDENTITY,
    };
    let p = Vector3::new(1., 2., 1.);
    let mut child = Part {
        extents: Vector::new(0.5, 0.5, 0.5),
        position: p,
        rotation: Quaternion::IDENTITY,
    };
    let _ = child.stamp(&parent);

    let pinks = vec![
        // Color::rgb_u8(253, 162, 231),
        // Color::rgb_u8(253, 53, 176),
        Color::rgb_u8(254, 134, 212),
    ];

    // Root cube
    let color: Color = *pinks.choose(&mut rng).unwrap();
    let mut parent_cube = commands
        .spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(parent.shape())),
                material: materials.add(color.into()),
                ..default()
            },
            // RigidBody::Static,
            RigidBody::Dynamic,
            Rotation(parent.rotation),
            Position(parent.position),
            parent.collider(),

            CollisionLayers::new([Layer::Part], [Layer::Ground]),
        ))
        .id();

    let density = 1.0;
    for (child, (p1, p2)) in make_snake(4, &parent) {
        let color: Color = *pinks.choose(&mut rng).unwrap();
        let child_cube = commands
            .spawn((
                PbrBundle {
                    mesh: meshes.add(Mesh::from(child.shape())),
                    material: materials.add(StandardMaterial {
                        base_color: color,
                        emissive: Color::WHITE * 0.1,
                        ..default()
                    }),
                    ..default()
                },
                // RigidBody::Static,
                RigidBody::Dynamic,
                Position(child.position()),
                MassPropertiesBundle::new_computed(&child.collider(), child.volume() * density),
                // c,
                child.collider(),
                CollisionLayers::new([Layer::Part], [Layer::Ground]),
            ))
            .id();

        commands.spawn(
            RevoluteJoint::new(parent_cube, child_cube)
                .with_local_anchor_1(p1)
                .with_local_anchor_2(p2)
                .with_aligned_axis(Vector::Z)
                .with_angle_limits(-FRAC_PI_3, FRAC_PI_3), // .with_linear_velocity_damping(0.1)
                                                           // .with_angular_velocity_damping(1.0)
                                                           // .with_compliance(1.0 / 1000.0),
        );
        let a1 = parent.extents * Vector::new(0.5, 0.5, 0.0);
        let a2 = child.extents * Vector::new(0.5, 0.5, 0.0);

        let rest_length = (parent.from_local(a1) - child.from_local(a2)).length();
        // println!("length {length}");

        let length_scale = 0.4;
        commands.spawn((
            DistanceJoint::new(parent_cube, child_cube)
                .with_local_anchor_1(a1)
                .with_local_anchor_2(a2)
                .with_rest_length(rest_length)
                // .with_limits(0.75, 2.5)
                // .with_linear_velocity_damping(0.1)
                // .with_angular_velocity_damping(1.0)
                .with_compliance(1.0 / 100.0),
            SpringOscillator {
                freq: 1.0,
                min: rest_length * length_scale,
                max: rest_length * (1.0 + length_scale),
            },
        ));
        parent = child;
        parent_cube = child_cube;
    }

    // Light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 0.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        PanOrbitCamera::default(),
    ));
}
