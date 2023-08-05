use bevy::prelude::*;
use std::f32::consts::FRAC_PI_3;
use bevy_xpbd_3d::{math::*, prelude::*, SubstepSchedule, SubstepSet};
use bevy_panorbit_camera::{PanOrbitCameraPlugin, PanOrbitCamera};
use parry3d::{query::*, math::Isometry};

fn main() {
    let mut app = App::new();

    // Add plugins and startup system
    app.add_plugins((DefaultPlugins, PhysicsPlugins::default()))
        .add_systems(Startup, setup)
        .add_systems(Update, bevy::window::close_on_esc)
        .add_systems(Update, oscillate_motors)
        .add_plugin(PanOrbitCameraPlugin);
    // Run the app
    app.run();
}

fn oscillate_motors(time: Res<Time>, mut joints: Query<&mut DistanceJoint>) {
    let seconds = time.elapsed_seconds();
    for mut joint in &mut joints {
        joint.rest_length = (12. * seconds).sin() * 0.5 + 0.1;
    }
}

// Copied from xpbd
fn make_isometry(pos: Vector, rot: &Rotation) -> Isometry<Scalar> {
    Isometry::<Scalar>::new(pos.into(), rot.to_scaled_axis().into())
}

struct Part {
    collider: Collider,
    position: Vector3,
    rotation: Quat,
}

impl Part {
    fn shape(&self) -> Option<shape::Box> {
        self.collider.as_cuboid().map(|c| {
            let v = c.half_extents;
            shape::Box::new(v[0] * 2., v[1] * 2., v[1] * 2.)
        })
    }

    fn unwrap(self) -> Collider {
        self.collider
    }
}

trait Stampable {
    fn stamp(&mut self, onto: &impl Stampable) -> Option<(Vector3, Vector3)>;
    fn cast_to(&self, point: Vector3) -> Option<Vector3>;
    fn position(&self) -> Vector3;
}

impl Stampable for Part {
    fn position(&self) -> Vector3 { self.position }

    fn stamp(&mut self, onto: &impl Stampable) -> Option<(Vector3, Vector3)> {
        if let Some(intersect1) = onto.cast_to(self.position()) {
            if let Some(intersect2) = self.cast_to(onto.position()) {
                // We can put ourself into the right place.
                let delta = intersect2 - intersect1;
                println!("i1 {intersect1} i2 {intersect2}");
                let p1 = intersect1;
                let p2 = intersect2;
                // let p1 = onto.to_local(intersect1);
                // let p2 = self.to_local(intersect2);
                self.position -= delta;
                return Some((p1, p2));
            }
        }
        None
    }

    fn cast_to(&self, point: Vector3) -> Option<Vector3> {
        let r = parry3d::query::details::Ray::new(self.position().into(), (point - self.position()).into());
        let m = make_isometry(self.position(), &Rotation(self.rotation));
        self.collider.cast_ray_and_get_normal(&m, &r, 100., false).map(|intersect| r.point_at(intersect.toi).into())
    }
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

    // Spawn a static cube and a dynamic cube that is outside of the rest length.
    commands
        .spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Box::new(10., 0.1, 10.))),
                material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
                ..default()
            },
            RigidBody::Static,
            Collider::cuboid(10., 0.1, 10.),
        ));

    let root = Part {
        collider: Collider::cuboid(1., 1., 1.),
        position: Vector::Y,
        rotation: Quat::IDENTITY
    };
    let p = Vector3::new(1., 2., 1.);
    let c = Collider::cuboid(0.5, 0.5, 0.5);
    let mut child = Part {
        collider: c,
        position: p,
        rotation: Quat::IDENTITY
    };
    let _ = child.stamp(&root);

    // Spawn a static cube and a dynamic cube that is outside of the rest length.
    let root_cube = commands
        .spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(root.shape().unwrap())),
                material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
                ..default()
            },
            RigidBody::Static,
            // RigidBody::Dynamic,
            Rotation(root.rotation),
            Position(root.position),
            root.unwrap()
        ))
        .id();

    // This doesn't work because commands haven't actually run yet.
    // if let Some(hit) = spatial_query.cast_ray(p,
    //                                           Vector::NEG_X,
    //                                           100.,
    //                                           true,
    //                                           SpatialQueryFilter::default()) {
    //     println!("First hit: {:?}", hit);
    // } else {
    //     println!("No hit");
    // }
    //


    // let a = Collider::cuboid(1., 1., 1.);
    // let r = parry3d::query::details::Ray::new(Vector3::new(0.,0.,0.).into(), Vector3::NEG_X.into());
    // let m = make_isometry(Vector3::ZERO, &Rotation(Quat::IDENTITY));
    // if let Some(hit) = a.cast_ray_and_get_normal(&m,
    //                                              &r,
    //                  100.,
    //                  false) {
    //     println!("First hit: {:?}", hit);
    // } else {
    //     println!("No hit");
    // }

    let child_cube = commands
        .spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(child.shape().unwrap())),
                material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
                ..default()
            },
            RigidBody::Static,
            // RigidBody::Dynamic,
            Position(child.position),
            MassPropertiesBundle::new_computed(&child.collider, 1.0),
            // c,
            child.unwrap()
        ))
        .id();

    // Add a distance joint to keep the cubes at a certain distance from each other.
    // The dynamic cube should bounce like it's on a spring.
    // commands.spawn(
    //     RevoluteJoint::new(root_cube, child_cube)
    //         .with_local_anchor_1(0.5 * Vector::X)
    //         .with_local_anchor_2(-0.25 * Vector::X)
    //         .with_aligned_axis(Vector::Z)
    //         .with_angle_limits(-FRAC_PI_3, FRAC_PI_3)
    //         // .with_linear_velocity_damping(0.1)
    //         // .with_angular_velocity_damping(1.0)
    //         .with_compliance(1.0 / 1000.0),
    // );

    // commands.spawn(
    //     DistanceJoint::new(root_cube, child_cube)
    //         .with_local_anchor_1(Vector::new(0.5, 0.5, 0.0))
    //         .with_local_anchor_2(Vector::new(0.25, 0.25, 0.0))
    //         .with_rest_length(0.5)
    //         // .with_limits(0.75, 2.5)
    //         // .with_linear_velocity_damping(0.1)
    //         // .with_angular_velocity_damping(1.0)
    //         .with_compliance(1.0 / 100.0),
    // );

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
    commands.spawn((Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    },
    PanOrbitCamera::default()));
}
