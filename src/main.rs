use bevy::prelude::{*, shape};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_xpbd_3d::{math::*, prelude::*, SubstepSchedule, SubstepSet};
use parry3d_f64 as parry3d;
use parry3d::{math::Isometry, query::*};
use rand::seq::SliceRandom;
use std::f64::consts::{FRAC_PI_4, FRAC_PI_3, PI, TAU};
use nalgebra::point;
use bevy_fundsp::prelude::*;

/// Use an even and odd part scheme so that the root part is even. Every part
/// successively attached is odd then even then odd. Then we don't allow even
/// and odd parts to collide. This is how we can create our own "no collisions
/// between objects that share a joint."
#[derive(PhysicsLayer)]
enum Layer {
    Ground,
    Part,
    PartEven,
    PartOdd,
}

fn white_noise() -> impl AudioUnit32 {
    // white() >> split::<U2>() * 0.2
    dc(50.0) >> sine() * 0.5
}

fn white_noise_mono() -> impl AudioUnit32 {
    // white() * 0.2
    dc(50.0) >> sine() * 0.5
}

fn main() {
    let mut app = App::new();

    let blue = Color::rgb_u8(27, 174, 228);
    // Add plugins and startup system
    app.add_plugins((DefaultPlugins, PhysicsPlugins::default()))
        .add_plugins(DspPlugin::default())
        .insert_resource(ClearColor(blue))
        .add_dsp_source(white_noise, SourceType::Dynamic)
        .add_systems(Startup, setup)
        .add_systems(PostStartup, play_noise)
        .add_systems(Update, bevy::window::close_on_esc)
        // .add_systems(Update, oscillate_motors)
        .add_systems(Update, flex_muscles)
        .add_plugin(PanOrbitCameraPlugin);
    // Run the app
    app.run();
}

fn play_noise(
    mut commands: Commands,
    mut assets: ResMut<Assets<DspSource>>,
    dsp_manager: Res<DspManager>,
) {
    let source = assets.add(
        dsp_manager
            .get_graph(white_noise)
            .unwrap_or_else(|| panic!("DSP source not found!"))
            // HACK: I'm cloning here and that may be wrong.
            .clone(),
    );
    commands.spawn(AudioSourceBundle {
        source,
        ..default()
    });
}

fn oscillate_motors(time: Res<Time>, mut joints: Query<(&mut DistanceJoint, &SpringOscillator)>) {
    let seconds = time.elapsed_seconds_f64();
    for (mut joint, oscillator) in &mut joints {
        joint.rest_length = (oscillator.max - oscillator.min)
            * ((TAU * oscillator.freq * seconds).sin() * 0.5 + 0.5)
            + oscillator.min;
    }
}

// fn flex_muscles<T: Iterator + Send + Sync + 'static>(time: Res<Time>, mut joints: Query<(&mut DistanceJoint, &mut MuscleIterator<f64>)>) {
// fn flex_muscles<T: Iterator<Item = [f32; 2]> + Send + Sync + 'static>(time: Res<Time>,
//                                                                       mut joints: Query<(&mut DistanceJoint, &mut MuscleIterator<T>)>) {
//     for (mut joint, mut muscle) in &mut joints {
//         if let Some(l) = muscle.iter.next() {
//             joint.rest_length = l[0] as f64;
//         }
//     }
// }
fn flex_muscles(time: Res<Time>,
                mut joints: Query<(&mut DistanceJoint, &mut MuscleUnit)>) {
    let input : [f32; 0] = [];
    let mut output : [f32; 1] = [0.];

    for (mut joint, mut muscle) in &mut joints {
        debug_assert!(muscle.unit.inputs() == 0);
        debug_assert!(muscle.unit.outputs() >= 1);
        muscle.unit.tick(&input, &mut output);
        let length = muscle.min.lerp(muscle.max, output[0] as f64 / 2.0 + 0.5);
        // let length = output[0].abs() as f64;
        println!("Setting muscle to {length}");
        joint.rest_length = length;
    }
}

// Copied from xpbd. It's only pub(crate) there.
fn make_isometry(pos: Vector, rot: &Rotation) -> Isometry<Scalar> {
    Isometry::<Scalar>::new(pos.into(), rot.to_scaled_axis().into())
}

#[derive(Component, Debug)]
struct SpringOscillator {
    freq: Scalar,
    min: Scalar,
    max: Scalar,
}

// #[derive(Component, Debug)]
// struct MuscleIterator<T : Iterator<Item = [f32; 2]>> {
//     iter: T,
// }


// #[derive(Component, Debug)]
#[derive(Component)]
struct MuscleUnit {
    unit : Box<dyn AudioUnit32>,
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
        // We prefer this method to bevy's `Transform` because it can be done
        // with f64 just as easily as f32.
        make_isometry(self.position, &Rotation(self.rotation))
            // FIXME: Is there a From or Into defined somewhere?
            .transform_point(&point![ point.x, point.y, point.z ]).into()
    }
}

trait Stampable {
    /// Return object position.
    fn position(&self) -> Vector3;
    /// Return object orientation.
    fn rotation(&self) -> Quaternion;
    /// Stamp object onto another, return the local vectors of each where they connect.
    fn stamp(&mut self, onto: &impl Stampable) -> Option<(Vector3, Vector3)>;
    /// Raycast from within the object to determine a point on its surface.
    fn cast_to(&self, point: Vector3) -> Option<Vector3>;
    /// Convert a world point to a local point.
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
        make_isometry(self.position, &Rotation(self.rotation))
            .inverse()
            .transform_point(&point![ point.x, point.y, point.z ]).into()
    }

    fn stamp(&mut self, onto: &impl Stampable) -> Option<(Vector3, Vector3)> {
        if let Some(intersect1) = onto.cast_to(self.position()) {
            if let Some(intersect2) = self.cast_to(onto.position()) {
                // We can put ourself into the right place.
                let delta = intersect2 - intersect1;
                let p1 = onto.to_local(intersect1);
                let p2 = self.to_local(intersect2);
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
    for _ in 0..n {
        let mut child: Part = parent.clone();
        child.position += 5. * Vector3::X;
        child.extents *= 0.6;
        if let Some((p1, p2)) = child.stamp(&parent) {
            results.push((child.clone(), (p1, p2)));
        }
        parent = child;
    }
    results
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut dsp_sources: ResMut<Assets<DspSource>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    dsp_manager: Res<DspManager>,
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
        CollisionLayers::new([Layer::Ground], [Layer::Part, Layer::PartEven, Layer::PartOdd]),
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
        Color::rgb_u8(253, 53, 176),
        // Color::rgb_u8(254, 134, 212),
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

            CollisionLayers::new([Layer::PartEven], [Layer::Ground, Layer::PartEven]),
        ))
        .id();

    let density = 1.0;
    for (i, (child, (p1, p2))) in make_snake(4, &parent).into_iter().enumerate() {
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
                if (i + 1) % 2 == 0 {
                    CollisionLayers::new([Layer::PartEven], [Layer::Ground, Layer::PartEven])
                } else {
                    CollisionLayers::new([Layer::PartOdd], [Layer::Ground, Layer::PartOdd])
                },
            ))
            .id();

        // commands.spawn(
        //     RevoluteJoint::new(parent_cube, child_cube)
        //         .with_local_anchor_1(p1)
        //         .with_local_anchor_2(p2)
        //         .with_aligned_axis(Vector::Z)
        //         .with_angle_limits(-FRAC_PI_3, FRAC_PI_3), // .with_linear_velocity_damping(0.1)
        //                                                    // .with_angular_velocity_damping(1.0)
        //                                                    // .with_compliance(1.0 / 1000.0),
        // );
        commands.spawn(
            SphericalJoint::new(parent_cube, child_cube)
                .with_swing_axis(Vector::Y)
                .with_twist_axis(Vector::X)
                .with_local_anchor_1(p1)
                .with_local_anchor_2(p2)
                // .with_aligned_axis(Vector::Z)
                .with_swing_limits(-FRAC_PI_4, FRAC_PI_4) // .with_linear_velocity_damping(0.1)
                .with_twist_limits(-FRAC_PI_4, FRAC_PI_4), // .with_linear_velocity_damping(0.1)
                                                           // .with_angular_velocity_damping(1.0)
                                                           // .with_compliance(1.0 / 1000.0),
        );
        let a1 = parent.extents * Vector::new(0.5, 0.5, 0.0);
        let a2 = child.extents * Vector::new(0.5, 0.5, 0.0);

        let rest_length = (parent.from_local(a1) - child.from_local(a2)).length();

        let length_scale = 0.4;

        let sample_rate = 44_100.0; // This should come from somewhere else.
        // let dsp = DspSource::new(white_noise_mono,
        //                          sample_rate,
        //                          SourceType::Dynamic);

        let dsp = dsp_manager
                    .get_graph(white_noise)
                    .unwrap()
                    // HACK: This doesn't feel right.
                    .clone();
        commands.spawn(
            DistanceJoint::new(parent_cube, child_cube)
                .with_local_anchor_1(a1)
                .with_local_anchor_2(a2)
                .with_rest_length(rest_length)
                // .with_limits(0.75, 2.5)
                // .with_linear_velocity_damping(0.1)
                // .with_angular_velocity_damping(1.0)
                .with_compliance(1.0 / 100.0))
            .insert(
            MuscleUnit {
                // iter: dsp_sources.add(dsp)
                // iter: dsp.into_iter()
                unit: Box::new({ let mut unit = white_noise_mono();
                                 unit.set_sample_rate(1000.0);
                                 unit}),
                min: rest_length * length_scale,
                max: rest_length * (1.0 + length_scale),
            }
            // SpringOscillator {
            //     freq: 1.0,
            //     min: rest_length * length_scale,
            //     max: rest_length * (1.0 + length_scale),
            // },
        );
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
