use avian3d::{math::*, prelude::*};
use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use rand::seq::SliceRandom;
use std::f32::consts::FRAC_PI_4; //, FRAC_PI_3, PI, TAU};

use muscley_wusaley::*;

// fn white_noise() -> impl AudioUnit32 {
//     // white() >> split::<U2>() * 0.2
//     dc(50.0) >> sine() * 0.5
// }

// fn white_noise_mono() -> impl AudioUnit32 {
//     // white() * 0.2
//     dc(50.0) >> sine() * 0.5
// }

// fn play_noise(
//     mut commands: Commands,
//     mut assets: ResMut<Assets<DspSource>>,
//     dsp_manager: Res<DspManager>,
// ) {
//     let source = assets.add(
//         dsp_manager
//             .get_graph(white_noise)
//             .unwrap_or_else(|| panic!("DSP source not found!"))
//             // HACK: I'm cloning here and that may be wrong.
//             .clone(),
//     );
//     commands.spawn(AudioSourceBundle {
//         source,
//         ..default()
//     });
// }
//
fn main() {
    let mut app = App::new();

    let blue = Color::srgb_u8(27, 174, 228);
    // Add plugins and startup system
    app.add_plugins((
        DefaultPlugins,
        PhysicsDebugPlugin::default(),
        PhysicsPlugins::default(),
        plugin,
    ))
    // .add_plugins(DspPlugin::default())
    .insert_resource(ClearColor(blue))
    // .add_dsp_source(white_noise, SourceType::Dynamic)
    .add_systems(Startup, setup)
    // .add_systems(PostStartup, play_noise)
    // .add_systems(Update, bevy::window::close_on_esc)
    // .add_systems(Update, oscillate_motors)
    // .add_systems(FixedUpdate, sync_muscles)
    // .add_systems(Update, graph::flex_muscles)
    .add_plugins(PanOrbitCameraPlugin);
    // Run the app
    app.run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    // mut dsp_sources: ResMut<Assets<DspSource>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    // dsp_manager: Res<DspManager>,
) {
    let mut rng = rand::thread_rng();
    let ground_color = Color::srgb_u8(226, 199, 184);
    // Ground
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Cuboid::new(10., 0.1, 10.)),
            material: materials.add(ground_color),
            ..default()
        },
        CollisionLayers::new(
            [Layer::Ground],
            [Layer::Part, Layer::PartEven, Layer::PartOdd],
        ),
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

    let pinks = [Color::srgb_u8(253, 53, 176)];

    // Root cube
    let color: Color = *pinks.choose(&mut rng).unwrap();
    let mut parent_cube = commands
        .spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(parent.shape())),
                material: materials.add(color),
                ..default()
            },
            // RigidBody::Static,
            RigidBody::Dynamic,
            Rotation(parent.rotation()),
            Position(parent.position()),
            parent.collider(),
            CollisionLayers::new([Layer::PartEven], [Layer::Ground, Layer::PartEven]),
        ))
        .id();

    let density = 1.0;
    let scaling = 0.6;
    let mut muscles = vec![];
    for (i, (child, (p1, p2))) in make_snake(3, scaling, &parent).into_iter().enumerate() {
        let color: Color = *pinks.choose(&mut rng).unwrap();
        let child_cube = commands
            .spawn((
                PbrBundle {
                    mesh: meshes.add(Mesh::from(child.shape())),
                    material: materials.add(StandardMaterial {
                        base_color: color,
                        // emissive: Color::WHITE.into(),
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
            {
                let mut j = SphericalJoint::new(parent_cube, child_cube)
                    // .with_swing_axis(Vector::Y)
                    // .with_twist_axis(Vector::X)
                    .with_local_anchor_1(p1)
                    .with_local_anchor_2(p2)
                    // .with_aligned_axis(Vector::Z)
                    .with_swing_limits(-FRAC_PI_4, FRAC_PI_4) // .with_linear_velocity_damping(0.1)
                    .with_twist_limits(-FRAC_PI_4, FRAC_PI_4); // .with_linear_velocity_damping(0.1)
                j.swing_axis = Vector::Y;
                j.twist_axis = Vector::X;
                j
            }, // .with_angular_velocity_damping(1.0)
               // .with_compliance(1.0 / 1000.0),
        );
        let a1 = parent.extents * Vector::new(0.5, 0.5, 0.0);
        let a2 = child.extents * Vector::new(0.5, 0.5, 0.0);

        let rest_length = (parent.from_local(a1) - child.from_local(a2)).length();

        // let length_scale = 0.4;

        // let sample_rate = 44_100.0; // This should come from somewhere else.
        // let dsp = DspSource::new(white_noise_mono,
        //                          sample_rate,
        //                          SourceType::Dynamic);

        // let dsp = dsp_manager
        //             .get_graph(white_noise)
        //             .unwrap()
        //             // HACK: This doesn't feel right.
        //             .clone();
        let muscle_id = commands
            .spawn(
                DistanceJoint::new(parent_cube, child_cube)
                    .with_local_anchor_1(a1)
                    .with_local_anchor_2(a2)
                    .with_rest_length(rest_length)
                    // .with_limits(rest_length * length_scale, rest_length * (1.0 + length_scale))
                    // .with_limits(0.0, 2.0 * rest_length)
                    // .with_linear_velocity_damping(0.1)
                    // .with_angular_velocity_damping(1.0)
                    .with_compliance(1.0 / 100.0),
            )
            .insert(
                // MuscleUnit {
                //     // iter: dsp_sources.add(dsp)
                //     // iter: dsp.into_iter()
                //     unit: Box::new({ let mut unit = white_noise_mono();
                //                      unit.set_sample_rate(1000.0);
                //                      unit}),
                //     min: rest_length * length_scale,
                //     max: rest_length * (1.0 + length_scale),
                // }

                // SpringOscillator {
                //     freq: 1.0,
                //     min: rest_length * length_scale,
                //     max: rest_length * (1.0 + length_scale),
                // },
                Muscle::default(),
            )
            .insert(MuscleRange {
                min: 0.0,
                max: rest_length * 2.0,
            })
            .id();
        muscles.push(muscle_id);
        parent = child;
        parent_cube = child_cube;
    }
    commands.spawn((
        NervousSystem {
            muscles,
            sensors: vec![],
        },
        // KeyboardBrain,
        // SpringOscillator {
        //     freq: 0.5,
        //     phase: 0.0,
        // },
        OscillatorBrain {
            oscillators: vec![
                SpringOscillator {
                    freq: 0.5,
                    phase: 0.0,
                },
                SpringOscillator {
                    freq: 0.5,
                    phase: 0.2,
                },
                SpringOscillator {
                    freq: 0.5,
                    phase: 0.4,
                },
                SpringOscillator {
                    freq: 0.5,
                    phase: 0.6,
                },
            ],
        },
    ));

    // Light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 1000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(1.0, 8.0, 1.0).looking_at(Vec3::ZERO, Dir3::Y),
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
