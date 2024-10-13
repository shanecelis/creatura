use avian3d::{math::*, prelude::*};
use avian_pickup::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use bevy::{app::RunFixedMainLoop, prelude::*, time::run_fixed_main_schedule, window::WindowResolution};

use creatura::{brain::*, graph::*, *, operator::*, stamp::*};
use petgraph::prelude::*;

fn main() {
    let mut app = App::new();

    let blue = Color::srgb_u8(27, 174, 228);

        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: WindowResolution::new(400.0, 400.0),
                        ..default()
                    }),
                    ..default()
                })
        );
    // Add plugins and startup system
    app.add_plugins((
        // DefaultPlugins,
        PhysicsDebugPlugin::default(),
        PhysicsPlugins::default(),
        AvianPickupPlugin::default(),
        // Add interpolation
        // AvianInterpolationPlugin::default(),
        CreaturaPlugin,
    ))
    .insert_resource(ClearColor(blue))
    .add_systems(Startup, setup_env)
    .add_systems(Startup, construct_creature)
        .add_systems(Update, mutate_on_space)
    .add_plugins(PanOrbitCameraPlugin)
    //
    .add_systems(
        RunFixedMainLoop,
        (handle_pickup_input).before(run_fixed_main_schedule),
    );
    // Run the app
    app.run();
}

/// Pass player input along to `avian_pickup`
fn handle_pickup_input(
    mut avian_pickup_input_writer: EventWriter<AvianPickupInput>,
    key_input: Res<ButtonInput<MouseButton>>,
    actors: Query<Entity, With<AvianPickupActor>>,
) {
    for actor in &actors {
        if key_input.just_pressed(MouseButton::Left) {
            avian_pickup_input_writer.send(AvianPickupInput {
                action: AvianPickupAction::Throw,
                actor,
            });
        }
        if key_input.just_pressed(MouseButton::Right) {
            avian_pickup_input_writer.send(AvianPickupInput {
                action: AvianPickupAction::Drop,
                actor,
            });
        }
        if key_input.pressed(MouseButton::Right) {
            avian_pickup_input_writer.send(AvianPickupInput {
                action: AvianPickupAction::Pull,
                actor,
            });
        }
    }
}

#[derive(Component)]
struct RootBody;

fn construct_creature(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let pink = Color::srgb_u8(253, 53, 176);
    let density = 1.0;
    let (genotype, root) = snake_graph(3);
    let mut is_root = true;
    let mut muscles = vec![];
    for _entity in construct_phenotype(
        &genotype,
        root,
        BuildState::default(),
        Vector3::Y,
        Vector3::X,
        Vector3::Y,
        &mut commands,
        move |part, commands| {
            let id = cube_body(
                part,
                pink,
                density,
                &mut meshes,
                &mut materials,
                commands,
            );
            if is_root {
                commands.entity(id).insert(RootBody);
                is_root = false;
                // root_id = Some(id);
            }
            Some(id)
        },
        |joint: &JointConfig, commands| Some(spherical_joint(joint, commands)),
        |a, b, commands| {
            Some({
                let id = build_muscle(a, b, commands);
                muscles.push(id);
                id
            })
        },
    )
    .expect("creature")
    {}

    let mut g = DiGraph::new();
    let a = g.add_node(Neuron::Sin {
        amp: 1.0,
        freq: 1.0,
        phase: 0.0,
    });
    // let a = g.add_node(Neuron::Const(1.0));
    let b = g.add_node(Neuron::Muscle);
    g.add_edge(a, b, ());
    let brain = BitBrain::new(&g).unwrap();
    let genotype: DiGraph<NVec4, ()> = g.map(|ni, n| (*n).into(), |_, _| ());

    commands.spawn((
        NervousSystem {
            muscles,
            sensors: vec![],
        },
        // KeyboardBrain,
        brain,
        Genotype(genotype),
    ));
}

fn mutate_on_space(mut query: Query<(&mut BitBrain, &mut Genotype<DiGraph<NVec4, ()>>)>,
                   input: Res<ButtonInput<KeyCode>>) {
    if input.just_pressed(KeyCode::Space) {

        let mut rng = rand::thread_rng();
        let (mut brain, mut genotype) = query.single_mut();
        let mut g = genotype.0.clone();
        let count = nvec4_brain_mutator.mutate(&mut g, &mut rng);

        let h: DiGraph<Neuron, ()> = g.map(|ni, n| (*n).into(), |_, _| ());
        genotype.0 = g;

        if let Some(new_brain) = BitBrain::new(&h) {
            *brain = new_brain;
            info!("brain {count} mutated; brain replaced.");
        } else {
            warn!("brain {count} mutated; brain NOT replaced.");
        }
    }
}

// fn hill_climber(last_fitness: Local<Option<

#[derive(Component)]
struct Genotype<T>(T);

fn setup_env(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
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
        // Add this to set up the camera as the entity that can pick up
        // objects.
        AvianPickupActor {
            // Increase the maximum distance a bit to show off the
            // prop changing its distance on scroll.
            interaction_distance: 15.0,
            ..default()
        },
        // InputAccumulation::default(),
    ));
}

fn build_muscle(parent: &MuscleSite, child: &MuscleSite, commands: &mut Commands) -> Entity {
    let rest_length = (parent.part.from_local(parent.anchor_local)
        - child.part.from_local(child.anchor_local))
    .length();
    commands
        .spawn(
            DistanceJoint::new(parent.id, child.id)
                .with_local_anchor_1(parent.anchor_local)
                .with_local_anchor_2(child.anchor_local)
                .with_rest_length(dbg!(rest_length))
                // .with_limits(rest_length, rest_length)
                // .with_linear_velocity_damping(0.1)
                // .with_angular_velocity_damping(1.0)
                .with_compliance(1.0 / 100.0),
        )
        .insert(Muscle::default())
        .insert(MuscleRange {
            min: 0.0,
            max: rest_length * 2.0,
        })
        .id()
}

// fn setup(
//     mut commands: Commands,
//     mut meshes: ResMut<Assets<Mesh>>,
//     mut materials: ResMut<Assets<StandardMaterial>>,
// ) {
//     let mut rng = rand::thread_rng();
//     let ground_color = Color::srgb_u8(226, 199, 184);
//     // Ground
//     commands.spawn((
//         PbrBundle {
//             mesh: meshes.add(Cuboid::new(10., 0.1, 10.)),
//             material: materials.add(ground_color),
//             ..default()
//         },
//         CollisionLayers::new(
//             [Layer::Ground],
//             [Layer::Part, Layer::PartEven, Layer::PartOdd],
//         ),
//         RigidBody::Static,
//         Collider::cuboid(10., 0.1, 10.),
//     ));

//     let mut parent = Part {
//         extents: Vector::new(1., 1., 1.),
//         position: Vector::Y,
//         rotation: Quaternion::IDENTITY,
//     };
//     let pinks = [Color::srgb_u8(253, 53, 176)];

//     // Root cube
//     let color: Color = *pinks.choose(&mut rng).unwrap();
//     let mut parent_cube = commands
//         .spawn((
//             PbrBundle {
//                 mesh: meshes.add(Mesh::from(parent.shape())),
//                 material: materials.add(color),
//                 ..default()
//             },
//             // RigidBody::Static,
//             RigidBody::Dynamic,
//             Rotation(parent.rotation()),
//             Position(parent.position()),
//             parent.collider(),
//             CollisionLayers::new([Layer::PartEven], [Layer::Ground, Layer::PartEven]),
//         ))
//         .id();

//     let density = 1.0;
//     let scaling = 0.6;
//     let mut muscles = vec![];
//     for (i, (child, (p1, p2))) in make_snake(3, scaling, &parent).into_iter().enumerate() {
//         let color: Color = *pinks.choose(&mut rng).unwrap();
//         let child_cube = commands
//             .spawn((
//                 PbrBundle {
//                     mesh: meshes.add(Mesh::from(child.shape())),
//                     material: materials.add(StandardMaterial {
//                         base_color: color,
//                         // emissive: Color::WHITE.into(),
//                         ..default()
//                     }),
//                     ..default()
//                 },
//                 // RigidBody::Static,
//                 RigidBody::Dynamic,
//                 Position(child.position()),
//                 MassPropertiesBundle::new_computed(&child.collider(), child.volume() * density),
//                 // c,
//                 child.collider(),
//                 if (i + 1) % 2 == 0 {
//                     CollisionLayers::new([Layer::PartEven], [Layer::Ground, Layer::PartEven])
//                 } else {
//                     CollisionLayers::new([Layer::PartOdd], [Layer::Ground, Layer::PartOdd])
//                 },
//             ))
//             .id();

//         // commands.spawn(
//         //     RevoluteJoint::new(parent_cube, child_cube)
//         //         .with_local_anchor_1(p1)
//         //         .with_local_anchor_2(p2)
//         //         .with_aligned_axis(Vector::Z)
//         //         .with_angle_limits(-FRAC_PI_3, FRAC_PI_3), // .with_linear_velocity_damping(0.1)
//         //                                                    // .with_angular_velocity_damping(1.0)
//         //                                                    // .with_compliance(1.0 / 1000.0),
//         // );
//         commands.spawn(
//             {
//                 let mut j = SphericalJoint::new(parent_cube, child_cube)
//                     // .with_swing_axis(Vector::Y)
//                     // .with_twist_axis(Vector::X)
//                     .with_local_anchor_1(p1)
//                     .with_local_anchor_2(p2)
//                     // .with_aligned_axis(Vector::Z)
//                     .with_swing_limits(-FRAC_PI_4, FRAC_PI_4) // .with_linear_velocity_damping(0.1)
//                     .with_twist_limits(-FRAC_PI_4, FRAC_PI_4); // .with_linear_velocity_damping(0.1)
//                 j.swing_axis = Vector::Y;
//                 j.twist_axis = Vector::X;
//                 j
//             },
//         );
//         let a1 = parent.extents * Vector::new(0.5, 0.5, 0.0);
//         let a2 = child.extents * Vector::new(0.5, 0.5, 0.0);

//         let rest_length = (parent.from_local(a1) - child.from_local(a2)).length();

//         let muscle_id = commands
//             .spawn(
//                 DistanceJoint::new(parent_cube, child_cube)
//                     .with_local_anchor_1(a1)
//                     .with_local_anchor_2(a2)
//                     .with_rest_length(rest_length)
//                     // .with_limits(rest_length, rest_length)
//                     // .with_linear_velocity_damping(0.1)
//                     // .with_angular_velocity_damping(1.0)
//                     .with_compliance(1.0 / 100.0),
//             )
//             .insert(
//                 // MuscleUnit {
//                 //     // iter: dsp_sources.add(dsp)
//                 //     // iter: dsp.into_iter()
//                 //     unit: Box::new({ let mut unit = white_noise_mono();
//                 //                      unit.set_sample_rate(1000.0);
//                 //                      unit}),
//                 //     min: rest_length * length_scale,
//                 //     max: rest_length * (1.0 + length_scale),
//                 // }

//                 // SpringOscillator {
//                 //     freq: 1.0,
//                 //     min: rest_length * length_scale,
//                 //     max: rest_length * (1.0 + length_scale),
//                 // },
//                 Muscle::default(),
//             )
//             .insert(MuscleRange {
//                 min: 0.0,
//                 max: rest_length * 2.0,
//             })
//             .id();
//         muscles.push(muscle_id);
//         parent = child;
//         parent_cube = child_cube;
//     }
//     commands.spawn((
//         NervousSystem {
//             muscles,
//             sensors: vec![],
//         },
//         // KeyboardBrain,
//         // SpringOscillator {
//         //     freq: 0.5,
//         //     phase: 0.0,
//         // },
//         OscillatorBrain {
//             oscillators: vec![
//                 SpringOscillator {
//                     freq: 0.5,
//                     phase: 0.0,
//                 },
//                 SpringOscillator {
//                     freq: 0.5,
//                     phase: 0.2,
//                 },
//                 SpringOscillator {
//                     freq: 0.5,
//                     phase: 0.4,
//                 },
//                 SpringOscillator {
//                     freq: 0.5,
//                     phase: 0.6,
//                 },
//             ],
//         },
//     ));

// }
