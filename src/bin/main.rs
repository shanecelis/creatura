use avian3d::{math::*, prelude::*};
use avian_pickup::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use bevy::{
    app::RunFixedMainLoop, prelude::*, time::run_fixed_main_schedule, window::WindowResolution,
};

use creatura::{body::*, brain::*, graph::*, operator::*, *};
use petgraph::prelude::*;
use rand::thread_rng;

fn main() {
    let mut app = App::new();

    let blue = Color::srgb_u8(27, 174, 228);

    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            resolution: WindowResolution::new(400.0, 400.0),
            ..default()
        }),
        ..default()
    }));
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
    .add_systems(Update, (mutate_on_space,
delete_on_backspace))
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

#[derive(Component)]
struct Creature(Vec<Entity>);

fn construct_creature(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let pink = Color::srgb_u8(253, 53, 176);
    let density = 1.0;
    let mut rng = thread_rng();
    // let genotype = snake_graph(3);
    let genotype = BodyGenotype::generate(&mut rng);
    let mut is_root = true;
    let mut root_id = None;
    let mut muscles = vec![];
    let entities: Vec<Entity> = construct_phenotype(
        &genotype.graph,
        genotype.start,
        BuildState::default(),
        Vector3::Y,
        Vector3::X,
        Vector3::Y,
        &mut commands,
        move |part, commands| {
            let id = cube_body(
                part,
                part.position,
                pink,
                density,
                &mut meshes,
                &mut materials,
                commands,
            );
            if is_root {
                commands.entity(id).insert(RootBody);
                is_root = false;
                root_id = Some(id);
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
    .expect("creature").into_iter()
                       //.filter(|x| root_id.map(|y| y != *x).unwrap_or(true))
                       .collect();

    commands.spawn(Creature(entities));

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
    let genotype: DiGraph<NVec4, ()> = g.map(|_ni, n| (*n).into(), |_, _| ());

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

fn delete_on_backspace(
    mut query: Query<(Entity, &Creature)>,
    mut commands: Commands,
    input: Res<ButtonInput<KeyCode>>,
) {
    if input.just_pressed(KeyCode::Backspace) {
        if let Ok((id, creature)) = query.get_single() {
            for id in &creature.0 {
                commands.entity(*id).despawn_recursive();
            }
            commands.entity(id).despawn_recursive();
        }
    }
}

fn mutate_on_space(
    mut query: Query<(&mut BitBrain, &mut Genotype<DiGraph<NVec4, ()>>)>,
    input: Res<ButtonInput<KeyCode>>,
) {
    if input.just_pressed(KeyCode::Space) {
        let mut rng = rand::thread_rng();
        let (mut brain, mut genotype) = query.single_mut();
        let mut g = genotype.0.clone();
        let count = brain_mutator.mutate(&mut g, &mut rng);

        let h: DiGraph<Neuron, ()> = g.map(|_ni, n| (*n).into(), |_, _| ());
        genotype.0 = g;

        if let Some(new_brain) = BitBrain::new(&h) {
            *brain = new_brain;
            info!("brain {count} mutated; brain replaced.");
        } else {
            warn!("brain {count} mutated; brain NOT replaced.");
        }
    }

    if input.just_pressed(KeyCode::Enter) {
        let mut rng = rand::thread_rng();
        let (mut brain, mut genotype) = query.single_mut();
        let g = brain_generator.generate(&mut rng);

        let h: DiGraph<Neuron, ()> = g.map(|_ni, n| (*n).into(), |_, _| ());
        genotype.0 = g;

        if let Some(new_brain) = BitBrain::new(&h) {
            *brain = new_brain;
            info!("brain generated; brain replaced.");
        } else {
            warn!("brain generated; brain NOT replaced.");
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
    let p_parent = parent.part.transform().transform_point(parent.anchor_local);
    let p_child = child.part.transform().transform_point(child.anchor_local);
    let rest_length = (p_parent - p_child).length();
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

