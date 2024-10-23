use avian3d::{math::*, prelude::*};
#[cfg(feature = "pickup")]
use avian_pickup::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use bevy::{
    app::RunFixedMainLoop, prelude::*, time::run_fixed_main_schedule, window::WindowResolution,
};

use creatura::{body::*, brain::*, graph::*, operator::*, *};
use petgraph::prelude::*;
use rand::{thread_rng, rngs::StdRng, SeedableRng};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::{
    path::PathBuf,
    ffi::OsString,
    fs::File,
    io::{Write, BufWriter},
};
use serde::{Serialize, Deserialize};

#[derive(Debug, Subcommand)]
enum Subcommands {
    /// Write creature to file path
    #[command(arg_required_else_help = true)]
    Write {
        /// The path to write
        #[arg(required = true, value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
        path: PathBuf,
    },
    #[command(external_subcommand)]
    External(Vec<OsString>),
}

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    subcommand: Subcommands,
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Creature {
    body: BodyGenotype,
    brain: DiGraph<Neuron, ()>,
}

fn main() {
    let cli = Cli::parse();
    let mut app = App::new();
    // if let Some(seed) = args.seed {
    //     app.insert_resource(Seed(seed));
    // }
    let creature = Creature {
        body: match cli.seed {
            Some(seed) => {
                let mut rng = StdRng::seed_from_u64(seed);
                BodyGenotype::generate(&mut rng)
            }
            None => {
                snake_graph(3)
            }
        },
        brain: {
            let mut g = DiGraph::new();
            let a = g.add_node(Neuron::Sin {
                amp: 1.0,
                freq: 1.0,
                phase: 0.0,
            });
            // let a = g.add_node(Neuron::Const(1.0));
            let b = g.add_node(Neuron::Muscle);
            g.add_edge(a, b, ());
            g
        }
    };
    match cli.subcommand {
        Subcommands::Write { path } => {
            let file = File::create(path).expect("file");
            let mut writer = BufWriter::new(file);
            // let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
            serde_json::to_writer_pretty(&mut writer, &creature).expect("write");
            writer.flush().expect("flush");
            return ();
        }
        _ => {}
    }

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

        #[cfg(feature = "pickup")]
        AvianPickupPlugin::default(),
        // Add interpolation
        // AvianInterpolationPlugin::default(),
        CreaturaPlugin,
    ))
    .insert_resource(ClearColor(blue))
    .add_systems(Startup, setup_env)
    .add_systems(Startup, (move || creature.clone()).pipe(construct_creature))
    .add_systems(Update, (mutate_on_space, delete_on_backspace))
    .add_plugins(PanOrbitCameraPlugin);
    //
    //
    #[cfg(feature = "pickup")]
    app.add_systems(
        RunFixedMainLoop,
        (handle_pickup_input).before(run_fixed_main_schedule),
    );
    // Run the app
    app.run();
}

/// Pass player input along to `avian_pickup`
#[cfg(feature = "pickup")]
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
struct RootBody(Vec<Entity>);

#[derive(Resource)]
struct Seed(u64);

fn rng_from_seed(seed: Option<Res<Seed>>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed.0),
        None => StdRng::from_entropy()
    }
}

fn construct_creature(
    In(input): In<Creature>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    // seed: Option<Res<Seed>>,
) {
    let pink = Color::srgb_u8(253, 53, 176);
    let density = 1.0;
    // let mut rng = rng_from_seed(seed);
    // let genotype = snake_graph(3);
    let genotype = input.body;
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
            if root_id.is_none() {
                root_id = Some(id);
                // commands.entity(id).insert(RootBody);
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
                       .collect();

    commands.entity(root_id.unwrap()).insert(RootBody(entities));
    // commands.spawn(Creature(entities));

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
    mut query: Query<&RootBody>,
    mut commands: Commands,
    input: Res<ButtonInput<KeyCode>>,
) {
    if input.just_pressed(KeyCode::Backspace) {
        if let Ok(creature) = query.get_single() {
            for id in &creature.0 {
                commands.entity(*id).despawn_recursive();
            }
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
    let thing = commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 0.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        PanOrbitCamera::default(),
        ));

    #[cfg(feature = "pickup")]
    thing.insert(
        // Add this to set up the camera as the entity that can pick up
        // objects.
        AvianPickupActor {
            // Increase the maximum distance a bit to show off the
            // prop changing its distance on scroll.
            interaction_distance: 15.0,
            ..default()
        },
        // InputAccumulation::default(),
    );
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

