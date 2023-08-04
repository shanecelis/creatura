use bevy::prelude::*;
use std::f32::consts::FRAC_PI_3;
use bevy_xpbd_3d::{math::*, prelude::*, SubstepSchedule, SubstepSet};
use bevy_panorbit_camera::{PanOrbitCameraPlugin, PanOrbitCamera};

fn main() {
    let mut app = App::new();

    // Add plugins and startup system
    app.add_plugins((DefaultPlugins, PhysicsPlugins::default()))
        .add_system(bevy::window::close_on_esc)
        .add_systems(Startup, setup)
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

    // Spawn a static cube and a dynamic cube that is outside of the rest length.
    let root_cube = commands
        .spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Box::new(1., 1., 1.))),
                material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
                ..default()
            },
            RigidBody::Dynamic,
            Position(Vector::new(0.0, 1.0, 0.0)),
            Collider::cuboid(1., 1., 1.),
        ))
        .id();

    let c = Collider::cuboid(0.5, 0.5, 0.5);
    let child_cube = commands
        .spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Box::new(0.5, 0.5, 0.5))),
                material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
                ..default()
            },
            RigidBody::Dynamic,
            Position(Vector::new(1.0, 1.0, 0.0)),
            MassPropertiesBundle::new_computed(&c, 1.0),
            // c,
        ))
        .id();

    // Add a distance joint to keep the cubes at a certain distance from each other.
    // The dynamic cube should bounce like it's on a spring.
    commands.spawn(
        RevoluteJoint::new(root_cube, child_cube)
            .with_local_anchor_1(0.5 * Vector::X)
            .with_local_anchor_2(-0.25 * Vector::X)
            .with_aligned_axis(Vector::Z)
            .with_angle_limits(-FRAC_PI_3, FRAC_PI_3)
            // .with_linear_velocity_damping(0.1)
            // .with_angular_velocity_damping(1.0)
            .with_compliance(1.0 / 1000.0),
    );

    commands.spawn(
        DistanceJoint::new(root_cube, child_cube)
            .with_local_anchor_1(Vector::new(0.5, 0.5, 0.0))
            .with_local_anchor_2(Vector::new(0.25, 0.25, 0.0))
            .with_rest_length(0.5)
            // .with_limits(0.75, 2.5)
            // .with_linear_velocity_damping(0.1)
            // .with_angular_velocity_damping(1.0)
            .with_compliance(1.0 / 100.0),
    );

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
