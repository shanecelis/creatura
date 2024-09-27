#![allow(dead_code)]
#![allow(unused_imports)]
use avian3d::{math::*, prelude::*};
use bevy::prelude::*;
use nalgebra::{point, Isometry};
mod repeat_visit_map;
mod dfs;
mod bfs;

#[cfg(feature = "dsp")]
mod dsp;
// pub mod graph;
//
#[derive(Component)]
pub struct MuscleRange {
    pub min: Scalar,
    pub max: Scalar,
}

/// Use an even and odd part scheme so that the root part is even. Every part
/// successively attached is odd then even then odd. Then we don't allow even
/// and odd parts to collide. This is how we can create our own "no collisions
/// between objects that share a joint."
#[derive(PhysicsLayer, Default)]
pub enum Layer {
    Ground,
    #[default]
    Part,
    PartEven,
    PartOdd,
}

#[derive(Component)]
pub struct Sensor {
    pub value: Scalar,
}

/// Muscle `value` $\in [-1, 1]$.
#[derive(Component, Default)]
pub struct Muscle {
    pub value: Scalar,
}

pub fn plugin(app: &mut App) {
    app.add_systems(Update, keyboard_brain)
        .add_systems(Update, oscillate_muscles)
        .add_systems(Update, oscillate_brain)
        .add_systems(FixedUpdate, sync_muscles);
}

pub fn sync_muscles(
    mut joints: Query<(&mut DistanceJoint, &Muscle, Option<&MuscleRange>), Changed<Muscle>>,
) {
    for (mut joint, muscle, range) in &mut joints {
        if let Some(range) = range {
            let delta = range.max - range.min;
            joint.rest_length = (muscle.value / 2.0 + 0.5) * delta + range.min;
        } else {
            joint.rest_length = muscle.value;
        }
    }
}

pub fn oscillate_muscles(
    time: Res<Time>,
    nervous_systems: Query<(&NervousSystem, &SpringOscillator)>,
    mut muscles: Query<&mut Muscle>,
) {
    let seconds = time.elapsed_seconds();
    for (nervous_system, oscillator) in &nervous_systems {
        let v = oscillator.eval(seconds);
        for muscle_id in &nervous_system.muscles {
            if let Ok(mut muscle) = muscles.get_mut(*muscle_id) {
                // eprintln!("set muscle value {v}");
                muscle.value = v;
            }
        }
    }
}

pub fn oscillate_brain(
    time: Res<Time>,
    nervous_systems: Query<(&NervousSystem, &OscillatorBrain)>,
    mut muscles: Query<&mut Muscle>,
) {
    let seconds = time.elapsed_seconds();
    for (nervous_system, brain) in &nervous_systems {
        let n = brain.oscillators.len();
        for (i, muscle_id) in nervous_system.muscles.iter().enumerate() {
            if let Ok(mut muscle) = muscles.get_mut(*muscle_id) {
                let v = brain.oscillators[i % n].eval(seconds);
                muscle.value = v;
            }
        }
    }
}

// pub fn oscillate_motors(time: Res<Time>, mut joints: Query<(&mut DistanceJoint, &SpringOscillator)>) {
//     let seconds = time.elapsed_seconds();
//     for (mut joint, oscillator) in &mut joints {
//         joint.rest_length = (oscillator.max - oscillator.min)
//             * ((TAU * oscillator.freq * seconds).sin() * 0.5 + 0.5)
//             + oscillator.min;
//     }
// }

#[derive(Component)]
pub struct KeyboardBrain;

pub fn keyboard_brain(
    time: Res<Time>,
    input: Res<ButtonInput<KeyCode>>,
    nervous_systems: Query<&NervousSystem, With<KeyboardBrain>>,
    mut joints: Query<&mut Muscle>,
    disabled: Query<&JointDisabled>,
    mut commands: Commands,
) {
    use KeyCode::*;
    let keys = [[KeyQ, KeyA, KeyZ, Digit1],
                [KeyW, KeyS, KeyX, Digit2],
                [KeyE, KeyD, KeyC, Digit3],
                [KeyR, KeyF, KeyV, Digit4],
                [KeyT, KeyG, KeyB, Digit5],
                [KeyY, KeyJ, KeyN, Digit6],
    ];
    let delta = 1.0;
    for nervous_system in &nervous_systems {
        let muscles = &nervous_system.muscles;
        for i in 0..muscles.len().min(keys.len()) {
            if let Ok(mut muscle) = joints.get_mut(muscles[i]) {
                if input.pressed(keys[i][0]) {
                    muscle.value += delta * time.delta_seconds();
                    eprintln!("set muscle value {}", muscle.value);
                } else if input.pressed(keys[i][1]) {
                    muscle.value = 0.0;
                } else if input.pressed(keys[i][2]) {
                    muscle.value -= delta * time.delta_seconds();
                } else if input.just_pressed(keys[i][3]) {
                    if disabled.get(muscles[i]).is_ok() {
                        commands.entity(muscles[i]).remove::<JointDisabled>();
                    } else {
                        commands.entity(muscles[i]).insert(JointDisabled);
                    }
                }
            }
        }
    }
}

#[derive(Component)]
pub struct OscillatorBrain {
    pub oscillators: Vec<SpringOscillator>,
}

/// Contain sensors and muscles
#[derive(Component)]
pub struct NervousSystem {
    pub sensors: Vec<Entity>,
    pub muscles: Vec<Entity>,
}

#[derive(Component, Debug)]
pub struct SpringOscillator {
    pub freq: Scalar,
    pub phase: Scalar,
    // pub min: Scalar,
    // pub max: Scalar,
}

impl SpringOscillator {
    pub fn eval(&self, seconds: f32) -> f32 {
        (TAU * self.freq * seconds + self.phase).sin()
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Part {
    pub extents: Vector3,
    pub position: Vector3,
    pub rotation: Quaternion,
}

impl Default for Part {
    fn default() -> Self {
        Self {
            extents: Vector3::ONE,
            position: Vector3::ZERO,
            rotation: Quaternion::IDENTITY,
        }
    }
}

impl Part {
    pub fn shape(&self) -> Cuboid {
        let v = self.extents.f32();
        Cuboid::new(v[0], v[1], v[2])
    }

    pub fn collider(&self) -> Collider {
        let v = self.extents;
        Collider::cuboid(v[0], v[1], v[2])
    }

    pub fn volume(&self) -> Scalar {
        let v = self.extents;
        v.x * v.y * v.z
    }

    pub fn from_local(&self, point: Vector3) -> Vector3 {
        ColliderTransform {
            translation: self.position,
            rotation: self.rotation.into(),
            scale: Vector::ONE,
        }
        .transform_point(point)
    }
}

pub trait Stampable {
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
        Mat4::from_scale_rotation_translation(Vector::ONE, self.rotation, self.position)
            .inverse()
            .transform_point(point)
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
        let dir = self.position() - point;
        self.collider()
            .cast_ray(self.position(), self.rotation, point, dir, 100., false)
            .map(|(toi, _normal)| dir * toi + point)
    }
}

/// Returns a vector of parts and positions `(parent, child)`.
pub fn make_snake(n: u8, scale: f32, parent: &Part) -> Vec<(Part, (Vector3, Vector3))> {
    let mut results = Vec::new();
    let mut parent = *parent;
    for _ in 0..n {
        let mut child: Part = parent;
        child.position += 5. * Vector3::X;
        child.extents *= scale; //0.6;
        if let Some((p1, p2)) = child.stamp(&parent) {
            results.push((child, (p1, p2)));
        }
        parent = child;
    }
    results
}

#[derive(PartialEq, Clone, Copy)]
enum PartParity {
    Even,
    Odd,
}

impl PartParity {
    fn next(&self) -> Self {
        match self {
            PartParity::Even => PartParity::Odd,
            PartParity::Odd => PartParity::Even,
        }
    }
}

struct PartData {
    part: Part,
    id: Option<Entity>,
    parity: Option<PartParity>,
}
