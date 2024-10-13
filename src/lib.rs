#![allow(dead_code)]
#![allow(unused_imports)]
use avian3d::{math::*, prelude::*};
use bevy::prelude::*;
use nalgebra::{point, Isometry};
use crate::stamp::*;
pub mod brain;
pub mod operator;
pub mod stamp;
mod repeat_visit_map;

#[cfg(feature = "dsp")]
mod dsp;

pub mod graph;
pub mod rdfs;
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

/// Sensor `value` $\in [0, 1]$.
#[derive(Component, Default)]
pub struct Sensor {
    pub value: Scalar,
}

/// Muscle `value` $\in [0, 1]$.
#[derive(Component, Default)]
pub struct Muscle {
    pub value: Scalar,
}

pub struct CreaturaPlugin;

impl Plugin for CreaturaPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(brain::plugin)
            .add_systems(Update, keyboard_brain)
            .add_systems(Update, oscillate_muscles)
            .add_systems(Update, oscillate_brain)
            .add_systems(FixedUpdate, sync_muscles);
    }
}

pub fn sync_muscles(
    mut joints: Query<
        (
            &mut DistanceJoint,
            &Muscle,
            Option<&MuscleRange>,
            Option<&mut Sensor>,
        ),
        Changed<Muscle>,
    >,
) {
    for (mut joint, muscle, range, sensor) in &mut joints {
        if let Some(mut sensor) = sensor {
            if let Some(range) = range {
                let delta = range.max - range.min;
                sensor.value = (joint.rest_length - range.min) / delta;
            } else {
                sensor.value = joint.rest_length;
            }
        }
        if let Some(range) = range {
            let delta = range.max - range.min;
            joint.rest_length = muscle.value * delta + range.min;
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
    let keys = [
        [KeyQ, KeyA, KeyZ, Digit1],
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
                    eprintln!("inc muscle value {}", muscle.value);
                } else if input.pressed(keys[i][1]) {
                    muscle.value = 0.0;
                    eprintln!("reset muscle value {}", muscle.value);
                } else if input.pressed(keys[i][2]) {
                    muscle.value -= delta * time.delta_seconds();
                    eprintln!("dec muscle value {}", muscle.value);
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
}

impl SpringOscillator {
    pub fn eval(&self, seconds: f32) -> f32 {
        (TAU * self.freq * seconds + self.phase).sin()
    }
}

#[derive(Clone, Debug, Copy)]
pub struct MuscleGene {
    parent: Quaternion,
    child: Quaternion,
    max_strength: f32,
}


#[derive(Clone, Debug, Copy)]
pub enum EdgeOp {
    /// Relfect the edge through a plane
    Reflect { normal: Vector3 },
    /// Use a radial symmetry with a rotation normal.
    Radial { normal: Vector3, symmetry: u8 },
}

impl Default for MuscleGene {
    fn default() -> Self {
        Self {
            parent: Quaternion::IDENTITY,
            child: Quaternion::IDENTITY,
            max_strength: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PartEdge {
    // joint_type:
    pub joint_rotation: Quaternion,
    pub rotation: Quaternion,
    pub scale: Vector3,
    pub iteration_count: u8,
    pub op: Option<EdgeOp>,
    pub muscles: Vec<MuscleGene>,
}

