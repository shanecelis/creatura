//! This is a pretty funky idea. What if we used a music Digital Signal
//! Processor (DSP) as a creature's brain? It does a lot of what we want. There
//! are input signals; they're transformed and they're output somewhere. It also
//! is performant since it expects to be sampled 44,100 times a second.
//!
//! We could use a slow DSP to control a creature. And we could speed up the DSP
//! if we wanted to "listen" to its brain. Yeah, weird idea, I know.
use bevy::prelude::*;
use bevy_fundsp::prelude::*;
use bevy_xpbd_3d::{math::*, prelude::*};

#[derive(Component)]
pub struct MuscleUnit {
    pub unit: Box<dyn AudioUnit32>,
    pub min: Scalar,
    pub max: Scalar,
}

pub fn flex_muscles(
    //time: Res<Time>,
    mut joints: Query<(&mut DistanceJoint, &mut MuscleUnit)>,
) {
    let input: [f32; 0] = [];
    let mut output: [f32; 1] = [0.];

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
