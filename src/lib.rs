#![allow(dead_code)]
#![allow(unused_imports)]
use bevy::prelude::*;
use avian3d::{prelude::*, math::*};
use nalgebra::{point, Isometry};

#[cfg(feature = "dsp")]
mod dsp;
// pub mod graph;

/// Use an even and odd part scheme so that the root part is even. Every part
/// successively attached is odd then even then odd. Then we don't allow even
/// and odd parts to collide. This is how we can create our own "no collisions
/// between objects that share a joint."
#[derive(PhysicsLayer)]
pub enum Layer {
    Ground,
    Part,
    PartEven,
    PartOdd,
}

#[derive(Component)]
pub struct Sensor {
    pub value: Scalar,
}

#[derive(Component)]
pub struct Muscle {
    pub value: Scalar,
    pub min: Scalar,
    pub max: Scalar,
}

pub fn sync_muscles(mut joints: Query<(&mut DistanceJoint, &Muscle), Changed<Muscle>>) {
    for (mut joint, muscle) in &mut joints {
        joint.rest_length = muscle.value;
    }
}

pub fn oscillate_motors(time: Res<Time>, mut joints: Query<(&mut DistanceJoint, &SpringOscillator)>) {
    let seconds = time.elapsed_seconds();
    for (mut joint, oscillator) in &mut joints {
        joint.rest_length = (oscillator.max - oscillator.min)
            * ((TAU * oscillator.freq * seconds).sin() * 0.5 + 0.5)
            + oscillator.min;
    }
}

pub struct NervousSystem {
    muscles: Vec<Entity>,
}

#[derive(Component, Debug)]
pub struct SpringOscillator {
    pub freq: Scalar,
    pub min: Scalar,
    pub max: Scalar,
}

// #[derive(Component, Debug)]
// struct MuscleIterator<T : Iterator<Item = [f32; 2]>> {
//     iter: T,
// }


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
            rotation: Quaternion::IDENTITY
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
            scale: Vector::ONE
        }.transform_point(point)
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

pub fn make_snake(n: u8, scale: f32, parent: &Part) -> Vec<(Part, (Vector3, Vector3))> {
    let mut results = Vec::new();
    let mut parent = *parent;
    for _ in 0..n {
        let mut child: Part = parent;
        child.position += 5. * Vector3::X;
        child.extents *= scale;//0.6;
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
            PartParity::Odd  => PartParity::Even,
        }
    }
}

struct PartData {
    part: Part,
    id: Option<Entity>,
    parity: Option<PartParity>
}
