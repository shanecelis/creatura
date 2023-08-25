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
pub enum Layer {
    Ground,
    Part,
    PartEven,
    PartOdd,
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
pub fn flex_muscles(time: Res<Time>,
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
pub struct MuscleUnit {
    pub unit : Box<dyn AudioUnit32>,
    pub min: Scalar,
    pub max: Scalar,
}

#[derive(Clone, Debug)]
pub struct Part {
    pub extents: Vector3,
    pub position: Vector3,
    pub rotation: Quaternion,
}

impl Part {
    pub fn shape(&self) -> shape::Box {
        let v = self.extents.as_f32();
        shape::Box::new(v[0], v[1], v[2])
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
        // We prefer this method to bevy's `Transform` because it can be done
        // with f64 just as easily as f32.
        make_isometry(self.position, &Rotation(self.rotation))
            // FIXME: Is there a From or Into defined somewhere?
            .transform_point(&point![ point.x, point.y, point.z ]).into()
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

pub fn make_snake(n: u8, parent: &Part) -> Vec<(Part, (Vector3, Vector3))> {
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

