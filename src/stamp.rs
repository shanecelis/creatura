#![allow(dead_code)]
#![allow(unused_imports)]
use avian3d::{math::*, prelude::*};
use bevy::prelude::*;
use nalgebra::{point, Isometry};

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
}

/// Stamp info.
pub struct StampInfo {
    /// In order to move the stamp so that the stamp and surface anchors are in
    /// the same world position, one would do this:
    ///
    /// `stamp.position = surface.position + stamp_delta`
    pub stamp_delta: Vector3,
    /// Anchor of the stamp in its local space.
    pub stamp_anchor: Vector3,
    /// Anchor of the surface in its local space.
    pub surface_anchor: Vector3,
}

pub trait Surface {
    /// Return object orientation.
    fn rotation(&self) -> Quaternion;
    /// Raycast from within the object to determine a point on its surface in
    /// local space.
    fn cast_to(&self, dir: Dir3) -> Option<Vector3>;
}

/// A stampable is an object that can be "stamped" onto a surface (also
/// `Stampable`). This means that it is moved so that its exterior is just
/// touching the object it was stamped onto.
pub trait Stampable: Surface {
    /// Return object position.
    fn position(&self) -> Vector3;
    /// Stamp object onto another, return the local vectors of each where they touch.
    fn stamp(&self, onto: &impl Surface, in_dir: Dir3) -> Option<StampInfo>;
    /// Convert a world point to a local point.
    fn to_local(&self, point: Vector3) -> Vector3;
    #[allow(clippy::wrong_self_convention)]
    /// Convert a local point to a world point.
    fn from_local(&self, point: Vector3) -> Vector3;
}

impl Surface for Part {
    fn rotation(&self) -> Quaternion {
        self.rotation
    }

    fn cast_to(&self, dir: Dir3) -> Option<Vector3> {
        self.collider()
            .cast_ray(
                Vec3::ZERO,
                Quaternion::IDENTITY,
                Vec3::ZERO,
                (self.rotation.inverse() * dir).as_vec3(),
                100.,
                false,
            )
            .map(|(toi, _normal)| dir * toi)
    }
}

impl Stampable for Part {
    fn position(&self) -> Vector3 {
        self.position
    }
    fn to_local(&self, point: Vector3) -> Vector3 {
        Mat4::from_scale_rotation_translation(Vector::ONE, self.rotation, self.position)
            .inverse()
            .transform_point(point)
    }

    fn from_local(&self, point: Vector3) -> Vector3 {
        ColliderTransform {
            translation: self.position,
            rotation: self.rotation.into(),
            scale: Vector::ONE,
        }
        .transform_point(point)
    }

    fn stamp(&self, onto: &impl Surface, in_dir: Dir3) -> Option<StampInfo> {
        if let Some(p1) = onto.cast_to(-in_dir) {
            if let Some(p2) = self.cast_to(in_dir) {
                // let intersect1 = onto.from_local(p1);
                // let intersect2 = self.from_local(p2);
                // We can put ourself into the right place.
                // let delta = intersect2 - intersect1;

                let deltap = onto.rotation() * p1 - self.rotation() * p2;
                // dbg!(delta, deltap);

                // let p1 = onto.to_local(intersect1);
                // let p2 = self.to_local(intersect2);
                // self.position -= delta;
                // let positionp = onto.position() + deltap;
                // self.position = positionp;
                // dbg!(self.position, positionp);
                return Some(StampInfo {
                    stamp_delta: deltap,
                    stamp_anchor: p2,
                    surface_anchor: p1
                });
            }
        }
        None
    }

}
