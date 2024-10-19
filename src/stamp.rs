#![allow(dead_code)]
#![allow(unused_imports)]
use avian3d::{math::*, prelude::*};
use bevy::prelude::*;

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

/// A surface can be stamped on.
pub trait Surface {
    /// Return object orientation.
    fn rotation(&self) -> Quaternion;
    /// Raycast from within the object to determine a point on its surface in
    /// local space.
    fn cast_to(&self, dir: Dir3) -> Option<Vector3>;
}

/// A stampable is an object that can be "stamped" onto a surface (also
/// `Stamp`). This means that it is moved so that its exterior is just
/// touching the object it was stamped onto.
pub trait Stamp: Surface {
    /// Stamp object onto another, return the local vectors of each where they
    /// touch.
    fn stamp(&self, onto: &impl Surface, in_dir: Dir3) -> Option<StampInfo>;
}

