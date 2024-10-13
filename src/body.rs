use avian3d::{math::*, prelude::*};
use bevy::prelude::*;
use crate::stamp::*;
use core::f32::consts::FRAC_PI_4;

use petgraph::{
    graph::DefaultIx,
    prelude::*,
};

#[derive(Clone, Debug, Copy)]
pub struct MuscleGene {
    pub parent: Quaternion,
    pub child: Quaternion,
    pub max_strength: f32,
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

    pub fn transform(&self) -> Transform {
        Transform {
            translation: self.position,
            rotation: self.rotation,
            scale: Vector::ONE,
        }
    }
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

impl Stamp for Part {
    fn stamp(&self, onto: &impl Surface, in_dir: Dir3) -> Option<StampInfo> {
        if let Some(p1) = onto.cast_to(-in_dir) {
            if let Some(p2) = self.cast_to(in_dir) {
                let delta = onto.rotation() * p1 - self.rotation() * p2;
                return Some(StampInfo {
                    stamp_delta: delta,
                    stamp_anchor: p2,
                    surface_anchor: p1,
                });
            }
        }
        None
    }
}

pub fn snake_graph(part_count: u8) -> (DiGraph<Part, PartEdge>, NodeIndex<DefaultIx>) {
    let part = Part {
        extents: Vector::new(1., 1., 1.),
        position: Vector::ZERO,
        rotation: Quaternion::IDENTITY,
    };
    let mut graph = DiGraph::new();
    let index = graph.add_node(part);
    graph.add_edge(
        index,
        index,
        PartEdge {
            joint_rotation: Quaternion::IDENTITY,
            rotation: Quaternion::IDENTITY,
            scale: 0.6 * Vector3::ONE,
            iteration_count: part_count,
            op: None,
            muscles: vec![MuscleGene {
                parent: Quaternion::from_axis_angle(Vec3::Z, FRAC_PI_4),
                child: Quaternion::IDENTITY,
                max_strength: 1.0,
            }],
        },
    );
    (graph, index)
}
