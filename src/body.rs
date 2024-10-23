use avian3d::{math::*, prelude::*};
use bevy::prelude::*;
use crate::{
    stamp::*,
    brain::Neuron,
    operator::{*, graph::*},
};
use core::f32::consts::FRAC_PI_4;
use rand::Rng;

use petgraph::{
    graph::DefaultIx,
    prelude::*,
};

use rand_distr::{Distribution, Standard};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
pub struct MuscleGene {
    pub parent: Quaternion,
    pub child: Quaternion,
    pub max_strength: f32,
}

impl MuscleGene {
    fn generate<R>(rng: &mut R) -> MuscleGene
    where R: Rng {
        Self {
            parent: Quat::from_rng(rng),
            child: Quat::from_rng(rng),
            max_strength: rng.gen_range(0.1..2.0),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
pub enum EdgeOp {
    /// Relfect the edge through a plane
    Reflect { normal: Dir3 },
    /// Use a radial symmetry with a rotation normal.
    Radial { normal: Dir3, symmetry: u8 },
}

impl EdgeOp {
    pub fn generate<R: Rng>(rng: &mut R) -> EdgeOp {
        if rng.with_prob(0.5) {
            EdgeOp::Reflect { normal: Dir3::from_rng(rng) }
        } else {
            EdgeOp::Radial { normal: Dir3::from_rng(rng),
                             symmetry: rng.gen_range(1..=8) }
        }
    }
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


#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PartEdge {
    // joint_type:
    pub joint_rotation: Quaternion,
    pub rotation: Quaternion,
    pub scale: Vector3,
    pub iteration_count: u8,
    pub op: Option<EdgeOp>,
    pub muscles: Vec<MuscleGene>,
}

impl PartEdge {
    fn generate<R: Rng>(rng: &mut R) -> PartEdge {
        let s = to_vec3(uniform_generator(0.1, 1.2));
        let m = rng.gen_range(0..3);
        Self {
            joint_rotation: Quat::from_rng(rng),
            rotation: Quat::from_rng(rng),
            scale: s.generate(rng),
            iteration_count: uniform_generator(1, 5).generate(rng),
            op: rng.with_prob(0.2).then_some(EdgeOp::generate(rng)),
            muscles: MuscleGene::generate.into_iter(rng).take(m).collect(),
        }
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
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

    pub fn generate<R>(rng: &mut R) -> Part where R: Rng {
        let v = uniform_generator(0.1, 2.0);
        let w = uniform_generator(0.0, TAU);
        Part {
            extents: Vec3::new(v.generate(rng),
                               v.generate(rng),
                               v.generate(rng)),
            position: Vector3::ZERO,
            rotation: Quat::from_euler(EulerRot::XYZ,
                                       w.generate(rng),
                                       w.generate(rng),
                                       w.generate(rng))
        }
    }

    pub fn mutate<R>(&mut self, rng: &mut R) -> u32 where R: Rng {
        let v = uniform_mutator(-0.2, 0.2);
        // let w = uniform_mutator(-FRAC_PI_4, FRAC_PI_4);
        let mut count = 0;
        count += v.mutate(&mut self.extents.x, rng);
        count += v.mutate(&mut self.extents.y, rng);
        count += v.mutate(&mut self.extents.z, rng);
        let q = Quat::from_rng(rng);
        self.rotation *= q;
        count += 1;
        count
    }
}


/// Convert a `FromRng` into a `Generator`.
pub fn into_generator<T, R>() -> impl Generator<T,R>
    where
    T: FromRng,
    Standard: Distribution<T>,
    R: Rng {
    move |rng: &mut R| T::from_rng(rng)
}

pub fn to_vec3<R>(v: impl Generator<f32, R>) -> impl Generator<Vec3, R> {
    move |rng: &mut R|
    Vec3::new(v.generate(rng),
              v.generate(rng),
              v.generate(rng))
}

pub fn quat_generator<R>(generator: impl Generator<f32, R>) -> impl Generator<Quat, R> {
    move |rng: &mut R|
    Quat::from_euler(EulerRot::XYZ,
                     generator.generate(rng),
                     generator.generate(rng),
                     generator.generate(rng))

}

// impl<R,S,T> From<Generator<f32, R>> for Generator<Vec3, R>
// where
//     R: Rng,
//     Self: Sized,
//     Generator<f32, R>: Sized,
// {
//     fn from(v: Generator<f32, R>) -> Self {
//         move |rng: &mut R|
//         Vec3::new(v.generate(rng),
//                   v.generate(rng),
//                   v.generate(rng))
//     }
// }

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

pub fn snake_graph(part_count: u8) -> BodyGenotype {
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
    BodyGenotype { graph, start: index }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BodyGenotype {
    pub graph: DiGraph<Part, PartEdge>,
    pub start: NodeIndex<DefaultIx>,
}

impl BodyGenotype {
    pub fn generate<R>(rng: &mut R) -> Self where R: Rng {
        let mut graph = DiGraph::new();
        let node_gen = Part::generate;
        let edge_gen = PartEdge::generate;
        let start = graph.add_node(node_gen.generate(rng));
        let add_nodes = add_connecting_node(node_gen, edge_gen).repeat(5);
        let count = add_nodes.mutate(&mut graph, rng);
        info!("Generate body genotype with {count} mutations.");

        println!("{:?}", petgraph::dot::Dot::with_config(&graph, &[]));
        BodyGenotype {
            graph,
            start
        }
    }

    pub fn sensor_count(&self) -> usize {
        0
    }

    pub fn muscle_count(&self) -> usize {
        self.graph.edge_weights().map(|w| w.muscles.len()).sum()
    }
}
