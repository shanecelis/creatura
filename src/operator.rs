use crate::{Muscle, NervousSystem};
use bevy::prelude::*;
use genevo::{mutation::value::RandomValueMutation, operator::MutationOp};
use petgraph::{
    algo::{tarjan_scc, toposort, Cycle, DfsSpace},
    graph::DefaultIx,
    prelude::*,
    visit::{
        GraphBase, IntoEdgesDirected, IntoNeighbors, IntoNeighborsDirected, IntoNodeIdentifiers,
        IntoNodeReferences, NodeIndexable, Visitable,
    },
};
use rand::{
    distributions::uniform::{SampleRange, SampleUniform},
    Rng,
};
use rand_distr::{Distribution, Normal, StandardNormal};
use std::cmp::Ordering;
use std::f32::consts::TAU;
use std::ops::AddAssign;

trait Generator<R> {
    type G;
    /// Generate a genome.
    fn generate(&self, rng: &mut R) -> Self::G;

    fn into_iter(self, rng: &mut R) -> impl Iterator<Item = Self::G>
    where
        Self: Sized,
    {
        std::iter::repeat_with(move || self.generate(rng))
    }

    fn into_mutator<F>(self, f: F) -> impl Mutator<Self::G, R>
    where
        Self: Sized,
        F: Fn(Self::G, &mut Self::G),
    {
        move |genome: &mut Self::G, rng: &mut R| {
            let generated = self.generate(rng);
            f(generated, genome);
            1
        }
    }
}

impl<F, G, R> Generator<R> for F
where
    F: Fn(&mut R) -> G,
{
    type G = G;

    fn generate(&self, rng: &mut R) -> G {
        self(rng)
    }
}

trait Mutator<G, R> {
    /// Mutate the `genome` returning the number of mutations that occurred.
    fn mutate(&self, genome: &mut G, rng: &mut R) -> u32;

    // fn map<F,T>(self, f: F) -> impl Mutator<T, R> where Self: Sized,
    //                                                     F: Fn(G) -> T {
    //     move |genome: &mut G, rng: &mut R| {
    //         f(self.mutate(genome, rng))
    //     }
    // }

    fn repeat(self, repeat_count: usize) -> impl Mutator<G, R>
    where
        Self: Sized,
    {
        move |genome: &mut G, rng: &mut R| {
            let mut count = 0u32;
            for _ in 0..repeat_count {
                count += self.mutate(genome, rng);
            }
            count
        }
    }

    fn for_vec(self) -> impl Mutator<Vec<G>, R>
    where
        Self: Sized,
    {
        move |genomes: &mut Vec<G>, rng: &mut R| {
            let mut count = 0u32;
            for genome in genomes {
                count += self.mutate(genome, rng);
            }
            count
        }
    }
}

// struct SliceMutator<T>(T);

// impl<T> Mutator<G,R> for SliceMutator<

impl<F, G, R> Mutator<G, R> for F
where
    F: Fn(&mut G, &mut R) -> u32,
{
    // type G = G;
    fn mutate(&self, value: &mut G, rng: &mut R) -> u32 {
        self(value, rng)
    }
}

pub fn uniform_generator<T, R>(min: T, max: T) -> impl Generator<R, G = T>
where
    T: SampleUniform + PartialOrd + Copy,
    R: Rng,
{
    move |rng: &mut R| rng.gen_range(min..max)
}

pub fn normal_generator<T, R>(mean: T, stddev: T) -> Option<impl Generator<R, G = T>>
where
    T: PartialOrd + Copy + rand_distr::num_traits::Float,
    StandardNormal: Distribution<T>,
    R: Rng,
{
    Normal::new(mean, stddev)
        .map(|n| move |rng: &mut R| n.sample(rng))
        .ok()
}

pub fn uniform_mutator<T, R>(min: T, max: T) -> impl Mutator<T, R>
where
    T: SampleUniform + PartialOrd + Copy + AddAssign<T>,
    R: Rng,
{
    move |value: &mut T, rng: &mut R| {
        *value += rng.gen_range(min..max);
        1
    }
}

pub fn normal_mutator<T, R>(mean: T, stddev: T) -> Option<impl Mutator<T, R>>
where
    T: PartialOrd + Copy + rand_distr::num_traits::Float + AddAssign<T>,
    StandardNormal: Distribution<T>,
    R: Rng,
{
    normal_generator(mean, stddev)
        .map(|generator| generator.into_mutator(|generated, mutated| *mutated += generated))
}

pub fn rnd_prob<R>(rng: &mut R) -> f64
where
    R: Rng,
{
    rng.sample(rand::distributions::Open01)
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_generator() {
        let mut rng = rand::thread_rng();
        let v: Vec<f64> = rnd_prob.into_iter(&mut rng).take(2).collect();
        assert!(v[0] > 0.0 && v[0] < 1.0);
        assert!(v[1] > 0.0 && v[1] < 1.0);
    }

    #[test]
    fn test_uniform_generator() {
        let mut rng = rand::thread_rng();
        let g = uniform_generator(0, 100);
        let x = g.generate(&mut rng);
        assert!(x > 0 && x < 100);
    }
}
