use rand::{
    distributions::uniform::SampleUniform,
    Rng,
};
use rand_distr::{Distribution, Normal, StandardNormal};
use std::ops::AddAssign;

pub mod graph;

/// Generate a type `G` in association with type `R`, which is usually a random
/// number generator.
pub trait Generator<G, R> {
    /// Generate a genome.
    fn generate(&self, rng: &mut R) -> G;

    /// Create an iterator of the given generator.
    fn into_iter(self, rng: &mut R) -> impl Iterator<Item = G>
    where
        Self: Sized,
    {
        std::iter::repeat_with(move || self.generate(rng))
    }

    /// Turn this generator into a mutator via the given closure.
    fn into_mutator<F>(self, f: F) -> impl Mutator<G, R>
    where
        Self: Sized,
        F: Fn(G, &mut G),
    {
        move |genome: &mut G, rng: &mut R| {
            let generated = self.generate(rng);
            f(generated, genome);
            1
        }
    }
}

impl<F, G, R> Generator<G, R> for F
where
    F: Fn(&mut R) -> G,
{
    fn generate(&self, rng: &mut R) -> G {
        self(rng)
    }
}

// pub trait Crosser<G, R> : Mutator<(G, G), R> {
//     fn cross(&self, a: &mut G, b: &mut G, rng: &mut R) -> u32;

//     fn mutate(&self, genome: &mut (G, G), rng: &mut R) -> u32 {
//         self.cross(&mut genome.0, &mut genome.1, rng)
//     }
// }

/// A crosser for type `G`. It crosses over two values of type `G`. This trait is
/// associated with type `R`, which is typically a random number generator.
pub trait Crosser<G, R> {
    /// Mutate the `genome` returning the number of mutations that occurred.
    fn cross(&self, a: &mut G, b: &mut G, rng: &mut R) -> u32;
}

/// A mutator for type `G` in association with type `R`, which is typically a
/// random number generator.
pub trait Mutator<G, R> {
    /// Mutate the `genome` returning the number of mutations that occurred.
    fn mutate(&self, genome: &mut G, rng: &mut R) -> u32;

    /// Repeat this mutator a set number of times.
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

    /// Create a vector mutator. Mutates all entries.
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

    /// Return a mutator that only applies itself with probability $p \in [0,
    /// 1]$.
    fn with_prob(self, p: f32) -> impl Mutator<G, R>
        where
        Self: Sized,
        R: Rng,
    {
        move |genome: &mut G, rng: &mut R| {
            if rng.with_prob(p) {
                self.mutate(genome, rng)
            } else {
                0
            }
        }
    }
}

// struct SliceMutator<T>(T);

// impl<T> Mutator<G,R> for SliceMutator<

impl<F, G, R> Mutator<G, R> for F
where
    F: Fn(&mut G, &mut R) -> u32,
{
    fn mutate(&self, value: &mut G, rng: &mut R) -> u32 {
        self(value, rng)
    }
}

impl<F, G, R> Crosser<G, R> for F
where
    F: Fn(&mut G, &mut G, &mut R) -> u32,
{
    fn cross(&self, a: &mut G, b: &mut G, rng: &mut R) -> u32 {
        self(a, b, rng)
    }
}

/// Generate a uniform distribution $U ~ [min, max)$.
pub fn uniform_generator<T, R>(min: T, max: T) -> impl Generator<T, R>
where
    T: SampleUniform + PartialOrd + Copy,
    R: Rng,
{
    move |rng: &mut R| rng.gen_range(min..max)
}

/// Generate a normal distribution with the given $mean$ and $stddev$.
pub fn normal_generator<T, R>(mean: T, stddev: T) -> Option<impl Generator<T, R>>
where
    T: PartialOrd + Copy + rand_distr::num_traits::Float,
    StandardNormal: Distribution<T>,
    R: Rng,
{
    Normal::new(mean, stddev)
        .map(|n| move |rng: &mut R| n.sample(rng))
        .ok()
}

/// Add a value drawn from a uniform distribution $U ~ [min, max)$.
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

/// Add a value drawn from a normal distribution with the given $mean$ and
/// $stddev$.
pub fn normal_mutator<T, R>(mean: T, stddev: T) -> Option<impl Mutator<T, R>>
where
    T: PartialOrd + Copy + rand_distr::num_traits::Float + AddAssign<T>,
    StandardNormal: Distribution<T>,
    R: Rng,
{
    normal_generator(mean, stddev)
        .map(|generator| generator.into_mutator(|generated, mutated| *mutated += generated))
}

/// Swap the two genomes. Doesn't do much by itself but useful in composition.
pub fn swapper<G, R>() -> impl Crosser<G, R>
where
    R: Rng,
{
    |a: &mut G, b: &mut G, _rng: &mut R| {
        std::mem::swap(a, b);
        1
    }
}

/// Random number generator extensions.
pub trait RngExt {
    /// Return a probability (0, 1).
    fn prob(&mut self) -> f32;
    /// Return true with a probability $p \in [0, 1]$.
    fn with_prob(&mut self, p: f32) -> bool;
}

impl<R: Rng> RngExt for R {
    fn prob(&mut self) -> f32 {
        self.sample(rand::distributions::Open01)
    }

    fn with_prob(&mut self, p: f32) -> bool {
        p > self.prob()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_generator() {
        let mut rng = rand::thread_rng();
        let v: Vec<f32> = RngExt::prob.into_iter(&mut rng).take(2).collect();
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
