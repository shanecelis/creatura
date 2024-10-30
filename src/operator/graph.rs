//! Generic graph operators
use super::*;
use petgraph::{graph::IndexType, prelude::*, EdgeType};
use std::collections::HashMap;
use weighted_rand::{
    builder::{NewBuilder, WalkerTableBuilder},
    table::WalkerTable,
};

use rand::{seq::IteratorRandom, Rng};

fn nodes_of_subtree<N, E, Ty, Ix>(
    graph: &mut Graph<N, E, Ty, Ix>,
    start: NodeIndex<Ix>,
) -> Vec<NodeIndex<Ix>>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    let mut dfs = Dfs::new(&*graph, start);
    let _ = dfs.next(&*graph); // skip the start node.
    let mut v = vec![];
    while let Some(node) = dfs.next(&*graph) {
        v.push(node);
    }
    v
}

#[allow(dead_code)]
fn prune_subtree<N, E, Ty, Ix>(graph: &mut Graph<N, E, Ty, Ix>, start: NodeIndex<Ix>)
where
    Ty: EdgeType,
    Ix: IndexType,
{
    let mut dfs = Dfs::new(&*graph, start);
    let _ = dfs.next(&*graph); // skip the start node.
    while let Some(node) = dfs.next(&*graph) {
        graph.remove_node(node);
    }
}

fn add_subtree<N, E, Ty, Ix>(
    source: &Graph<N, E, Ty, Ix>,
    source_root: NodeIndex<Ix>,
    dest: &mut Graph<N, E, Ty, Ix>,
    dest_root: NodeIndex<Ix>,
) where
    N: Clone,
    E: Clone,
    Ty: EdgeType,
    Ix: IndexType,
{
    let mut nodes = HashMap::new();
    let mut dfs = Dfs::new(source, source_root);

    let _ = dfs.next(source); // skip the start node.
    nodes.insert(source_root, dest_root);
    while let Some(src_idx) = dfs.next(source) {
        let dst_idx = dest.add_node(source[src_idx].clone());
        nodes.insert(src_idx, dst_idx);
    }
    // Go through all the edges.
    for edge in source.edge_references() {
        if let Some((a, b)) = nodes.get(&edge.source()).zip(nodes.get(&edge.target())) {
            dest.add_edge(*a, *b, edge.weight().clone());
        }
    }
}

/// Cross two random subtrees.
pub fn tree_crosser<N, E, Ty, Ix, R>(
    a: &mut Graph<N, E, Ty, Ix>,
    b: &mut Graph<N, E, Ty, Ix>,
    rng: &mut R,
) -> u32
where
    R: Rng,
    N: Clone,
    E: Clone,
    Ty: EdgeType,
    Ix: IndexType,
{
    if let Some(x) = a.node_indices().choose(rng) {
        if let Some(y) = b.node_indices().choose(rng) {
            cross_subtree(a, x, b, y);
            return 2;
        }
    }
    0
}

fn cross_subtree<N, E, Ty, Ix>(
    source: &mut Graph<N, E, Ty, Ix>,
    source_root: NodeIndex<Ix>,
    dest: &mut Graph<N, E, Ty, Ix>,
    dest_root: NodeIndex<Ix>,
) where
    N: Clone,
    E: Clone,
    Ty: EdgeType,
    Ix: IndexType,
{
    let source_prune = nodes_of_subtree(source, source_root);
    let dest_prune = nodes_of_subtree(dest, dest_root);
    add_subtree(source, source_root, dest, dest_root);
    add_subtree(dest, dest_root, source, source_root);
    for n in source_prune {
        source.remove_node(n);
    }

    for n in dest_prune {
        dest.remove_node(n);
    }
}

/// Remove a random edge if available.
pub fn remove_edge<N, E, Ty, Ix, R>(graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R) -> u32
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    if let Some(edge) = graph.edge_indices().choose(rng) {
        graph.remove_edge(edge);
        1
    } else {
        0
    }
}

/// Add a random node (no edges).
pub fn add_node<N, E, Ty, Ix, R>(
    generator: impl Generator<N, R>,
) -> impl Mutator<Graph<N, E, Ty, Ix>, R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    move |graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R| {
        graph.add_node(generator.generate(rng));
        1
    }
}

/// Generate an edge and add to two random nodes.
pub fn add_edge<N, E, Ty, Ix, R>(
    generator: impl Generator<E, R>,
) -> impl Mutator<Graph<N, E, Ty, Ix>, R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    move |graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R| {
        let a = graph.node_indices().choose(rng);
        let b = graph.node_indices().choose(rng);
        if let Some((i, j)) = a.zip(b) {
            graph.add_edge(i, j, generator.generate(rng));
            1
        } else {
            0
        }
    }
}

/// Add a node and connect it to a distinct random node.
pub fn add_connecting_node<N, E, Ty, Ix, R>(
    node_generator: impl Generator<N, R>,
    edge_generator: impl Generator<E, R>,
) -> impl Mutator<Graph<N, E, Ty, Ix>, R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    move |graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R| {
        if let Some(i) = graph.node_indices().choose(rng) {
            let j = graph.add_node(node_generator.generate(rng));
            graph.add_edge(i, j, edge_generator.generate(rng));
            2
        } else {
            let _ = graph.add_node(node_generator.generate(rng));
            1
        }
    }
}

/// Mutate one random node.
pub fn mutate_one_node<N, E, Ty, Ix, R>(
    mutator: impl Mutator<N, R>,
) -> impl Mutator<Graph<N, E, Ty, Ix>, R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    move |graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R| {
        if let Some(i) = graph.node_indices().choose(rng) {
            let n = graph.node_weight_mut(i).unwrap();
            mutator.mutate(n, rng)
        } else {
            0
        }
    }
}

/// Mutate all nodes.
///
/// To mutate nodes with a 10% certain probability:
///
/// ```ignore
/// mutate_all_nodes(mutator.with_prob(0.1));
/// ```
pub fn mutate_all_nodes<N, E, Ty, Ix, R>(
    mutator: impl Mutator<N, R>,
) -> impl Mutator<Graph<N, E, Ty, Ix>, R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    move |graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R| {
        let mut count = 0u32;
        for node in graph.node_weights_mut() {
            count += mutator.mutate(node, rng);
        }
        count
    }
}

/// Use one of a collection of weighted mutators when called upon.
pub struct WeightedMutator<'a, G, R> {
    mutators: Vec<&'a dyn Mutator<G, R>>,
    table: WalkerTable,
}

impl<'a, G, R> WeightedMutator<'a, G, R> {
    pub fn new<T>(mutators: Vec<&'a dyn Mutator<G, R>>, weights: &[T]) -> Self
    where
        WalkerTableBuilder: NewBuilder<T>,
    {
        let builder = WalkerTableBuilder::new(weights);
        assert_eq!(
            mutators.len(),
            weights.len(),
            "Mutators and weights different lengths."
        );
        Self {
            table: builder.build(),
            mutators,
        }
    }
}

impl<'a, G, R> Mutator<G, R> for WeightedMutator<'a, G, R>
where
    R: Rng,
{
    fn mutate(&self, genome: &mut G, rng: &mut R) -> u32 {
        self.mutators[dbg!(self.table.next_rng(rng))].mutate(genome, rng)
    }
}

/// Mutate one random edge.
pub fn mutate_one_edge<N, E, Ty, Ix, R>(
    mutator: impl Mutator<E, R>,
) -> impl Mutator<Graph<N, E, Ty, Ix>, R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    move |graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R| {
        if let Some(i) = graph.edge_indices().choose(rng) {
            mutator.mutate(&mut graph[i], rng)
        } else {
            0
        }
    }
}

/// Mutate all the edges.
///
/// To mutate all edges with a 10% probability:
///
/// ```ignore
/// mutate_all_edges(mutator.with_prob(0.1));
/// ```
pub fn mutate_all_edges<N, E, Ty, Ix, R>(
    mutator: impl Mutator<E, R>,
) -> impl Mutator<Graph<N, E, Ty, Ix>, R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    move |graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R| {
        let mut count = 0u32;
        for edge in graph.edge_weights_mut() {
            count += mutator.mutate(edge, rng);
        }
        count
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::brain::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn weighted_mutator() {
        let i = 2;
        // for i in 0..100 {
        let mut rng = StdRng::seed_from_u64(i);
        let a = uniform_mutator(0.0, 1.0);
        let b = uniform_mutator(2.0, 10.0);
        let w = WeightedMutator::new(vec![&a, &b], &[0.0, 1.0]);
        let mut v = 0.1;
        assert_eq!(w.mutate(&mut v, &mut rng), 1);
        assert!(v > 2.0, "v {v} > 2.0, seed {i}");
        // }
    }

    #[test]
    fn test_prune_subtree() {
        let mut a = lessin::fig4_3();
        assert_eq!(
            a.node_weights().filter(|w| *w == &Neuron::Muscle).count(),
            3
        );
        assert_eq!(
            a.node_weights()
                .filter(|w| *w == &Neuron::Complement)
                .count(),
            1
        );
        let sin_idx = a
            .node_indices()
            .find(|n| matches!(a[*n], Neuron::Sin { .. }))
            .unwrap();
        prune_subtree(&mut a, sin_idx);
        assert_eq!(
            a.node_weights().filter(|w| *w == &Neuron::Muscle).count(),
            2
        );
        assert_eq!(
            a.node_weights()
                .filter(|w| *w == &Neuron::Complement)
                .count(),
            0
        );
    }

    #[test]
    fn test_add_subtree() {
        let a = lessin::fig4_3();
        let mut b = Graph::new();
        let s = b.add_node(Neuron::Sensor);
        let idx = a
            .node_indices()
            .find(|n| matches!(a[*n], Neuron::Complement))
            .unwrap();
        add_subtree(&a, idx, &mut b, s);
        assert_eq!(b.node_count(), 2);
        assert_eq!(b.edge_count(), 1);
        // assert_eq!(format!("{:?}", Dot::with_config(&b, &[])), "");
    }

    #[test]
    fn test_cross_subtree() {
        let mut a = lessin::fig4_3();
        let mut b = Graph::new();
        let s = b.add_node(Neuron::Sensor);
        let t = b.add_node(Neuron::Mult);
        let _ = b.add_edge(s, t, ());
        let idx = a
            .node_indices()
            .find(|n| matches!(a[*n], Neuron::Complement))
            .unwrap();
        cross_subtree(&mut a, idx, &mut b, s);
        assert_eq!(b.node_count(), 2);
        assert_eq!(b.edge_count(), 1);
        // assert_eq!(format!("{:?}", Dot::with_config(&b, &[])), "");
        // assert_eq!(format!("{:?}", Dot::with_config(&a, &[])), "");
    }
}
