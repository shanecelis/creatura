use petgraph::{visit::{VisitMap, GraphRef, GraphBase, Visitable, IntoNeighbors, IntoEdgesDirected, EdgeIndexable}, EdgeType, graph::IndexType, prelude::*};

use std::{
    hash::{Hash, DefaultHasher, Hasher},
    collections::{HashMap, HashSet, VecDeque},
    marker::PhantomData,
    ops::Index,
};
/// A breadth first search (BFS) of a graph.
///
/// The traversal starts at a given edge and only traverses nodes reachable
/// from it.
///
/// `Dfs` is not recursive.
///
/// `Dfs` does not itself borrow the graph, and because of this you can run
/// a traversal over a graph while still retaining mutable access to it, if you
/// use it like the following example:
///
/// ```
/// use petgraph::Graph;
/// use petgraph::visit::Dfs;
///
/// let mut graph = Graph::<_,()>::new();
/// let a = graph.add_node(0);
///
/// let mut bfs = Dfs::new(&graph, a);
/// while let Some(nx) = bfs.next(&graph) {
///     // we can access `graph` mutably here still
///     graph[nx] += 1;
/// }
///
/// assert_eq!(graph[a], 1);
/// ```
///
/// **Note:** The algorithm may not behave correctly if nodes are removed
/// during iteration. It may not necessarily visit added nodes or edges.
#[derive(Clone)]
pub struct Dfs<E,N, F,T> {
    /// The queue of nodes to visit
    pub stack: Vec<(E, usize)>,
    /// The map of discovered nodes
    // pub discovered: VM,
    pub edge_traverses: F,
    pub edge_target: T,
    pub node: PhantomData<N>
}

// impl<E, VM> Default for Dfs<E, VM>
// where
//     VM: Default,
// {
//     fn default() -> Self {
//         Dfs {
//             stack: VecDeque::new(),
//             discovered: VM::default(),
//         }
//     }
// }

impl<E, N, F, T> Dfs<E, N, F, T>
where
    E: Copy,
    F: Fn(E) -> Option<u8>,
    T: Fn(E) -> N,
{
    /// Create a new **Dfs**, using the graph's visitor map, and put **start**
    /// in the stack of nodes to visit.
    pub fn new<G>(graph: G, start: E, edge_traverses: F, edge_target: T) -> Self
        where G: GraphRef + Visitable<NodeId = N, EdgeId = E>,

    {
        // let root_edge = graph.edges_directed(start.into(), Direction::Incoming).next().unwrap();
        // let discovered = graph.visit_map();
        // discovered.visit(start);
        Dfs { stack: vec![(start, 0)], edge_traverses, edge_target, node: PhantomData }
    }

    /// Return the next edge in the bfs, or **None** if the traversal is done.
    pub fn next<G>(&mut self, graph: G) -> Option<E>
    where
        G: GraphRef + Visitable<NodeId =N, EdgeId = E> + IntoEdgesDirected
        // E: EdgeRef
    //std::ops::Index<EdgeIndex<petgraph::stable_graph::DefaultIx>>
    {
        if let Some((edge, depth)) = self.stack.pop() {


            // let edge = &graph[EdgeIndexable::to_index(&graph, edge)];
            // if let Some(edge) = graph.next_edge(edge, Direction::Outgoing) {
                for succ in graph.edges_directed((self.edge_target)(edge), Direction::Outgoing) {
                // for succ in graph.neighbors(edge) {
                    // if self.discovered.visit(succ) {
                        self.stack.push((succ.id(), depth + 1));
                    // }
                }
            // }

            return Some(edge);
        }
        None
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn node1() {
        let mut g = Graph::<usize, ()>::new();
        let a = g.add_node(0);
        let e = g.add_edge(a, a, ());
        let mut dfs = Dfs::new(&g, e, |_| Some(1), |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap());
        assert_eq!(dfs.next(&g), Some(e));
        assert_eq!(dfs.next(&g), Some(e));
    }

}
