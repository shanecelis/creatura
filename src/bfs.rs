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
    pub path: Vec<E>,
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
    E: Copy + Eq,
    F: Fn(E) -> Option<u8>,
    T: Fn(E) -> N,
{
    /// Create a new **Dfs**, using the graph's visitor map, and put **start**
    /// in the stack of nodes to visit.
    pub fn new<G>(graph: G, start: E, edge_traverses: F, edge_target: T) -> Self
        where G: GraphRef + Visitable<NodeId = N, EdgeId = E>,

    {
        Dfs { stack: vec![(start, 0)], path: Vec::new(), edge_traverses, edge_target, node: PhantomData }
    }

    /// Return the next edge in the bfs, or **None** if the traversal is done.
    pub fn next<G>(&mut self, graph: G) -> Option<E>
    where
        G: GraphRef + Visitable<NodeId = N, EdgeId = E> + IntoEdgesDirected
    {
        if let Some((edge, depth)) = self.stack.pop() {
            let _ = self.path.drain(depth..);
            self.path.push(edge);
            let allowance = (self.edge_traverses)(edge).unwrap_or(1);
            if self.path.iter().filter(|e| edge == **e).count() as u8 <= allowance {
                for succ in graph.edges_directed((self.edge_target)(edge), Direction::Outgoing) {
                    self.stack.push((succ.id(), depth + 1));
                }
                return Some(edge);
            }
        }
        self.path.clear();
        None
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn node1() {
        let mut g = Graph::<isize, ()>::new();
        let r = g.add_node(-1);
        let a = g.add_node(0);
        let e = g.add_edge(r, a, ());
        let mut dfs = Dfs::new(&g, e, |_| Some(1), |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap());
        assert_eq!(dfs.next(&g), Some(e));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_cycle0() {
        let mut g = Graph::<isize, ()>::new();
        let r = g.add_node(-1);
        let a = g.add_node(0);
        let e0 = g.add_edge(r, a, ());
        let e1 = g.add_edge(a, a, ());
        let mut dfs = Dfs::new(&g, e0, |_| Some(0), |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap());
        // assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_cycle1() {
        let mut g = Graph::<isize, ()>::new();
        let r = g.add_node(-1);
        let a = g.add_node(0);
        let e0 = g.add_edge(r, a, ());
        let e1 = g.add_edge(a, a, ());
        let mut dfs = Dfs::new(&g, e0, |_| Some(1), |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap());
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_cycle2() {
        let mut g = Graph::<isize, ()>::new();
        let r = g.add_node(-1);
        let a = g.add_node(0);
        let e0 = g.add_edge(r, a, ());
        let e1 = g.add_edge(a, a, ());
        let mut dfs = Dfs::new(&g, e0, |_| Some(2), |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap());
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_root() {
        let mut g = Graph::<isize, ()>::new();
        let r = g.add_node(-1);
        let a = g.add_node(0);
        let b = g.add_node(1);
        let c = g.add_node(2);
        let d = g.add_node(3);
        let e = g.add_node(4);
        let e0 = g.add_edge(r, a, ());
        let e1 = g.add_edge(a, b, ());
        let e2 = g.add_edge(b, c, ());
        let e3 = g.add_edge(b, d, ());
        let e4 = g.add_edge(a, e, ());
        let mut dfs = Dfs::new(&g, e0, |_| Some(1), |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap());
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.path, vec![e0]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e0, e1]);
        assert_eq!(dfs.next(&g), Some(e2));
        assert_eq!(dfs.path, vec![e0, e1, e2]);
        assert_eq!(dfs.next(&g), Some(e3));
        assert_eq!(dfs.path, vec![e0, e1, e3]);
        assert_eq!(dfs.next(&g), Some(e4));
        assert_eq!(dfs.path, vec![e0, e4]);
        assert_eq!(dfs.next(&g), None);
        assert_eq!(dfs.path, vec![]);
    }

}
