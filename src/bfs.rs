use petgraph::{
    graph::IndexType,
    prelude::*,
    visit::{
        EdgeIndexable, GraphBase, GraphRef, IntoEdgesDirected, IntoNeighbors, VisitMap, Visitable,
    },
    EdgeType,
};

use std::{
    collections::{HashMap, HashSet, VecDeque},
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
    ops::Index,
};
/// A depth first search (DFS) of a graph.
///
/// The traversal starts at the edges of a given node and only traverses nodes
/// reachable from it.
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
pub struct Dfs<E, N, F, T> {
    /// The stack of edges to visit
    pub stack: Vec<(E, usize)>,
    /// The path of edge for last visit
    pub path: Vec<E>,
    /// Function that returns the number of traversals permitted for edge
    pub edge_traverses: F,
    /// Closure that returns the target of an edge
    ///
    /// TODO: Consider exposing this as a trait in petgraph so it could be
    /// avoided.
    pub edge_target: T,
    pub node: PhantomData<N>,
}

impl<E, N, F, T> Dfs<E, N, F, T>
where
    E: Copy + Eq,
    F: Fn(E) -> u8,
    T: Fn(E) -> N,
{
    /// Create a new `Dfs`, and put `start`'s edges onthe stack of edges to visit.
    pub fn new<G>(graph: G, start: N, edge_traverses: F, edge_target: T) -> Self
    where
        G: GraphRef + Visitable<NodeId = N, EdgeId = E> + IntoEdgesDirected,
    {
        let mut stack = vec![];
        for succ in graph.edges_directed(start, Direction::Outgoing) {
            if edge_traverses(succ.id()) > 0 {
                stack.push((succ.id(), 0));
            }
        }
        Dfs {
            stack,
            path: Vec::new(),
            edge_traverses,
            edge_target,
            node: PhantomData,
        }
    }

    /// Return the next edge in the dfs, or **None** if the traversal is done.
    pub fn next<G>(&mut self, graph: G) -> Option<E>
    where
        G: GraphRef + Visitable<NodeId = N, EdgeId = E> + IntoEdgesDirected,
    {
        if let Some((edge, depth)) = self.stack.pop() {
            let _ = self.path.drain(depth..);
            self.path.push(edge);
            for succ in graph.edges_directed((self.edge_target)(edge), Direction::Outgoing) {
                let succ = succ.id();
                let allowance = (self.edge_traverses)(succ);
                if (self.path.iter().filter(|e| succ == **e).count() as u8)
                    <= allowance.saturating_sub(1)
                {
                    self.stack.push((succ, depth + 1));
                }
            }
            return Some(edge);
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
        let a = g.add_node(0);
        let mut dfs = Dfs::new(
            &g,
            a,
            |_| 1,
            |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap(),
        );
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_cycle0() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let _ = g.add_edge(a, a, ());
        let mut dfs = Dfs::new(
            &g,
            a,
            |_| 0,
            |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap(),
        );
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_cycle1() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let e1 = g.add_edge(a, a, ());
        let mut dfs = Dfs::new(
            &g,
            a,
            |_| 1,
            |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap(),
        );
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_cycle2() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let e1 = g.add_edge(a, a, ());
        let mut dfs = Dfs::new(
            &g,
            a,
            |_| 2,
            |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap(),
        );
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_cycle3() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let e1 = g.add_edge(a, a, ());
        let mut dfs = Dfs::new(
            &g,
            a,
            |_| 3,
            |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap(),
        );
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_root() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let b = g.add_node(1);
        let c = g.add_node(2);
        let d = g.add_node(3);
        let e = g.add_node(4);
        let e0 = g.add_edge(a, b, ());
        let e1 = g.add_edge(b, c, ());
        let e2 = g.add_edge(b, d, ());
        let e3 = g.add_edge(a, e, ());
        let mut dfs = Dfs::new(
            &g,
            a,
            |_| 1,
            |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap(),
        );
        // assert_eq!(dfs.next(&g), Some(e0));
        // assert_eq!(dfs.path, vec![e0]);
        //
        assert_eq!(dfs.stack, vec![(e3, 0), (e0, 0)]);
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.path, vec![e0]);

        assert_eq!(dfs.stack, vec![(e3, 0), (e2, 1), (e1, 1)]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e0, e1]);
        assert_eq!(dfs.next(&g), Some(e2));
        assert_eq!(dfs.path, vec![e0, e2]);
        assert_eq!(dfs.next(&g), Some(e3));
        assert_eq!(dfs.path, vec![e3]);
        assert_eq!(dfs.next(&g), None);
        assert_eq!(dfs.path, vec![]);
    }

    #[test]
    fn node_tree1() {
        let mut g = Graph::<isize, isize>::new();
        let a = g.add_node(0);
        let e0 = g.add_edge(a, a, 0);
        let e1 = g.add_edge(a, a, 1);
        assert_eq!(g[e0], 0);
        assert_eq!(g[e1], 1);
        assert_ne!(e0, e1);
        let mut dfs = Dfs::new(
            &g,
            a,
            |_| 1,
            |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap(),
        );
        assert_eq!(dfs.stack, vec![(e1, 0), (e0, 0)]);
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.stack, vec![(e1, 0), (e1, 1)], "wrong path");
        assert_eq!(dfs.path, vec![e0]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e0, e1]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e1]);
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.path, vec![e1, e0]);
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_tree2() {
        let mut g = Graph::<isize, isize>::new();
        let a = g.add_node(0);
        let e0 = g.add_edge(a, a, 0);
        let e1 = g.add_edge(a, a, 1);
        assert_eq!(g[e0], 0);
        assert_eq!(g[e1], 1);
        assert_ne!(e0, e1);
        let mut dfs = Dfs::new(
            &g,
            a,
            |_| 2,
            |e| g.edge_endpoints(e).map(|(_, target)| target).unwrap(),
        );
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.path, vec![e0]);
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.path, vec![e0, e0]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e0, e0, e1]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e0, e0, e1, e1]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e0, e1]);
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.path, vec![e0, e1, e0]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e0, e1, e0, e1]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e0, e1, e1]);
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.path, vec![e0, e1, e1, e0]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e1]);
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.path, vec![e1, e0]);
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.path, vec![e1, e0, e0]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e1, e0, e0, e1]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e1, e0, e1]);
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.path, vec![e1, e0, e1, e0]);
        assert_eq!(dfs.next(&g), Some(e1));
        assert_eq!(dfs.path, vec![e1, e1]);
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.path, vec![e1, e1, e0]);
        assert_eq!(dfs.next(&g), Some(e0));
        assert_eq!(dfs.next(&g), None);
    }
}
