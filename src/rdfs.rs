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

/// An recurrent depth first search (RDFS) of a graph.
///
/// The traversal starts at the edges of a given node and only traverses nodes
/// reachable from it. It returns the current edge rather than the current node.
///
/// Instead of keeping track of visited nodes, it keeps track of the current
/// edge path. It takes a closure `edge_permits` that declares how many
/// traversals of the edge are permitted for any given path. If `edge_permits =
/// |_, _| 1` it behaves similarly to the conventional DFS.
///
/// If `edge_permits = |_, _| 0` it will return `None` on the call to `next()`
/// since it will not traverse any edges.
///
/// The interesting case is where `edge_permits = |_, _| 2` now edges will be
/// traversed at most two times on any path. So for a simple graph with one node
/// and a self-edge, the RDFS will return that edge twice. This allows one to
/// intentionally traverse recurrent graphs like those present in
/// [Sims](https://www.karlsims.com/papers/siggraph94.pdf)' work without
/// traversing infinitely.
///
/// `Rdfs` is not recursive.
///
/// `Rdfs` does not itself borrow the graph, and because of this you can run
/// a traversal over a graph while still retaining mutable access to it, if you
/// use it like the following example:
///
/// ```
/// use petgraph::Graph;
/// use muscley_wusaley::Rdfs;
///
/// let mut graph = Graph::<isize,isize>::new();
/// let a = graph.add_node(0);
/// let e = graph.add_edge(a, a, 0);
/// let mut cdfs = Rdfs::new(&graph, a, |_, _| 2);
///
/// while let Some(n) = cdfs.next(&graph) {
///     // We can access `graph` mutably here still.
///     graph[n] += 1;
/// }
///
/// assert_eq!(graph[a], 3);
/// ```
///
/// **Note:** The algorithm may not behave correctly if nodes are removed
/// during iteration. It may not necessarily visit added nodes or edges.
#[derive(Clone)]
pub struct Rdfs<E, N, G, F> {
    /// The stack of edges to visit
    pub stack: Vec<(E, usize)>,
    /// The path of edge for last visit
    pub path: Vec<E>,
    /// Closure that returns the number of traversals permitted for an edge for
    /// any path.
    pub edge_permits: F,
    pub start: Option<N>,
    pub node: PhantomData<G>,
}

impl<E, N, G, F> Rdfs<E, N, G, F>
where
    E: Copy + Eq,
    N: Copy + Eq,
    F: Fn(&G, E) -> u8,
    G: GraphBase,
    for<'a> &'a G: Visitable<NodeId = N, EdgeId = E> + IntoEdgesDirected + EdgeEndpoints<N, E>,
{
    /// Create a new `Rdfs`, and put `start`'s edges on the stack of edges to visit.
    pub fn new(graph: &G, start: N, edge_permits: F) -> Self {
        // let mut stack = vec![];
        // for succ in graph.edges_directed(start, Direction::Outgoing) {
        //     if edge_permits(graph, succ.id()) > 0 {
        //         stack.push((succ.id(), 0));
        //     }
        // }
        Rdfs {
            stack: Vec::new(),
            start: Some(start),
            path: Vec::new(),
            edge_permits,
            node: PhantomData,
        }
    }

    /// Return the next edge in the cdfs, or `None` if the traversal is done.
    pub fn next(&mut self, graph: &G) -> Option<N> {
        if let Some(start) = self.start.take() {
            for succ in graph.edges_directed(start, Direction::Outgoing) {
                if (self.edge_permits)(graph, succ.id()) > 0 {
                    self.stack.push((succ.id(), 0));
                }
            }
            return Some(start);
        }
        if let Some((edge, depth)) = self.stack.pop() {
            let _ = self.path.drain(depth..);
            self.path.push(edge);
            if let Some((_, target)) = graph.edge_endpoints(edge) {
                for succ in graph.edges_directed(target, Direction::Outgoing) {
                    let succ = succ.id();
                    let allowance = (self.edge_permits)(graph, succ);
                    if (self.path.iter().filter(|e| succ == **e).count() as u8)
                        <= allowance.saturating_sub(1)
                    {
                        self.stack.push((succ, depth + 1));
                    }
                }
                Some(target)
            } else {
                panic!("edge has no end points");
            }
        } else {
            self.path.clear();
            None
        }
    }

    /// Return the depth of the last edge given by `next()`.
    pub fn depth(&self) -> usize {
        self.path.len()
    }
}

/// Get the nodes of an edge.
///
/// NOTE: Functionality is already present in petgraph but not generic.
pub trait EdgeEndpoints<N, E> {
    /// Return the `(source, target)` of an edge if present.
    fn edge_endpoints(&self, edge_id: E) -> Option<(N, N)>;
}

impl<N, E, Ty, Ix> EdgeEndpoints<NodeIndex<Ix>, EdgeIndex<Ix>> for &Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn edge_endpoints(&self, edge_id: EdgeIndex<Ix>) -> Option<(NodeIndex<Ix>, NodeIndex<Ix>)> {
        Graph::edge_endpoints(self, edge_id)
    }
}


pub fn unfurl<N, E, Ty, Ix, N2, E2>(graph: &Graph<N, E, Ty, Ix>, start: NodeIndex<Ix>, permits: impl Fn(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> u8, mut node_fn: impl FnMut(&N) -> N2, mut edge_fn: impl FnMut(&E) -> E2) -> Graph<N2, E2, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    let mut cdfs = Rdfs::new(graph, start, permits);
    let mut unfurled = Graph::<N2, E2, Ty, Ix>::default();
    let mut new_nodes: HashMap<u64, NodeIndex<Ix>> = HashMap::new();

    let mut get_or_insert_node = |node: NodeIndex<Ix>, hash: u64, unfurled: &mut Graph<N2, E2, Ty, Ix>| {
        if let Some(node) = new_nodes.get(&hash) {
            *node
        } else {
            let n = unfurled.add_node(node_fn(&graph[node]));
            new_nodes.insert(hash, n);
            n
        }
    };
    while let Some(node) = cdfs.next(graph) {
        if let Some(&edge) = cdfs.path.last() {
        let depth = cdfs.depth().saturating_sub(1);
        if let Some((source, target)) = graph.edge_endpoints(edge) {
            let mut hash = DefaultHasher::new();
            for i in 0..depth {
                cdfs.path[i].hash(&mut hash);
            }
            // Copy the source, Luke!
            // let source = get_or_insert_node((source, cdfs.path.iter().nth(depth.saturating_sub(1)).copied(), depth), &mut unfurled);
            let source = get_or_insert_node(source, hash.finish(), &mut unfurled);
            cdfs.path.last().unwrap().hash(&mut hash);
            let target = get_or_insert_node(target, hash.finish(), &mut unfurled);
            unfurled.add_edge(source, target, edge_fn(&graph[edge]));
            //
        }
        }
    }
    unfurled
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn check_hash_assumptions() {
        let mut hash = DefaultHasher::new();
        0.hash(&mut hash);
        1.hash(&mut hash);
        let a = hash.finish();
        let mut hash = DefaultHasher::new();
        0.hash(&mut hash);
        1.hash(&mut hash);
        let b = hash.finish();
        assert_eq!(a, b);
    }

    #[test]
    fn node1() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let mut dfs = Rdfs::new(&g, a, |_, _| 1);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_cycle0() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let _ = g.add_edge(a, a, ());
        let mut dfs = Rdfs::new(&g, a, |_, _| 0);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_unfurl() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let _ = g.add_edge(a, a, ());
        let g2 = unfurl(&g, a, |_, _| 2, |n| n.clone(), |e| e.clone());
        assert_eq!(g2.node_count(), 3);
        assert_eq!(g2.edge_count(), 2);
    }

    #[test]
    fn node_cycle1() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let e1 = g.add_edge(a, a, ());
        let mut dfs = Rdfs::new(&g, a, |_, _| 1);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_cycle2() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let e1 = g.add_edge(a, a, ());
        let mut dfs = Rdfs::new(&g, a, |_, _| 2);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_cycle3() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let e1 = g.add_edge(a, a, ());
        let mut dfs = Rdfs::new(&g, a, |_, _| 3);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), Some(a));
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
        let mut dfs = Rdfs::new(&g, a, |_, _| 1);
        // assert_eq!(dfs.next(&g), Some(e0));
        // assert_eq!(dfs.path, vec![e0]);
        //
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.stack, vec![(e3, 0), (e0, 0)]);
        assert_eq!(dfs.path, vec![]);

        assert_eq!(dfs.next(&g), Some(b));
        assert_eq!(dfs.stack, vec![(e3, 0), (e2, 1), (e1, 1)]);
        assert_eq!(dfs.path, vec![e0]);
        assert_eq!(dfs.next(&g), Some(c));
        assert_eq!(dfs.path, vec![e0, e1]);
        assert_eq!(dfs.next(&g), Some(d));
        assert_eq!(dfs.path, vec![e0, e2]);
        assert_eq!(dfs.next(&g), Some(e));
        assert_eq!(dfs.path, vec![e3]);
        assert_eq!(dfs.next(&g), None);
        assert_eq!(dfs.path, vec![]);
        assert_eq!(dfs.next(&g), None);
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
        let mut dfs = Rdfs::new(&g, a, |_, _| 1);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.stack, vec![(e1, 0), (e0, 0)]);
        assert_eq!(dfs.path, vec![]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.stack, vec![(e1, 0), (e1, 1)], "wrong path");
        assert_eq!(dfs.path, vec![e0]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e0, e1]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e1]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e1, e0]);
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
        let mut dfs = Rdfs::new(&g, a, |_, _| 2);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e0]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e0, e0]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e0, e0, e1]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e0, e0, e1, e1]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e0, e1]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e0, e1, e0]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e0, e1, e0, e1]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e0, e1, e1]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e0, e1, e1, e0]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e1]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e1, e0]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e1, e0, e0]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e1, e0, e0, e1]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e1, e0, e1]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e1, e0, e1, e0]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e1, e1]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.path, vec![e1, e1, e0]);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn node_tree_dot() {
        use petgraph::dot::{Dot, Config};
        let mut g = Graph::<isize, isize>::new();
        let a = g.add_node(0);
        let e0 = g.add_edge(a, a, 0);
        let e1 = g.add_edge(a, a, 1);
        let tree = unfurl(&g, a, |_, _| 2, |n| *n, |e| *e);
        eprintln!("{:?}", Dot::with_config(&tree, &[Config::EdgeNoLabel]));
    }

    #[test]
    fn dfs_doc_test() {
        use petgraph::visit::Dfs;
        use petgraph::Graph;

        let mut graph = Graph::<_, ()>::new();
        let a = graph.add_node(0);

        let mut dfs = Dfs::new(&graph, a);
        while let Some(nx) = dfs.next(&graph) {
            // we can access `graph` mutably here still
            graph[nx] += 1;
        }

        assert_eq!(graph[a], 1);
    }

    #[test]
    fn cdfs_doc_test() {
        use crate::Rdfs;
        use petgraph::Graph;

        let mut graph = Graph::<isize, isize>::new();
        let a = graph.add_node(0);
        let x = graph.add_edge(a, a, 0);

        let mut cdfs = Rdfs::new(&graph, a, |_, _| 2);
        while let Some(n) = cdfs.next(&graph) {
            graph[n] += 1;
        }
        assert_eq!(graph[a], 3);
    }
}
