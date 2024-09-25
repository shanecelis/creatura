use petgraph::prelude::*;
use petgraph::visit::{EdgeRef, GraphBase, IntoEdgesDirected, GraphRef, IntoNeighbors, IntoNeighborsDirected, VisitMap, Visitable};
use petgraph::{Direction, Incoming};
use std::{
    hash::{Hash, DefaultHasher, Hasher},
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

/// Visit nodes of a graph in a depth-first-search (DFS) emitting nodes in
/// preorder (when they are first discovered).
///
/// The traversal starts at a given node and only traverses nodes reachable
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
/// let mut dfs = Dfs::new(&graph, a);
/// while let Some(nx) = dfs.next(&graph) {
///     // we can access `graph` mutably here still
///     graph[nx] += 1;
/// }
///
/// assert_eq!(graph[a], 1);
/// ```
///
/// **Note:** The algorithm may not behave correctly if nodes are removed
/// during iteration. It may not necessarily visit added nodes or edges.
#[derive(Clone, Debug)]
pub struct Dfs<N, E, VM, VEM> {
    /// The stack of nodes to visit
    pub stack: Vec<(N, Vec<E>)>,
    /// The map of discovered nodes
    pub discovered: VM,
    pub edge_discovered: VEM
}

// impl<N, E, VM, VEM> Default for Dfs<N, E, VM, VEM>
// where
//     VM: Default,
//     // VEM: Default,
// {
//     fn default() -> Self {
//         Dfs {
//             stack: Vec::new(),
//             discovered: VM::default(),
//             edge_discovered: VEM::default(),
//         }
//     }
// }

impl<N, E, VM, VEM> Dfs<N, E, VM, VEM>
where
    N: Copy + PartialEq,
    E: Copy,
    VM: VisitMap<N>,
    VEM: VisitEdgeMap<E>,
{
    /// Create a new **Dfs**, using the graph's visitor map, and put **start**
    /// in the stack of nodes to visit.
    pub fn new<G>(graph: G, start: N, edge_discovered: VEM) -> Self
    where
        G: GraphRef + Visitable<NodeId = N, EdgeId = E, Map = VM>,
    {
        let mut dfs = Dfs::empty(graph, edge_discovered);
        dfs.move_to(start);
        dfs
    }

    /// Create a `Dfs` from a vector and a visit map
    pub fn from_parts(stack: Vec<(N, Vec<E>)>, discovered: VM, edge_discovered: VEM) -> Self {
        Dfs { stack, discovered, edge_discovered }
    }

    /// Clear the visit state
    pub fn reset<G>(&mut self, graph: G)
    where
        G: GraphRef + Visitable<NodeId = N, Map = VM>,
    {
        graph.reset_map(&mut self.discovered);
        self.stack.clear();
    }

    /// Create a new **Dfs** using the graph's visitor map, and no stack.
    pub fn empty<G>(graph: G, edge_discovered: VEM) -> Self
    where
        G: GraphRef + Visitable<NodeId = N, Map = VM>,
    {
        Dfs {
            stack: Vec::new(),
            discovered: graph.visit_map(),
            edge_discovered
        }
    }

    /// Keep the discovered map, but clear the visit stack and restart
    /// the dfs from a particular node.
    pub fn move_to(&mut self, start: N) {
        self.stack.clear();
        self.stack.push((start, Vec::new()));
    }

    /// Return the next node in the dfs, or **None** if the traversal is done.
    pub fn next<G>(&mut self, graph: G) -> Option<N>
    where
        G: IntoEdgesDirected<NodeId = N, EdgeId = E>,
    {
        while let Some((node, mut edges)) = self.stack.pop() {
            if //self.discovered.visit(node) &&
                self.edge_discovered.visit_edge(&edges) {
                for edge in graph.edges_directed(node, Direction::Outgoing) {
                    edges.push(edge.id());
                    let succ = edge.target();
                // for succ in graph.neighbors(node) {
                    if // !self.discovered.is_visited(&succ) &&
                      !self.edge_discovered.is_visited_edge(&edges) {
                        self.stack.push((succ, edges.clone()));
                    }
                    edges.pop();
                }
                // return Some(node);
            }
            return Some(node);
        }
        None
    }
}

pub trait VisitEdgeMap<E> {
    // Required methods
    fn visit_edge(&mut self, e: &[E]) -> bool;
    fn is_visited_edge(&self, e: &[E]) -> bool;
}

pub struct RevisitEdgeMap<E> {//where F: Fn(&[E]) -> Option<u8> {
    pub counts: HashMap<u64, u8>,
    pub func: Box<dyn Fn(&[E]) -> Option<u8>>,
    pub edge: PhantomData<E>,
}

pub trait EdgeVisitable: GraphBase {
    type Map: VisitEdgeMap<Self::EdgeId>;

    fn visit_edge_map(&self, f: impl Fn(&[Self::EdgeId]) -> Option<u8> + 'static) -> Self::Map;
}

impl<G: GraphBase> EdgeVisitable for G where G::EdgeId: Hash + Eq{
    type Map = RevisitEdgeMap<G::EdgeId>;

    fn visit_edge_map(&self, f: impl Fn(&[G::EdgeId]) -> Option<u8> + 'static) -> Self::Map {
        RevisitEdgeMap::new(f)
    }
}

impl<E> RevisitEdgeMap<E>
{
    fn new(f: impl Fn(&[E]) -> Option<u8> + 'static) -> Self {
        Self {
            counts: HashMap::default(),
            func: Box::new(f),
            edge: PhantomData,
        }
    }

    // fn hash_edges(edges: &[E]) -> u64 where E: Hash + Eq {
    //     let mut hash = DefaultHasher::new();
    //     let mut set = HashSet::new();
    //     for edge in &edges[0..edges.len().saturating_sub(1)] {
    //         if set.insert(edge) {
    //             edge.hash(&mut hash);
    //         }
    //     }
    //     // Always has the last edge.
    //     edges.last().map(|e| e.hash(&mut hash));
    //     let h = hash.finish();
    //     eprintln!("hash {h}");
    //     h
    // }
    fn hash_edges(edges: &[E]) -> u64 where E: Hash + Eq {
        let mut hash = DefaultHasher::new();
        let mut set = HashSet::new();
        for edge in edges {
            if set.insert(edge) {
                edge.hash(&mut hash);
            }
        }
        let h = hash.finish();
        eprintln!("hash {h}");
        h
    }
}

impl<E> VisitEdgeMap<E> for RevisitEdgeMap<E> where E: Hash + Eq {
    fn visit_edge(&mut self, e: &[E]) -> bool {
        let key = Self::hash_edges(e);
        if let Some(count) = self.counts.get_mut(&key) {
            count.checked_sub(1).map(|c| *count = c).is_some()
        } else {
            let mut count = (self.func)(e).unwrap_or(1);
            let result = count.checked_sub(1).map(|c| count = c).is_some();
            self.counts.insert(key, count);
            result
        }
    }

    fn is_visited_edge(&self, e: &[E]) -> bool {
        let key = Self::hash_edges(e);
        matches!(self.counts.get(&key), Some(0))
    }
}

pub struct DummyVisit<N> {
    visit: bool,
    node: PhantomData<N>,
}

impl<N> VisitMap<N> for DummyVisit<N>
{
    fn visit(&mut self, _x: N) -> bool {
        self.visit
    }

    fn is_visited(&self, _x: &N) -> bool {
        self.visit
    }
}




#[cfg(test)]
mod test {
    use super::*;
    use petgraph::prelude::Graph;
    #[test]
    fn normal_visit_edge_map() {
        let mut g = Graph::<usize, ()>::new();
        let a = g.add_node(0);
        g.add_edge(a, a, ());
        let edge_map = g.visit_edge_map(|_| Some(1u8));

        let mut dfs = super::Dfs::new(&g, a, edge_map);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn visit_edge_map_0() {
        let mut g = Graph::<usize, ()>::new();
        let a = g.add_node(0);
        g.add_edge(a, a, ());
        let edge_map = g.visit_edge_map(|_| Some(0u8));
        // assert!(edge_map.is_visited_edge(&[a]))

        let mut dfs = super::Dfs::new(&g, a, edge_map);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn visit_edge_map_2() {
        let mut g = Graph::<usize, ()>::new();
        let a = g.add_node(0);
        g.add_edge(a, a, ());
        let edge_map = g.visit_edge_map(|_| Some(2u8));
        // assert!(edge_map.is_visited_edge(&[a]))

        let mut dfs = super::Dfs::new(&g, a, edge_map);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }
}
