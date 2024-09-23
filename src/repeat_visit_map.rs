use petgraph::visit::VisitMap;
use std::{collections::HashMap, hash::Hash};

/// Allow a node to be visited repeatedly.
///
/// By default every node will be visited once, as though
/// each node `x` were set with `repeat_visit_map.set_visits(x, 1)`.
///
/// To repeat the visit two times, `repeat_visit_map.set_visits(x, 2)`.
///
/// To avoid a visit, `repeat_visit_map.set_visits(x, 0)`.
#[derive(Default)]
struct RepeatVisitMap<N> {
    pub counts: HashMap<N, usize>,
}

impl<N> RepeatVisitMap<N>
where
    N: Hash + Eq,
{

    fn set_visits(&mut self, x: N, count: usize) {
        self.counts.insert(x, count);
    }

    fn with_visits(mut self, x: N, count: usize) -> Self {
        self.counts.insert(x, count);
        self
    }
}

impl<N> VisitMap<N> for RepeatVisitMap<N>
where
    N: Hash + Eq,
{
    fn visit(&mut self, x: N) -> bool {
        if let Some(count) = self.counts.get_mut(&x) {
            count.checked_sub(1).map(|c| *count = c).is_some()
        } else {
            self.counts.insert(x, 0);
            true
        }
    }

    fn is_visited(&self, x: &N) -> bool {
        matches!(self.counts.get(x), Some(0))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use petgraph::prelude::*;
    #[test]
    fn normal_visit_map() {
        let mut g = Graph::<usize, ()>::new();
        let a = g.add_node(0);
        g.add_edge(a, a, ());

        let mut dfs = Dfs::new(&g, a);
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn repeat_visit_map_default() {
        let mut g = Graph::<usize, ()>::new();
        let a = g.add_node(0);
        g.add_edge(a, a, ());

        let mut dfs = Dfs {
            stack: vec![a],
            discovered: RepeatVisitMap::default(),
        };
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn repeat_visit_map_zero() {
        let mut g = Graph::<usize, ()>::new();
        let a = g.add_node(0);
        g.add_edge(a, a, ());

        let mut dfs = Dfs {
            stack: vec![a],
            discovered: RepeatVisitMap::default().with_visits(a, 0),
        };
        // assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn repeat_visit_map_one() {
        let mut g = Graph::<usize, ()>::new();
        let a = g.add_node(0);
        g.add_edge(a, a, ());

        let mut dfs = Dfs {
            stack: vec![a],
            discovered: RepeatVisitMap::default().with_visits(a, 1),
        };
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }

    /// I have a simple cylic graph with one node 'a' that has a edge connecting
    /// to itself. I want to be able to do a depth-first traversal, which
    /// normally would result in visiting 'a' once. However, with
    /// `RepeatVisitMap` we can say we want to visit it more than once, say two
    /// times, and petgraph's builtin depth-first search will traverse the edge
    /// and visit the node 'a' twice.
    ///
    /// +--------+
    /// |        |
    /// |   a    |- +
    /// |        |
    /// +--------+  | 2
    ///      ^
    ///      + - - -
    ///
    /// (Unfortunately, I just realized I want this property determined by the
    /// edge not the node. Doh!)
    #[test]
    fn repeat_visit_map_two() {
        let mut g = Graph::<usize, ()>::new();
        let a = g.add_node(0);
        g.add_edge(a, a, ());

        let mut dfs = Dfs {
            stack: vec![a],
            discovered: RepeatVisitMap::default().with_visits(a, 2),
        };
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }

    #[test]
    fn repeat_visit_map_three() {
        let mut g = Graph::<usize, ()>::new();
        let a = g.add_node(0);
        g.add_edge(a, a, ());

        let mut dfs = Dfs {
            stack: vec![a],
            discovered: RepeatVisitMap::default().with_visits(a, 3),
        };
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), Some(a));
        assert_eq!(dfs.next(&g), None);
    }
}
