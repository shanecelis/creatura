use petgraph::visit::VisitMap;
use std::{collections::HashMap, hash::Hash};

#[derive(Default)]
struct RepeatVisitMap<N> {
    counts: HashMap<N, usize>,
}

impl<N> RepeatVisitMap<N>
where
    N: Hash + Eq,
{
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

    #[test]
    fn repeat_visit_map() {
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
    fn repeat_visit_map_3() {
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
