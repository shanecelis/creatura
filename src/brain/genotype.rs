use petgraph::{prelude::*, graph::DefaultIx, visit::IntoNodeReferences, algo::toposort};
use std::cmp::Ordering;

#[derive(Clone, Debug, Copy, PartialEq)]
pub enum Neuron {
    Sensor,
    Muscle,
    Sin { amp: f32, freq: f32, phase: f32 },
    Complement,
    Const(f32),
    Scale(f32),
    Mult,
    /// Divide first input by second
    Div,
    /// Sums inputs
    Sum,
    /// Subtracts first input from second.
    Diff,
    /// Outputs difference between current and previous input, scaled to units of change 0.1 sec with evolvable direction flag
    Deriv { dir: bool },
    /// Outputs 1 if >= .0; otherwise outputs 0.
    Threshold(f32),
    /// If first input is >= .0, output second input; otherwise outputs 0.
    Switch(f32),
    /// Applies an evolvable delay to input signal.
    Delay(u8),
    /// Outputs the absolute difference of input units.
    AbsDiff,
}

/// Order sensors first, inner nodes second, muscles last.
fn order_neurons(a: &Neuron, ai: usize, b: &Neuron, bi: usize) -> Ordering {
    use Neuron::*;
    match (a, b) {
        (&Sensor, &Sensor) => ai.cmp(&bi),
        (&Sensor, _) => Ordering::Less,
        (_, &Sensor) => Ordering::Greater,
        (&Muscle, &Muscle) => bi.cmp(&ai),
        (&Muscle, _) => Ordering::Greater,
        (_, &Muscle) => Ordering::Less,
        (_, _) => Ordering::Equal,
    }
}

impl Neuron {
    fn storage(&self) -> u8 {
        use Neuron::*;
        match self {
            Delay(x) => *x,
            _ => 1,
        }
    }
}

pub struct Brain {
    graph: DiGraph<(Neuron, usize), ()>,
    update: Vec<NodeIndex<DefaultIx>>,
    storage: Vec<f32>,
}

impl Brain {
    fn new(graph: DiGraph<Neuron, ()>) -> Option<Brain> {
        let count: u8 = graph.node_references().map(|(_i, n)| n.storage()).sum();
        let mut g = graph.clone();
        let mut cycles = vec![];
        for edge in g.edge_references() {
            if edge.source() == edge.target() {
                cycles.push(edge.id());
            }
        }
        for edge_id in cycles {
            g.remove_edge(edge_id);
        }
        let mut update = toposort(&g, None).ok()?;
        update.sort_by(|ai, bi| order_neurons(&g[*ai], ai.index(), &g[*bi], bi.index()));
        let mut index = 0;

        let mut brain = graph.map(|_i, n| (*n, 0), |_i, e| *e);

        for i in &update {
            let node = &mut brain[*i];
            node.1 = index;
            index += node.0.storage() as usize;
        }

        Some(Brain {
            graph: brain,
            update,
            storage: vec![0.0; count.into()],
        })
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use Neuron::*;
    #[test]
    fn compact_node_indices() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let b = g.add_node(1);
        let c = g.add_node(2);
        let d = g.add_node(3);
        let e = g.add_node(4);
        assert_eq!(a.index(), 0);
        assert_eq!(b.index(), 1);
        assert_eq!(c.index(), 2);
        assert_eq!(d.index(), 3);
        assert_eq!(e.index(), 4);
    }

    #[test]
    fn topo_sort() {
        let graph = crate::brain::lessin::figure4_3();
        let brain = Brain::new(graph).unwrap();
        let indices: Vec<usize> = brain.update.iter().map(|i| i.index()).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 6, 5, 4, 9, 8, 7]);
        let nodes: Vec<Neuron> = brain.update.iter().map(|i| brain.graph[*i].0).collect();
        assert_eq!(nodes, vec![Sensor, Sensor, Sensor, Sin { amp: 1.0, freq: 1.97, phase: 0.83 },
                               Complement, Mult, Mult, Muscle, Muscle, Muscle]);
    }

}
