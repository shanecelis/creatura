use super::*;
use petgraph::{algo::toposort, graph::DefaultIx, prelude::*, visit::IntoNodeReferences};

/// Lessin, D. Evolved Virtual Creatures as Content: Increasing Behavioral and
/// Morphological Complexity. (2015).
pub fn fig4_3() -> DiGraph<Neuron, ()> {
    use Neuron::*;
    let mut g = Graph::new();
    g.add_node(Sensor);
    let s2 = g.add_node(Sensor);
    g.add_node(Sensor);
    let n1 = g.add_node(Sin {
        amp: 1.0,
        freq: 1.97,
        phase: 0.83,
    });
    let n2 = g.add_node(Mult);
    let n3 = g.add_node(Mult);
    let n4 = g.add_node(Complement);
    let m1 = g.add_node(Muscle);
    let m2 = g.add_node(Muscle);
    let m3 = g.add_node(Muscle);
    g.add_edge(s2, n2, ());
    g.add_edge(n1, n2, ());
    g.add_edge(n1, n3, ());
    g.add_edge(n1, m2, ());
    g.add_edge(n1, n4, ());
    g.add_edge(n4, m3, ());
    g.add_edge(n2, m1, ());
    g
}
