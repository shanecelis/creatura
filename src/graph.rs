use petgraph::prelude::*;
use super::*;

#[derive(Clone, Debug, Copy)]
struct Edge {
    // joint_type:
    parent_attach: Vector3,
    child_scale: Vector3,
    iteration_count: u8,
}

fn snake_graph(part_count: u8) -> DiGraph<Part, Edge> {
    let part = Part {
        extents: Vector::new(1., 1., 1.),
        position: Vector::Y,
        rotation: Quaternion::IDENTITY,
    };
    let mut graph = DiGraph::new();
    let index = graph.add_node(part);
    graph.add_edge(index, index,
        Edge {
            parent_attach: Vector3::X,
            child_scale: 0.6 * Vector3::ONE,
            iteration_count: part_count
        });
    graph
}

// fn unfurl_graph(graph: DiGraph<Part, Edge>) -> DiGraph<Part, Edge> {
// }
