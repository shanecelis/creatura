use petgraph::prelude::*;
use super::*;

#[derive(Clone, Debug, Copy)]
struct GraphEdge {
    // joint_type:
    parent_attach: Vector3,
    child_attach: Vector3,
    child_scale: Vector3,
    iteration_count: u8,
}

fn snake_graph(part_count: u8) -> DiGraph<Part, GraphEdge> {
    let part = Part {
        extents: Vector::new(1., 1., 1.),
        position: Vector::ZERO,
        rotation: Quaternion::IDENTITY,
    };
    let mut graph = DiGraph::new();
    let index = graph.add_node(part);
    graph.add_edge(index, index,
        GraphEdge {
            parent_attach: Vector3::X,
            child_attach: -Vector3::X,
            child_scale: 0.6 * Vector3::ONE,
            iteration_count: part_count
        });
    graph
}

fn unfurl_graph(graph: &DiGraph<Part, GraphEdge>) -> DiGraph<Part, ()> {


}

pub fn make_graph(graph: &DiGraph<Part,GraphEdge>,
                  root: petgraph::prelude::NodeIndex<u32>,
                  mut meshes: ResMut<Assets<Mesh>>,
                  pbr: PbrBundle,
                  mut commands: Commands) {
    let mut graph = graph.map(|i, p| PartData { part: p.clone(), id: None, parity: None }, |i, e| e);
    let mut dfs = Dfs::new(&graph, root);
    let density = 1.0;
    while let Some(nx) = dfs.next(&graph) {
        let node = &mut graph[nx];
        let child = node.part;

        if node.parity == None {
            let mut edges = graph.neighbors_directed(nx, Incoming);
            while let Some(ex) = edges.next() {
                node.parity = Some(graph[ex].parity.unwrap().next());
            }
        }

        if node.id == None {
            node.id = Some(commands
            .spawn((
                PbrBundle {
                    mesh: meshes.add(Mesh::from(child.shape())),
                    ..pbr
                },
                // RigidBody::Static,
                RigidBody::Dynamic,
                Position(child.position()),
                MassPropertiesBundle::new_computed(&child.collider(), child.volume() * density),
                // c,
                child.collider(),
                if node.parity == Some(PartParity::Even) {
                    CollisionLayers::new([Layer::PartEven], [Layer::Ground, Layer::PartEven])
                } else {
                    CollisionLayers::new([Layer::PartOdd], [Layer::Ground, Layer::PartOdd])
                },
            ))
            .id());


        }

        let mut edges = graph.edges_directed(nx, Incoming);
        while let Some(ex) = edges.next() {
        }
    }
}
