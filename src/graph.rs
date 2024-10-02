use petgraph::{prelude::*, graph::DefaultIx};
use core::f32::consts::FRAC_PI_4;
use crate::rdfs::*;
use super::*;


fn snake_graph(part_count: u8) -> DiGraph<Part, PartEdge> {
    let part = Part {
        extents: Vector::new(1., 1., 1.),
        position: Vector::ZERO,
        rotation: Quaternion::IDENTITY,
    };
    let mut graph = DiGraph::new();
    let index = graph.add_node(part);
    graph.add_edge(index, index,
        PartEdge {
            joint_rotation: Quaternion::IDENTITY,
            rotation: Quaternion::IDENTITY,
            scale: 0.6 * Vector3::ONE,
            iteration_count: part_count,
            op: None,
        });
    graph
}

pub enum ConstructError {
}

#[derive(Clone, Copy, Debug)]
pub struct State {
    pub scale: Vector3,
    pub rotation: Quaternion,
}

impl Default for State {
    fn default() -> Self {
        State {
            scale: Vector3::ONE,
            rotation: Quaternion::IDENTITY,
        }
    }
}

pub fn construct_phenotype<F, G>(graph: &DiGraph<Part,PartEdge>,
                                 root: NodeIndex<DefaultIx>,
                                 state: State,
                                 principle_axis: Vector3,
                                 secondary_axis: Vector3,
                                 make_part: F,
                                 make_joint: G,
                                 commands: &mut Commands
) -> Result<Vec<Entity>, ConstructError>
where F: Fn(&Part, &mut Commands) -> Option<Entity>,
      G: Fn(&JointConfig, &mut Commands) -> Option<Entity>,
{
    let mut rdfs = Rdfs::new(graph, root, |g, _n, e| Permit::EdgeCount(g[e].iteration_count));
    let mut states = vec![];
    let mut parts: Vec<(Part, Entity)> = vec![];
    let mut entities = vec![];
    while let Some(node) = rdfs.next(graph) {
        let depth = rdfs.depth();
        let _ = states.drain(depth..);
        let _ = parts.drain(depth..);
        let mut state: State = *states.last().unwrap_or(&state);
        let mut joint_rotation = None;
        if let Some(edge) = rdfs.path.last() {
            let part_edge = graph[*edge];
            state.rotation *= part_edge.rotation;
            state.scale *= part_edge.scale;
            joint_rotation = Some(part_edge.joint_rotation);
        }
        let mut child: Part = graph[node];
        let child_id = if let Some((parent, parent_id)) = parts.last() {
            // Child object, has a parent.
            let joint_rotation = joint_rotation.map(|q| q * state.rotation).unwrap_or(state.rotation);
            let joint_dir = joint_rotation * principle_axis;
            child.position = parent.position + 10.0 * joint_dir;
            // Position child
            if let Some((p1, p2)) = child.stamp(parent) {
                let child_id = make_part(&child, commands).unwrap();
                make_joint(&JointConfig { parent: *parent_id,
                                            parent_anchor: p1,
                                            child: child_id,
                                            child_anchor: p2,
                                            normal: joint_dir,
                                            tangent: joint_rotation * secondary_axis
                }, commands)
                    .map(|e| entities.push(e));
                child_id
            } else {
                panic!();
            }
        } else {
            // Root object, no parent.
            make_part(&child, commands).unwrap()
        };
        commands.entity(child_id).insert(if depth % 2 == 0 {
            CollisionLayers::new([Layer::PartEven], [Layer::Ground, Layer::PartEven])
        } else {
            CollisionLayers::new([Layer::PartOdd], [Layer::Ground, Layer::PartOdd])
        });
        entities.push(child_id);
        parts.push((child, child_id));
        states.push(state);
    }
    Ok(entities)
}

pub struct JointConfig {
    pub parent: Entity,
    /// Local coordinates
    pub parent_anchor: Vector3,
    pub child: Entity,
    /// Local coordinates
    pub child_anchor: Vector3,
    /// Normal axis
    pub normal: Vector3,
    /// Tangent axis
    pub tangent: Vector3,
}

pub fn spherical_joint(joint: &JointConfig, commands: &mut Commands) -> Entity{
    commands.spawn({
        let mut j = SphericalJoint::new(joint.parent, joint.child)
        // .with_swing_axis(Vector::Y)
        // .with_twist_axis(Vector::X)
            .with_local_anchor_1(joint.parent_anchor)
            .with_local_anchor_2(joint.child_anchor)
        // .with_aligned_axis(Vector::Z)
            .with_swing_limits(-FRAC_PI_4, FRAC_PI_4) // .with_linear_velocity_damping(0.1)
            .with_twist_limits(-FRAC_PI_4, FRAC_PI_4); // .with_linear_velocity_damping(0.1)
        j.swing_axis = joint.tangent; //Vector::Y;
        j.twist_axis = joint.normal; //Vector::X;
        j}).id()
}

pub fn cube_body(child: &Part,
                 color: Color,
                 density: f32,
                 meshes: &mut Assets<Mesh>,
                 materials: &mut Assets<StandardMaterial>,
                 commands: &mut Commands) -> Entity {
    commands
        .spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(child.shape())),
                material: materials.add(StandardMaterial {
                    base_color: color,
                    // emissive: Color::WHITE.into(),
                    ..default()
                }),
                ..default()
            },
            // RigidBody::Static,
            RigidBody::Dynamic,
            Position(child.position()),
            MassPropertiesBundle::new_computed(&child.collider(), child.volume() * density),
            // c,
            child.collider(),
        ))
        .id()
}

