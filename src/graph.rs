use super::*;
use crate::{rdfs::*, operator::*};
use core::f32::consts::FRAC_PI_4;
use petgraph::{graph::{DefaultIx, IndexType}, prelude::*, EdgeType};
use rand::Rng;

fn mutate_nodes<N,E,Ty,Ix,R>(mutator: impl Mutator<N, R>, mutation_rate: f32)
                             -> impl Mutator<Graph<N,E,Ty,Ix>,R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng {
    move |graph: &mut Graph<N,E,Ty,Ix>, rng: &mut R| {
        let mut count = 0u32;
        for node in graph.node_weights_mut() {
            if mutation_rate < rnd_prob(rng) {
                mutator.mutate(node, rng);
                count += 1;
            }
        }
        count
    }
}

fn mutate_edges<N,E,Ty,Ix,R>(mutator: impl Mutator<E, R>, mutation_rate: f32)
                             -> impl Mutator<Graph<N,E,Ty,Ix>,R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng {
    move |graph: &mut Graph<N,E,Ty,Ix>, rng: &mut R| {
        let mut count = 0u32;
        for edge in graph.edge_weights_mut() {
            if mutation_rate < rnd_prob(rng) {
                count += mutator.mutate(edge, rng);
            }
        }
        count
    }
}

pub fn snake_graph(part_count: u8) -> (DiGraph<Part, PartEdge>, NodeIndex<DefaultIx>) {
    let part = Part {
        extents: Vector::new(1., 1., 1.),
        position: Vector::ZERO,
        rotation: Quaternion::IDENTITY,
    };
    let mut graph = DiGraph::new();
    let index = graph.add_node(part);
    graph.add_edge(
        index,
        index,
        PartEdge {
            joint_rotation: Quaternion::IDENTITY,
            rotation: Quaternion::IDENTITY,
            scale: 0.6 * Vector3::ONE,
            iteration_count: part_count,
            op: None,
            muscles: vec![MuscleGene {
                parent: Quaternion::from_axis_angle(Vec3::Z, FRAC_PI_4),
                child: Quaternion::IDENTITY,
                max_strength: 1.0,
            }],
        },
    );
    (graph, index)
}

#[derive(Clone, Copy, Debug)]
pub enum ConstructError {}

#[derive(Clone, Copy, Debug)]
pub struct BuildState {
    pub scale: Vector3,
    pub rotation: Quaternion,
}

impl Default for BuildState {
    fn default() -> Self {
        BuildState {
            scale: Vector3::ONE,
            rotation: Quaternion::IDENTITY,
        }
    }
}

#[derive(Debug)]
pub struct MuscleSite<'a> {
    pub id: Entity,
    pub anchor_local: Vector3,
    pub part: &'a Part,
}

#[allow(clippy::too_many_arguments)]
pub fn construct_phenotype<F, G, H>(
    graph: &DiGraph<Part, PartEdge>,
    root: NodeIndex<DefaultIx>,
    state: BuildState,
    position: Vector3,
    principle_axis: Vector3,
    secondary_axis: Vector3,
    commands: &mut Commands,
    mut make_part: F,
    mut make_joint: G,
    mut make_muscle: H,
) -> Result<Vec<Entity>, ConstructError>
where
    F: FnMut(&Part, &mut Commands) -> Option<Entity>,
    G: FnMut(&JointConfig, &mut Commands) -> Option<Entity>,
    H: FnMut(&MuscleSite, &MuscleSite, &mut Commands) -> Option<Entity>,
{
    let mut rdfs = Rdfs::new(graph, root, |g, _n, e| {
        Permit::EdgeCount(g[e].iteration_count)
    });
    let mut states = vec![];
    let mut parts: Vec<(Part, Entity)> = vec![];
    let mut entities = vec![];
    while let Some(node) = rdfs.next(graph) {
        let depth = rdfs.depth();
        let _ = states.drain(depth..);
        let _ = parts.drain(depth..);
        let mut state: BuildState = *states.last().unwrap_or(&state);
        let mut joint_rotation = None;
        if let Some(edge) = rdfs.path.last() {
            let part_edge = &graph[*edge];
            state.rotation *= part_edge.rotation;
            state.scale *= part_edge.scale;
            joint_rotation = Some(part_edge.joint_rotation);
        }
        let mut child: Part = graph[node];
        let child_id = if let Some((parent, parent_id)) = parts.last() {
            // Child object, has a parent.
            let joint_rotation = joint_rotation
                .map(|q| q * state.rotation)
                .unwrap_or(state.rotation);
            let joint_dir = joint_rotation * principle_axis;
            child.position = parent.position + 10.0 * joint_dir;
            child.extents = state.scale * graph[node].extents;
            // Position child
            if let Some((p1, p2)) = child.stamp(parent) {
                let child_id = make_part(&child, commands).unwrap();
                if let Some(e) = make_joint(
                    &JointConfig {
                        parent: *parent_id,
                        parent_anchor: p1,
                        child: child_id,
                        child_anchor: p2,
                        normal: joint_dir,
                        tangent: joint_rotation * secondary_axis,
                    },
                    commands,
                ) {
                    entities.push(e);
                }
                // Make muscles
                if let Some(edge) = rdfs.path.last() {
                    for muscle in &graph[*edge].muscles {
                        let r = joint_rotation * muscle.parent;
                        let parent_anchor_dir: Dir3 = Dir3::new(r * principle_axis).expect("dir3");
                        let child_anchor_dir: Dir3 =
                            Dir3::new(r * muscle.child * principle_axis).expect("dir3");
                        dbg!(parent);
                        dbg!(parent_anchor_dir);
                        dbg!(child);
                        dbg!(child_anchor_dir);
                        if let Some((a1, a2)) = parent
                            .cast_to(parent_anchor_dir)
                            .zip(child.cast_to(child_anchor_dir))
                        {
                            let parent_site = MuscleSite {
                                id: *parent_id,
                                anchor_local: a1,
                                part: parent,
                            };

                            let child_site = MuscleSite {
                                id: child_id,
                                anchor_local: a2,
                                part: &child,
                            };
                            dbg!(&parent_site);
                            dbg!(&child_site);
                            if let Some(e) = make_muscle(&parent_site, &child_site, commands) {
                                entities.push(e);
                            }
                        }
                    }
                }
                child_id
            } else {
                panic!();
            }
        } else {
            // Root object, no parent.
            child.position = position;
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

pub fn spherical_joint(joint: &JointConfig, commands: &mut Commands) -> Entity {
    commands
        .spawn({
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
            j
        })
        .id()
}

pub fn cube_body(
    child: &Part,
    color: Color,
    density: f32,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    commands: &mut Commands,
) -> Entity {
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cast_to() {
        let parent = Part {
            extents: Vec3::ONE,
            position: Vec3::Y,
            rotation: Quat::IDENTITY,
        };

        let anchor_dir = Dir3::Y;
        let hit = parent.cast_to(anchor_dir).unwrap();
        assert_eq!(hit, Vec3::Y / 2.0);
    }

    #[test]
    fn quat_div() {
        let a = Quat::from_axis_angle(Vec3::X, PI);
        let b = Quat::from_axis_angle(Vec3::X, PI / 2.0);
        let c = a * b.inverse();
        assert!(c.angle_between(b) < 0.01);
    }
}
