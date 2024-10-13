use super::*;
use crate::{operator::*, rdfs::*, body::*};
use core::f32::consts::FRAC_PI_4;
use petgraph::{
    graph::{DefaultIx, IndexType},
    prelude::*,
    EdgeType,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use weighted_rand::{
    builder::{NewBuilder, WalkerTableBuilder},
    table::WalkerTable,
};

fn rand_elem<T, R>(iter: impl Iterator<Item = T>, rng: &mut R) -> Option<T>
where
    R: Rng,
{
    let mut result = None;
    let mut i = 1;
    for item in iter {
        if rng.with_prob(1.0 / (i as f32)) {
            result = Some(item);
            i += 1;
        }
    }
    result
}

fn nodes_of_subtree<N, E, Ty, Ix>(
    graph: &mut Graph<N, E, Ty, Ix>,
    start: NodeIndex<Ix>,
) -> Vec<NodeIndex<Ix>>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    let mut dfs = Dfs::new(&*graph, start);
    let _ = dfs.next(&*graph); // skip the start node.
    let mut v = vec![];
    while let Some(node) = dfs.next(&*graph) {
        v.push(node);
    }
    v
}

fn prune_subtree<N, E, Ty, Ix>(graph: &mut Graph<N, E, Ty, Ix>, start: NodeIndex<Ix>)
where
    Ty: EdgeType,
    Ix: IndexType,
{
    let mut dfs = Dfs::new(&*graph, start);
    let _ = dfs.next(&*graph); // skip the start node.
    while let Some(node) = dfs.next(&*graph) {
        graph.remove_node(node);
    }
}

fn add_subtree<N, E, Ty, Ix>(
    source: &Graph<N, E, Ty, Ix>,
    source_root: NodeIndex<Ix>,
    dest: &mut Graph<N, E, Ty, Ix>,
    dest_root: NodeIndex<Ix>,
) where
    N: Clone,
    E: Clone,
    Ty: EdgeType,
    Ix: IndexType,
{
    let mut nodes = HashMap::new();
    let mut dfs = Dfs::new(source, source_root);

    let _ = dfs.next(source); // skip the start node.
    nodes.insert(source_root, dest_root);
    while let Some(src_idx) = dfs.next(source) {
        let dst_idx = dest.add_node(source[src_idx].clone());
        nodes.insert(src_idx, dst_idx);
    }
    // Go through all the edges.
    for edge in source.edge_references() {
        if let Some((a, b)) = nodes.get(&edge.source()).zip(nodes.get(&edge.target())) {
            dest.add_edge(*a, *b, edge.weight().clone());
        }
    }
}

fn tree_crosser<N, E, Ty, Ix, R>(
    a: &mut Graph<N, E, Ty, Ix>,
    b: &mut Graph<N, E, Ty, Ix>,
    rng: &mut R,
) -> u32
where
    R: Rng,
    N: Clone,
    E: Clone,
    Ty: EdgeType,
    Ix: IndexType,
{
    if let Some(x) = rand_elem(a.node_indices(), rng) {
        if let Some(y) = rand_elem(b.node_indices(), rng) {
            cross_subtree(a, x, b, y);
            return 2;
        }
    }
    0
}

fn cross_subtree<N, E, Ty, Ix>(
    source: &mut Graph<N, E, Ty, Ix>,
    source_root: NodeIndex<Ix>,
    dest: &mut Graph<N, E, Ty, Ix>,
    dest_root: NodeIndex<Ix>,
) where
    N: Clone,
    E: Clone,
    Ty: EdgeType,
    Ix: IndexType,
{
    let source_prune = nodes_of_subtree(source, source_root);
    let dest_prune = nodes_of_subtree(dest, dest_root);
    add_subtree(source, source_root, dest, dest_root);
    add_subtree(dest, dest_root, source, source_root);
    for n in source_prune {
        source.remove_node(n);
    }

    for n in dest_prune {
        dest.remove_node(n);
    }
}

fn prune_connection<N, E, Ty, Ix, R>() -> impl Mutator<Graph<N, E, Ty, Ix>, R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    move |graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R| {
        if let Some(edge) = rand_elem(graph.edge_indices(), rng) {
            graph.remove_edge(edge);
            return 1;
        }
        0
    }
}

fn add_connection<N, E, Ty, Ix, R>(
    generator: impl Generator<E, R>,
) -> impl Mutator<Graph<N, E, Ty, Ix>, R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    move |graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R| {
        if let Some(a) = rand_elem(graph.node_indices(), rng) {
            if let Some(b) = rand_elem(graph.node_indices(), rng) {
                graph.add_edge(a, b, generator.generate(rng));
                return 1;
            }
        }
        0
    }
}

fn add_node<N, E, Ty, Ix, R>(
    generator: impl Generator<N, R>,
) -> impl Mutator<Graph<N, E, Ty, Ix>, R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    move |graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R| {
        graph.add_node(generator.generate(rng));
        1
    }
}

fn mutate_nodes<N, E, Ty, Ix, R>(
    mutator: impl Mutator<N, R>,
    mutation_rate: f32,
) -> impl Mutator<Graph<N, E, Ty, Ix>, R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    move |graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R| {
        let mut count = 0u32;
        for node in graph.node_weights_mut() {
            if rng.with_prob(mutation_rate) {
                mutator.mutate(node, rng);
                count += 1;
            }
        }
        count
    }
}

/// Use one of a collection of weighted mutators when called upon.
struct WeightedMutator<'a, G, R> {
    mutators: Vec<&'a dyn Mutator<G, R>>,
    table: WalkerTable,
}

impl<'a, G, R> WeightedMutator<'a, G, R> {
    fn new<T>(mutators: Vec<&'a dyn Mutator<G, R>>, weights: &[T]) -> Self
    where
        WalkerTableBuilder: NewBuilder<T>,
    {
        let builder = WalkerTableBuilder::new(weights);
        assert_eq!(
            mutators.len(),
            weights.len(),
            "Mutators and weights different lengths."
        );
        Self {
            table: builder.build(),
            mutators,
        }
    }
}

impl<'a, G, R> Mutator<G, R> for WeightedMutator<'a, G, R>
where
    R: Rng,
{
    fn mutate(&self, genome: &mut G, rng: &mut R) -> u32 {
        self.mutators[dbg!(self.table.next_rng(rng))].mutate(genome, rng)
    }
}

fn mutate_edges<N, E, Ty, Ix, R>(
    mutator: impl Mutator<E, R>,
    mutation_rate: f32,
) -> impl Mutator<Graph<N, E, Ty, Ix>, R>
where
    Ty: EdgeType,
    Ix: IndexType,
    R: Rng,
{
    move |graph: &mut Graph<N, E, Ty, Ix>, rng: &mut R| {
        let mut count = 0u32;
        for edge in graph.edge_weights_mut() {
            if rng.with_prob(mutation_rate) {
                count += mutator.mutate(edge, rng);
            }
        }
        count
    }
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
            let joint_dir = Dir3::new(joint_rotation * principle_axis).expect("joint_dir");
            // child.position = parent.position + 10.0 * joint_dir;
            child.extents = state.scale * graph[node].extents;
            // Position child
            if let Some(stamp_info) = child.stamp(parent, joint_dir) {
                child.position = parent.position + stamp_info.stamp_delta;
                let child_id = make_part(&child, commands).unwrap();
                if let Some(e) = make_joint(
                    &JointConfig {
                        parent: *parent_id,
                        parent_anchor: stamp_info.surface_anchor,
                        child: child_id,
                        child_anchor: stamp_info.stamp_anchor,
                        normal: *joint_dir,
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
                        // dbg!(parent);
                        // dbg!(parent_anchor_dir);
                        // dbg!(child);
                        // dbg!(child_anchor_dir);
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
                            // dbg!(&parent_site);
                            // dbg!(&child_site);
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
    position: Vec3,
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
            Position(position),
            MassPropertiesBundle::new_computed(&child.collider(), child.volume() * density),
            // c,
            child.collider(),
        ))
        .id()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::brain::{lessin, Neuron};
    use petgraph::dot::Dot;

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

    #[test]
    fn weighted_mutator() {
        let i = 2;
        // for i in 0..100 {
        let mut rng = StdRng::seed_from_u64(i);
        let a = uniform_mutator(0.0, 1.0);
        let b = uniform_mutator(2.0, 10.0);
        let w = WeightedMutator::new(vec![&a, &b], &[0.0, 1.0]);
        let mut v = 0.1;
        assert_eq!(w.mutate(&mut v, &mut rng), 1);
        assert!(v > 2.0, "v {v} > 2.0, seed {i}");
        // }
    }

    #[test]
    fn test_prune_subtree() {
        let mut a = lessin::fig4_3();
        assert_eq!(
            a.node_weights().filter(|w| *w == &Neuron::Muscle).count(),
            3
        );
        assert_eq!(
            a.node_weights()
                .filter(|w| *w == &Neuron::Complement)
                .count(),
            1
        );
        let sin_idx = a
            .node_indices()
            .find(|n| matches!(a[*n], Neuron::Sin { .. }))
            .unwrap();
        prune_subtree(&mut a, sin_idx);
        assert_eq!(
            a.node_weights().filter(|w| *w == &Neuron::Muscle).count(),
            2
        );
        assert_eq!(
            a.node_weights()
                .filter(|w| *w == &Neuron::Complement)
                .count(),
            0
        );
    }

    #[test]
    fn test_add_subtree() {
        let a = lessin::fig4_3();
        let mut b = Graph::new();
        let s = b.add_node(Neuron::Sensor);
        let idx = a
            .node_indices()
            .find(|n| matches!(a[*n], Neuron::Complement))
            .unwrap();
        add_subtree(&a, idx, &mut b, s);
        assert_eq!(b.node_count(), 2);
        assert_eq!(b.edge_count(), 1);
        // assert_eq!(format!("{:?}", Dot::with_config(&b, &[])), "");
    }

    #[test]
    fn test_cross_subtree() {
        let mut a = lessin::fig4_3();
        let mut b = Graph::new();
        let s = b.add_node(Neuron::Sensor);
        let t = b.add_node(Neuron::Mult);
        let _ = b.add_edge(s, t, ());
        let idx = a
            .node_indices()
            .find(|n| matches!(a[*n], Neuron::Complement))
            .unwrap();
        cross_subtree(&mut a, idx, &mut b, s);
        assert_eq!(b.node_count(), 2);
        assert_eq!(b.edge_count(), 1);
        // assert_eq!(format!("{:?}", Dot::with_config(&b, &[])), "");
        // assert_eq!(format!("{:?}", Dot::with_config(&a, &[])), "");
    }
}
