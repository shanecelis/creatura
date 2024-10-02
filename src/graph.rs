use petgraph::prelude::*;
use core::f32::consts::FRAC_PI_4;
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

pub fn construct_phenotype<F, G>(graph: &DiGraph<Part,PartEdge>,
                              root: NodeIndex<u32>,
                              make_part: F,
                              make_joint: G,
                              mut commands: Commands
)
where F: Fn(&Part, &mut Commands) -> Entity,
      G: Fn(&JointConfig) -> Option<Entity>,
{

}

struct JointConfig {
    parent: Entity,
    /// Local coordinates
    parent_attach: Vector3,
    child: Entity,
    /// Local coordinates
    child_attach: Vector3,
}

pub fn spherical_joint(joint: &JointConfig, commands: &mut Commands) -> Entity{
    commands.spawn(
        {

    let mut j = SphericalJoint::new(joint.parent, joint.child)
        // .with_swing_axis(Vector::Y)
        // .with_twist_axis(Vector::X)
        .with_local_anchor_1(joint.parent_attach)
        .with_local_anchor_2(joint.child_attach)
        // .with_aligned_axis(Vector::Z)
        .with_swing_limits(-FRAC_PI_4, FRAC_PI_4) // .with_linear_velocity_damping(0.1)
        .with_twist_limits(-FRAC_PI_4, FRAC_PI_4); // .with_linear_velocity_damping(0.1)
    j.swing_axis = Vector::Y;
    j.twist_axis = Vector::X;
            j}).id()
}
