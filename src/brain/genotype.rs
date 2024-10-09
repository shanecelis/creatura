use petgraph::{
    prelude::*,
    graph::DefaultIx,
    visit::{
        GraphBase,
        IntoNodeReferences,
        IntoNeighborsDirected,
        IntoNodeIdentifiers,
        Visitable,
        NodeIndexable,
        IntoNeighbors,
        IntoEdgesDirected
    },
    algo::{
        toposort,
        tarjan_scc,
        DfsSpace,
        Cycle,
    }};
use std::cmp::Ordering;
use std::f32::consts::TAU;

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

pub struct Context {
    time: f32,
}

impl Neuron {
    fn eval(&self, context: &Context, state: f32, inputs: &[f32]) -> f32 {
        use Neuron::*;
        match self {
            Sensor => state,
            Muscle => inputs.into_iter().sum(),
            Sin { amp, freq, phase } => (context.time * freq * TAU + phase).sin() * amp / 2.0 + 0.5,
            Complement => 1.0 - inputs.into_iter().sum::<f32>(),
            Const(c) => *c,
            Scale(s) => s * inputs.into_iter().sum::<f32>(),
            Sum => inputs.into_iter().sum::<f32>(),
            Mult => inputs.into_iter().product(),
            Div => inputs.first().map(|f| f / inputs.into_iter().skip(1).sum::<f32>()).unwrap_or(0.0),
            Diff => inputs.first().map(|f| f - inputs.into_iter().skip(1).sum::<f32>()).unwrap_or(0.0),
            Deriv { dir } => todo!("Deriv"),
            Threshold(t) => inputs.first().and_then(|f| (*f >= 0.0).then_some(1.0)).unwrap_or(0.0),
            Switch(t) => inputs.first().and_then(|f| (*f >= 0.0).then_some(inputs.into_iter().skip(1).sum::<f32>())).unwrap_or(0.0),
            Delay(count) => todo!("Delay"),
            AbsDiff => inputs.first().map(|f| (f - inputs.into_iter().skip(1).sum::<f32>()).abs()).unwrap_or(0.0),
        }
    }
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

pub struct BitBrain {
    neurons: Vec<Neuron>,
    /// This "bit code" follows a simple format:
    ///
    /// ```ignore
    /// <input count>, <input 1>, <input 2>, ..., <input count>.
    /// ```
    ///
    /// Neurons and code are read only.
    ///
    /// Storage a and b are double-buffered storage.
    code: Vec<u8>,
    eval_count: usize,
    storage_a: Vec<f32>,
    storage_b: Vec<f32>,
}

impl BitBrain {

    fn new(graph: &DiGraph<Neuron, ()>) -> Option<BitBrain> {
        // let count: u8 = graph.node_references().map(|(_i, n)| n.storage()).sum();
        let count: usize = graph.node_count();
        // let mut g = graph.clone();
        // let mut cycles = vec![];
        // for edge in g.edge_references() {
        //     if edge.source() == edge.target() {
        //         cycles.push(edge.id());
        //     }
        // }
        // for edge_id in cycles {
        //     g.remove_edge(edge_id);
        // }
        // TODO: This could still have cycles. We can find the strongly
        // connected components (scc) and try to take one of the edges between
        // the nodes out.
        let mut update = {
            let mut u = None;
            let mut g = graph.clone();
            for _ in 0..5 {
                match toposort(&g, None) {
                    Ok(list) => {
                        u = Some(list);
                        break;
                    }
                    Err(cycle) => {
                        let edges: Vec<_> = g.edges_connecting(cycle.node_id(), cycle.neighbor_id()).map(|e| e.id()).collect();
                        if edges.is_empty() {
                            break;
                        }
                        for edge in edges {
                            g.remove_edge(edge);
                        }
                    }
                }
            }
            u?
        };
        // let mut update = toposort_lossy(&mut g, |g: &mut DiGraph<Neuron, ()>, e| { g.remove_edge(e); Ok(g) }).ok()?;
        update.sort_by(|ai, bi| order_neurons(&graph[*ai], ai.index(), &graph[*bi], bi.index()));
        let mut index = 0;

        // let mut brain = graph.clone();//map(|_i, n| (*n, 0), |_i, e| *e);

        let mut neurons: Vec<Neuron> = vec![];
        let mut code = vec![];
        for (i, node_index) in update.iter().enumerate() {
            neurons.push(graph[*node_index]);
            // This is implicit in its ordering.
            // code.push(i as u8);
            code.push(graph.edges_directed(*node_index, Direction::Incoming).count() as u8);
            for edge in graph.edges_directed(*node_index, Direction::Incoming) {
                code.push(update.iter().position(|n| *n == edge.source()).expect("neuron position") as u8);
            }
        }

        Some(BitBrain {
            neurons,
            code,
            eval_count: 0,
            storage_a: vec![0.0; count.into()],
            storage_b: vec![0.0; count.into()],
        })
    }
    /// Return the read storage.
    fn read(&mut self) -> &[f32] {
        if self.eval_count % 2 == 0 {
            &self.storage_a
        } else {
            &self.storage_b
        }
    }

    /// Return the write storage.
    fn write(&mut self) -> &mut [f32] {
        if self.eval_count % 2 == 1 {
            &mut self.storage_a
        } else {
            &mut self.storage_b
        }
    }

    /// Evaluate all nodes in the network in topological order.
    fn eval(&mut self, ctx: &Context) {
        let mut i: usize = 0;
        let mut scratch = vec![];
        let mut j = 0;
        while i < self.code.len() {
            // let j = self.code[i] as usize;
            let neuron = self.neurons[j];
            let count = self.code[i] as usize;
            i += 1;
            scratch.clear();
            for _ in 0..count {
                let k = self.code[i] as usize;
                scratch.push(self.read()[k]);
                i += 1;
            }
            self.write()[j] = neuron.eval(ctx, self.read()[j], &scratch);
            j += 1;
        }
        self.eval_count += 1;
    }
}

fn try_repeat<F, R, I, O, E>(attempts: usize,
                             mut attempt: F,
                             mut input: &mut I,
                             mut remedy: R)
                             -> Result<(O, usize), E>
where F: FnMut(&I) -> Result<O, E>,
      R: FnMut(&mut I, E) -> Result<(), E>
{
    for i in 0..attempts {
        match attempt(&input) {
            Ok(output) => return Ok((output, i)),
            Err(error) => {
                if i + 1 == attempts {
                    return Err(error);
                }
                if let Err(error) = remedy(&mut input, error) {
                    return Err(error);
                }
            }
        }
    }
    unreachable!();
}

// pub fn toposort_lossy<N, E, Ty, Ix, F>(
//     mut graph: &mut Graph<N, E, Ty, Ix>,
//     mut remove_edge: F
//     // space: Option<&mut DfsSpace<G::NodeId, G::Map>>
// // ) -> Result<Vec<NodeIndex<Ix>>, Cycle<EdgeIndex<Ix>>>
// ) -> Result<Vec<<petgraph::Graph<N, E, Ty, Ix> as GraphBase>::NodeId>,
//             Cycle<<petgraph::Graph<N, E, Ty, Ix> as GraphBase>::NodeId>>
// where
// Ix: petgraph::adj::IndexType,
//     // F: FnMut(&mut Graph<N, E, Ty, Ix>, EdgeIndex<Ix>),
//     F: FnMut(&mut Graph<N, E, Ty, Ix>, <&petgraph::Graph<N, E, Ty, Ix> as GraphBase>::EdgeId),
//     for<'a> &'a Graph<N, E, Ty, Ix>: IntoNeighborsDirected + IntoNodeIdentifiers + Visitable + NodeIndexable + IntoNeighbors + IntoEdgesDirected
//     {
//     try_repeat(3,
//                |g| toposort(&g, None),
//                graph,
//                |g, e| {
//                    let troublemaker = e.node_id();
//                    for nodes in tarjan_scc(&*g) {
//                        if nodes.contains(&e.node_id()) {
//                            // This scc contains our trouble maker.
//                            for edge in g.edges_directed(troublemaker, Direction::Incoming) {
//                                if nodes.contains(&edge.source()) {
//                                    remove_edge(g, edge.id());
//                                    break;
//                                }
//                            }

//                            for edge in g.edges_directed(troublemaker, Direction::Outgoing) {
//                                if nodes.contains(&edge.target()) {
//                                    remove_edge(g, edge.id());
//                                    break;
//                                }
//                            }
//                            // for neighbor in g.neighbors(troublemaker) {
//                            //     if nodes.contains(&neighbor) {
//                            //         for edge in (*g).edges_connecting(troublemaker, neighbor) {
//                            //             g.edge_remove(edge.id());
//                            //         }
//                            //     }
//                            // }
//                        }
//                    }
//                    Ok(())
//                },
//     ).map(|r| r.0)
// }

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
        // TODO: This could still have cycles. We can find the strongly
        // connected components (scc) and try to take one of the edges between
        // the nodes out.
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
    fn cross_graph_indices() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let b = g.add_node(1);
        let h = g.clone();
        assert_eq!(h[a], 0);
        assert_eq!(h[b], 1);
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

    #[test]
    fn neuron_eval() {
        let ctx = Context {
            time: 0.0,
        };
        let inputs = [2.0, 1.0];
        assert_eq!(Sensor.eval(&ctx, 1.0, &inputs), 1.0);
        assert_eq!(Sum.eval(&ctx, 1.0, &inputs), 3.0);
        assert_eq!(Diff.eval(&ctx, 1.0, &inputs), 1.0);

    }

    #[test]
    fn bitbrain_eval() {
        let ctx = Context {
            time: 0.0,
        };
        let mut g = Graph::<Neuron, ()>::new();
        let a = g.add_node(Const(1.0));
        let mut brain = BitBrain::new(&g).unwrap();
        assert_eq!(brain.storage_a.len(), 1);
        assert_eq!(brain.read()[0], 0.0);
        brain.eval(&ctx);
        assert_eq!(brain.read()[0], 1.0);
    }

    #[test]
    fn bitbrain_eval_cycle() {
        let ctx = Context {
            time: 0.0,
        };
        let mut g = Graph::<Neuron, ()>::new();
        let a = g.add_node(Const(1.0));
        let _e = g.add_edge(a, a, ());
        let mut brain = BitBrain::new(&g).unwrap();
        assert_eq!(brain.storage_a.len(), 1);
        assert_eq!(brain.read()[0], 0.0);
        brain.eval(&ctx);
        assert_eq!(brain.read()[0], 1.0);
    }

    #[test]
    fn bitbrain_eval_scc() {
        let ctx = Context {
            time: 0.0,
        };
        let mut g = Graph::<Neuron, ()>::new();
        let a = g.add_node(Const(1.0));
        let b = g.add_node(Sum);
        let _ = g.add_edge(a, b, ());
        let _ = g.add_edge(b, a, ());
        let _ = g.add_edge(b, b, ());
        let mut brain = BitBrain::new(&g).unwrap();
        assert_eq!(brain.code, [1,1,2,1,0]);
        assert_eq!(brain.storage_a, [0.0, 0.0]);
        assert_eq!(brain.storage_b, [0.0, 0.0]);
        assert_eq!(brain.storage_a.len(), 2);
        assert_eq!(brain.read()[0], 0.0);
        brain.eval(&ctx);
        assert_eq!(brain.read()[0], 1.0);
        assert_eq!(brain.storage_a, [0.0, 0.0]);
        assert_eq!(brain.storage_b, [1.0, 0.0]);
        brain.eval(&ctx);
        assert_eq!(brain.read()[0], 1.0);
        assert_eq!(brain.storage_a, [1.0, 1.0]);
        assert_eq!(brain.storage_b, [1.0, 0.0]);
        brain.eval(&ctx);
        assert_eq!(brain.read()[0], 1.0);
        assert_eq!(brain.read()[1], 2.0);
        assert_eq!(brain.storage_a, [1.0, 1.0]);
        assert_eq!(brain.storage_b, [1.0, 2.0]);
    }

}
