use petgraph::{
    prelude::*,
    Graph,
    dot::{Dot, Config}
};
use muscley_wusaley::*;

/// Run in shell like this:
///
/// ```ignore
/// cargo run --bin make-dot  > tree.dot && dot -Tpdf tree.dot > tree.pdf && open tree.pdf
/// ```
fn main() {
        let mut g = Graph::<isize, isize>::new();
        let a = g.add_node(0);
        let e0 = g.add_edge(a, a, 0);
        let e1 = g.add_edge(a, a, 100);
        let mut count = 0;
        let tree = unfurl(&g, a, |_, _| 2, |n| {
            count += 1;
            *n + count
        }, |e| *e + 1);
        println!("{:?}", Dot::with_config(&tree, &[Config::EdgeIndexLabel]));
}
