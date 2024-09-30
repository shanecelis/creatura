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
        let mut g = Graph::<_, usize>::new();
        let a = g.add_node("a");
        let e0 = g.add_edge(a, a, 2);
        let e1 = g.add_edge(a, a, 2);
        let mut count = 0;
        let tree = unfurl(&g, a, |g, e| g[e] as u8, |n| n.clone(), |e| e.clone());
        // println!("{:?}", Dot::with_config(&g, &[]));
        // println!("{}", &g);
        println!("{}", Dot::with_config(&tree, &[]));
}
