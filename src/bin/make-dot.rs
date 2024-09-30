use petgraph::{
    Graph,
    dot::Dot
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
        let _e0 = g.add_edge(a, a, 2);
        let _e1 = g.add_edge(a, a, 2);
        let tree = unfurl(&g, a, |g, e| g[e] as u8, |n| n.to_owned(), |e| *e);
        // println!("{:?}", Dot::with_config(&g, &[]));
        // println!("{}", &g);
        println!("{}", Dot::with_config(&tree, &[]));
}
