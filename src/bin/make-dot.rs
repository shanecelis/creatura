use std::fmt;
use petgraph::{
    Graph,
    dot::Dot
};
use muscley_wusaley::*;

struct Edge {
    index: usize,
    repeat: usize
}

impl From<(usize, usize)> for Edge {
    fn from((index, repeat): (usize, usize)) -> Self {
        Edge {index, repeat }
    }
}

impl fmt::Display for Edge {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Write strictly the first element into the supplied output
        // stream: `f`. Returns `fmt::Result` which indicates whether the
        // operation succeeded or failed. Note that `write!` uses syntax which
        // is very similar to `println!`.
        write!(f, "{} repeat {}", self.index, self.repeat)
    }
}

/// Run in shell like this:
///
/// ```ignore
/// cargo run --bin make-dot  > tree.dot && dot -Tpdf tree.dot > tree.pdf && open tree.pdf
/// ```
fn main() {
        let mut g = Graph::<_, Edge>::new();
        let a = g.add_node("a");
        let _e0 = g.add_edge(a, a, (0, 2).into());
        let _e1 = g.add_edge(a, a, (1, 2).into());
        let tree = unfurl(&g, a, |g, e| g[e].repeat as u8, |g, n| g[n].to_owned(), |g,e| e.index());
        // println!("{}", Dot::with_config(&g, &[]));
        // println!("{}", &g);
        println!("{}", Dot::with_config(&tree, &[]));
}
