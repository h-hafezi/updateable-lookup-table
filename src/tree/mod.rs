pub mod srs_tree;

pub mod commitment_tree;

/// Alias for node indices in the tree
type Index = usize;


/// Generic Tree struct
pub struct Tree<Node, Params: TreeParams> {
    pub nodes: Vec<Node>,  // The nodes in the tree
    pub params: Params,    // The parameters defining the tree structure
}

impl<Node, Params: TreeParams> Tree<Node, Params> {
    /// Constructs a new tree
    pub fn new(params: Params, nodes: Vec<Node>) -> Self {
        Self { nodes, params }
    }

    /// Returns all nodes in a specific row `i`
    pub fn row(&self, i: Index) -> Vec<&Node> {
        let depth = self.depth();
        assert!(i < depth, "Row index out of range. Valid range: [0, {})", depth);

        let start = (1 << i) - 1; // Start index for row `i`
        let end = (1 << (i + 1)) - 2; // End index for row `i`

        self.nodes[start..=end.min(self.nodes.len() - 1)]
            .iter()
            .collect()
    }

    /// Returns all rows in the tree as a vector of vectors
    pub fn rows(&self) -> Vec<Vec<&Node>> {
        (0..self.depth())
            .map(|i| self.row(i))
            .collect()
    }

    /// Returns the depth of the tree (requires `params` to provide depth)
    pub fn depth(&self) -> usize {
        self.params.depth()
    }

    /// Checks if the node at the given index is a leaf
    pub fn is_leaf(&self, i: Index) -> bool {
        let depth = self.depth();
        let start = (1 << (depth - 1)) - 1; // Start index of the last row (leaves)
        i >= start && i < self.nodes.len()
    }

    /// Returns the path from a leaf node to the root
    pub fn get_path_to_root(&self, mut i: Index) -> Vec<Index> {
        assert!(self.is_leaf(i), "Index must correspond to a leaf node");

        let mut path = Vec::new();
        while i > 0 {
            path.push(i);
            i = (i - 1) / 2; // Move to parent node
        }
        path.push(0); // Include the root
        path
    }

    /// Returns the sibling index of a given node index
    fn get_sibling_index(&self, i: Index) -> Option<Index> {
        if i == 0 {
            None // Root node has no sibling
        } else if i % 2 == 0 {
            Some(i - 1) // Right child sibling is the left child
        } else {
            Some(i + 1) // Left child sibling is the right child
        }
    }

    /// Returns the sibling indices along the path from a leaf to the root
    pub fn get_sibling_path_to_root(&self, leaf_index: Index) -> Vec<Index> {
        let path_to_root = self.get_path_to_root(leaf_index);
        path_to_root
            .into_iter()
            .filter_map(|index| self.get_sibling_index(index))
            .collect()
    }
}

/// Trait to define the parameters of a tree
pub trait TreeParams {
    fn depth(&self) -> usize; // Retrieve the depth of the tree
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockParams {
        depth: usize,
    }

    impl TreeParams for MockParams {
        fn depth(&self) -> usize {
            self.depth
        }
    }

    #[test]
    fn test_get_sibling_index() {
        let params = MockParams { depth: 4 };
        let nodes: Vec<usize> = (0..15).collect(); // Nodes [0, 1, ..., 14]
        let tree = Tree::new(params, nodes);

        // Root has no sibling
        assert_eq!(tree.get_sibling_index(0), None);

        // Pairs of siblings in different rows
        let sibling_pairs = vec![
            (1, 2),
            (3, 4), (5, 6),
            (7, 8), (9, 10), (11, 12), (13, 14),
        ];

        for (a, b) in sibling_pairs {
            assert_eq!(tree.get_sibling_index(a), Some(b), "Sibling of {} should be {}", a, b);
            assert_eq!(tree.get_sibling_index(b), Some(a), "Sibling of {} should be {}", b, a);
        }
    }

    #[test]
    fn test_get_row() {
        let params = MockParams { depth: 4 };
        let nodes: Vec<usize> = (0..15).collect();  // Nodes [0, 1, ..., 14]
        let tree = Tree::new(params, nodes);

        // First row (depth 0)
        assert_eq!(tree.row(0), vec![&0]);

        // Second row (depth 1)
        assert_eq!(tree.row(1), vec![&1, &2]);

        // Third row (depth 2)
        assert_eq!(tree.row(2), vec![&3, &4, &5, &6]);

        // Fourth row (depth 3)
        assert_eq!(tree.row(3), vec![&7, &8, &9, &10, &11, &12, &13, &14]);
    }

    #[test]
    fn test_is_leaf() {
        let params = MockParams { depth: 4 };
        let nodes: Vec<usize> = (0..15).collect();  // Nodes [0, 1, ..., 14]
        let tree = Tree::new(params, nodes);

        // Nodes 0 to 6 are not leaves
        for i in 0..7 {
            assert!(!tree.is_leaf(i), "Node {} should not be a leaf", i);
        }

        // Nodes 7 to 14 are leaves
        for i in 7..15 {
            assert!(tree.is_leaf(i), "Node {} should be a leaf", i);
        }
    }

    #[test]
    fn test_get_path_to_root() {
        let params = MockParams { depth: 4 };
        let nodes: Vec<usize> = (0..15).collect();  // Nodes [0, 1, ..., 14]
        let tree = Tree::new(params, nodes);

        // Expected paths for each leaf node
        let expected_paths = vec![
            (7, vec![7, 3, 1, 0]),
            (8, vec![8, 3, 1, 0]),
            (9, vec![9, 4, 1, 0]),
            (10, vec![10, 4, 1, 0]),
            (11, vec![11, 5, 2, 0]),
            (12, vec![12, 5, 2, 0]),
            (13, vec![13, 6, 2, 0]),
            (14, vec![14, 6, 2, 0]),
        ];

        for (leaf, expected_path) in expected_paths {
            let path = tree.get_path_to_root(leaf);
            assert_eq!(path, expected_path, "Path for leaf {} is incorrect", leaf);
        }
    }

    #[test]
    fn test_get_sibling_path_to_root() {
        let params = MockParams { depth: 4 };
        let nodes: Vec<usize> = (0..15).collect(); // Nodes [0, 1, ..., 14]
        let tree = Tree::new(params, nodes);

        // Expected sibling paths for each leaf node
        let expected_sibling_paths = vec![
            (7, vec![8, 4, 2]),
            (8, vec![7, 4, 2]),
            (9, vec![10, 3, 2]),
            (10, vec![9, 3, 2]),
            (11, vec![12, 6, 1]),
            (12, vec![11, 6, 1]),
            (13, vec![14, 5, 1]),
            (14, vec![13, 5, 1]),
        ];

        for (leaf, expected_sibling_path) in expected_sibling_paths {
            let sibling_path = tree.get_sibling_path_to_root(leaf);
            assert_eq!(sibling_path, expected_sibling_path, "Sibling path for leaf {} is incorrect", leaf);
        }
    }
}

