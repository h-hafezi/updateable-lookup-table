use crate::kzg::SRS;
use crate::lagrange_basis::lagrange_basis::{split_subgroup, split_vector};
use crate::tree::{Tree, TreeParams};
use ark_ec::pairing::Pairing;
use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::rand::Rng;
use ark_std::UniformRand;
use std::ops::Mul;


pub type SRSTree<F, E> = Tree<SRSTreeNode<F, E>, SRSTreeParams<E>>;

/// Struct representing the tree parameters
#[derive(Clone)]
pub struct SRSTreeParams<E: Pairing> {
    pub subgroup_size: usize,
    pub depth: usize,
    pub g1_powers: Vec<E::G1>,
    pub g2_powers: Vec<E::G2>,
}

impl<E: Pairing> TreeParams for SRSTreeParams<E> {
    fn depth(&self) -> usize {
        self.depth
    }
}

impl<E: Pairing> SRSTreeParams<E> {
    /// Validates the SRS tree parameters
    pub fn is_valid(&self) {
        assert!(self.subgroup_size.is_power_of_two(), "Subgroup size must be a power of two");
        assert!(self.depth > 0, "Depth must be positive");
        assert_eq!(self.g1_powers.len(), self.subgroup_size, "g1_powers length must be equal to subgroup_size");
        assert_eq!(self.g2_powers.len(), self.subgroup_size, "g2_powers length must be equal to subgroup_size");
    }
}

impl<E: Pairing> SRSTreeParams<E> {
    /// Generate the setup parameters for the binary tree
    pub fn setup<R: Rng>(rng: &mut R, subgroup_size: usize, depth: usize) -> Self {
        let tau = E::ScalarField::rand(rng); // Random scalar tau

        // Generate powers of tau in G1
        let generator_g1 = E::G1Affine::generator(); // Generator g in G1
        let g1_powers: Vec<E::G1> = (0..subgroup_size)
            .map(|i| generator_g1.mul(tau.pow([i as u64])))
            .collect();

        // Generate powers of tau in G2
        let generator_g2 = E::G2Affine::generator(); // Generator h in G2
        let g2_powers: Vec<E::G2> = (0..subgroup_size)
            .map(|i| generator_g2.mul(tau.pow([i as u64])))
            .collect();

        Self {
            subgroup_size,
            depth,
            g1_powers,
            g2_powers,
        }
    }
}

/// Struct representing a node in the binary tree
pub struct SRSTreeNode<F: FftField, E: Pairing<ScalarField=F>> {
    /// subgroup or coset corresponding to this node
    pub subgroup: GeneralEvaluationDomain<F>,

    /// kzg commitment to L_0(tau), L_1(tau), L_2(tau), ...
    pub commitments: Vec<E::G1>,
}

/// Constructs a binary tree with the given parameters
pub fn new_srs_tree<F, E>(params: SRSTreeParams<E>) -> SRSTree<F, E>
where
    F: FftField,
    E: Pairing<ScalarField = F>,
{
    // checking the parameters given is valid
    params.is_valid();

    // Total nodes in a binary tree: 2^depth - 1
    let total_nodes = (1 << params.depth) - 1;
    let mut nodes = Vec::with_capacity(total_nodes);

    // Generate the original subgroup
    let subgroup = GeneralEvaluationDomain::new(params.subgroup_size).unwrap();

    for d in 0..params.depth {
        let split_factor = 1 << d;
        let split_subgroups = split_subgroup(&subgroup, split_factor);
        let split_g1_vector = split_vector(&params.g1_powers, split_factor);

        for (sub_index, subgroup) in split_subgroups.iter().enumerate() {
            let powers_of_g = split_g1_vector[sub_index].clone();
            let commitments = SRS::<E> { powers_of_g }.compute_kzg_commitments_to_lagrange_basis(subgroup);

            nodes.push(SRSTreeNode {
                subgroup: subgroup.clone(),
                commitments,
            });
        }
    }

    SRSTree { nodes, params }
}


#[cfg(test)]
mod test {
    use crate::constant_curve::E;
    use crate::kzg::SRS;
    use crate::tree::srs_tree::{new_srs_tree, SRSTreeParams};
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use ark_std::test_rng;

    #[test]
    fn test_correctness_of_tree() {
        let mut rng = test_rng();

        let srs_tree_params = SRSTreeParams::<E>::setup(&mut rng, 8, 2);

        let tree = new_srs_tree(srs_tree_params.clone());

        let [g0, g1, g2, g3, g4, g5, g6, g7] = srs_tree_params.g1_powers.as_slice() else {
            panic!("Expected srs_tree_params.g1_powers to have exactly 8 elements.");
        };

        let subgroup_8 = GeneralEvaluationDomain::new(8).unwrap();
        assert_eq!(tree.nodes[0].subgroup, subgroup_8);
        assert_eq!(tree.nodes[0].commitments, SRS::<E> { powers_of_g: vec![*g0, *g1, *g2, *g3, *g4, *g5, *g6, *g7] }.compute_kzg_commitments_to_lagrange_basis(&subgroup_8));

        let subgroup_4 = GeneralEvaluationDomain::new(4).unwrap();
        assert_eq!(tree.nodes[1].subgroup, subgroup_4);
        assert_eq!(tree.nodes[1].commitments, SRS::<E> { powers_of_g: vec![*g0, *g2, *g4, *g6] }.compute_kzg_commitments_to_lagrange_basis(&subgroup_4));

        let coset_4 = subgroup_4.get_coset(subgroup_8.element(1)).unwrap();
        assert_eq!(tree.nodes[2].subgroup, coset_4);
        assert_eq!(tree.nodes[2].commitments, SRS::<E> { powers_of_g: vec![*g1, *g3, *g5, *g7] }.compute_kzg_commitments_to_lagrange_basis(&coset_4));
    }
}
