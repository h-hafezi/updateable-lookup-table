use crate::lagrange_basis::lagrange_basis::{split_subgroup, split_vector};
use ark_ec::pairing::Pairing;
use ark_ec::AffineRepr;
use ark_ec::CurveGroup;
use ark_ff::{FftField, Field};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::rand::Rng;
use ark_std::UniformRand;
use std::ops::Mul;
use crate::kzg::SRS;

/// Struct representing the tree parameters
pub struct TreeParams<E: Pairing> {
    pub subgroup_size: usize,
    pub depth: usize,
    pub g1_powers: Vec<E::G1>,
    pub g2_powers: Vec<E::G2>,
}


impl<E: Pairing> TreeParams<E> {
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
pub struct BinaryTreeNode<F: FftField, E: Pairing<ScalarField=F>> {
    /// subgroup or coset corresponding to this node
    pub subgroup: GeneralEvaluationDomain<F>,

    /// kzg commitment to L_0(tau), L_1(tau), L_2(tau), ...
    pub commitments: Vec<E::G1>,
}

/// Struct representing the entire binary tree
pub struct BinaryTree<F: FftField, E: Pairing<ScalarField=F>> {
    pub nodes: Vec<BinaryTreeNode<F, E>>,
    pub params: TreeParams<E>,
}


impl<F: FftField, E: Pairing<ScalarField=F>> BinaryTree<F, E> {
    /// Constructs a binary tree with the given parameters
    pub fn new(params: TreeParams<E>) -> Self {
        // 2^depth - 1 nodes in a binary tree
        let total_nodes = (1 << params.depth) - 1;
        // vector including all the nodes
        let mut nodes = Vec::with_capacity(total_nodes);

        // the original subgroup
        let subgroup = GeneralEvaluationDomain::new(params.subgroup_size).unwrap();

        for d in 0..params.depth {
            let split_factor = 1 << d;
            let split_subgroups = split_subgroup(&subgroup, split_factor);
            let split_g1_vector = split_vector(params.g1_powers.as_slice(), split_factor);

            for sub_index in 0..split_factor {
                nodes.push(
                    BinaryTreeNode {
                        // Borrowing the reference of split_subgroups[sub_index]
                        subgroup: split_subgroups[sub_index].clone(),
                        commitments: {
                            SRS::<E> {
                                powers_of_g: split_g1_vector[sub_index]
                                    .iter()
                                    .map(|elem| elem.clone())
                                    .collect(),
                            }.compute_kzg_commitments_to_lagrange_basis(&split_subgroups[sub_index])
                        },
                    }
                );
            }
        }

        BinaryTree {
            nodes,
            params,
        }
    }

    /// Returns the left child index of a given node index
    pub fn left_child(index: usize) -> usize {
        2 * index + 1
    }

    /// Returns the right child index of a given node index
    pub fn right_child(index: usize) -> usize {
        2 * index + 2
    }
}
