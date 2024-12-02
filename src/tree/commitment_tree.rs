use crate::lagrange_basis::lagrange_basis::split_vector;
use crate::tree::srs_tree::SRSTree;
use crate::tree::{Tree, TreeParams};
use ark_ec::pairing::Pairing;
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::FftField;


pub type CommitmentTree<F, E> = Tree<CommitmentTreeNode<F, E>, CommitmentTreeParams<F, E>>;

pub struct CommitmentTreeParams<F: FftField, E: Pairing<ScalarField=F>> {
    pub srs_tree: SRSTree<F, E>,
    pub opening_vector: Vec<F>,
}

impl<F: FftField, E: Pairing<ScalarField=F>> TreeParams for CommitmentTreeParams<F, E> {
    fn depth(&self) -> usize {
        self.srs_tree.params.depth
    }
}

impl<F, E> CommitmentTreeParams<F, E>
where
    F: FftField,
    E: Pairing<ScalarField=F>,
{
    pub fn is_valid(&self) {
        self.srs_tree.params.is_valid();
        assert_eq!(self.opening_vector.len(), self.srs_tree.params.subgroup_size);
    }
}


/// Struct representing a node in the binary tree
pub struct CommitmentTreeNode<F: FftField, E: Pairing<ScalarField=F>> {
    /// opening
    pub opening: Vec<F>,
    /// kzg commitment
    pub commitment: E::G1,
}

pub fn new_commitment_tree<F, E>(params: CommitmentTreeParams<F, E>) -> CommitmentTree<F, E>
where
    F: FftField,
    E: Pairing<ScalarField=F>,
{
    params.is_valid();

    let depth = params.srs_tree.params.depth;
    let total_nodes = (1 << depth) - 1;
    let mut nodes = Vec::with_capacity(total_nodes);

    for (d, srs_row) in (0..depth).zip(params.srs_tree.rows()) {
        let split_opening = split_vector(&params.opening_vector, 1 << d);

        for (scalar, srs_node) in split_opening.iter().zip(srs_row.iter()) {
            let commitments_affine: Vec<E::G1Affine> = srs_node
                .commitments
                .iter()
                .map(|c| c.into_affine())
                .collect();
            let commitment = E::G1::msm(&commitments_affine, scalar).unwrap();
            nodes.push(CommitmentTreeNode {
                opening: scalar.clone(),
                commitment,
            });
        }
    }

    CommitmentTree { nodes, params }
}
