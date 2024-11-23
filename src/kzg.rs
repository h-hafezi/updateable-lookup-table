use crate::lagrange_basis::lagrange_basis::LagrangeSubgroup;
use crate::poly::polynomial::GeneralDensePolynomial;
use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{FftField, Field};
use ark_std::rand::RngCore as Rng;
use ark_std::UniformRand;
use std::ops::Mul;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SRS<E: Pairing> {
    /// vec![g^{tau^0}, g^{tau^1}, ..., g^{tau^n}]
    pub powers_of_g: Vec<E::G1>,
}

#[derive(
    Default,
    Hash,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq
)]
/// The commitment is a group element.
pub struct KZGCommitment<E: Pairing> {
    pub com: E::G1,
}

impl<E: Pairing> SRS<E> {
    pub fn setup<R: Rng>(n: usize, rng: &mut R) -> Self {
        let tau = E::ScalarField::rand(rng); // Random scalar tau
        let generator = E::G1Affine::generator(); // Generator g in G1

        let mut powers_of_g = Vec::with_capacity(n);

        // Compute g^{tau^i} for i = 0, 1, ..., n - 1
        for i in 0..n {
            let tau_i = tau.pow([i as u64]); // tau^i
            let g_tau_i = generator.mul(tau_i); // g^{tau^i}
            powers_of_g.push(g_tau_i);
        }

        SRS { powers_of_g }
    }

    /// it also returns secret power tau which is used for writing unit tests
    pub fn unsafe_setup<R: Rng>(n: usize, rng: &mut R) -> (E::ScalarField, Self) {
        let tau = E::ScalarField::rand(rng); // Random scalar tau
        let generator = E::G1Affine::generator(); // Generator g in G1

        let mut powers_of_g = Vec::with_capacity(n);

        // Compute g^{tau^i} for i = 0, 1, ..., n -1
        for i in 0..n {
            let tau_i = tau.pow([i as u64]); // tau^i
            let g_tau_i = generator.mul(tau_i); // g^{tau^i}
            powers_of_g.push(g_tau_i);
        }

        (tau, SRS { powers_of_g })
    }

    pub fn commit(&self, coefficients: Vec<E::ScalarField>) -> KZGCommitment<E> {
        // assert that coefficients is less than srs size
        assert!(coefficients.len() > self.powers_of_g.len(), "coefficients length should be less than srs size");

        KZGCommitment {
            com: E::G1::msm(
                {
                    let mut res = Vec::new();
                    for g in &self.powers_of_g[..coefficients.len()] {
                        res.push(g.into_affine());
                    }
                    res
                }.as_slice(),
                coefficients.as_slice(),
            ).unwrap()
        }
    }


    /// Given a set of roots of unity, this function compute all Lagrange basis
    pub fn compute_kzg_commitments_to_lagrange_basis<F>(&self, subgroup: &LagrangeSubgroup<F>) -> Vec<E::G1>
    where
        F: FftField,
        E: Pairing<ScalarField=F>,
    {
        assert_eq!(self.powers_of_g.len(), subgroup.size(), "kzg srs should be equal to ");
        let normalization_factor = F::from(subgroup.size() as u64).inverse().unwrap();

        let mut poly = GeneralDensePolynomial::from_coeff_vec(
            {
                let mut res = Vec::new();
                for g in self.powers_of_g.clone() {
                    res.push(g.mul(normalization_factor));
                }
                res
            }
        );

        let p_evals = poly.batch_evaluate_rou(&subgroup.domain);

        let n = p_evals.len();
        let mut l_values = Vec::with_capacity(n);

        for i in 0..n {
            let index = (n - i) % n;
            l_values.push(p_evals[index].clone());
        }

        l_values

    }
}


#[cfg(test)]
mod test {
    use crate::constant_curve::{G1Affine, G1Projective, ScalarField, E};
    use crate::kzg::{SRS};
    use crate::lagrange_basis::lagrange_basis::LagrangeSubgroup;
    use ark_ec::AffineRepr;
    use ark_std::{test_rng};
    use std::ops::Mul;

    type F = ScalarField;

    #[test]
    pub fn test_compute_kzg_commitments_to_lagrange_basis() {
        let n = 4usize;
        let mut rng = test_rng();

        // get the srs and tau
        let (tau, srs): (F, SRS<E>) = SRS::unsafe_setup(n, &mut rng);

        // make sure srs is well-formatted
        assert_eq!(srs.powers_of_g.len(), n);
        assert_eq!(srs.powers_of_g[0], G1Affine::generator());
        for (i, g_i) in srs.powers_of_g.iter().enumerate() {
            if i < srs.powers_of_g.len() - 1 {
                assert_eq!(srs.powers_of_g[i + 1], g_i.mul(tau));
            }
        }

        // generate a subgroup of size n
        let subgroup = LagrangeSubgroup::<F>::new(n);

        // compute vec![g ^ L_0(tau), g ^ L_1(tau), ...] through FFT
        let kzg_commitments_to_lagrange_basis: Vec<G1Projective> = srs.compute_kzg_commitments_to_lagrange_basis(&subgroup);

        // compute L_i(tau) by directly compute L_i(tau) and scalar multiplication
        let expected_kzg_commitment_to_lagrange_basis: Vec<G1Projective> = {
            let evals = subgroup.evaluate_all_basis(&tau);
            let mut res = Vec::new();
            for l_i_tau in evals {
                res.push(G1Affine::generator().mul(l_i_tau));
            }
            res
        };

        // assert computing g ^ L_i(tau) in both ways is equal
        assert_eq!(
            kzg_commitments_to_lagrange_basis,
            expected_kzg_commitment_to_lagrange_basis
        );
    }
}
