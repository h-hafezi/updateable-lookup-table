use crate::poly::polynomial::GeneralDensePolynomial;
use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{FftField, Field};
use ark_std::rand::RngCore as Rng;
use ark_std::UniformRand;
use std::ops::Mul;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

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


    /// Given a set of roots of unity, this function computes all KZG commitments to the Lagrange basis
    /// The code is optimised such that does FFT for a subgroup instead of coset directly
    pub fn compute_kzg_commitments_to_lagrange_basis<F>(&self, subgroup: &GeneralEvaluationDomain<F>) -> Vec<E::G1>
    where
        F: FftField,
        E: Pairing<ScalarField = F>,
    {
        assert_eq!(
            self.powers_of_g.len(),
            subgroup.size(),
            "KZG SRS size must match subgroup size"
        );

        // Normalization factor and coset factor
        let normalization_factor = F::from(subgroup.size() as u64).inverse().unwrap();
        let coset_factor = subgroup.coset_offset(); // Coset offset (z)

        // Adjust coefficients based on normalization and coset scaling
        let adjusted_coeffs: Vec<_> = self
            .powers_of_g
            .iter()
            .enumerate()
            .map(|(i, g_i)| g_i.mul(normalization_factor / coset_factor.pow([i as u64])))
            .collect();

        // Evaluate the polynomial on the coset
        let mut poly = GeneralDensePolynomial::from_coeff_vec(adjusted_coeffs);
        let p_evals = poly.batch_evaluate_rou(&subgroup.get_coset(F::ONE).unwrap());

        let n = p_evals.len();
        (0..n).map(|i| p_evals[(n - i) % n].clone()).collect()
    }
}


#[cfg(test)]
mod test {
    use crate::constant_curve::{G1Affine, G1Projective, ScalarField, E};
    use crate::kzg::{SRS};
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_std::{test_rng, UniformRand};
    use std::ops::Mul;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

    type F = ScalarField;

    #[test]
    pub fn test_compute_kzg_commitments_to_lagrange_basis() {
        let n = 1024usize;
        let mut rng = test_rng();

        // get the srs and tau
        let (tau, srs): (F, SRS<E>) = SRS::unsafe_setup(n, &mut rng);

        // make sure srs is well-formatted
        assert_eq!((srs.powers_of_g[0].into_affine(), srs.powers_of_g.len()), (G1Affine::generator(), n));
        for i in 0..(n - 1) {
            assert_eq!(srs.powers_of_g[i + 1], srs.powers_of_g[i].mul(tau));
        }

        // generate a subgroup of size n
        let coset = GeneralEvaluationDomain::<F>::new(n).unwrap().get_coset(F::rand(&mut rng)).unwrap();

        // compute vec![g ^ L_0(tau), g ^ L_1(tau), ...] through FFT
        let kzg_commitments_to_lagrange_basis: Vec<G1Projective> = srs.compute_kzg_commitments_to_lagrange_basis(&coset);

        // compute L_i(tau) by directly compute L_i(tau) and scalar multiplication
        let expected_kzg_commitment_to_lagrange_basis: Vec<G1Projective> = {
            let evals = coset.evaluate_all_lagrange_coefficients(tau);
            let mut res = Vec::new();
            for (i, l_i_tau) in evals.iter().enumerate() {
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
