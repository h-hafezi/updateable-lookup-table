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

        let powers_of_g = (0..n)
            .map(|i| generator.mul(tau.pow([i as u64]))) // Compute g^{tau^i}
            .collect();

        SRS { powers_of_g }
    }

    /// it also returns secret power tau which is used for writing unit tests
    /// It also returns the secret power tau, which is used for writing unit tests.
    pub fn unsafe_setup<R: Rng>(n: usize, rng: &mut R) -> (E::ScalarField, Self) {
        let tau = E::ScalarField::rand(rng); // Random scalar tau
        let generator = E::G1Affine::generator(); // Generator g in G1

        let powers_of_g = (0..n)
            .map(|i| generator.mul(tau.pow([i as u64]))) // Compute g^{tau^i}
            .collect();

        (tau, SRS { powers_of_g })
    }

    pub fn commit(&self, coefficients: Vec<E::ScalarField>) -> KZGCommitment<E> {
        // Ensure coefficients length is less than SRS size
        assert!(
            coefficients.len() <= self.powers_of_g.len(),
            "coefficients length should be less than or equal to SRS size"
        );

        // Perform multi-scalar multiplication directly
        let com = E::G1::msm(
            &self.powers_of_g[..coefficients.len()]
                .iter()
                .map(|g| g.into_affine())
                .collect::<Vec<_>>(),
            &coefficients,
        )
            .unwrap();

        KZGCommitment { com }
    }


    /// Given a set of roots of unity, this function computes all KZG commitments to the Lagrange basis.
    /// Optimized to perform FFT on a subgroup instead of directly on a coset.
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

        // Compute normalization and coset scaling factors
        let normalization_inv = F::from(subgroup.size() as u64).inverse().unwrap();
        let coset_offset = subgroup.coset_offset(); // Coset offset (z)

        // Scale the SRS powers by normalization and coset offset
        let scaled_srs_powers: Vec<_> = self
            .powers_of_g
            .iter()
            .enumerate()
            .map(|(i, power)| power.mul(normalization_inv / coset_offset.pow([i as u64])))
            .collect();

        // Evaluate the polynomial on the coset points
        let mut poly = GeneralDensePolynomial::from_coeff_vec(scaled_srs_powers);
        let coset_evaluations = poly.batch_evaluate_rou(&subgroup.get_coset(F::ONE).unwrap());

        // Reverse the order of evaluations
        let num_evaluations = coset_evaluations.len();
        (0..num_evaluations)
            .map(|i| coset_evaluations[(num_evaluations - i) % num_evaluations].clone())
            .collect()
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
        let expected_kzg_commitment_to_lagrange_basis: Vec<G1Projective> =
            coset.evaluate_all_lagrange_coefficients(tau)
                .into_iter()
                .map(|l_i_tau| G1Affine::generator().mul(l_i_tau))
                .collect();

        // assert computing g ^ L_i(tau) in both ways is equal
        assert_eq!(
            kzg_commitments_to_lagrange_basis,
            expected_kzg_commitment_to_lagrange_basis
        );
    }
}
