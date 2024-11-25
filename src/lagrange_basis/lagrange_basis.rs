use crate::util::is_positive_power_of_two;
use ark_ff::FftField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::CanonicalSerialize;

#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize)]
pub struct LagrangeSubgroup<F: FftField> {
    pub domain: GeneralEvaluationDomain<F>,
}

impl<F: FftField> LagrangeSubgroup<F> {
    /// given a power of two it returns a subgroup of that size
    pub fn new(n: usize) -> Self {
        // make sure n is a positive power of two
        is_positive_power_of_two(n);

        LagrangeSubgroup {
            domain: GeneralEvaluationDomain::<F>::new(n).unwrap()
        }
    }

    /// it returns L_0(tau), L_1(tau), L_2(tau), ...
    pub fn evaluate_all_basis(&self, tau: &F) -> Vec<F> {
        self.domain.evaluate_all_lagrange_coefficients(*tau)
    }

    pub fn size(&self) -> usize {
        self.domain.size()
    }

}

#[cfg(test)]
mod tests {
    use crate::constant_curve::ScalarField;
    use crate::lagrange_basis::lagrange_basis::LagrangeSubgroup;
    use ark_ff::{FftField, Field, One, Zero};
    use ark_poly::{DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
    use ark_poly::univariate::DensePolynomial;
    use ark_std::{test_rng, UniformRand};

    type F = ScalarField;

    // Function to compute Lagrange basis polynomials L_i
    fn compute_lagrange_basis<F: FftField>(roots: &GeneralEvaluationDomain<F>) -> Vec<DensePolynomial<F>> {
        let mut lagrange_polys = Vec::new();

        for i in 0..roots.size() {
            // Initialize numerator and denominator
            let mut numerator = DensePolynomial::from_coefficients_vec(vec![F::one()]);
            let mut denominator = F::one();

            for j in 0..roots.size() {
                if i != j {
                    // (x - w^j)
                    let poly_x_minus_root = DensePolynomial::from_coefficients_vec(vec![-roots.element(j), F::one()]);
                    numerator = &numerator * &poly_x_minus_root;

                    // Denominator (w^i - w^j)
                    denominator *= roots.element(i) - roots.element(j);
                }
            }

            // L_i(x) = numerator / denominator
            let li = numerator * denominator.inverse().unwrap();
            lagrange_polys.push(li);
        }

        lagrange_polys
    }

    #[test]
    fn lagrange_test() {
        let subgroup_16: LagrangeSubgroup<F> = LagrangeSubgroup::new(16);
        assert_eq!(subgroup_16.domain.size(), 16);

        // Check subgroup elements
        let elements: Vec<_> = (0..4).map(|i| subgroup_16.domain.element(i)).collect();
        assert_eq!(elements[0], F::ONE);
        assert_eq!(elements[1] * elements[1], elements[2]);
        assert_eq!(elements[2] * elements[1], elements[3]);
        assert_eq!(elements[3], subgroup_16.domain.element(19));

        // Verify coset offset
        assert_eq!(subgroup_16.domain.coset_offset(), F::ONE);

        // Verify subgroup_8 elements within subgroup_16
        let subgroup_8: LagrangeSubgroup<F> = LagrangeSubgroup::new(8);
        for i in 0..8 {
            assert_eq!(subgroup_8.domain.element(i), subgroup_16.domain.element(2 * i));
        }
    }

    #[test]
    fn test_lagrange_basis() {
        let mut rng = test_rng();

        let tau = F::rand(&mut rng);
        let subgroup_16 = LagrangeSubgroup {
            domain: GeneralEvaluationDomain::<F>::new(16).unwrap().get_coset(tau).unwrap(),
        };

        let subgroup_16 =LagrangeSubgroup {
            domain: subgroup_16.domain.get_coset(F::ONE).unwrap(),
        };
        // a subgroup should have an offset of one
        assert_eq!(subgroup_16.domain.coset_offset(), F::ONE);

        let lagrange_polynomials = compute_lagrange_basis::<F>(&subgroup_16.domain);

        // Verify the Lagrange properties: L_i(w^j)
        for (i, poly) in lagrange_polynomials.iter().enumerate() {
            for (j, element) in subgroup_16.domain.elements().enumerate() {
                let result = poly.evaluate(&element);
                let expected = if i == j { F::one() } else { F::zero() };
                assert_eq!(result, expected, "L_{}(w^{}) should be {}", i, j, expected);
            }
        }

        let tau = F::rand(&mut rng);

        // verify that evaluate_all_basis works well
        assert_eq!(
            subgroup_16.evaluate_all_basis(&tau),
            lagrange_polynomials.iter().map(|poly| poly.evaluate(&tau)).collect::<Vec<_>>()
        );
    }
}