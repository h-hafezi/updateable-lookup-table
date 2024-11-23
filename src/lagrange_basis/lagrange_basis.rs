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

    pub fn get_coset(&self, f: F) -> LagrangeCoset<F> {
        LagrangeCoset {
            domain: self.domain.get_coset(f).unwrap(),
        }
    }

    pub fn size(&self) -> usize {
        self.domain.size()
    }

}

#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize)]
pub struct LagrangeCoset<F: FftField> {
    pub domain: GeneralEvaluationDomain<F>,
}

impl<F: FftField> LagrangeCoset<F> {
    pub fn size(&self) -> usize {
        self.domain.size()
    }

    pub fn get_offset(&self) -> F {
        self.domain.coset_offset()
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
        let subgroup_16 = LagrangeSubgroup {
            domain: GeneralEvaluationDomain::<F>::new(16).unwrap()
        };

        assert_eq!(subgroup_16.domain.size(), 16);
        let w_0 = subgroup_16.domain.element(0);
        let w_1 = subgroup_16.domain.element(1);
        let w_2 = subgroup_16.domain.element(2);
        let w_3 = subgroup_16.domain.element(3);
        let w_19 = subgroup_16.domain.element(19);

        assert_eq!(w_0, F::ONE);
        assert_eq!(w_1 * w_1, w_2);
        assert_eq!(w_2 * w_1, w_3);
        assert_eq!(w_3, w_19);

        // every subgroup coset_offset is one
        assert_eq!(subgroup_16.domain.coset_offset(), F::ONE);

        let coset = subgroup_16.get_coset(F::ONE);
        assert_eq!(coset.domain, subgroup_16.domain);

        let subgroup_8 = LagrangeSubgroup {
            domain: GeneralEvaluationDomain::<F>::new(8).unwrap()
        };

        for i in 0..8usize {
            assert_eq!(subgroup_8.domain.element(i), subgroup_16.domain.element(2 * i));
        }
    }

    #[test]
    fn test_lagrange_basis() {
        let mut rng = test_rng();

        let subgroup_16 = LagrangeSubgroup {
            domain: GeneralEvaluationDomain::<F>::new(16).unwrap()
        };

        let lagrange_polynomials = compute_lagrange_basis::<F>(&subgroup_16.domain);

        // Verify the Lagrange properties: L_i(w^j)
        for i in 0..subgroup_16.size() {
            for j in 0..subgroup_16.size() {
                let result = lagrange_polynomials[i].evaluate(&subgroup_16.domain.element(j));
                if i == j {
                    assert_eq!(result, F::one(), "L_{}(w^{}) should be 1", i, j);
                } else {
                    assert_eq!(result, F::zero(), "L_{}(w^{}) should be 0", i, j);
                }
            }
        }

        let tau = F::rand(&mut rng);

        // verify that evaluate_all_basis works well
        assert_eq!(
            subgroup_16.evaluate_all_basis(&tau),
            {
                let mut res = Vec::new();
                for poly in lagrange_polynomials {
                    res.push(poly.evaluate(&tau));
                }
                res
            }
        )
    }
}