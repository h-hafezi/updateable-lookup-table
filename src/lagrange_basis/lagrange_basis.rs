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
    use ark_ff::Field;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

    type F = ScalarField;

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
}