// borrowed from Espresso/JellyFish


use ark_ff::{FftField, Field, Zero};
use ark_poly::domain::DomainCoeff;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::UniformRand;
use itertools::{
    EitherOrBoth::{Both, Left, Right},
    Itertools,
};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Mul, MulAssign};
use ark_std::rand::RngCore as Rng;

// `https://doc.rust-lang.org/unstable-book/language-features/trait-alias.html`
/// A trait bound alias for generalized coefficient type used in
/// `GeneralDensePolynomial`. Concrete instantiations can be both field or
/// group elements.
pub trait GroupCoeff<F: Field>:
Clone + fmt::Debug + Copy + Add<Self, Output=Self> + MulAssign<F> + Zero + UniformRand + Sized
{}

impl<T, F: Field> GroupCoeff<F> for T
where
    T: Clone
    + fmt::Debug
    + Copy
    + Add<Self, Output=Self>
    + MulAssign<F>
    + Zero
    + UniformRand
    + Sized,
{}

/// Stores a polynomial in the coefficient form.
#[derive(Clone, PartialEq, Eq, Default)]
pub struct GeneralDensePolynomial<T: GroupCoeff<F>, F: Field> {
    /// The coefficient of `x^i` is stored at location `i` in `self.coeffs`.
    pub coeffs: Vec<T>,
    _phantom: PhantomData<F>,
}

// struct because that trait assumes the coeffs are field elements. Therefore, we
// need to generalize that trait first which is left for future work or even
// upstream PR.
impl<T, F> GeneralDensePolynomial<T, F>
where
    T: GroupCoeff<F>,
    F: Field,
{
    /// Constructs a new polynomial from a list of coefficients
    pub fn from_coeff_slice(coeffs: &[T]) -> Self {
        Self::from_coeff_vec(coeffs.to_vec())
    }

    /// Constructs a new polynomial from a list of coefficients
    pub fn from_coeff_vec(coeffs: Vec<T>) -> Self {
        let mut result = Self {
            coeffs,
            _phantom: PhantomData,
        };
        result.truncate_leading_zeros();
        assert!(result.coeffs.last().map_or(true, |coeff| !coeff.is_zero()));
        result
    }

    /// Outputs a univariate polynomial of degree `d` where
    /// each coefficient is sampled uniformly at random.
    #[allow(dead_code)]
    pub fn rand<R: Rng>(d: usize, rng: &mut R) -> Self {
        let mut random_coeffs = Vec::new();
        for _ in 0..=d {
            random_coeffs.push(T::rand(rng));
        }
        Self::from_coeff_vec(random_coeffs)
    }

    /// Returns the total degree of the polynomial
    #[allow(dead_code)]
    pub fn degree(&self) -> usize {
        if self.is_zero() {
            0
        } else {
            assert!(self.coeffs.last().map_or(false, |coeff| !coeff.is_zero()));
            self.coeffs.len() - 1
        }
    }

    /// Evaluate `self` at a given `point`
    pub fn evaluate(&self, point: &F) -> T {
        if self.is_zero() {
            return T::zero();
        } else if point.is_zero() {
            return self.coeffs[0];
        }
        self.horner_evaluate(point)
    }
}

impl<T, F> GeneralDensePolynomial<T, F>
where
    T: GroupCoeff<F> + DomainCoeff<F> + for<'a> Mul<&'a F, Output=T>,
    F: FftField,
{
    /// Evaluate `self` at a list of arbitrary `points`.
    pub fn batch_evaluate(&self, points: &[F]) -> Vec<T> {
        // Horner: n * d
        // FFT: ~ d*log^2(d)  (independent of n, as long as n<=d)
        // naive cutoff-point, not taking parallelism into consideration:
        let cutoff_size = <F as FftField>::TWO_ADICITY.pow(2);

        if points.is_empty() {
            Vec::new()
        } else if points.len() < cutoff_size as usize {
            points.iter().map(|x| self.evaluate(x)).collect()
        } else {
            unimplemented!("TODO: (alex) implements Appendix A of FK23");
        }
    }

    /// Similar task as [`Self::batch_evaluate()`], except the points are
    /// [roots of unity](https://en.wikipedia.org/wiki/Root_of_unity) of `domain`.
    /// By leveraging FFT algorithms, we have a much lower amortized cost.
    /// It returns f(w^0), f(w^1), ..., f(w^n)
    pub fn batch_evaluate_rou(
        &mut self,
        domain: &GeneralEvaluationDomain<F>,
    ) -> Vec<T> {
        assert!(self.coeffs.len() <= domain.size(),
                "Polynomial with {} num_of_coeffs can't be evaluated on a smaller domain with size {}",
                self.coeffs.len(), domain.size(),
        );
        self.coeffs.resize_with(domain.size(), Zero::zero);

        domain.fft(&self.coeffs)
    }
}

impl<T, F> GeneralDensePolynomial<T, F>
where
    T: GroupCoeff<F>,
    F: Field,
{
    fn truncate_leading_zeros(&mut self) {
        while self.coeffs.last().map_or(false, |c| c.is_zero()) {
            self.coeffs.pop();
        }
    }

    // Horner's method for polynomial evaluation with cost O(n).
    fn horner_evaluate(&self, point: &F) -> T {
        self.coeffs
            .iter()
            .rfold(T::zero(), move |mut result, coeff| {
                result *= *point;
                result + *coeff
            })
    }
}

impl<T, F> Zero for GeneralDensePolynomial<T, F>
where
    T: GroupCoeff<F>,
    F: Field,
{
    /// returns the zero polynomial
    fn zero() -> Self {
        Self {
            coeffs: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// checks if the given polynomial is zero
    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(|coeff| coeff.is_zero())
    }
}

impl<T, F> Add<Self> for GeneralDensePolynomial<T, F>
where
    T: GroupCoeff<F>,
    F: Field,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut res = if self.is_zero() {
            rhs
        } else if rhs.is_zero() {
            self
        } else {
            let coeffs = self
                .coeffs
                .into_iter()
                .zip_longest(rhs.coeffs)
                .map(|pair| match pair {
                    Both(x, y) => x + y,
                    Left(x) | Right(x) => x,
                })
                .collect();
            Self::from_coeff_vec(coeffs)
        };

        res.truncate_leading_zeros();
        res
    }
}

impl<T, F> fmt::Debug for GeneralDensePolynomial<T, F>
where
    T: GroupCoeff<F>,
    F: Field,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        for (i, coeff) in self.coeffs.iter().enumerate().filter(|(_, c)| !c.is_zero()) {
            if i == 0 {
                write!(f, "\n{:?}", coeff)?;
            } else if i == 1 {
                write!(f, " + \n{:?} * x", coeff)?;
            } else {
                write!(f, " + \n{:?} * x^{}", coeff, i)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cmp::max;
    use super::*;
    use crate::constant_curve::{G1Projective, ScalarField, E};
    use ark_ec::{short_weierstrass::SWCurveConfig, CurveGroup};
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
    use ark_std::iter::successors;
    use ark_std::test_rng;
    use ark_std::rand::Rng;
    use crate::kzg::SRS;

    type Fr = ScalarField;

    #[test]
    fn test_poly_eval_single_point() {
        let mut rng = test_rng();
        for _ in 0..5 {
            let degree = rng.gen_range(5..40);
            let f = GeneralDensePolynomial::<Fr, Fr>::rand(degree, &mut rng);
            let g = GeneralDensePolynomial::<G1Projective, Fr>::rand(degree, &mut rng);
            let point = Fr::rand(&mut rng);

            let f_x = f.evaluate(&point);
            let expected_f_x: Fr =
                DensePolynomial::from_coefficients_slice(&f.coeffs).evaluate(&point);
            assert_eq!(f_x, expected_f_x);

            let g_x = g.evaluate(&point);
            // g(X) = G0 + G1 * X + G2 * X^2 + G3 * X^3 + ...
            // turn into an MSM between [G0, G1, G2, ...] and [1, x, x^2, x^3, ...]
            let scalars: Vec<Fr> = successors(Some(Fr::from(1u32)), |&prev| Some(prev * point))
                .take(g.degree() + 1)
                .collect();
            let expected_g_x =
                SWCurveConfig::msm(&CurveGroup::normalize_batch(&g.coeffs), &scalars).unwrap();
            assert_eq!(g_x, expected_g_x);
        }
    }

    #[test]
    fn test_poly_add() {
        let mut rng = test_rng();
        for _ in 0..5 {
            let degree = rng.gen_range(5..40);
            let f_1 = GeneralDensePolynomial::<Fr, Fr>::rand(degree, &mut rng);
            let mut f_2 = GeneralDensePolynomial::<Fr, Fr>::rand(degree, &mut rng);
            let expected_f_3 = DensePolynomial::from_coefficients_slice(&f_1.coeffs)
                + DensePolynomial::from_coefficients_slice(&f_2.coeffs);
            let f_3 = f_1.clone() + f_2.clone();
            assert_eq!(f_3.coeffs, expected_f_3.coeffs);

            f_2.coeffs[degree] = -f_1.coeffs[degree];
            let f_4 = f_1 + f_2;
            assert_eq!(f_4.degree(), degree - 1,);

            let g_1 = GeneralDensePolynomial::<G1Projective, Fr>::rand(degree, &mut rng);
            let mut g_2 = GeneralDensePolynomial::<G1Projective, Fr>::rand(degree, &mut rng);
            let expected_g_3: Vec<G1Projective> = g_1
                .coeffs
                .iter()
                .zip(g_2.coeffs.iter())
                .map(|(a, b)| a + b)
                .collect();
            let g_3 = g_1.clone() + g_2.clone();
            assert_eq!(g_3.coeffs, expected_g_3);

            g_2.coeffs[degree] = -g_1.coeffs[degree];
            g_2.coeffs[degree - 1] = -g_1.coeffs[degree - 1];
            g_2.coeffs[degree - 2] = -g_1.coeffs[degree - 2];
            let g_4 = g_1 + g_2;
            assert_eq!(g_4.degree(), degree - 3);
        }
    }

    #[test]
    fn test_multi_open() {
        let mut rng = test_rng();
        let degrees = [14, 15, 16, 17, 18];

        for degree in degrees {
            // FFT-based eval on arbitrary points
            let num_points = rng.gen_range(degree - 4..degree + 4); // should allow more points than degree
            let mut f = GeneralDensePolynomial::<Fr, Fr>::rand(degree, &mut rng);
            let mut g = GeneralDensePolynomial::<G1Projective, Fr>::rand(degree, &mut rng);

            let points: Vec<Fr> = (0..num_points).map(|_| Fr::rand(&mut rng)).collect();
            ark_std::println!("degree: {}, num_points: {}", degree, num_points);

            // First, test general points
            assert_eq!(
                f.batch_evaluate(&points),
                points.iter().map(|x| f.evaluate(x)).collect::<Vec<_>>()
            );
            assert_eq!(
                g.batch_evaluate(&points),
                points.iter().map(|x| g.evaluate(x)).collect::<Vec<_>>()
            );

            // Second, test points at roots-of-unity
            let domain = GeneralEvaluationDomain::new(
                max(degree + 1, num_points)
            ).unwrap();
            let roots = domain.elements();

            assert_eq!(
                f.batch_evaluate_rou(&domain)[..num_points],
                roots
                    .take(num_points)
                    .map(|x| f.evaluate(&x))
                    .collect::<Vec<_>>()
            );

            assert_eq!(
                g.batch_evaluate_rou(&domain)[..num_points],
                domain
                    .elements()
                    .take(num_points)
                    .map(|x| g.evaluate(&x))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_fft() {
        let mut rng = test_rng();
        let srs: SRS<E> = SRS::setup(4, &mut rng);

        let mut poly = GeneralDensePolynomial::from_coeff_vec(srs.powers_of_g.clone());

        let subgroup: GeneralEvaluationDomain<Fr> = GeneralEvaluationDomain::new(4).unwrap();

        let evals = poly.batch_evaluate_rou(&subgroup);

        let eval_w0 = poly.evaluate(&subgroup.element(0));
        let eval_w1 = poly.evaluate(&subgroup.element(1));
        let eval_w2 = poly.evaluate(&subgroup.element(2));
        let eval_w3 = poly.evaluate(&subgroup.element(3));

        assert_eq!(evals, vec![eval_w0, eval_w1, eval_w2, eval_w3]);
    }
}