use crate::util::is_positive_power_of_two;
use ark_ff::FftField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::CanonicalSerialize;

pub fn split_subgroup<F: FftField>(domain : &GeneralEvaluationDomain<F>, split_factor: usize) -> Vec<GeneralEvaluationDomain<F>> {
    // 1. Check that the domain is indeed a subgroup (not a coset)
    assert_eq!(domain.coset_offset(), F::ONE, "Domain is a coset, not a subgroup.");

    // 2. Ensure `n` is a positive power of two
    is_positive_power_of_two(split_factor);

    // 3. If n == 1, return the current subgroup
    if split_factor == 1 {
        return vec![domain.clone()];
    }

    // 4. Calculate the number of cosets
    let n = domain.size() / split_factor;
    assert_eq!(domain.size() % split_factor, 0, "Size of subgroup must be divisible by n.");

    let mut subgroups = Vec::with_capacity(split_factor);

    // First subgroup (H) is of size `n`
    let first_subgroup_domain = GeneralEvaluationDomain::<F>::new(n).unwrap();
    subgroups.push(first_subgroup_domain);

    // Generate remaining subgroups as cosets of the first subgroup
    let generator = domain.element(1); // ω, the primitive root
    for i in 1..split_factor {
        let offset = generator.pow([i as u64]); // ω^i as offset
        let coset_domain = first_subgroup_domain.get_coset(offset).unwrap();
        subgroups.push(coset_domain);
    }

    subgroups
}

fn print_subgroups<F: FftField>(domain: &GeneralEvaluationDomain<F>, subgroups: &[GeneralEvaluationDomain<F>], split_factor: usize) {
    let generator_symbol = "ω"; // Symbolic representation for ω
    let subgroup_size = domain.size();  // Original subgroup size
    let coset_size = subgroup_size / split_factor;

    println!("Original subgroup elements:");
    let original_elements: Vec<String> = (0..subgroup_size)
        .map(|i| format!("{}^{}", generator_symbol, i))
        .collect();
    println!("{:?}", original_elements);

    println!("\nSplit cosets:");
    for (coset_index, coset) in subgroups.iter().enumerate() {
        let offset_exponent = coset_index;

        assert_eq!(coset.coset_offset(), domain.element(1).pow([offset_exponent as u64]));
        let coset_elements: Vec<String> = (0..coset_size)
            .map(|j| format!("{}^{}", generator_symbol, offset_exponent + j * split_factor))
            .collect();
        println!("Coset {} (offset: {}^{}) : {:?}", coset_index , generator_symbol, coset_index, coset_elements);
    }
}

pub fn split_vector<T: Clone>(vector: &[T], split_factor: usize) -> Vec<Vec<T>> {
    let length = vector.len();
    assert!(length.is_power_of_two(), "Vector length must be a power of two.");
    assert!(split_factor > 0 && (length % split_factor) == 0, "Invalid split factor.");

    let subgroup_size = length / split_factor; // Size of each split vector
    let mut result = Vec::with_capacity(split_factor);

    // Debug mode to print information
    #[cfg(debug_assertions)]  // This ensures the block is only compiled in debug mode
    {
        println!("\nDebugging: Splitting the vector...");
    }

    for i in 0..split_factor {
        let mut subvector = Vec::with_capacity(subgroup_size);
        for j in 0..subgroup_size {
            subvector.push(vector[i + j * split_factor].clone()); // Picking elements based on the pattern
        }

        result.push(subvector);

        // Debug print the subgroup
        #[cfg(debug_assertions)]  // Only print in debug mode
        {
            let subvector_elements: Vec<String> = (0..subgroup_size)
                .map(|j| format!("g{}", i + j * split_factor))
                .collect();
            println!("Vector {}: {:?}", i, subvector_elements);
        }
    }

    result
}


#[cfg(test)]
mod tests {
    use crate::constant_curve::{ScalarField, E};
    use ark_ec::pairing::Pairing;
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_ff::{FftField, Field, One, Zero};
    use ark_poly::univariate::DensePolynomial;
    use ark_poly::{DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
    use ark_std::{test_rng, UniformRand};
    use std::ops::Mul;
    use crate::lagrange_basis::lagrange_basis::{print_subgroups, split_vector, split_subgroup};

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
        let domain_16: GeneralEvaluationDomain<F> = GeneralEvaluationDomain::new(16).unwrap();
        assert_eq!(domain_16.size(), 16);

        // Check subgroup elements
        let elements: Vec<_> = (0..4).map(|i| domain_16.element(i)).collect();
        assert_eq!(elements[0], F::ONE);
        assert_eq!(elements[1] * elements[1], elements[2]);
        assert_eq!(elements[2] * elements[1], elements[3]);
        assert_eq!(elements[3], domain_16.element(19));

        // Verify coset offset
        assert_eq!(domain_16.coset_offset(), F::ONE);

        // Verify subgroup_8 elements within subgroup_16
        let domain_8: GeneralEvaluationDomain<F> = GeneralEvaluationDomain::new(8).unwrap();
        for i in 0..8 {
            assert_eq!(domain_8.element(i), domain_16.element(2 * i));
        }
    }

    #[test]
    fn test_lagrange_basis() {
        let mut rng = test_rng();

        let tau = F::rand(&mut rng);
        let coset_16 = GeneralEvaluationDomain::<F>::new(16).unwrap().get_coset(tau).unwrap();

        let subgroup_16 = coset_16.get_coset(F::ONE).unwrap();

        // a subgroup should have an offset of one
        assert_eq!(subgroup_16.coset_offset(), F::ONE);

        let lagrange_polynomials = compute_lagrange_basis::<F>(&subgroup_16);

        // Verify the Lagrange properties: L_i(w^j)
        for (i, poly) in lagrange_polynomials.iter().enumerate() {
            for (j, element) in subgroup_16.elements().enumerate() {
                let result = poly.evaluate(&element);
                let expected = if i == j { F::one() } else { F::zero() };
                assert_eq!(result, expected, "L_{}(w^{}) should be {}", i, j, expected);
            }
        }

        let tau = F::rand(&mut rng);

        // verify that evaluate_all_basis works well
        assert_eq!(
            subgroup_16.evaluate_all_lagrange_coefficients(tau),
            lagrange_polynomials.iter().map(|poly| poly.evaluate(&tau)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_split_subgroup_offsets() {
        let (subgroup_size, split_factor) = (16usize, 1usize);

        // Generate a subgroup of size 1024
        let subgroup: GeneralEvaluationDomain<F> = GeneralEvaluationDomain::new(subgroup_size).unwrap();

        // Split the subgroup into 16 subgroups of size 64
        let subgroups = split_subgroup(&subgroup, split_factor);

        // Extract the first element (offset) from each split subgroup
        let split_offsets: Vec<F> = subgroups.iter()
            .map(|sg| sg.coset_offset())  // The first element in each subgroup
            .collect();

        // Extract the first 16 elements of the original subgroup
        let original_elements: Vec<F> = (0..split_factor)
            .map(|i| subgroup.element(i))
            .collect();

        // Verify that the offsets match the expected elements
        assert_eq!(
            split_offsets,
            original_elements,
            "The split subgroup offsets do not match the expected elements from the original subgroup."
        );

        print_subgroups(&subgroup, subgroups.as_slice(), split_factor)
    }

    #[test]
    fn test_split_vector() {
        let vector = (0..16).collect::<Vec<_>>(); // Vector from 0 to 15

        let split_factor_2 = split_vector(&vector, 2);
        println!("Split with factor 2: {:?}", split_factor_2);

        let split_factor_4 = split_vector(&vector, 4);
        println!("Split with factor 4: {:?}", split_factor_4);
    }
}