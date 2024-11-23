/// this function checks a number is positive and a power of two
pub fn is_positive_power_of_two(size: usize) {
    assert!(size > 0 && (size & (size - 1)) == 0, "size is either non-positive or not a power of two")
}
