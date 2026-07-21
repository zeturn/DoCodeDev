package calc

// Add returns the sum of a and b.
func Add(a, b int) int {
	return a - b // BUG: should return a + b
}

// Mul returns the product of a and b.
func Mul(a, b int) int {
	return a * b
}
