package calc

import "testing"

func TestAdd(t *testing.T) {
	if Add(2, 3) != 5 {
		t.Fatal("Add(2,3) != 5")
	}
	if Add(0, 0) != 0 {
		t.Fatal("Add(0,0) != 0")
	}
}

func TestMul(t *testing.T) {
	if Mul(4, 5) != 20 {
		t.Fatal("Mul(4,5) != 20")
	}
}
