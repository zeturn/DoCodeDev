function add(a, b) {
  return a - b; // BUG: should return a + b
}

function mul(a, b) {
  return a * b;
}

module.exports = { add, mul };
