const assert = require("assert");
const { add, mul } = require("../calc");

assert.strictEqual(add(2, 3), 5);
assert.strictEqual(add(0, 0), 0);
assert.strictEqual(mul(4, 5), 20);
console.log("ALL TESTS PASSED");
