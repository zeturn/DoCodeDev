import assert from "node:assert/strict";
import test from "node:test";

import { foldSignals } from "../morrow_mesh.mjs";

test("foldSignals totals signed values", () => {
  assert.equal(foldSignals([4, -2, 9]), 11);
});
