import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("the public API and shared input contract agree", async () => {
  const [axiom, keystone, sequence] = await Promise.all([
    readFile(new URL("../src/axiom.ts", import.meta.url), "utf8"),
    readFile(new URL("../src/keystone.ts", import.meta.url), "utf8"),
    readFile(new URL("../src/sequence.ts", import.meta.url), "utf8"),
  ]);
  assert.match(axiom, /label:\s*string/);
  assert.doesNotMatch(keystone, /input\.title/);
  assert.match(keystone, /caption:\s*input\.label/);
  assert.doesNotMatch(sequence, /\{\s*title:/);
  assert.match(sequence, /export function compileEnvelope\(label: string, samples: number\[\]\)/);
  assert.match(sequence, /buildEnvelope\(\{\s*label,\s*samples\s*\}\)/);
});
