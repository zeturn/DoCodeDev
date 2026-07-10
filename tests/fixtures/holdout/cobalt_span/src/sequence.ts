import { buildEnvelope } from "./keystone.js";
import type { EnvelopeView } from "./axiom.js";

export function compileEnvelope(label: string, samples: number[]): EnvelopeView {
  return buildEnvelope({ title: label, samples });
}
