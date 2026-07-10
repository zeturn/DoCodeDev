import type { EnvelopeInput, EnvelopeView } from "./axiom.js";

export function buildEnvelope(input: EnvelopeInput): EnvelopeView {
  return {
    caption: input.title,
    total: input.samples.reduce((sum, value) => sum + value, 0),
  };
}
