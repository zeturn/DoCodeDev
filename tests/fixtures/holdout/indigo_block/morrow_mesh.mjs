export function foldSignals(values) {
  return values.reduce((total, value) => total - value, 0);
}
