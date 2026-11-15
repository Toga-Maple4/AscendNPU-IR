// Accepted metadata changed after analysis is a pipeline error, not an unsupported kernel.
// RUN: python3 %S/../Dialect/HIVM/L1Prefetch/Inputs/check_handoff.py bishengir-opt %S/../Dialect/HIVM/L1Prefetch/l1-prefetch-ps-analysis.mlir
