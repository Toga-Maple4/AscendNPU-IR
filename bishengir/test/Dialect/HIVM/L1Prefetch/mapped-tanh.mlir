// RUN: python3 %S/Inputs/check_mapped_tanh.py bishengir-opt %S/l1-prefetch-ps-nested.mlir
// Exercise Maple4's external scalar tanh representation on the existing real
// GELU fixture, preserving its original shape and partial-tile proof.
// RUN: python3 %S/Inputs/check_mapped_tanh_compile.py bishengir-compile %S/l1-prefetch-ps-nested.mlir
