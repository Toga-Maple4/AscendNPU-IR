// RUN: bishengir-opt %s --hfusion-flatten-ops="flatten-mode=tidy register-based=true skip-scope=true skip-host=true" --split-input-file | FileCheck %s
// RUN: bishengir-opt %s --propagate-reshape="for-regbased=true" --hfusion-flatten-ops="flatten-mode=tidy register-based=true skip-scope=true skip-host=true" --split-input-file | FileCheck %s

// Only the split New-L1 vector consumer opts into late flattening.

// -----

// CHECK-LABEL: func.func @new_l1_aiv(
// CHECK: linalg.elemwise_binary
// CHECK-SAME: tensor<128xf32>
func.func @new_l1_aiv(%a: tensor<4x32xf32>, %b: tensor<4x32xf32>) -> tensor<4x32xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, program_split.transformed} {
  %empty = tensor.empty() : tensor<4x32xf32>
  %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<4x32xf32>, tensor<4x32xf32>) outs(%empty : tensor<4x32xf32>) -> tensor<4x32xf32>
  return %sum : tensor<4x32xf32>
}

// -----

// CHECK-LABEL: func.func @ordinary_aiv(
// CHECK-NOT: tensor.collapse_shape
// CHECK: linalg.elemwise_binary
// CHECK-SAME: tensor<4x32xf32>
func.func @ordinary_aiv(%a: tensor<4x32xf32>, %b: tensor<4x32xf32>) -> tensor<4x32xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %empty = tensor.empty() : tensor<4x32xf32>
  %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<4x32xf32>, tensor<4x32xf32>) outs(%empty : tensor<4x32xf32>) -> tensor<4x32xf32>
  return %sum : tensor<4x32xf32>
}

// -----

// CHECK-LABEL: func.func @other_mix_aiv(
// CHECK-NOT: tensor.collapse_shape
// CHECK: linalg.elemwise_binary
// CHECK-SAME: tensor<4x32xf32>
func.func @other_mix_aiv(%a: tensor<4x32xf32>, %b: tensor<4x32xf32>) -> tensor<4x32xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
  %empty = tensor.empty() : tensor<4x32xf32>
  %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<4x32xf32>, tensor<4x32xf32>) outs(%empty : tensor<4x32xf32>) -> tensor<4x32xf32>
  return %sum : tensor<4x32xf32>
}

// -----

// CHECK-LABEL: func.func @not_split_aiv(
// CHECK-NOT: tensor.collapse_shape
// CHECK: linalg.elemwise_binary
// CHECK-SAME: tensor<4x32xf32>
func.func @not_split_aiv(%a: tensor<4x32xf32>, %b: tensor<4x32xf32>) -> tensor<4x32xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, program_split.transformed} {
  %empty = tensor.empty() : tensor<4x32xf32>
  %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<4x32xf32>, tensor<4x32xf32>) outs(%empty : tensor<4x32xf32>) -> tensor<4x32xf32>
  return %sum : tensor<4x32xf32>
}

// -----

// CHECK-LABEL: func.func @new_l1_aic(
// CHECK-NOT: tensor.collapse_shape
// CHECK: linalg.elemwise_binary
// CHECK-SAME: tensor<4x32xf32>
func.func @new_l1_aic(%a: tensor<4x32xf32>, %b: tensor<4x32xf32>) -> tensor<4x32xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, program_split.transformed} {
  %empty = tensor.empty() : tensor<4x32xf32>
  %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<4x32xf32>, tensor<4x32xf32>) outs(%empty : tensor<4x32xf32>) -> tensor<4x32xf32>
  return %sum : tensor<4x32xf32>
}

// -----

// CHECK-LABEL: func.func @untyped(
// CHECK: linalg.elemwise_binary
// CHECK-SAME: tensor<128xf32>
func.func @untyped(%a: tensor<4x32xf32>, %b: tensor<4x32xf32>) -> tensor<4x32xf32> {
  %empty = tensor.empty() : tensor<4x32xf32>
  %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<4x32xf32>, tensor<4x32xf32>) outs(%empty : tensor<4x32xf32>) -> tensor<4x32xf32>
  return %sum : tensor<4x32xf32>
}

// -----

// Elementwise flattening must not merge a reduction axis into parallel work.
// CHECK-LABEL: func.func @new_l1_reduction(
// CHECK-NOT: tensor.collapse_shape
// CHECK: linalg.reduce
// CHECK-SAME: dimensions = [1]
// CHECK-NOT: tensor.collapse_shape
// CHECK: return
func.func @new_l1_reduction(%a: tensor<4x32xf32>, %init: tensor<4xf32>) -> tensor<4xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, program_split.transformed} {
  %result = linalg.reduce ins(%a : tensor<4x32xf32>) outs(%init : tensor<4xf32>) dimensions = [1]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
  return %result : tensor<4xf32>
}

// -----

// The New-L1 exception does not authorize propagation across opaque scopes.
// CHECK-LABEL: func.func @new_l1_opaque_scope(
// CHECK-NOT: tensor.collapse_shape
// CHECK: scope.scope
// CHECK: linalg.elemwise_binary
// CHECK-SAME: tensor<4x32xf32>
// CHECK-NOT: tensor.collapse_shape
// CHECK: return
func.func @new_l1_opaque_scope(%a: tensor<4x32xf32>, %b: tensor<4x32xf32>) -> tensor<4x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, program_split.transformed} {
  %result = scope.scope : () -> tensor<4x32xf32> {
    %empty = tensor.empty() : tensor<4x32xf32>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<4x32xf32>, tensor<4x32xf32>) outs(%empty : tensor<4x32xf32>) -> tensor<4x32xf32>
    scope.return %sum : tensor<4x32xf32>
  }
  return %result : tensor<4x32xf32>
}

// -----

// Retain skip-host even if a malformed/intermediate host carries split tags.
// CHECK-LABEL: func.func @skip_host(
// CHECK-NOT: tensor.collapse_shape
// CHECK: linalg.elemwise_binary
// CHECK-SAME: tensor<4x32xf32>
func.func @skip_host(%a: tensor<4x32xf32>, %b: tensor<4x32xf32>) -> tensor<4x32xf32>
    attributes {hacc.function_kind = #hacc.function_kind<HOST>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, program_split.transformed} {
  %empty = tensor.empty() : tensor<4x32xf32>
  %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<4x32xf32>, tensor<4x32xf32>) outs(%empty : tensor<4x32xf32>) -> tensor<4x32xf32>
  return %sum : tensor<4x32xf32>
}
