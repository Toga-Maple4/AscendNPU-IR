// RUN: bishengir-opt %s -hivm-split-mixed-if-conditionals -hivm-split-mix-kernel -verify-diagnostics | FileCheck %s

// A runtime guard selects a mixed CUBE+VECTOR L1 fast path or the original
// VECTOR-only fallback loop. Both split kernels recompute the same scalar
// predicate from the same arguments. The AIC keeps only CUBE work; the AIV
// keeps both the fast-path VECTOR work and the fallback loop's VECTOR work.

// CHECK-LABEL: func.func @runtime_versioned_prefetch_mix_aic(
// CHECK-SAME:    %[[AIC_TRIPS:[^:]+]]: index, %[[AIC_MIN:[^:]+]]: index
// CHECK:         %[[AIC_GUARD:.*]] = arith.cmpi sge, %[[AIC_TRIPS]], %[[AIC_MIN]] : index
// CHECK:         scf.if %[[AIC_GUARD]] {
// CHECK-NOT:       hivm.hir.vadd
// CHECK:           scope.scope
// CHECK:             hivm.hir.mmadL1
// CHECK-NOT:       hivm.hir.vadd
// CHECK:         scf.if %[[AIC_GUARD]] {
// CHECK:         } else {
// CHECK:           scf.for
// CHECK-NOT:       hivm.hir.vadd
// CHECK:         return

// CHECK-LABEL: func.func @runtime_versioned_prefetch_mix_aiv(
// CHECK-SAME:    %[[AIV_TRIPS:[^:]+]]: index, %[[AIV_MIN:[^:]+]]: index
// CHECK:         %[[AIV_GUARD:.*]] = arith.cmpi sge, %[[AIV_TRIPS]], %[[AIV_MIN]] : index
// CHECK:         scf.if %[[AIV_GUARD]] {
// CHECK:           scope.scope
// CHECK-NOT:         hivm.hir.mmadL1
// CHECK:             %[[FAST:.*]] = hivm.hir.vadd
// CHECK:             "test.consume"(%[[FAST]])
// CHECK:         scf.if %[[AIV_GUARD]] {
// CHECK:         } else {
// CHECK:           scf.for
// CHECK-NOT:         hivm.hir.mmadL1
// CHECK:             %[[FALLBACK:.*]] = hivm.hir.vadd
// CHECK:             "test.consume"(%[[FALLBACK]]
// CHECK-NOT:     hivm.hir.mmadL1

module {
  func.func @runtime_versioned_prefetch(
      %trip_count: index, %minimum_trips: index,
      %lb: index, %ub: index, %step: index, %matrix_extent: index,
      %a: tensor<32x32xf16>, %v0: tensor<32x32xf32>,
      %v1: tensor<32x32xf32>)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %guard =
      arith.cmpi sge, %trip_count, %minimum_trips : index
    scf.if %guard {
      scope.scope : () -> () {
        %out = tensor.empty() : tensor<32x32xf32>
        %mm = hivm.hir.mmadL1
          {already_set_real_mkn, fixpipe_already_inserted = true}
          ins(%a, %a, %guard, %matrix_extent, %matrix_extent, %matrix_extent
              : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index)
          outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>
        "test.consume"(%mm) : (tensor<32x32xf32>) -> ()
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scope.scope : () -> () {
        %out = tensor.empty() : tensor<32x32xf32>
        %sum = hivm.hir.vadd
          ins(%v0, %v1 : tensor<32x32xf32>, tensor<32x32xf32>)
          outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>
        "test.consume"(%sum) : (tensor<32x32xf32>) -> ()
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    } else {
      scf.for %iv = %lb to %ub step %step {
        %out = tensor.empty() : tensor<32x32xf32>
        %fallback = hivm.hir.vadd
          ins(%v0, %v1 : tensor<32x32xf32>, tensor<32x32xf32>)
          outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>
        "test.consume"(%fallback, %iv)
          : (tensor<32x32xf32>, index) -> ()
      }
    }
    return
  }
}
