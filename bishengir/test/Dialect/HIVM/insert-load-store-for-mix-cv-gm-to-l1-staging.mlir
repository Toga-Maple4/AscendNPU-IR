// RUN: bishengir-opt %s \
// RUN:   -hivm-insert-load-store-for-mix-cv="target=Ascend950PR_9589" \
// RUN:   | FileCheck %s

// Regression test for the GM->L1 staging-load core-type fix in
// InsertPropagation.cpp (InsertLoadStoreForMixCV).
//
// A hivm.hir.load that stages GM into L1 writes into a *plain* alloc (no address
// space) that is bound to a cbuf buffer only by a `bind_buffer` annotation.mark
// on its to_tensor view -- BindBuffer, which would put the address space on the
// memref, does not run until after SplitMixKernel. Such a load runs on the CUBE.
//
// The load sits directly in a `hivm.parallel_loop` scf.for. Without the fix the
// "unstructured load in a parallel loop is always a UB load" branch labels it
// VECTOR/UB; SplitMixKernel then drops it from both the AIC clone (wrong core)
// and the AIV clone (its buffer is cube-side), leaving L1 read uninitialised
// (GELU/SwiGLU returned garbage after the upstream `enablemixedcv removal`).
// The fix recognises the staging pattern and propagates CUBE/L1 first.

// CHECK-LABEL: func.func @stage_gm_to_l1_in_parallel_loop
// The staging load must land on the cube, not the vector core:
// CHECK:      hivm.hir.load
// CHECK-SAME:   core_type = <CUBE>
// CHECK-NOT:    core_type = <VECTOR>
// CHECK:      hivm.parallel_loop

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @stage_gm_to_l1_in_parallel_loop(%gm: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c2048 = arith.constant 2048 : index
    %true = arith.constant true
    %cbuf = memref.alloc() : memref<16x2048xf16, #hivm.address_space<cbuf>>
    annotation.mark %cbuf {effects = ["write", "read"]} : memref<16x2048xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0 to %c16 step %c1 {
      %gm2d = memref.reinterpret_cast %gm to offset: [0], sizes: [16, 2048], strides: [16384, 1] : memref<?xf16> to memref<16x2048xf16, strided<[16384, 1], offset: ?>>
      %staging = memref.alloc() : memref<16x2048xf16>
      %src = memref.subview %gm2d[0, 0] [%c16, %c2048] [1, 1] : memref<16x2048xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
      %dst = memref.subview %staging[0, 0] [%c16, %c2048] [1, 1] : memref<16x2048xf16> to memref<?x?xf16, strided<[2048, 1]>>
      hivm.hir.load ins(%src : memref<?x?xf16, strided<[16384, 1], offset: ?>>) outs(%dst : memref<?x?xf16, strided<[2048, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %true : i1 eviction_policy = <EvictFirst>
      %t = bufferization.to_tensor %staging restrict writable : memref<16x2048xf16>
      annotation.mark %t keys = ["bind_buffer"] values = [%cbuf : memref<16x2048xf16, #hivm.address_space<cbuf>>] : tensor<16x2048xf16>
    } {hivm.parallel_loop}
    return
  }
}
