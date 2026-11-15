// RUN: bishengir-opt %s -convert-hivm-to-std="mark-libcall-noinline=true" -split-input-file | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: bishengir-opt %s -convert-hivm-to-std -split-input-file | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: bishengir-opt %s -convert-hivm-to-std="mark-libcall-noinline=false" -split-input-file | FileCheck %s --check-prefixes=CHECK,INLINE

// The RegBase 2D output-store helper must inline even when other AIV helpers
// retain the default no-inline policy. Dynamic dimensions/strides match the
// failing hardware path; the exception is independent of shape and datatype.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @store_f32(%src: memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>,
                        %dst: memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>) {
    hivm.hir.store ins(%src : memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>)
                  outs(%dst : memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>)
    return
  }
}
// CHECK-LABEL: func.func private @store_ubuf_to_gm_2d_float
// CHECK-SAME: hacc.always_inline
// CHECK-NOT: hacc.noinline

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @store_f16(%src: memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<ub>>,
                        %dst: memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>>) {
    hivm.hir.store ins(%src : memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
                  outs(%dst : memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>>)
    return
  }
}
// CHECK-LABEL: func.func private @store_ubuf_to_gm_2d_half
// CHECK-SAME: hacc.always_inline
// CHECK-NOT: hacc.noinline

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @store_bf16(%src: memref<?x?xbf16, strided<[?, 1]>, #hivm.address_space<ub>>,
                        %dst: memref<?x?xbf16, strided<[?, 1]>, #hivm.address_space<gm>>) {
    hivm.hir.store ins(%src : memref<?x?xbf16, strided<[?, 1]>, #hivm.address_space<ub>>)
                  outs(%dst : memref<?x?xbf16, strided<[?, 1]>, #hivm.address_space<gm>>)
    return
  }
}
// CHECK-LABEL: func.func private @store_ubuf_to_gm_2d_bfloat16
// CHECK-SAME: hacc.always_inline
// CHECK-NOT: hacc.noinline

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @other_helpers(%gm: memref<?xf32, #hivm.address_space<gm>>,
                           %ub: memref<?xf32, #hivm.address_space<ub>>) {
    hivm.hir.load ins(%gm : memref<?xf32, #hivm.address_space<gm>>)
                 outs(%ub : memref<?xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%ub : memref<?xf32, #hivm.address_space<ub>>)
                  outs(%gm : memref<?xf32, #hivm.address_space<gm>>)
    return
  }
}
// CHECK-LABEL: func.func private @load_gm_to_ubuf_1d_float
// DEFAULT-SAME: hacc.noinline
// INLINE-SAME: hacc.always_inline
// CHECK-LABEL: func.func private @store_ubuf_to_gm_1d_float
// DEFAULT-SAME: hacc.noinline
// INLINE-SAME: hacc.always_inline

// -----

// The same rank and datatype do not exempt an input-load helper.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @load_2d(%gm: memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>,
                     %ub: memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>) {
    hivm.hir.load ins(%gm : memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>)
                 outs(%ub : memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>)
    return
  }
}
// CHECK-LABEL: func.func private @load_gm_to_ubuf_2d_float
// DEFAULT-SAME: hacc.noinline
// INLINE-SAME: hacc.always_inline
