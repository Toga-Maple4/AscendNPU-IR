// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-std | FileCheck %s

// Only explicitly marked staging loads select the grouped helper.
// Legacy/manual L1 and ordinary GM->UB keep their existing entry points.
module {
  // CHECK-LABEL: func.func @new_l1_fp32(
  // CHECK: call @load_gm_to_cbuf_2d_float_program_split(
  // CHECK: call @load_gm_to_cbuf_1d_float(
  // CHECK: call @load_gm_to_ubuf_2d_float(
  func.func @new_l1_fp32(%src: memref<16x2048xf32, #hivm.address_space<gm>>, %l1: memref<16x2048xf32, #hivm.address_space<cbuf>>, %ub: memref<16x2048xf32, #hivm.address_space<ub>>) attributes {program_split.transformed} {
    hivm.hir.load ins(%src : memref<16x2048xf32, #hivm.address_space<gm>>) outs(%l1 : memref<16x2048xf32, #hivm.address_space<cbuf>>) {hivm.grouped_l1_load}
    hivm.hir.load ins(%src : memref<16x2048xf32, #hivm.address_space<gm>>) outs(%l1 : memref<16x2048xf32, #hivm.address_space<cbuf>>)
    hivm.hir.load ins(%src : memref<16x2048xf32, #hivm.address_space<gm>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>)
    return
  }
  // CHECK-LABEL: func.func @new_l1_fp16(
  // CHECK: call @load_gm_to_cbuf_2d_half_program_split(
  func.func @new_l1_fp16(%src: memref<16x4096xf16, #hivm.address_space<gm>>, %l1: memref<16x4096xf16, #hivm.address_space<cbuf>>) attributes {program_split.transformed} {
    hivm.hir.load ins(%src : memref<16x4096xf16, #hivm.address_space<gm>>) outs(%l1 : memref<16x4096xf16, #hivm.address_space<cbuf>>) {hivm.grouped_l1_load}
    return
  }
  // CHECK-LABEL: func.func @new_l1_bf16(
  // CHECK: call @load_gm_to_cbuf_2d_bfloat16_t_program_split(
  // CHECK-NOT: load_gm_to_cbuf_1d_bfloat16_t_program_split
  func.func @new_l1_bf16(%src: memref<1x16384xbf16, #hivm.address_space<gm>>, %l1: memref<1x16384xbf16, #hivm.address_space<cbuf>>) attributes {program_split.transformed} {
    hivm.hir.load ins(%src : memref<1x16384xbf16, #hivm.address_space<gm>>) outs(%l1 : memref<1x16384xbf16, #hivm.address_space<cbuf>>) {hivm.grouped_l1_load}
    return
  }
  // CHECK-LABEL: func.func @legacy_l1(
  // CHECK: call @load_gm_to_cbuf_1d_float(
  // CHECK-NOT: _program_split
  // CHECK: return
  func.func @legacy_l1(%src: memref<16x2048xf32, #hivm.address_space<gm>>, %l1: memref<16x2048xf32, #hivm.address_space<cbuf>>) {
    hivm.hir.load ins(%src : memref<16x2048xf32, #hivm.address_space<gm>>) outs(%l1 : memref<16x2048xf32, #hivm.address_space<cbuf>>)
    return
  }
}
