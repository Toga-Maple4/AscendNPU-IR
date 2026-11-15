// RUN: bishengir-opt %s --hfusion-flatten-ops="flatten-mode=tidy register-based=true" --canonicalize | FileCheck %s
// Saved SiLU New-L1 consumer immediately before delayed flattening. Keep its
// inactive-worker branch, partial store bounds, and cross-core handshakes.
// CHECK-LABEL: func.func @_silu_fwd_kernel_mix_aiv
// CHECK-SAME: program_split.launch
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_V>] flag = 1
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
// CHECK: scf.if
// CHECK: memref.alloc() : memref<16x4096xf16, #hivm.address_space<ub>>
// CHECK: program_split.selected_input = 0
// CHECK: tensor.collapse_shape
// CHECK-SAME: tensor<16x4096xf16> into tensor<65536xf16>
// CHECK: linalg.exp
// CHECK-SAME: tensor<65536xf32>
// CHECK: tensor.expand_shape
// CHECK-SAME: tensor<65536xf16> into tensor<16x4096xf16>
// CHECK: tensor.extract_slice
// CHECK-SAME: tensor<16x4096xf16> to tensor<?x?xf16>
// CHECK: hivm.hir.store
// CHECK-SAME: tensor<?x?xf16>
// CHECK-SAME: memref<?x?xf16, strided<[16384, 1], offset: ?>>
// CHECK-SAME: hivm.sync_memory_read_ids = array<i32: 0>
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_MTE1>] flag = 3
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2

func.func @_silu_fwd_kernel_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, func_dyn_memref_args = dense<[true, true, true, true, false, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd", program_split.launch = {element_bits = 16 : i32, input_args = array<i32: 2>, output_arg = 3 : i32, partial_tiles = 1 : i32, row_stride = 16384 : i32, rows_arg = 4 : i32, static_cols = 16384 : i32, tile_cols = 4096 : i32, tile_rows = 16 : i32}, program_split.transformed} {
  %c16_i32 = arith.constant 16 : i32
  %c15_i32 = arith.constant 15 : i32
  %c0_i32 = arith.constant 0 : i32
  %c4096_i32 = arith.constant 4096 : i32
  %cst = arith.constant 1.000000e+00 : f32
  %c16384 = arith.constant 16384 : index
  %c16 = arith.constant 16 : index
  %c4096 = arith.constant 4096 : index
  %c0 = arith.constant 0 : index
  %c4_i32 = arith.constant 4 : i32
  %cst_0 = arith.constant -1.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 {program_split.physical_id} : i64 to i32
  %4 = arith.divsi %1, %c2_i32 {program_split.physical_count} : i32
  %5 = hivm.hir.get_sub_block_idx -> i64
  %6 = arith.trunci %5 : i64 to i32
  %7 = arith.muli %6, %4 : i32
  %8 = arith.addi %3, %7 : i32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %9 = arith.remsi %8, %arg5 : i32
  %10 = arith.addi %arg4, %c15_i32 : i32
  %11 = arith.divsi %10, %c16_i32 : i32
  %12 = arith.addi %11, %arg5 : i32
  %13 = arith.subi %12, %c1_i32 : i32
  %14 = arith.divsi %13, %arg5 : i32
  %15 = arith.muli %14, %c4_i32 : i32
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_MTE1>] flag = 3
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
  scf.for %arg8 = %c0_i32 to %15 step %c1_i32  : i32 {
    %16 = arith.divsi %arg8, %c4_i32 : i32
    %17 = arith.remsi %arg8, %c4_i32 : i32
    %18 = arith.muli %16, %arg5 : i32
    %19 = arith.addi %9, %18 : i32
    %20 = arith.muli %17, %c4096_i32 : i32
    %21 = arith.cmpi slt, %19, %11 : i32
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_V>] flag = 1
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    scf.if %21 {
      %22 = arith.muli %19, %c16_i32 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = arith.index_cast %20 : i32 to index
      %25 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 16384)>()[%24, %23]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [16, 4096], strides: [16384, 1] : memref<?xf16> to memref<16x4096xf16, strided<[16384, 1], offset: ?>>
      %alloc = memref.alloc() : memref<16x4096xf16, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x4096xf16, #hivm.address_space<ub>>
      annotation.mark %alloc {hivm.sync_memory_read_id = 0 : i32, program_split.selected_input = 0 : i32} : memref<16x4096xf16, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc : memref<16x4096xf16, #hivm.address_space<ub>> to memref<16x4096xf16>
      %26 = affine.apply affine_map<()[s0] -> (s0 + 16)>()[%23]
      %27 = arith.index_cast %arg4 : i32 to index
      %28 = arith.maxsi %23, %27 : index
      %29 = arith.minsi %26, %28 : index
      %30 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%29, %23]
      %31 = affine.apply affine_map<()[s0] -> (s0 + 4096)>()[%24]
      %32 = arith.maxsi %24, %c16384 : index
      %33 = arith.minsi %31, %32 : index
      %34 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%33, %24]
      %35 = arith.minsi %30, %c16 : index
      %36 = arith.maxsi %35, %c0 : index
      %37 = arith.minsi %34, %c4096 : index
      %38 = arith.maxsi %37, %c0 : index
      %39 = bufferization.to_tensor %memspacecast restrict writable : memref<16x4096xf16>
      %40 = tensor.empty() : tensor<16x4096xf32>
      %41 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%39 : tensor<16x4096xf16>) outs(%40 : tensor<16x4096xf32>) -> tensor<16x4096xf32>
      %42 = linalg.fill ins(%cst_0 : f32) outs(%40 : tensor<16x4096xf32>) -> tensor<16x4096xf32>
      %43 = linalg.mul ins(%41, %42 : tensor<16x4096xf32>, tensor<16x4096xf32>) outs(%40 : tensor<16x4096xf32>) -> tensor<16x4096xf32>
      %44 = linalg.exp ins(%43 : tensor<16x4096xf32>) outs(%40 : tensor<16x4096xf32>) -> tensor<16x4096xf32>
      %45 = linalg.fill ins(%cst : f32) outs(%40 : tensor<16x4096xf32>) -> tensor<16x4096xf32>
      %46 = linalg.add ins(%44, %45 : tensor<16x4096xf32>, tensor<16x4096xf32>) outs(%40 : tensor<16x4096xf32>) -> tensor<16x4096xf32>
      %47 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%41, %46 : tensor<16x4096xf32>, tensor<16x4096xf32>) outs(%40 : tensor<16x4096xf32>) -> tensor<16x4096xf32>
      %48 = tensor.empty() : tensor<16x4096xf16>
      %49 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%47 : tensor<16x4096xf32>) outs(%48 : tensor<16x4096xf16>) -> tensor<16x4096xf16>
      %extracted_slice = tensor.extract_slice %49[0, 0] [%36, %38] [1, 1] : tensor<16x4096xf16> to tensor<?x?xf16>
      %subview = memref.subview %reinterpret_cast[0, 0] [%36, %38] [1, 1] : memref<16x4096xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
      hivm.hir.store ins(%extracted_slice : tensor<?x?xf16>) outs(%subview : memref<?x?xf16, strided<[16384, 1], offset: ?>>) {hivm.sync_memory_read_ids = array<i32: 0>}
    }
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_MTE1>] flag = 3
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
  }
  hivm.hir.set_ctrl true at ctrl[60]
  return
}
