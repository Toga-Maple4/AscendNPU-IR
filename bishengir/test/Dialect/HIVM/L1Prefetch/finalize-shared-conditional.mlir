// RUN: bishengir-opt %s -l1-prefetch-balance-sync | FileCheck %s
// Captured from l1-prefetch-ps-retry.mlir with shared UB conditional intact.
// Both cores must participate in the existing handshake on inactive work.
// CHECK-LABEL: func.func @_gelu_direct_56aiv_mix_aic(
// CHECK: scf.for
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_MTE1>] flag = 3
// CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
// CHECK-NEXT: scf.if
// CHECK: hivm.hir.l12ub
// CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_V>] flag = 1
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
// CHECK-LABEL: func.func @_gelu_direct_56aiv_mix_aiv(
// CHECK: scf.for
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_V>] flag = 1
// CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
// CHECK-NEXT: scf.if
// CHECK: hivm.hir.store
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_MTE1>] flag = 3
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @_gelu_direct_56aiv_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd", program_split.launch = {input_args = array<i32: 2>, logical_programs = 56 : i32, output_arg = 3 : i32, persistent_steps = 148 : i32, physical_programs = 28 : i32, row_tasks = 2048 : i32}, program_split.transformed} {
  %c2_i32 = arith.constant 2 : i32
  %c148_i32 = arith.constant 148 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c2048_i32 = arith.constant 2048 : i32
  %c56_i32 = arith.constant 56 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c28_i32 = arith.constant {program_split.physical_count} 28 : i32
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 {program_split.physical_id} : i64 to i32
  %alloc = memref.alloc() : memref<16x2048xf32, #hivm.address_space<cbuf>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<16x2048xf32, #hivm.address_space<cbuf>>
  %alloc_0 = memref.alloc() : memref<16x2048xf32, #hivm.address_space<cbuf>>
  annotation.mark %alloc_0 {effects = ["write", "read"], hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<16x2048xf32, #hivm.address_space<cbuf>>
  %alloc_1 = memref.alloc() : memref<16x2048xf32, #hivm.address_space<cbuf>>
  annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<16x2048xf32, #hivm.address_space<cbuf>>
  %alloc_2 = memref.alloc() : memref<16x2048xf32, #hivm.address_space<cbuf>>
  annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<16x2048xf32, #hivm.address_space<cbuf>>
  %4 = hivm.hir.get_sub_block_idx -> i64
  %5 = arith.trunci %4 : i64 to i32
  %6 = arith.muli %5, %c28_i32 : i32
  %7 = arith.addi %3, %6 : i32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %8 = arith.remsi %7, %arg5 : i32
  %9 = arith.remsi %3, %arg5 : i32
  %10 = arith.cmpi slt, %9, %c2048_i32 : i32
  scf.if %10 {
    %14 = arith.muli %9, %c16_i32 : i32
    %15 = arith.index_cast %14 : i32 to index
    %16 = affine.apply affine_map<()[s0] -> (s0 * 8192)>()[%15]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<16x2048xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_3 : memref<16x2048xf32>) {hivm.grouped_l1_load} eviction_policy = <EvictFirst> core_type = <CUBE>
    %17 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x2048xf32>
    annotation.mark %17 keys = ["bind_buffer"] values = [%alloc : memref<16x2048xf32, #hivm.address_space<cbuf>>] : tensor<16x2048xf32>
  } {hivm.cube_only}
  %11 = arith.addi %3, %c28_i32 : i32
  %12 = arith.remsi %11, %arg5 : i32
  %13 = arith.cmpi slt, %12, %c2048_i32 : i32
  scf.if %13 {
    %14 = arith.muli %12, %c16_i32 : i32
    %15 = arith.index_cast %14 : i32 to index
    %16 = affine.apply affine_map<()[s0] -> (s0 * 8192)>()[%15]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<16x2048xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_3 : memref<16x2048xf32>) {hivm.grouped_l1_load} eviction_policy = <EvictFirst> core_type = <CUBE>
    %17 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x2048xf32>
    annotation.mark %17 keys = ["bind_buffer"] values = [%alloc_1 : memref<16x2048xf32, #hivm.address_space<cbuf>>] : tensor<16x2048xf32>
  } {hivm.cube_only}
  scf.for %arg8 = %c0_i32 to %c148_i32 step %c1_i32  : i32 {
    %14 = arith.divsi %arg8, %c4_i32 : i32
    %15 = arith.muli %14, %c56_i32 : i32
    %16 = arith.addi %8, %15 : i32
    %17 = arith.cmpi slt, %16, %c2048_i32 : i32
    %18 = arith.remsi %arg8, %c2_i32 : i32
    %19 = arith.cmpi eq, %18, %c0_i32 : i32
    scf.if %17 {
      %alloc_3 = memref.alloc() : memref<16x2048xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x2048xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_3 {hivm.sync_memory_read_id = 0 : i32, program_split.selected_input = 0 : i32} : memref<16x2048xf32, #hivm.address_space<ub>>
      %20 = arith.addi %arg8, %c1_i32 : i32
      %21 = arith.cmpi slt, %20, %c148_i32 : i32
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_MTE1>] flag = 3
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
      scf.if %19 {
        scf.if %21 {
          %22 = arith.divsi %20, %c4_i32 : i32
          %23 = arith.muli %22, %c56_i32 : i32
          %24 = arith.addi %9, %23 : i32
          %25 = arith.cmpi slt, %24, %c2048_i32 : i32
          scf.if %25 {
            %28 = arith.muli %24, %c16_i32 : i32
            %29 = arith.index_cast %28 : i32 to index
            %30 = arith.remsi %20, %c4_i32 : i32
            %31 = arith.muli %30, %c2048_i32 : i32
            %32 = arith.index_cast %31 : i32 to index
            %33 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8192)>()[%32, %29]
            %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%33], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
            %alloc_4 = memref.alloc() : memref<16x2048xf32>
            hivm.hir.load ins(%reinterpret_cast : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_4 : memref<16x2048xf32>) {hivm.grouped_l1_load} eviction_policy = <EvictFirst> core_type = <CUBE>
            %34 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x2048xf32>
            annotation.mark %34 keys = ["bind_buffer"] values = [%alloc_0 : memref<16x2048xf32, #hivm.address_space<cbuf>>] : tensor<16x2048xf32>
          } {hivm.cube_only}
          %26 = arith.addi %12, %23 : i32
          %27 = arith.cmpi slt, %26, %c2048_i32 : i32
          scf.if %27 {
            %28 = arith.muli %26, %c16_i32 : i32
            %29 = arith.index_cast %28 : i32 to index
            %30 = arith.remsi %20, %c4_i32 : i32
            %31 = arith.muli %30, %c2048_i32 : i32
            %32 = arith.index_cast %31 : i32 to index
            %33 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8192)>()[%32, %29]
            %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%33], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
            %alloc_4 = memref.alloc() : memref<16x2048xf32>
            hivm.hir.load ins(%reinterpret_cast : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_4 : memref<16x2048xf32>) {hivm.grouped_l1_load} eviction_policy = <EvictFirst> core_type = <CUBE>
            %34 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x2048xf32>
            annotation.mark %34 keys = ["bind_buffer"] values = [%alloc_2 : memref<16x2048xf32, #hivm.address_space<cbuf>>] : tensor<16x2048xf32>
          } {hivm.cube_only}
        } {hivm.cube_only}
        hivm.hir.l12ub ins(%alloc : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%alloc_3 : memref<16x2048xf32, #hivm.address_space<ub>>)
        hivm.hir.l12ub ins(%alloc_1 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%alloc_3 : memref<16x2048xf32, #hivm.address_space<ub>>) sub_blockid = true
      } else {
        scf.if %21 {
          %22 = arith.divsi %20, %c4_i32 : i32
          %23 = arith.muli %22, %c56_i32 : i32
          %24 = arith.addi %9, %23 : i32
          %25 = arith.cmpi slt, %24, %c2048_i32 : i32
          scf.if %25 {
            %28 = arith.muli %24, %c16_i32 : i32
            %29 = arith.index_cast %28 : i32 to index
            %30 = arith.remsi %20, %c4_i32 : i32
            %31 = arith.muli %30, %c2048_i32 : i32
            %32 = arith.index_cast %31 : i32 to index
            %33 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8192)>()[%32, %29]
            %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%33], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
            %alloc_4 = memref.alloc() : memref<16x2048xf32>
            hivm.hir.load ins(%reinterpret_cast : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_4 : memref<16x2048xf32>) {hivm.grouped_l1_load} eviction_policy = <EvictFirst> core_type = <CUBE>
            %34 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x2048xf32>
            annotation.mark %34 keys = ["bind_buffer"] values = [%alloc : memref<16x2048xf32, #hivm.address_space<cbuf>>] : tensor<16x2048xf32>
          } {hivm.cube_only}
          %26 = arith.addi %12, %23 : i32
          %27 = arith.cmpi slt, %26, %c2048_i32 : i32
          scf.if %27 {
            %28 = arith.muli %26, %c16_i32 : i32
            %29 = arith.index_cast %28 : i32 to index
            %30 = arith.remsi %20, %c4_i32 : i32
            %31 = arith.muli %30, %c2048_i32 : i32
            %32 = arith.index_cast %31 : i32 to index
            %33 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8192)>()[%32, %29]
            %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%33], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
            %alloc_4 = memref.alloc() : memref<16x2048xf32>
            hivm.hir.load ins(%reinterpret_cast : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_4 : memref<16x2048xf32>) {hivm.grouped_l1_load} eviction_policy = <EvictFirst> core_type = <CUBE>
            %34 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x2048xf32>
            annotation.mark %34 keys = ["bind_buffer"] values = [%alloc_1 : memref<16x2048xf32, #hivm.address_space<cbuf>>] : tensor<16x2048xf32>
          } {hivm.cube_only}
        } {hivm.cube_only}
        hivm.hir.l12ub ins(%alloc_0 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%alloc_3 : memref<16x2048xf32, #hivm.address_space<ub>>)
        hivm.hir.l12ub ins(%alloc_2 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%alloc_3 : memref<16x2048xf32, #hivm.address_space<ub>>) sub_blockid = true
      } {hivm.cube_only}
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_V>] flag = 1
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    }
  }
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_MTE1>] flag = 3
  hivm.hir.set_ctrl true at ctrl[60]
  return
}
func.func @_gelu_direct_56aiv_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd", program_split.launch = {input_args = array<i32: 2>, logical_programs = 56 : i32, output_arg = 3 : i32, persistent_steps = 148 : i32, physical_programs = 28 : i32, row_tasks = 2048 : i32}, program_split.transformed} {
  %cst = arith.constant 1.000000e+00 : f32
  %c148_i32 = arith.constant 148 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c2048_i32 = arith.constant 2048 : i32
  %c56_i32 = arith.constant 56 : i32
  %c4_i32 = arith.constant 4 : i32
  %cst_0 = arith.constant -1.59576917 : f32
  %cst_1 = arith.constant 4.471500e-02 : f32
  %c1_i32 = arith.constant 1 : i32
  %c28_i32 = arith.constant {program_split.physical_count} 28 : i32
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 {program_split.physical_id} : i64 to i32
  %4 = hivm.hir.get_sub_block_idx -> i64
  %5 = arith.trunci %4 : i64 to i32
  %6 = arith.muli %5, %c28_i32 : i32
  %7 = arith.addi %3, %6 : i32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %8 = arith.remsi %7, %arg5 : i32
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_MTE1>] flag = 3
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
  scf.for %arg8 = %c0_i32 to %c148_i32 step %c1_i32  : i32 {
    %9 = arith.divsi %arg8, %c4_i32 : i32
    %10 = arith.remsi %arg8, %c4_i32 : i32
    %11 = arith.muli %9, %c56_i32 : i32
    %12 = arith.addi %8, %11 : i32
    %13 = arith.cmpi slt, %12, %c2048_i32 : i32
    %14 = arith.muli %12, %c16_i32 : i32
    %15 = arith.muli %10, %c2048_i32 : i32
    scf.if %13 {
      %16 = arith.index_cast %14 : i32 to index
      %17 = arith.index_cast %15 : i32 to index
      %18 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8192)>()[%17, %16]
      %alloc = memref.alloc() : memref<16x2048xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x2048xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {hivm.sync_memory_read_id = 0 : i32, program_split.selected_input = 0 : i32} : memref<16x2048xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc : memref<16x2048xf32, #hivm.address_space<ub>> to memref<16x2048xf32>
      %19 = bufferization.to_tensor %memspacecast restrict writable : memref<16x2048xf32>
      %20 = tensor.empty() : tensor<16x2048xf32>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_V>] flag = 1
      %21 = hivm.hir.vmul ins(%19, %19 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%20 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %22 = hivm.hir.vmul ins(%21, %19 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%20 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %23 = hivm.hir.vmul ins(%22, %cst_1 : tensor<16x2048xf32>, f32) outs(%20 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %24 = hivm.hir.vadd ins(%19, %23 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%20 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %25 = hivm.hir.vmul ins(%24, %cst_0 : tensor<16x2048xf32>, f32) outs(%20 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %26 = hivm.hir.vexp ins(%25 : tensor<16x2048xf32>) outs(%20 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %27 = hivm.hir.vadd ins(%26, %cst : tensor<16x2048xf32>, f32) outs(%20 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %28 = hivm.hir.vdiv ins(%19, %27 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%20 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_MTE1>] flag = 3
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%18], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
      hivm.hir.store ins(%28 : tensor<16x2048xf32>) outs(%reinterpret_cast : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) {hivm.sync_memory_read_ids = array<i32: 0>}
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
    }
  }
  hivm.hir.set_ctrl true at ctrl[60]
  return
}
}
