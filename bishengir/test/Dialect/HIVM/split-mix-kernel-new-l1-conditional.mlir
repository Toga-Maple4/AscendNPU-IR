// RUN: bishengir-opt %s -hivm-split-mixed-if-conditionals -hivm-split-mix-kernel | FileCheck %s
// Captured from the unchanged l1-prefetch-ps-retry.mlir New-L1 attempt.
// The core-partitioned conditional yields shared UB storage to vector work.
// It must retain the allocation scope and not substitute a tensor.empty.
// CHECK-LABEL: func.func @_gelu_direct_56aiv_mix_aic(
// CHECK: tightly_coupled_buffer<0>
// CHECK: hivm.hir.l12ub
// CHECK-LABEL: func.func @_gelu_direct_56aiv_mix_aiv(
// CHECK: scf.if
// CHECK: %[[UB:.*]] = memref.alloc() : memref<16x2048xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[UB]] {{.*}}tightly_coupled_buffer<0>
// CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[UB]]
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[CAST]]
// CHECK: hivm.hir.vmul ins(%[[T]], %[[T]]

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @_gelu_direct_56aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd", program_split.launch = {input_args = array<i32: 2>, logical_programs = 56 : i32, output_arg = 3 : i32, persistent_steps = 148 : i32, physical_programs = 28 : i32, row_tasks = 2048 : i32}, program_split.transformed} {
  %c2_i32 = arith.constant 2 : i32
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
  %alloc = memref.alloc() : memref<16x2048xf32, #hivm.address_space<cbuf>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<16x2048xf32, #hivm.address_space<cbuf>>
  %alloc_2 = memref.alloc() : memref<16x2048xf32, #hivm.address_space<cbuf>>
  annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<16x2048xf32, #hivm.address_space<cbuf>>
  %alloc_3 = memref.alloc() : memref<16x2048xf32, #hivm.address_space<cbuf>>
  annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<16x2048xf32, #hivm.address_space<cbuf>>
  %alloc_4 = memref.alloc() : memref<16x2048xf32, #hivm.address_space<cbuf>>
  annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<16x2048xf32, #hivm.address_space<cbuf>>
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
    %alloc_5 = memref.alloc() : memref<16x2048xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_5 : memref<16x2048xf32>) {hivm.grouped_l1_load} eviction_policy = <EvictFirst> core_type = <CUBE>
    %17 = bufferization.to_tensor %alloc_5 restrict writable : memref<16x2048xf32>
    annotation.mark %17 keys = ["bind_buffer"] values = [%alloc : memref<16x2048xf32, #hivm.address_space<cbuf>>] : tensor<16x2048xf32>
  }
  %11 = arith.addi %3, %c28_i32 : i32
  %12 = arith.remsi %11, %arg5 : i32
  %13 = arith.cmpi slt, %12, %c2048_i32 : i32
  scf.if %13 {
    %14 = arith.muli %12, %c16_i32 : i32
    %15 = arith.index_cast %14 : i32 to index
    %16 = affine.apply affine_map<()[s0] -> (s0 * 8192)>()[%15]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
    %alloc_5 = memref.alloc() : memref<16x2048xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_5 : memref<16x2048xf32>) {hivm.grouped_l1_load} eviction_policy = <EvictFirst> core_type = <CUBE>
    %17 = bufferization.to_tensor %alloc_5 restrict writable : memref<16x2048xf32>
    annotation.mark %17 keys = ["bind_buffer"] values = [%alloc_3 : memref<16x2048xf32, #hivm.address_space<cbuf>>] : tensor<16x2048xf32>
  }
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_MTE1>] flag = 3
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
  scf.for %arg8 = %c0_i32 to %c148_i32 step %c1_i32  : i32 {
    %14 = arith.divsi %arg8, %c4_i32 : i32
    %15 = arith.remsi %arg8, %c4_i32 : i32
    %16 = arith.muli %14, %c56_i32 : i32
    %17 = arith.addi %8, %16 : i32
    %18 = arith.cmpi slt, %17, %c2048_i32 : i32
    %19 = arith.muli %17, %c16_i32 : i32
    %20 = arith.muli %15, %c2048_i32 : i32
    %21 = arith.remsi %arg8, %c2_i32 : i32
    %22 = arith.cmpi eq, %21, %c0_i32 : i32
    scf.if %18 {
      %23 = arith.index_cast %19 : i32 to index
      %24 = arith.index_cast %20 : i32 to index
      %25 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8192)>()[%24, %23]
      %alloc_5 = memref.alloc() : memref<16x2048xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x2048xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5 {hivm.sync_memory_read_id = 0 : i32, program_split.selected_input = 0 : i32} : memref<16x2048xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc_5 : memref<16x2048xf32, #hivm.address_space<ub>> to memref<16x2048xf32>
      %26 = arith.addi %arg8, %c1_i32 : i32
      %27 = arith.cmpi slt, %26, %c148_i32 : i32
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_MTE1>] flag = 3
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
      scf.if %22 {
        scf.if %27 {
          %38 = arith.divsi %26, %c4_i32 : i32
          %39 = arith.muli %38, %c56_i32 : i32
          %40 = arith.addi %9, %39 : i32
          %41 = arith.cmpi slt, %40, %c2048_i32 : i32
          scf.if %41 {
            %44 = arith.muli %40, %c16_i32 : i32
            %45 = arith.index_cast %44 : i32 to index
            %46 = arith.remsi %26, %c4_i32 : i32
            %47 = arith.muli %46, %c2048_i32 : i32
            %48 = arith.index_cast %47 : i32 to index
            %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8192)>()[%48, %45]
            %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%49], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
            %alloc_7 = memref.alloc() : memref<16x2048xf32>
            hivm.hir.load ins(%reinterpret_cast_6 : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_7 : memref<16x2048xf32>) {hivm.grouped_l1_load} eviction_policy = <EvictFirst> core_type = <CUBE>
            %50 = bufferization.to_tensor %alloc_7 restrict writable : memref<16x2048xf32>
            annotation.mark %50 keys = ["bind_buffer"] values = [%alloc_2 : memref<16x2048xf32, #hivm.address_space<cbuf>>] : tensor<16x2048xf32>
          }
          %42 = arith.addi %12, %39 : i32
          %43 = arith.cmpi slt, %42, %c2048_i32 : i32
          scf.if %43 {
            %44 = arith.muli %42, %c16_i32 : i32
            %45 = arith.index_cast %44 : i32 to index
            %46 = arith.remsi %26, %c4_i32 : i32
            %47 = arith.muli %46, %c2048_i32 : i32
            %48 = arith.index_cast %47 : i32 to index
            %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8192)>()[%48, %45]
            %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%49], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
            %alloc_7 = memref.alloc() : memref<16x2048xf32>
            hivm.hir.load ins(%reinterpret_cast_6 : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_7 : memref<16x2048xf32>) {hivm.grouped_l1_load} eviction_policy = <EvictFirst> core_type = <CUBE>
            %50 = bufferization.to_tensor %alloc_7 restrict writable : memref<16x2048xf32>
            annotation.mark %50 keys = ["bind_buffer"] values = [%alloc_4 : memref<16x2048xf32, #hivm.address_space<cbuf>>] : tensor<16x2048xf32>
          }
        }
        hivm.hir.l12ub ins(%alloc : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%alloc_5 : memref<16x2048xf32, #hivm.address_space<ub>>)
        hivm.hir.l12ub ins(%alloc_3 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%alloc_5 : memref<16x2048xf32, #hivm.address_space<ub>>) sub_blockid = true
      } else {
        scf.if %27 {
          %38 = arith.divsi %26, %c4_i32 : i32
          %39 = arith.muli %38, %c56_i32 : i32
          %40 = arith.addi %9, %39 : i32
          %41 = arith.cmpi slt, %40, %c2048_i32 : i32
          scf.if %41 {
            %44 = arith.muli %40, %c16_i32 : i32
            %45 = arith.index_cast %44 : i32 to index
            %46 = arith.remsi %26, %c4_i32 : i32
            %47 = arith.muli %46, %c2048_i32 : i32
            %48 = arith.index_cast %47 : i32 to index
            %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8192)>()[%48, %45]
            %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%49], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
            %alloc_7 = memref.alloc() : memref<16x2048xf32>
            hivm.hir.load ins(%reinterpret_cast_6 : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_7 : memref<16x2048xf32>) {hivm.grouped_l1_load} eviction_policy = <EvictFirst> core_type = <CUBE>
            %50 = bufferization.to_tensor %alloc_7 restrict writable : memref<16x2048xf32>
            annotation.mark %50 keys = ["bind_buffer"] values = [%alloc : memref<16x2048xf32, #hivm.address_space<cbuf>>] : tensor<16x2048xf32>
          }
          %42 = arith.addi %12, %39 : i32
          %43 = arith.cmpi slt, %42, %c2048_i32 : i32
          scf.if %43 {
            %44 = arith.muli %42, %c16_i32 : i32
            %45 = arith.index_cast %44 : i32 to index
            %46 = arith.remsi %26, %c4_i32 : i32
            %47 = arith.muli %46, %c2048_i32 : i32
            %48 = arith.index_cast %47 : i32 to index
            %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8192)>()[%48, %45]
            %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%49], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
            %alloc_7 = memref.alloc() : memref<16x2048xf32>
            hivm.hir.load ins(%reinterpret_cast_6 : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_7 : memref<16x2048xf32>) {hivm.grouped_l1_load} eviction_policy = <EvictFirst> core_type = <CUBE>
            %50 = bufferization.to_tensor %alloc_7 restrict writable : memref<16x2048xf32>
            annotation.mark %50 keys = ["bind_buffer"] values = [%alloc_3 : memref<16x2048xf32, #hivm.address_space<cbuf>>] : tensor<16x2048xf32>
          }
        }
        hivm.hir.l12ub ins(%alloc_2 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%alloc_5 : memref<16x2048xf32, #hivm.address_space<ub>>)
        hivm.hir.l12ub ins(%alloc_4 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%alloc_5 : memref<16x2048xf32, #hivm.address_space<ub>>) sub_blockid = true
      }
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_V>] flag = 1
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
      %28 = bufferization.to_tensor %memspacecast restrict writable : memref<16x2048xf32>
      %29 = tensor.empty() : tensor<16x2048xf32>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_V>] flag = 1
      %30 = hivm.hir.vmul ins(%28, %28 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%29 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %31 = hivm.hir.vmul ins(%30, %28 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%29 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %32 = hivm.hir.vmul ins(%31, %cst_1 : tensor<16x2048xf32>, f32) outs(%29 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %33 = hivm.hir.vadd ins(%28, %32 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%29 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %34 = hivm.hir.vmul ins(%33, %cst_0 : tensor<16x2048xf32>, f32) outs(%29 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %35 = hivm.hir.vexp ins(%34 : tensor<16x2048xf32>) outs(%29 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %36 = hivm.hir.vadd ins(%35, %cst : tensor<16x2048xf32>, f32) outs(%29 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      %37 = hivm.hir.vdiv ins(%28, %36 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%29 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_MTE1>] flag = 3
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%25], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
      hivm.hir.store ins(%37 : tensor<16x2048xf32>) outs(%reinterpret_cast : memref<16x2048xf32, strided<[8192, 1], offset: ?>>) {hivm.sync_memory_read_ids = array<i32: 0>}
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
    }
  }
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_MTE1>] flag = 3
  hivm.hir.set_ctrl true at ctrl[60]
  return
}
}
