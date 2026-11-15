// RUN: bishengir-opt %s --program-split-l1-materialize | FileCheck %s
//
// Recovered GELU input captured after the standard entry adaptation, HIVM
// conversion and MIX blockification of Inputs/recovered_gelu.mlir.
// Opposite-bank prefetch must be inside the same parity branch as delivery.
// Otherwise GraphSyncSolver cannot prove the banks are mutually exclusive.
// Prefetch is issued before either current delivery (and its UB handoff wait).
// CHECK-LABEL: func.func @_gelu_fwd
// L1 reservation marks belong to the cube half; vector splitting must not
// retain otherwise unused L1 allocations through these marks.
// CHECK: annotation.mark {{.*}}effects = ["write", "read"], hivm.tcore_type = #hivm.tcore_type<CUBE>
// CHECK: scf.for
// CHECK: %[[PARITY:.*]] = arith.cmpi eq,
// CHECK: hivm.sync_memory_read_id = 0
// CHECK: %[[NEXT:.*]] = arith.addi
// CHECK: %[[HAS_NEXT:.*]] = arith.cmpi slt, %[[NEXT]],
// CHECK: scf.if %[[PARITY]] {
// CHECK-NEXT: scf.if %[[HAS_NEXT]] {
// CHECK: hivm.hir.load {{.*}}core_type = <CUBE>
// CHECK: hivm.hir.load {{.*}}core_type = <CUBE>
// CHECK: hivm.hir.l12ub
// CHECK-NEXT: hivm.hir.l12ub {{.*}}sub_blockid = true
// CHECK: } else {
// CHECK-NEXT: scf.if %[[HAS_NEXT]] {
// CHECK: hivm.hir.load {{.*}}core_type = <CUBE>
// CHECK: hivm.hir.load {{.*}}core_type = <CUBE>
// CHECK: hivm.hir.l12ub
// CHECK-NEXT: hivm.hir.l12ub {{.*}}sub_blockid = true
// CHECK: hivm.hir.store {{.*}}hivm.sync_memory_read_ids = array<i32: 0>
// CHECK-NOT: hivm.hir.load
// CHECK: return

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @_gelu_fwd(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, false, false, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd", program_split.launch = {input_args = array<i32: 2>, output_arg = 3 : i32, cols_arg = 6 : i32, element_bits = 32 : i32, rows_arg = 5 : i32, stride_arg = 4 : i32, tile_cols = 2048 : i32, tile_rows = 16 : i32}, program_split.transformed} {
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 {program_split.physical_id} : i64 to i32
  %c2_i32 = arith.constant 2 : i32
  %4 = arith.divsi %1, %c2_i32 {program_split.physical_count} : i32
  scope.scope : () -> () {
    %5 = hivm.hir.get_sub_block_idx -> i64
    %6 = arith.trunci %5 : i64 to i32
    %7 = arith.muli %6, %4 : i32
    %8 = arith.addi %3, %7 : i32
    %9 = arith.extsi %8 {program_split.logical_id} : i32 to i64
    %c1_i32 = arith.constant 1 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.797884583 : f32
    %cst_1 = arith.constant 4.471500e-02 : f32
    %c16_i32 = arith.constant 16 : i32
    %cst_2 = arith.constant 5.000000e-01 : f32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %10 = arith.trunci %9 : i64 to i32
    %11 = arith.remsi %10, %arg7 : i32
    %12 = tensor.empty() : tensor<16x2048xf32>
    %13 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%12 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %14 = tensor.empty() : tensor<16x2048xf32>
    %15 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%14 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %16 = tensor.empty() : tensor<16x2048xf32>
    %17 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%16 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %18 = tensor.empty() : tensor<16x2048xf32>
    %19 = hivm.hir.vbrc ins(%cst : f32) outs(%18 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %20 = arith.divsi %arg5, %c16_i32 : i32
    %21 = arith.divsi %arg6, %c2048_i32 : i32
    %22 = arith.addi %20, %arg7 : i32
    %23 = arith.subi %22, %c1_i32 : i32
    %24 = arith.divsi %23, %arg7 : i32
    %25 = arith.muli %24, %21 : i32
    scf.for %arg10 = %c0_i32 to %25 step %c1_i32  : i32 {
      %26 = arith.divsi %arg10, %21 : i32
      %27 = arith.remsi %arg10, %21 : i32
      %28 = arith.muli %26, %arg7 : i32
      %29 = arith.addi %11, %28 : i32
      %30 = arith.muli %27, %c2048_i32 : i32
      %31 = arith.cmpi slt, %29, %20 : i32
      scf.if %31 {
        %32 = arith.muli %29, %c16_i32 : i32
        %33 = arith.index_cast %32 : i32 to index
        %34 = arith.index_cast %arg4 : i32 to index
        %35 = arith.muli %33, %34 : index
        %36 = arith.index_cast %30 : i32 to index
        %37 = arith.addi %35, %36 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%37], sizes: [16, 2048], strides: [%34, 1] : memref<?xf32> to memref<16x2048xf32, strided<[?, 1], offset: ?>>
        %alloc = memref.alloc() : memref<16x2048xf32>
        annotation.mark %alloc {program_split.selected_input = 0 : i32} : memref<16x2048xf32>
        hivm.hir.load ins(%reinterpret_cast : memref<16x2048xf32, strided<[?, 1], offset: ?>>) outs(%alloc : memref<16x2048xf32>) eviction_policy = <EvictFirst> core_type = <VECTOR>
        %38 = bufferization.to_tensor %alloc restrict writable : memref<16x2048xf32>
        %39 = arith.mulf %38, %38 : tensor<16x2048xf32>
        %40 = arith.mulf %39, %38 : tensor<16x2048xf32>
        %41 = arith.mulf %38, %13 : tensor<16x2048xf32>
        %42 = arith.mulf %40, %15 : tensor<16x2048xf32>
        %43 = arith.addf %38, %42 : tensor<16x2048xf32>
        %44 = arith.mulf %43, %17 : tensor<16x2048xf32>
        %45 = math.tanh %44 : tensor<16x2048xf32>
        %46 = arith.addf %45, %19 : tensor<16x2048xf32>
        %47 = arith.mulf %41, %46 : tensor<16x2048xf32>
        %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%37], sizes: [16, 2048], strides: [%34, 1] : memref<?xf32> to memref<16x2048xf32, strided<[?, 1], offset: ?>>
        hivm.hir.store ins(%47 : tensor<16x2048xf32>) outs(%reinterpret_cast_3 : memref<16x2048xf32, strided<[?, 1], offset: ?>>)
      }
    }
    hivm.hir.set_ctrl true at ctrl[60]
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, no_inline, program_split.whole_program_vector}
  return
}
}
