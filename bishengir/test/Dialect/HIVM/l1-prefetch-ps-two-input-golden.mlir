// Generated from the real GELU backward pre-materialization compiler dump.
// Two-input schedule: eight 64 KiB L1 banks and two private input buffers;
// both inputs delivered before unchanged math, plural final-reader metadata.
// RUN: bishengir-opt %s -program-split-l1-materialize | FileCheck %s

// CHECK-LABEL: func.func @_gelu_bwd_kernel
// CHECK-COUNT-8: memref.alloc() : memref<16x2048xf16, #hivm.address_space<cbuf>>
// CHECK: hivm.grouped_l1_load
// CHECK: scf.for
// CHECK: hivm.sync_memory_read_id = 0 : i32
// CHECK: hivm.sync_memory_read_id = 1 : i32
// CHECK: hivm.hir.l12ub
// CHECK: hivm.hir.l12ub
// CHECK: hivm.hir.l12ub{{.*}}sub_blockid = true
// CHECK: hivm.hir.l12ub{{.*}}sub_blockid = true
// CHECK: hivm.hir.store{{.*}}hivm.sync_memory_read_ids = array<i32: 0, 1>

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
  func.func @_gelu_bwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd", program_split.launch = {cols_arg = 7 : i32, element_bits = 16 : i32, input_args = array<i32: 2, 3>, output_arg = 4 : i32, rows_arg = 6 : i32, stride_arg = 5 : i32, tile_cols = 2048 : i32, tile_rows = 16 : i32}, program_split.transformed} {
    %0 = arith.muli %arg9, %arg10 : i32
    %1 = arith.muli %0, %arg11 : i32
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
      %cst = arith.constant -1.000000e+00 : f32
      %cst_0 = arith.constant 2.000000e+00 : f32
      %cst_1 = arith.constant -8.800000e+00 : f32
      %cst_2 = arith.constant 8.800000e+00 : f32
      %c2048_i32 = arith.constant 2048 : i32
      %c0_i32 = arith.constant 0 : i32
      %cst_3 = arith.constant 1.341450e-01 : f32
      %cst_4 = arith.constant 5.000000e-01 : f32
      %cst_5 = arith.constant 1.000000e+00 : f32
      %cst_6 = arith.constant 0.797884583 : f32
      %c16_i32 = arith.constant 16 : i32
      %cst_7 = arith.constant 4.471500e-02 : f32
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %10 = arith.trunci %9 : i64 to i32
      %11 = arith.remsi %10, %arg9 : i32
      %12 = arith.divsi %arg7, %c16_i32 : i32
      %13 = arith.divsi %arg8, %c2048_i32 : i32
      %14 = arith.addi %12, %arg9 : i32
      %15 = arith.subi %14, %c1_i32 : i32
      %16 = arith.divsi %15, %arg9 : i32
      %17 = arith.muli %16, %13 : i32
      scf.for %arg12 = %c0_i32 to %17 step %c1_i32  : i32 {
        %18 = arith.divsi %arg12, %13 : i32
        %19 = arith.remsi %arg12, %13 : i32
        %20 = arith.muli %18, %arg9 : i32
        %21 = arith.addi %11, %20 : i32
        %22 = arith.muli %19, %c2048_i32 : i32
        %23 = arith.cmpi slt, %21, %12 : i32
        scf.if %23 {
          %24 = arith.muli %21, %c16_i32 : i32
          %25 = arith.index_cast %24 : i32 to index
          %26 = arith.index_cast %arg6 : i32 to index
          %27 = arith.muli %25, %26 : index
          %28 = arith.index_cast %22 : i32 to index
          %29 = arith.addi %27, %28 : index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [16, 2048], strides: [%26, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
          %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%29], sizes: [16, 2048], strides: [%26, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
          %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%29], sizes: [16, 2048], strides: [%26, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<16x2048xf16>
          annotation.mark %alloc {program_split.selected_input = 0 : i32} : memref<16x2048xf16>
          hivm.hir.load ins(%reinterpret_cast : memref<16x2048xf16, strided<[?, 1], offset: ?>>) outs(%alloc : memref<16x2048xf16>) eviction_policy = <EvictFirst> core_type = <VECTOR>
          %30 = bufferization.to_tensor %alloc restrict writable : memref<16x2048xf16>
          %alloc_10 = memref.alloc() : memref<16x2048xf16>
          annotation.mark %alloc_10 {program_split.selected_input = 1 : i32} : memref<16x2048xf16>
          hivm.hir.load ins(%reinterpret_cast_8 : memref<16x2048xf16, strided<[?, 1], offset: ?>>) outs(%alloc_10 : memref<16x2048xf16>) eviction_policy = <EvictFirst> core_type = <VECTOR>
          %31 = bufferization.to_tensor %alloc_10 restrict writable : memref<16x2048xf16>
          %32 = tensor.empty() : tensor<16x2048xf32>
          %33 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%31 : tensor<16x2048xf16>) outs(%32 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %34 = tensor.empty() : tensor<16x2048xf32>
          %35 = hivm.hir.vmul ins(%33, %33 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%34 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %36 = tensor.empty() : tensor<16x2048xf32>
          %37 = hivm.hir.vmul ins(%35, %33 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%36 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %38 = tensor.empty() : tensor<16x2048xf32>
          %39 = hivm.hir.vmul ins(%37, %cst_7 : tensor<16x2048xf32>, f32) outs(%38 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %40 = tensor.empty() : tensor<16x2048xf32>
          %41 = hivm.hir.vadd ins(%33, %39 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%40 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %42 = tensor.empty() : tensor<16x2048xf32>
          %43 = hivm.hir.vmul ins(%41, %cst_6 : tensor<16x2048xf32>, f32) outs(%42 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %44 = tensor.empty() : tensor<16x2048xf32>
          %45 = hivm.hir.vmin ins(%43, %cst_2 : tensor<16x2048xf32>, f32) outs(%44 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %46 = tensor.empty() : tensor<16x2048xf32>
          %47 = hivm.hir.vmax ins(%45, %cst_1 : tensor<16x2048xf32>, f32) outs(%46 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %48 = tensor.empty() : tensor<16x2048xf32>
          %49 = hivm.hir.vmul ins(%47, %cst_0 : tensor<16x2048xf32>, f32) outs(%48 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %50 = tensor.empty() : tensor<16x2048xf32>
          %51 = hivm.hir.vexp ins(%49 : tensor<16x2048xf32>) outs(%50 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %52 = tensor.empty() : tensor<16x2048xf32>
          %53 = hivm.hir.vadd ins(%51, %cst : tensor<16x2048xf32>, f32) outs(%52 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %54 = tensor.empty() : tensor<16x2048xf32>
          %55 = hivm.hir.vadd ins(%51, %cst_5 : tensor<16x2048xf32>, f32) outs(%54 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %56 = tensor.empty() : tensor<16x2048xf32>
          %57 = hivm.hir.vdiv ins(%53, %55 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%56 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %58 = tensor.empty() : tensor<16x2048xf32>
          %59 = hivm.hir.vadd ins(%57, %cst_5 : tensor<16x2048xf32>, f32) outs(%58 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %60 = tensor.empty() : tensor<16x2048xf32>
          %61 = hivm.hir.vmul ins(%59, %cst_4 : tensor<16x2048xf32>, f32) outs(%60 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %62 = tensor.empty() : tensor<16x2048xf32>
          %63 = hivm.hir.vmul ins(%57, %57 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%62 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %64 = tensor.empty() : tensor<16x2048xf32>
          %65 = hivm.hir.vmul ins(%33, %cst_4 : tensor<16x2048xf32>, f32) outs(%64 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %66 = tensor.empty() : tensor<16x2048xf32>
          %67 = hivm.hir.vmul ins(%63, %cst : tensor<16x2048xf32>, f32) outs(%66 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %68 = tensor.empty() : tensor<16x2048xf32>
          %69 = hivm.hir.vadd ins(%67, %cst_5 : tensor<16x2048xf32>, f32) outs(%68 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %70 = tensor.empty() : tensor<16x2048xf32>
          %71 = hivm.hir.vmul ins(%65, %69 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%70 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %72 = tensor.empty() : tensor<16x2048xf32>
          %73 = hivm.hir.vmul ins(%33, %cst_3 : tensor<16x2048xf32>, f32) outs(%72 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %74 = tensor.empty() : tensor<16x2048xf32>
          %75 = hivm.hir.vmul ins(%73, %33 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%74 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %76 = tensor.empty() : tensor<16x2048xf32>
          %77 = hivm.hir.vadd ins(%75, %cst_5 : tensor<16x2048xf32>, f32) outs(%76 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %78 = tensor.empty() : tensor<16x2048xf32>
          %79 = hivm.hir.vmul ins(%77, %cst_6 : tensor<16x2048xf32>, f32) outs(%78 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %80 = tensor.empty() : tensor<16x2048xf32>
          %81 = hivm.hir.vmul ins(%71, %79 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%80 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %82 = tensor.empty() : tensor<16x2048xf32>
          %83 = hivm.hir.vadd ins(%61, %81 : tensor<16x2048xf32>, tensor<16x2048xf32>) outs(%82 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
          %84 = tensor.empty() : tensor<16x2048xf16>
          %85 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%83 : tensor<16x2048xf32>) outs(%84 : tensor<16x2048xf16>) -> tensor<16x2048xf16>
          %86 = tensor.empty() : tensor<16x2048xf16>
          %87 = hivm.hir.vmul ins(%30, %85 : tensor<16x2048xf16>, tensor<16x2048xf16>) outs(%86 : tensor<16x2048xf16>) -> tensor<16x2048xf16>
          hivm.hir.store ins(%87 : tensor<16x2048xf16>) outs(%reinterpret_cast_9 : memref<16x2048xf16, strided<[?, 1], offset: ?>>)
        }
      }
      hivm.hir.set_ctrl true at ctrl[60]
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, no_inline, program_split.whole_program_vector}
    return
  }
}