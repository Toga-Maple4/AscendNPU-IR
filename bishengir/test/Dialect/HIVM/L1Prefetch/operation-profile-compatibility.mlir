// RUN: bishengir-opt %s --split-input-file --l1-prefetch-ps-analysis | FileCheck %s
// Preserve the previous ownership proofs' operation sets during consolidation.
// Each rejected input differs from an accepted real input by one pure operation.
// The rejection is compatibility policy, not a claim that the extra op is unsafe.

// CHECK-LABEL: func.func @persistent_exp
// CHECK-SAME: l1_prefetch_ps.accepted
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @persistent_exp(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", ascend.memory.pointer_args_noalias, mix_mode = "aiv", parallel_mode = "simd"} {
    %c8192 = arith.constant 8192 : index
    %cst = arith.constant 4.471500e-02 : f32
    %cst_0 = arith.constant -1.59576917 : f32
    %c4_i32 = arith.constant 4 : i32
    %c56_i32 = arith.constant 56 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x2048xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    scf.for %arg10 = %c0_i32 to %c148_i32 step %c1_i32  : i32 {
      %4 = arith.divsi %arg10, %c4_i32 : i32
      %5 = arith.remsi %arg10, %c4_i32 : i32
      %6 = arith.muli %4, %c56_i32 : i32
      %7 = arith.addi %arg7, %6 : i32
      %8 = arith.cmpi slt, %7, %c2048_i32 : i32
      %9 = arith.muli %7, %c16_i32 : i32
      %10 = arith.muli %5, %c2048_i32 : i32
      scf.if %8 {
        %11 = arith.index_cast %9 : i32 to index
        %12 = arith.muli %11, %c8192 : index
        %13 = arith.index_cast %10 : i32 to index
        %14 = arith.addi %12, %13 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
        %alloc = memref.alloc() : memref<16x2048xf32>
        memref.copy %reinterpret_cast, %alloc : memref<16x2048xf32, strided<[8192, 1], offset: ?>> to memref<16x2048xf32>
        %15 = bufferization.to_tensor %alloc restrict writable : memref<16x2048xf32>
        %16 = arith.mulf %15, %15 : tensor<16x2048xf32>
        %17 = arith.mulf %16, %15 : tensor<16x2048xf32>
        %18 = arith.mulf %17, %3 : tensor<16x2048xf32>
        %19 = arith.addf %15, %18 : tensor<16x2048xf32>
        %20 = arith.mulf %19, %2 : tensor<16x2048xf32>
        %21 = math.exp %20 : tensor<16x2048xf32>
        %22 = arith.addf %21, %1 : tensor<16x2048xf32>
        %23 = arith.divf %15, %22 : tensor<16x2048xf32>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
        bufferization.materialize_in_destination %23 in writable %reinterpret_cast_2 : (tensor<16x2048xf32>, memref<16x2048xf32, strided<[8192, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @persistent_tanh
// CHECK-SAME: l1_prefetch_ps.reject_reason = "unsupported_memory_or_control_flow"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @persistent_tanh(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", ascend.memory.pointer_args_noalias, mix_mode = "aiv", parallel_mode = "simd"} {
    %c8192 = arith.constant 8192 : index
    %cst = arith.constant 4.471500e-02 : f32
    %cst_0 = arith.constant -1.59576917 : f32
    %c4_i32 = arith.constant 4 : i32
    %c56_i32 = arith.constant 56 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x2048xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    scf.for %arg10 = %c0_i32 to %c148_i32 step %c1_i32  : i32 {
      %4 = arith.divsi %arg10, %c4_i32 : i32
      %5 = arith.remsi %arg10, %c4_i32 : i32
      %6 = arith.muli %4, %c56_i32 : i32
      %7 = arith.addi %arg7, %6 : i32
      %8 = arith.cmpi slt, %7, %c2048_i32 : i32
      %9 = arith.muli %7, %c16_i32 : i32
      %10 = arith.muli %5, %c2048_i32 : i32
      scf.if %8 {
        %11 = arith.index_cast %9 : i32 to index
        %12 = arith.muli %11, %c8192 : index
        %13 = arith.index_cast %10 : i32 to index
        %14 = arith.addi %12, %13 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
        %alloc = memref.alloc() : memref<16x2048xf32>
        memref.copy %reinterpret_cast, %alloc : memref<16x2048xf32, strided<[8192, 1], offset: ?>> to memref<16x2048xf32>
        %15 = bufferization.to_tensor %alloc restrict writable : memref<16x2048xf32>
        %16 = arith.mulf %15, %15 : tensor<16x2048xf32>
        %17 = arith.mulf %16, %15 : tensor<16x2048xf32>
        %18 = arith.mulf %17, %3 : tensor<16x2048xf32>
        %19 = arith.addf %15, %18 : tensor<16x2048xf32>
        %20 = arith.mulf %19, %2 : tensor<16x2048xf32>
        %21 = math.tanh %20 : tensor<16x2048xf32>
        %22 = arith.addf %21, %1 : tensor<16x2048xf32>
        %23 = arith.divf %15, %22 : tensor<16x2048xf32>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [16, 2048], strides: [8192, 1] : memref<?xf32> to memref<16x2048xf32, strided<[8192, 1], offset: ?>>
        bufferization.materialize_in_destination %23 in writable %reinterpret_cast_2 : (tensor<16x2048xf32>, memref<16x2048xf32, strided<[8192, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @jagged_copy
// CHECK-SAME: l1_prefetch_ps.accepted
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @jagged_copy(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {ascend.memory.pointer_args_noalias, SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %arg11 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    %1 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>
    %2 = arith.addi %arg11, %c1_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%3], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    %4 = memref.load %reinterpret_cast_0[%c0] : memref<1xi64, strided<[1], offset: ?>>
    %5 = arith.extsi %arg4 : i32 to i64
    %6 = arith.muli %1, %5 : i64
    %7 = arith.muli %arg11, %arg7 : i32
    %8 = arith.minsi %arg6, %arg4 : i32
    %9 = arith.divsi %arg7, %arg6 : i32
    %10 = arith.subi %4, %1 : i64
    %11 = arith.extsi %9 : i32 to i64
    %12 = arith.minsi %11, %10 : i64
    %13 = scf.for %arg14 = %1 to %4 step %c256_i64 iter_args(%arg15 = %c0) -> (index)  : i64 {
      %14 = arith.index_cast %arg6 : i32 to index
      %15 = arith.muli %arg15, %14 : index
      %16:2 = scf.for %arg16 = %c0_i32 to %8 step %c256_i32 iter_args(%arg17 = %c0, %arg18 = %15) -> (index, index)  : i32 {
        %18 = arith.index_cast %6 : i64 to index
        %19 = arith.addi %18, %arg18 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [256, 256], strides: [%14, %c1] : memref<?xbf16> to memref<256x256xbf16, strided<[?, ?], offset: ?>>
        %alloc = memref.alloc() : memref<256x256xbf16>
        %20 = arith.addi %arg15, %c256 : index
        %21 = arith.index_cast %12 : i64 to index
        %22 = arith.maxsi %arg15, %21 : index
        %23 = arith.minsi %20, %22 : index
        %24 = arith.subi %23, %arg15 : index
        %25 = arith.addi %arg17, %c256 : index
        %26 = arith.index_cast %8 : i32 to index
        %27 = arith.maxsi %arg17, %26 : index
        %28 = arith.minsi %25, %27 : index
        %29 = arith.subi %28, %arg17 : index
        %30 = arith.minsi %24, %c256 : index
        %31 = arith.maxsi %30, %c0 : index
        %32 = arith.minsi %29, %c256 : index
        %33 = arith.maxsi %32, %c0 : index
        %34 = arith.cmpi slt, %31, %c256 : index
        %35 = arith.cmpi slt, %33, %c256 : index
        %36 = arith.ori %34, %35 : i1
        scf.if %36 {
          linalg.fill ins(%cst : bf16) outs(%alloc : memref<256x256xbf16>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast_1[0, 0] [%31, %33] [1, 1] : memref<256x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
        %subview_2 = memref.subview %alloc[0, 0] [%31, %33] [1, 1] : memref<256x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
        memref.copy %subview, %subview_2 : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
        %37 = bufferization.to_tensor %alloc restrict writable : memref<256x256xbf16>
        %38 = arith.index_cast %7 : i32 to index
        %39 = arith.addi %38, %arg18 : index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%39], sizes: [256, 256], strides: [%14, %c1] : memref<?xbf16> to memref<256x256xbf16, strided<[?, ?], offset: ?>>
        %extracted_slice = tensor.extract_slice %37[0, 0] [%31, %33] [1, 1] : tensor<256x256xbf16> to tensor<?x?xbf16>
        %subview_4 = memref.subview %reinterpret_cast_3[0, 0] [%31, %33] [1, 1] : memref<256x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_4 : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, ?], offset: ?>>) -> ()
        %40 = arith.addi %arg17, %c256 : index
        %41 = arith.addi %arg18, %c256 : index
        scf.yield %40, %41 : index, index
      }
      %17 = arith.addi %arg15, %c256 : index
      scf.yield %17 : index
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @jagged_tensor_empty
// CHECK-SAME: l1_prefetch_ps.reject_reason = "jagged_unsupported_memory_or_control_flow"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @jagged_tensor_empty(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {ascend.memory.pointer_args_noalias, SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %unused = tensor.empty() : tensor<1xf32>
    %cst = arith.constant 0.000000e+00 : bf16
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %arg11 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    %1 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>
    %2 = arith.addi %arg11, %c1_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%3], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    %4 = memref.load %reinterpret_cast_0[%c0] : memref<1xi64, strided<[1], offset: ?>>
    %5 = arith.extsi %arg4 : i32 to i64
    %6 = arith.muli %1, %5 : i64
    %7 = arith.muli %arg11, %arg7 : i32
    %8 = arith.minsi %arg6, %arg4 : i32
    %9 = arith.divsi %arg7, %arg6 : i32
    %10 = arith.subi %4, %1 : i64
    %11 = arith.extsi %9 : i32 to i64
    %12 = arith.minsi %11, %10 : i64
    %13 = scf.for %arg14 = %1 to %4 step %c256_i64 iter_args(%arg15 = %c0) -> (index)  : i64 {
      %14 = arith.index_cast %arg6 : i32 to index
      %15 = arith.muli %arg15, %14 : index
      %16:2 = scf.for %arg16 = %c0_i32 to %8 step %c256_i32 iter_args(%arg17 = %c0, %arg18 = %15) -> (index, index)  : i32 {
        %18 = arith.index_cast %6 : i64 to index
        %19 = arith.addi %18, %arg18 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [256, 256], strides: [%14, %c1] : memref<?xbf16> to memref<256x256xbf16, strided<[?, ?], offset: ?>>
        %alloc = memref.alloc() : memref<256x256xbf16>
        %20 = arith.addi %arg15, %c256 : index
        %21 = arith.index_cast %12 : i64 to index
        %22 = arith.maxsi %arg15, %21 : index
        %23 = arith.minsi %20, %22 : index
        %24 = arith.subi %23, %arg15 : index
        %25 = arith.addi %arg17, %c256 : index
        %26 = arith.index_cast %8 : i32 to index
        %27 = arith.maxsi %arg17, %26 : index
        %28 = arith.minsi %25, %27 : index
        %29 = arith.subi %28, %arg17 : index
        %30 = arith.minsi %24, %c256 : index
        %31 = arith.maxsi %30, %c0 : index
        %32 = arith.minsi %29, %c256 : index
        %33 = arith.maxsi %32, %c0 : index
        %34 = arith.cmpi slt, %31, %c256 : index
        %35 = arith.cmpi slt, %33, %c256 : index
        %36 = arith.ori %34, %35 : i1
        scf.if %36 {
          linalg.fill ins(%cst : bf16) outs(%alloc : memref<256x256xbf16>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast_1[0, 0] [%31, %33] [1, 1] : memref<256x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
        %subview_2 = memref.subview %alloc[0, 0] [%31, %33] [1, 1] : memref<256x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
        memref.copy %subview, %subview_2 : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
        %37 = bufferization.to_tensor %alloc restrict writable : memref<256x256xbf16>
        %38 = arith.index_cast %7 : i32 to index
        %39 = arith.addi %38, %arg18 : index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%39], sizes: [256, 256], strides: [%14, %c1] : memref<?xbf16> to memref<256x256xbf16, strided<[?, ?], offset: ?>>
        %extracted_slice = tensor.extract_slice %37[0, 0] [%31, %33] [1, 1] : tensor<256x256xbf16> to tensor<?x?xbf16>
        %subview_4 = memref.subview %reinterpret_cast_3[0, 0] [%31, %33] [1, 1] : memref<256x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_4 : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, ?], offset: ?>>) -> ()
        %40 = arith.addi %arg17, %c256 : index
        %41 = arith.addi %arg18, %c256 : index
        scf.yield %40, %41 : index, index
      }
      %17 = arith.addi %arg15, %c256 : index
      scf.yield %17 : index
    }
    return
  }
}
