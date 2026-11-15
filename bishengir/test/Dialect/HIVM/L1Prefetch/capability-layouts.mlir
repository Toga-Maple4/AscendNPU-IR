// RUN: bishengir-opt %s --split-input-file --l1-prefetch-ps-analysis --l1-prefetch-ps-transform | FileCheck %s
// Unit fixtures preserving the access proofs while varying public roles.

// CHECK-LABEL: func.func @constant_rows
// CHECK-SAME: program_split.launch = {
// CHECK-SAME: static_rows = 32 : i32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @constant_rows(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", ascend.memory.pointer_args_noalias, mix_mode = "aiv", parallel_mode = "simd"} {
    %rows = arith.constant 32 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c16 = arith.constant 16 : index
    %c2048_i32 = arith.constant 2048 : i32
    %c0_i32 = arith.constant 0 : i32
    %c15_i32 = arith.constant 15 : i32
    %cst_0 = arith.constant 1.341450e-01 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 0.797884583 : f32
    %c16_i32 = arith.constant 16 : i32
    %cst_4 = arith.constant 4.471500e-02 : f32
    %0 = tensor.empty() : tensor<16x2048xf32>
    %1 = linalg.fill ins(%cst_4 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %2 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %3 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %6 = arith.addi %rows, %c15_i32 : i32
    %7 = arith.divsi %6, %c16_i32 : i32
    scf.for %arg14 = %arg11 to %7 step %arg8  : i32 {
      %8 = arith.muli %arg14, %c16_i32 : i32
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %arg5 : i32 to index
      %11 = arith.muli %9, %10 : index
      scf.for %arg15 = %c0_i32 to %arg7 step %c2048_i32  : i32 {
        %12 = arith.index_cast %arg15 : i32 to index
        %13 = arith.addi %11, %12 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [16, 2048], strides: [%10, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [16, 2048], strides: [%10, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
        %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%13], sizes: [16, 2048], strides: [%10, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
        %alloc = memref.alloc() : memref<16x2048xf16>
        %14 = arith.addi %9, %c16 : index
        %15 = arith.index_cast %rows : i32 to index
        %16 = arith.maxsi %9, %15 : index
        %17 = arith.minsi %14, %16 : index
        %18 = arith.subi %17, %9 : index
        %19 = arith.addi %12, %c2048 : index
        %20 = arith.index_cast %arg7 : i32 to index
        %21 = arith.maxsi %12, %20 : index
        %22 = arith.minsi %19, %21 : index
        %23 = arith.subi %22, %12 : index
        %24 = arith.minsi %18, %c16 : index
        %25 = arith.maxsi %24, %c0 : index
        %26 = arith.minsi %23, %c2048 : index
        %27 = arith.maxsi %26, %c0 : index
        %28 = arith.cmpi slt, %25, %c16 : index
        %29 = arith.cmpi slt, %27, %c2048 : index
        %30 = arith.ori %28, %29 : i1
        scf.if %30 {
          linalg.fill ins(%cst : f16) outs(%alloc : memref<16x2048xf16>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_7 = memref.subview %alloc[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16> to memref<?x?xf16, strided<[2048, 1]>>
        memref.copy %subview, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[2048, 1]>>
        %31 = bufferization.to_tensor %alloc restrict writable : memref<16x2048xf16>
        %alloc_8 = memref.alloc() : memref<16x2048xf16>
        scf.if %30 {
          linalg.fill ins(%cst : f16) outs(%alloc_8 : memref<16x2048xf16>)
        } {hivm.unlikely_condition}
        %subview_9 = memref.subview %reinterpret_cast_5[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_10 = memref.subview %alloc_8[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16> to memref<?x?xf16, strided<[2048, 1]>>
        memref.copy %subview_9, %subview_10 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[2048, 1]>>
        %32 = bufferization.to_tensor %alloc_8 restrict writable : memref<16x2048xf16>
        %33 = arith.extf %32 : tensor<16x2048xf16> to tensor<16x2048xf32>
        %34 = arith.mulf %33, %33 : tensor<16x2048xf32>
        %35 = arith.mulf %34, %33 : tensor<16x2048xf32>
        %36 = arith.mulf %35, %1 : tensor<16x2048xf32>
        %37 = arith.addf %33, %36 : tensor<16x2048xf32>
        %38 = arith.mulf %37, %2 : tensor<16x2048xf32>
        %39 = math.tanh %38 : tensor<16x2048xf32>
        %40 = arith.addf %39, %3 : tensor<16x2048xf32>
        %41 = arith.mulf %40, %4 : tensor<16x2048xf32>
        %42 = arith.mulf %39, %39 : tensor<16x2048xf32>
        %43 = arith.mulf %33, %4 : tensor<16x2048xf32>
        %44 = arith.subf %3, %42 : tensor<16x2048xf32>
        %45 = arith.mulf %43, %44 : tensor<16x2048xf32>
        %46 = arith.mulf %33, %5 : tensor<16x2048xf32>
        %47 = arith.mulf %46, %33 : tensor<16x2048xf32>
        %48 = arith.addf %47, %3 : tensor<16x2048xf32>
        %49 = arith.mulf %48, %2 : tensor<16x2048xf32>
        %50 = arith.mulf %45, %49 : tensor<16x2048xf32>
        %51 = arith.addf %41, %50 : tensor<16x2048xf32>
        %52 = arith.truncf %51 : tensor<16x2048xf32> to tensor<16x2048xf16>
        %53 = arith.mulf %31, %52 : tensor<16x2048xf16>
        %extracted_slice = tensor.extract_slice %53[0, 0] [%25, %27] [1, 1] : tensor<16x2048xf16> to tensor<?x?xf16>
        %subview_11 = memref.subview %reinterpret_cast_6[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_11 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @folded_constant_rows
// CHECK-SAME: program_split.launch = {
// CHECK-SAME: static_rows = 32 : i32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @folded_constant_rows(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", ascend.memory.pointer_args_noalias, mix_mode = "aiv", parallel_mode = "simd"} {
    %rows = arith.constant 32 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c16 = arith.constant 16 : index
    %c2048_i32 = arith.constant 2048 : i32
    %c0_i32 = arith.constant 0 : i32
    %c15_i32 = arith.constant 15 : i32
    %cst_0 = arith.constant 1.341450e-01 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 0.797884583 : f32
    %c16_i32 = arith.constant 16 : i32
    %cst_4 = arith.constant 4.471500e-02 : f32
    %0 = tensor.empty() : tensor<16x2048xf32>
    %1 = linalg.fill ins(%cst_4 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %2 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %3 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %7 = arith.constant 2 : i32
    scf.for %arg14 = %arg11 to %7 step %arg8  : i32 {
      %8 = arith.muli %arg14, %c16_i32 : i32
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %arg5 : i32 to index
      %11 = arith.muli %9, %10 : index
      scf.for %arg15 = %c0_i32 to %arg7 step %c2048_i32  : i32 {
        %12 = arith.index_cast %arg15 : i32 to index
        %13 = arith.addi %11, %12 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [16, 2048], strides: [%10, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [16, 2048], strides: [%10, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
        %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%13], sizes: [16, 2048], strides: [%10, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
        %alloc = memref.alloc() : memref<16x2048xf16>
        %14 = arith.addi %9, %c16 : index
        %15 = arith.index_cast %rows : i32 to index
        %16 = arith.maxsi %9, %15 : index
        %17 = arith.minsi %14, %16 : index
        %18 = arith.subi %17, %9 : index
        %19 = arith.addi %12, %c2048 : index
        %20 = arith.index_cast %arg7 : i32 to index
        %21 = arith.maxsi %12, %20 : index
        %22 = arith.minsi %19, %21 : index
        %23 = arith.subi %22, %12 : index
        %24 = arith.minsi %18, %c16 : index
        %25 = arith.maxsi %24, %c0 : index
        %26 = arith.minsi %23, %c2048 : index
        %27 = arith.maxsi %26, %c0 : index
        %28 = arith.cmpi slt, %25, %c16 : index
        %29 = arith.cmpi slt, %27, %c2048 : index
        %30 = arith.ori %28, %29 : i1
        scf.if %30 {
          linalg.fill ins(%cst : f16) outs(%alloc : memref<16x2048xf16>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_7 = memref.subview %alloc[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16> to memref<?x?xf16, strided<[2048, 1]>>
        memref.copy %subview, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[2048, 1]>>
        %31 = bufferization.to_tensor %alloc restrict writable : memref<16x2048xf16>
        %alloc_8 = memref.alloc() : memref<16x2048xf16>
        scf.if %30 {
          linalg.fill ins(%cst : f16) outs(%alloc_8 : memref<16x2048xf16>)
        } {hivm.unlikely_condition}
        %subview_9 = memref.subview %reinterpret_cast_5[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_10 = memref.subview %alloc_8[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16> to memref<?x?xf16, strided<[2048, 1]>>
        memref.copy %subview_9, %subview_10 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[2048, 1]>>
        %32 = bufferization.to_tensor %alloc_8 restrict writable : memref<16x2048xf16>
        %33 = arith.extf %32 : tensor<16x2048xf16> to tensor<16x2048xf32>
        %34 = arith.mulf %33, %33 : tensor<16x2048xf32>
        %35 = arith.mulf %34, %33 : tensor<16x2048xf32>
        %36 = arith.mulf %35, %1 : tensor<16x2048xf32>
        %37 = arith.addf %33, %36 : tensor<16x2048xf32>
        %38 = arith.mulf %37, %2 : tensor<16x2048xf32>
        %39 = math.tanh %38 : tensor<16x2048xf32>
        %40 = arith.addf %39, %3 : tensor<16x2048xf32>
        %41 = arith.mulf %40, %4 : tensor<16x2048xf32>
        %42 = arith.mulf %39, %39 : tensor<16x2048xf32>
        %43 = arith.mulf %33, %4 : tensor<16x2048xf32>
        %44 = arith.subf %3, %42 : tensor<16x2048xf32>
        %45 = arith.mulf %43, %44 : tensor<16x2048xf32>
        %46 = arith.mulf %33, %5 : tensor<16x2048xf32>
        %47 = arith.mulf %46, %33 : tensor<16x2048xf32>
        %48 = arith.addf %47, %3 : tensor<16x2048xf32>
        %49 = arith.mulf %48, %2 : tensor<16x2048xf32>
        %50 = arith.mulf %45, %49 : tensor<16x2048xf32>
        %51 = arith.addf %41, %50 : tensor<16x2048xf32>
        %52 = arith.truncf %51 : tensor<16x2048xf32> to tensor<16x2048xf16>
        %53 = arith.mulf %31, %52 : tensor<16x2048xf16>
        %extracted_slice = tensor.extract_slice %53[0, 0] [%25, %27] [1, 1] : tensor<16x2048xf16> to tensor<?x?xf16>
        %subview_11 = memref.subview %reinterpret_cast_6[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_11 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @reordered_unused
// CHECK-SAME: input_args = array<i32: 4, 5>
// CHECK-SAME: output_arg = 3 : i32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @reordered_unused(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %unused: memref<?xi64>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", ascend.memory.pointer_args_noalias, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c16 = arith.constant 16 : index
    %c2048_i32 = arith.constant 2048 : i32
    %c0_i32 = arith.constant 0 : i32
    %c15_i32 = arith.constant 15 : i32
    %cst_0 = arith.constant 1.341450e-01 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 0.797884583 : f32
    %c16_i32 = arith.constant 16 : i32
    %cst_4 = arith.constant 4.471500e-02 : f32
    %0 = tensor.empty() : tensor<16x2048xf32>
    %1 = linalg.fill ins(%cst_4 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %2 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %3 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<16x2048xf32>) -> tensor<16x2048xf32>
    %6 = arith.addi %arg6, %c15_i32 : i32
    %7 = arith.divsi %6, %c16_i32 : i32
    scf.for %arg14 = %arg11 to %7 step %arg8  : i32 {
      %8 = arith.muli %arg14, %c16_i32 : i32
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %arg5 : i32 to index
      %11 = arith.muli %9, %10 : index
      scf.for %arg15 = %c0_i32 to %arg7 step %c2048_i32  : i32 {
        %12 = arith.index_cast %arg15 : i32 to index
        %13 = arith.addi %11, %12 : index
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%13], sizes: [16, 2048], strides: [%10, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [16, 2048], strides: [%10, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
        %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [16, 2048], strides: [%10, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
        %alloc = memref.alloc() : memref<16x2048xf16>
        %14 = arith.addi %9, %c16 : index
        %15 = arith.index_cast %arg6 : i32 to index
        %16 = arith.maxsi %9, %15 : index
        %17 = arith.minsi %14, %16 : index
        %18 = arith.subi %17, %9 : index
        %19 = arith.addi %12, %c2048 : index
        %20 = arith.index_cast %arg7 : i32 to index
        %21 = arith.maxsi %12, %20 : index
        %22 = arith.minsi %19, %21 : index
        %23 = arith.subi %22, %12 : index
        %24 = arith.minsi %18, %c16 : index
        %25 = arith.maxsi %24, %c0 : index
        %26 = arith.minsi %23, %c2048 : index
        %27 = arith.maxsi %26, %c0 : index
        %28 = arith.cmpi slt, %25, %c16 : index
        %29 = arith.cmpi slt, %27, %c2048 : index
        %30 = arith.ori %28, %29 : i1
        scf.if %30 {
          linalg.fill ins(%cst : f16) outs(%alloc : memref<16x2048xf16>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_7 = memref.subview %alloc[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16> to memref<?x?xf16, strided<[2048, 1]>>
        memref.copy %subview, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[2048, 1]>>
        %31 = bufferization.to_tensor %alloc restrict writable : memref<16x2048xf16>
        %alloc_8 = memref.alloc() : memref<16x2048xf16>
        scf.if %30 {
          linalg.fill ins(%cst : f16) outs(%alloc_8 : memref<16x2048xf16>)
        } {hivm.unlikely_condition}
        %subview_9 = memref.subview %reinterpret_cast_5[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_10 = memref.subview %alloc_8[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16> to memref<?x?xf16, strided<[2048, 1]>>
        memref.copy %subview_9, %subview_10 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[2048, 1]>>
        %32 = bufferization.to_tensor %alloc_8 restrict writable : memref<16x2048xf16>
        %33 = arith.extf %32 : tensor<16x2048xf16> to tensor<16x2048xf32>
        %34 = arith.mulf %33, %33 : tensor<16x2048xf32>
        %35 = arith.mulf %34, %33 : tensor<16x2048xf32>
        %36 = arith.mulf %35, %1 : tensor<16x2048xf32>
        %37 = arith.addf %33, %36 : tensor<16x2048xf32>
        %38 = arith.mulf %37, %2 : tensor<16x2048xf32>
        %39 = math.tanh %38 : tensor<16x2048xf32>
        %40 = arith.addf %39, %3 : tensor<16x2048xf32>
        %41 = arith.mulf %40, %4 : tensor<16x2048xf32>
        %42 = arith.mulf %39, %39 : tensor<16x2048xf32>
        %43 = arith.mulf %33, %4 : tensor<16x2048xf32>
        %44 = arith.subf %3, %42 : tensor<16x2048xf32>
        %45 = arith.mulf %43, %44 : tensor<16x2048xf32>
        %46 = arith.mulf %33, %5 : tensor<16x2048xf32>
        %47 = arith.mulf %46, %33 : tensor<16x2048xf32>
        %48 = arith.addf %47, %3 : tensor<16x2048xf32>
        %49 = arith.mulf %48, %2 : tensor<16x2048xf32>
        %50 = arith.mulf %45, %49 : tensor<16x2048xf32>
        %51 = arith.addf %41, %50 : tensor<16x2048xf32>
        %52 = arith.truncf %51 : tensor<16x2048xf32> to tensor<16x2048xf16>
        %53 = arith.mulf %31, %52 : tensor<16x2048xf16>
        %extracted_slice = tensor.extract_slice %53[0, 0] [%25, %27] [1, 1] : tensor<16x2048xf16> to tensor<?x?xf16>
        %subview_11 = memref.subview %reinterpret_cast_6[0, 0] [%25, %27] [1, 1] : memref<16x2048xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_11 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @jagged_reordered
// CHECK-SAME: input_args = array<i32: 5>
// CHECK-SAME: output_arg = 2 : i32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @jagged_reordered(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {ascend.memory.pointer_args_noalias, SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
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
        %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%19], sizes: [256, 256], strides: [%14, %c1] : memref<?xbf16> to memref<256x256xbf16, strided<[?, ?], offset: ?>>
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
        %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [256, 256], strides: [%14, %c1] : memref<?xbf16> to memref<256x256xbf16, strided<[?, ?], offset: ?>>
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
