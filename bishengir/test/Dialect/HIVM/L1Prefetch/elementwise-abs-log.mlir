// RUN: bishengir-opt %s --split-input-file --l1-prefetch-analyze | FileCheck %s
// Derived from LogSigmoid FWD Excel 795's original frontend IR with the marked
// B=32768 tile variant. Shapes/math are unchanged; source locations are stripped.
// abs/log are pure lane-wise operations. Even exceptional padded-lane values
// cannot reach a valid output; integer conversions must still fail that proof.

// CHECK-LABEL: func.func @pointwise_abs_log
// CHECK-SAME: l1_prefetch_ps.accepted
// CHECK: math.absf
// CHECK: math.log
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pointwise_abs_log(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: f32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c32768 = arith.constant 32768 : index
    %cst = arith.constant 1.000000e+00 : f32
    %c32768_i32 = arith.constant 32768 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32768xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32768xf32>) -> tensor<32768xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32768xf32>) -> tensor<32768xf32>
    %3 = arith.muli %arg11, %c32768_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%4], sizes: [32768], strides: [1] : memref<?xf32> to memref<32768xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<32768xf32>
    %5 = arith.addi %4, %c32768 : index
    %6 = arith.index_cast %arg5 : i32 to index
    %7 = arith.maxsi %4, %6 : index
    %8 = arith.minsi %5, %7 : index
    %9 = arith.subi %8, %4 : index
    %10 = arith.cmpi slt, %9, %c32768 : index
    scf.if %10 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<32768xf32>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0] [%9] [1] : memref<32768xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_1 = memref.subview %alloc[0] [%9] [1] : memref<32768xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %11 = bufferization.to_tensor %alloc restrict writable : memref<32768xf32>
    %12 = arith.minnumf %11, %1 : tensor<32768xf32>
    %13 = math.absf %11 : tensor<32768xf32>
    %14 = arith.subf %1, %13 : tensor<32768xf32>
    %15 = math.exp %14 : tensor<32768xf32>
    %16 = arith.addf %15, %2 : tensor<32768xf32>
    %17 = math.log %16 : tensor<32768xf32>
    %18 = arith.subf %12, %17 : tensor<32768xf32>
    %19 = linalg.fill ins(%arg4 : f32) outs(%0 : tensor<32768xf32>) -> tensor<32768xf32>
    %20 = arith.divf %18, %19 : tensor<32768xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%4], sizes: [32768], strides: [1] : memref<?xf32> to memref<32768xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %20[0] [%9] [1] : tensor<32768xf32> to tensor<?xf32>
    %subview_3 = memref.subview %reinterpret_cast_2[0] [%9] [1] : memref<32768xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_3 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


// -----

// CHECK-LABEL: func.func @pointwise_log_integer_escape
// CHECK-SAME: l1_prefetch_ps.reject_reason = "flat_observable_or_unproved_padding"
// CHECK-NOT: program_split.selected_input
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pointwise_log_integer_escape(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: f32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c32768 = arith.constant 32768 : index
    %cst = arith.constant 1.000000e+00 : f32
    %c32768_i32 = arith.constant 32768 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32768xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32768xf32>) -> tensor<32768xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32768xf32>) -> tensor<32768xf32>
    %3 = arith.muli %arg11, %c32768_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%4], sizes: [32768], strides: [1] : memref<?xf32> to memref<32768xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<32768xf32>
    %5 = arith.addi %4, %c32768 : index
    %6 = arith.index_cast %arg5 : i32 to index
    %7 = arith.maxsi %4, %6 : index
    %8 = arith.minsi %5, %7 : index
    %9 = arith.subi %8, %4 : index
    %10 = arith.cmpi slt, %9, %c32768 : index
    scf.if %10 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<32768xf32>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0] [%9] [1] : memref<32768xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_1 = memref.subview %alloc[0] [%9] [1] : memref<32768xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %11 = bufferization.to_tensor %alloc restrict writable : memref<32768xf32>
    %12 = arith.minnumf %11, %1 : tensor<32768xf32>
    %as_int = arith.fptosi %11 : tensor<32768xf32> to tensor<32768xi32>
    %as_float = arith.sitofp %as_int : tensor<32768xi32> to tensor<32768xf32>
    %13 = math.absf %as_float : tensor<32768xf32>
    %14 = arith.subf %1, %13 : tensor<32768xf32>
    %15 = math.exp %14 : tensor<32768xf32>
    %16 = arith.addf %15, %2 : tensor<32768xf32>
    %17 = math.log %16 : tensor<32768xf32>
    %18 = arith.subf %12, %17 : tensor<32768xf32>
    %19 = linalg.fill ins(%arg4 : f32) outs(%0 : tensor<32768xf32>) -> tensor<32768xf32>
    %20 = arith.divf %18, %19 : tensor<32768xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%4], sizes: [32768], strides: [1] : memref<?xf32> to memref<32768xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %20[0] [%9] [1] : tensor<32768xf32> to tensor<?xf32>
    %subview_3 = memref.subview %reinterpret_cast_2[0] [%9] [1] : memref<32768xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_3 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
