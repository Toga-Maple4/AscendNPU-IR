// RUN: bishengir-opt %s --split-input-file --l1-prefetch-analyze --l1-prefetch-normalize | FileCheck %s
// CHECK: program_split.launch
// CHECK: math.sqrt
// CHECK: program_split.launch
// CHECK: math.sqrt
// CHECK: linalg.reduce
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @flat_sqrt(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: f32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
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
    %13 = math.sqrt %11 : tensor<32768xf32>
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
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @program_sqrt(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.muli %arg13, %c256_i32 : i32
    %1 = arith.muli %arg14, %arg6 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.index_cast %0 : i32 to index
    %4 = arith.index_cast %arg7 : i32 to index
    %5 = arith.muli %3, %4 : index
    %6 = arith.addi %2, %5 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [256, 128], strides: [%4, 1] : memref<?xbf16> to memref<256x128xbf16, strided<[?, 1], offset: ?>>
    %7 = arith.muli %arg14, %arg8 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.index_cast %arg9 : i32 to index
    %10 = arith.muli %3, %9 : index
    %11 = arith.addi %8, %10 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%11], sizes: [256, 128], strides: [%9, 1] : memref<?xbf16> to memref<256x128xbf16, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<256x128xbf16>
    %12 = arith.addi %3, %c256 : index
    %13 = arith.index_cast %arg5 : i32 to index
    %14 = arith.maxsi %3, %13 : index
    %15 = arith.minsi %12, %14 : index
    %16 = arith.subi %15, %3 : index
    %17 = arith.cmpi slt, %16, %c256 : index
    scf.if %17 {
      linalg.fill ins(%cst_0 : bf16) outs(%alloc : memref<256x128xbf16>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0, 0] [%16, 128] [1, 1] : memref<256x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
    %subview_2 = memref.subview %alloc[0, 0] [%16, 128] [1, 1] : memref<256x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
    memref.copy %subview, %subview_2 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
    %18 = bufferization.to_tensor %alloc restrict writable : memref<256x128xbf16>
    %alloc_3 = memref.alloc() : memref<256x128xbf16>
    scf.if %17 {
      linalg.fill ins(%cst_0 : bf16) outs(%alloc_3 : memref<256x128xbf16>)
    } {hivm.unlikely_condition}
    %subview_4 = memref.subview %reinterpret_cast_1[0, 0] [%16, 128] [1, 1] : memref<256x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
    %subview_5 = memref.subview %alloc_3[0, 0] [%16, 128] [1, 1] : memref<256x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
    memref.copy %subview_4, %subview_5 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
    %19 = bufferization.to_tensor %alloc_3 restrict writable : memref<256x128xbf16>
    %20 = arith.mulf %18, %19 : tensor<256x128xbf16>
    %sqrt = math.sqrt %20 : tensor<256x128xbf16>
    %21 = arith.extf %sqrt : tensor<256x128xbf16> to tensor<256x128xf32>
    %22 = tensor.empty() : tensor<256xf32>
    %23 = linalg.fill ins(%cst : f32) outs(%22 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%21 : tensor<256x128xf32>) outs(%23 : tensor<256xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %28 = arith.addf %in, %init : f32
        linalg.yield %28 : f32
      }
    %24 = arith.muli %arg14, %arg5 : i32
    %25 = arith.index_cast %24 : i32 to index
    %26 = arith.addi %25, %3 : index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%26], sizes: [256], strides: [1] : memref<?xbf16> to memref<256xbf16, strided<[1], offset: ?>>
    %27 = arith.truncf %reduced : tensor<256xf32> to tensor<256xbf16>
    %extracted_slice = tensor.extract_slice %27[0] [%16] [1] : tensor<256xbf16> to tensor<?xbf16>
    %subview_7 = memref.subview %reinterpret_cast_6[0] [%16] [1] : memref<256xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_7 : (tensor<?xbf16>, memref<?xbf16, strided<[1], offset: ?>>) -> ()
    return
  }
}
