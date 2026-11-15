// RUN: python3 %S/Inputs/check_flat.py %s bishengir-opt
// Real TTAdapter input: Q3 RecSys _silu_backward, FP16, 32768 elements.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_silu_backward(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c32768 = arith.constant 32768 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768_i32 = arith.constant 32768 : i32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<32768xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<32768xf32>) -> tensor<32768xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32768xf32>) -> tensor<32768xf32>
    %3 = arith.muli %arg9, %c32768_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%4], sizes: [32768], strides: [1] : memref<?xf16> to memref<32768xf16, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<32768xf16>
    %5 = arith.addi %4, %c32768 : index
    %6 = arith.index_cast %arg5 : i32 to index
    %7 = arith.maxsi %4, %6 : index
    %8 = arith.minsi %5, %7 : index
    %9 = arith.subi %8, %4 : index
    %10 = arith.cmpi slt, %9, %c32768 : index
    scf.if %10 {
      linalg.fill ins(%cst : f16) outs(%alloc : memref<32768xf16>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0] [%9] [1] : memref<32768xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
    %subview_2 = memref.subview %alloc[0] [%9] [1] : memref<32768xf16> to memref<?xf16, strided<[1]>>
    memref.copy %subview, %subview_2 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
    %11 = bufferization.to_tensor %alloc restrict writable : memref<32768xf16>
    %12 = arith.extf %11 : tensor<32768xf16> to tensor<32768xf32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%4], sizes: [32768], strides: [1] : memref<?xf16> to memref<32768xf16, strided<[1], offset: ?>>
    %alloc_4 = memref.alloc() : memref<32768xf16>
    scf.if %10 {
      linalg.fill ins(%cst : f16) outs(%alloc_4 : memref<32768xf16>)
    } {hivm.unlikely_condition}
    %subview_5 = memref.subview %reinterpret_cast_3[0] [%9] [1] : memref<32768xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
    %subview_6 = memref.subview %alloc_4[0] [%9] [1] : memref<32768xf16> to memref<?xf16, strided<[1]>>
    memref.copy %subview_5, %subview_6 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
    %13 = bufferization.to_tensor %alloc_4 restrict writable : memref<32768xf16>
    %14 = arith.extf %13 : tensor<32768xf16> to tensor<32768xf32>
    %15 = arith.subf %2, %14 : tensor<32768xf32>
    %16 = math.exp %15 : tensor<32768xf32>
    %17 = arith.addf %16, %1 : tensor<32768xf32>
    %18 = arith.divf %1, %17 : tensor<32768xf32>
    %19 = arith.mulf %14, %18 : tensor<32768xf32>
    %20 = arith.subf %1, %18 : tensor<32768xf32>
    %21 = arith.mulf %19, %20 : tensor<32768xf32>
    %22 = arith.addf %18, %21 : tensor<32768xf32>
    %23 = arith.mulf %12, %22 : tensor<32768xf32>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%4], sizes: [32768], strides: [1] : memref<?xf16> to memref<32768xf16, strided<[1], offset: ?>>
    %24 = arith.truncf %23 : tensor<32768xf32> to tensor<32768xf16>
    %extracted_slice = tensor.extract_slice %24[0] [%9] [1] : tensor<32768xf16> to tensor<?xf16>
    %subview_8 = memref.subview %reinterpret_cast_7[0] [%9] [1] : memref<32768xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_8 : (tensor<?xf16>, memref<?xf16, strided<[1], offset: ?>>) -> ()
    return
  }
}
