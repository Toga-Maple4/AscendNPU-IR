// RUN: bishengir-opt %s --l1-prefetch-ps-analysis --l1-prefetch-ps-transform | FileCheck %s
// CHECK: element_bits = 16 : i32
// CHECK-SAME: tile_cols = 128 : i32, tile_rows = 128 : i32
// CHECK: hivm.hir.load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @triton_jagged_to_dense(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {ascend.memory.pointer_args_noalias, SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c128_i64 = arith.constant 128 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
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
    %13 = scf.for %arg14 = %1 to %4 step %c128_i64 iter_args(%arg15 = %c0) -> (index)  : i64 {
      %14 = arith.index_cast %arg6 : i32 to index
      %15 = arith.muli %arg15, %14 : index
      %16:2 = scf.for %arg16 = %c0_i32 to %8 step %c128_i32 iter_args(%arg17 = %c0, %arg18 = %15) -> (index, index)  : i32 {
        %18 = arith.index_cast %6 : i64 to index
        %19 = arith.addi %18, %arg18 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [128, 128], strides: [%14, %c1] : memref<?xf16> to memref<128x128xf16, strided<[?, ?], offset: ?>>
        %alloc = memref.alloc() : memref<128x128xf16>
        %20 = arith.addi %arg15, %c128 : index
        %21 = arith.index_cast %12 : i64 to index
        %22 = arith.maxsi %arg15, %21 : index
        %23 = arith.minsi %20, %22 : index
        %24 = arith.subi %23, %arg15 : index
        %25 = arith.addi %arg17, %c128 : index
        %26 = arith.index_cast %8 : i32 to index
        %27 = arith.maxsi %arg17, %26 : index
        %28 = arith.minsi %25, %27 : index
        %29 = arith.subi %28, %arg17 : index
        %30 = arith.minsi %24, %c128 : index
        %31 = arith.maxsi %30, %c0 : index
        %32 = arith.minsi %29, %c128 : index
        %33 = arith.maxsi %32, %c0 : index
        %34 = arith.cmpi slt, %31, %c128 : index
        %35 = arith.cmpi slt, %33, %c128 : index
        %36 = arith.ori %34, %35 : i1
        scf.if %36 {
          linalg.fill ins(%cst : f16) outs(%alloc : memref<128x128xf16>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast_1[0, 0] [%31, %33] [1, 1] : memref<128x128xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
        %subview_2 = memref.subview %alloc[0, 0] [%31, %33] [1, 1] : memref<128x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview, %subview_2 : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %37 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf16>
        %38 = arith.index_cast %7 : i32 to index
        %39 = arith.addi %38, %arg18 : index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%39], sizes: [128, 128], strides: [%14, %c1] : memref<?xf16> to memref<128x128xf16, strided<[?, ?], offset: ?>>
        %extracted_slice = tensor.extract_slice %37[0, 0] [%31, %33] [1, 1] : tensor<128x128xf16> to tensor<?x?xf16>
        %subview_4 = memref.subview %reinterpret_cast_3[0, 0] [%31, %33] [1, 1] : memref<128x128xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_4 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, ?], offset: ?>>) -> ()
        %40 = arith.addi %arg17, %c128 : index
        %41 = arith.addi %arg18, %c128 : index
        scf.yield %40, %41 : index, index
      }
      %17 = arith.addi %arg15, %c128 : index
      scf.yield %17 : index
    }
    return
  }
}
