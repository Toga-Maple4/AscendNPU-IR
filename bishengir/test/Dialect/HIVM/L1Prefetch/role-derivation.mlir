// RUN: bishengir-opt %s --split-input-file --l1-prefetch-ps-analysis | FileCheck %s

// Layout variations use access-derived roles; pointer overlap still rejects.
// CHECK-LABEL: func.func @reordered_pointers
// CHECK-SAME: l1_prefetch_ps.accepted
// CHECK: program_split.selected_input
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @reordered_pointers(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", ascend.memory.pointer_args_noalias, mix_mode = "aiv", parallel_mode = "simd"} {
    %c4096 = arith.constant 4096 : index
    %cst = arith.constant 4.471500e-02 : f32
    %cst_0 = arith.constant -1.59576917 : f32
    %c4_i32 = arith.constant 4 : i32
    %c56_i32 = arith.constant 56 : i32
    %c2048_i32 = arith.constant 1024 : i32
    %c16_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x1024xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
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
        %12 = arith.muli %11, %c4096 : index
        %13 = arith.index_cast %10 : i32 to index
        %14 = arith.addi %12, %13 : index
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [32, 1024], strides: [4096, 1] : memref<?xf32> to memref<32x1024xf32, strided<[4096, 1], offset: ?>>
        %alloc = memref.alloc() : memref<32x1024xf32>
        memref.copy %reinterpret_cast, %alloc : memref<32x1024xf32, strided<[4096, 1], offset: ?>> to memref<32x1024xf32>
        %15 = bufferization.to_tensor %alloc restrict writable : memref<32x1024xf32>
        %16 = arith.mulf %15, %15 : tensor<32x1024xf32>
        %17 = arith.mulf %16, %15 : tensor<32x1024xf32>
        %18 = arith.mulf %17, %3 : tensor<32x1024xf32>
        %19 = arith.addf %15, %18 : tensor<32x1024xf32>
        %20 = arith.mulf %19, %2 : tensor<32x1024xf32>
        %21 = math.exp %20 : tensor<32x1024xf32>
        %22 = arith.addf %21, %1 : tensor<32x1024xf32>
        %23 = arith.divf %15, %22 : tensor<32x1024xf32>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [32, 1024], strides: [4096, 1] : memref<?xf32> to memref<32x1024xf32, strided<[4096, 1], offset: ?>>
        bufferization.materialize_in_destination %23 in writable %reinterpret_cast_2 : (tensor<32x1024xf32>, memref<32x1024xf32, strided<[4096, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @overlapping_output
// CHECK-SAME: l1_prefetch_ps.reject_reason = "unproved_pointer_overlap"
// CHECK-NOT: program_split.selected_input
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @overlapping_output(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", ascend.memory.pointer_args_noalias, mix_mode = "aiv", parallel_mode = "simd"} {
    %c4096 = arith.constant 4096 : index
    %cst = arith.constant 4.471500e-02 : f32
    %cst_0 = arith.constant -1.59576917 : f32
    %c4_i32 = arith.constant 4 : i32
    %c56_i32 = arith.constant 56 : i32
    %c2048_i32 = arith.constant 1024 : i32
    %c16_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x1024xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
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
        %12 = arith.muli %11, %c4096 : index
        %13 = arith.index_cast %10 : i32 to index
        %14 = arith.addi %12, %13 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [32, 1024], strides: [4096, 1] : memref<?xf32> to memref<32x1024xf32, strided<[4096, 1], offset: ?>>
        %alloc = memref.alloc() : memref<32x1024xf32>
        memref.copy %reinterpret_cast, %alloc : memref<32x1024xf32, strided<[4096, 1], offset: ?>> to memref<32x1024xf32>
        %15 = bufferization.to_tensor %alloc restrict writable : memref<32x1024xf32>
        %16 = arith.mulf %15, %15 : tensor<32x1024xf32>
        %17 = arith.mulf %16, %15 : tensor<32x1024xf32>
        %18 = arith.mulf %17, %3 : tensor<32x1024xf32>
        %19 = arith.addf %15, %18 : tensor<32x1024xf32>
        %20 = arith.mulf %19, %2 : tensor<32x1024xf32>
        %21 = math.exp %20 : tensor<32x1024xf32>
        %22 = arith.addf %21, %1 : tensor<32x1024xf32>
        %23 = arith.divf %15, %22 : tensor<32x1024xf32>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [32, 1024], strides: [4096, 1] : memref<?xf32> to memref<32x1024xf32, strided<[4096, 1], offset: ?>>
        bufferization.materialize_in_destination %23 in writable %reinterpret_cast_2 : (tensor<32x1024xf32>, memref<32x1024xf32, strided<[4096, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}

// -----

// Rows and columns share one proven scalar; schema 2 records both roles.
// CHECK-LABEL: func.func @shared_scalar_roles
// CHECK-SAME: l1_prefetch_ps.accepted
// CHECK: program_split.selected_input
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @shared_scalar_roles(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %c8192 = arith.constant 8192 : index
    %cst_0 = arith.constant 4.471500e-02 : f32
    %cst_1 = arith.constant 0.797884583 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x2048xf32>
    %1 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<32x2048xf32>) -> tensor<32x2048xf32>
    %2 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<32x2048xf32>) -> tensor<32x2048xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<32x2048xf32>) -> tensor<32x2048xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32x2048xf32>) -> tensor<32x2048xf32>
    %5 = arith.addi %arg4, %c31_i32 : i32
    %6 = arith.divsi %5, %c32_i32 : i32
    scf.for %arg12 = %arg9 to %6 step %arg6  : i32 {
      %7 = arith.muli %arg12, %c32_i32 : i32
      %8 = arith.index_cast %7 : i32 to index
      %9 = arith.muli %8, %c8192 : index
      scf.for %arg13 = %c0_i32 to %arg4 step %c2048_i32  : i32 {
        %10 = arith.index_cast %arg13 : i32 to index
        %11 = arith.addi %9, %10 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 2048], strides: [8192, 1] : memref<?xf16> to memref<32x2048xf16, strided<[8192, 1], offset: ?>>
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%11], sizes: [32, 2048], strides: [8192, 1] : memref<?xf16> to memref<32x2048xf16, strided<[8192, 1], offset: ?>>
        %alloc = memref.alloc() : memref<32x2048xf16>
        %12 = arith.addi %8, %c32 : index
        %13 = arith.index_cast %arg4 : i32 to index
        %14 = arith.maxsi %8, %13 : index
        %15 = arith.minsi %12, %14 : index
        %16 = arith.subi %15, %8 : index
        %17 = arith.addi %10, %c2048 : index
        %18 = arith.index_cast %arg4 : i32 to index
        %19 = arith.maxsi %10, %18 : index
        %20 = arith.minsi %17, %19 : index
        %21 = arith.subi %20, %10 : index
        %22 = arith.minsi %16, %c32 : index
        %23 = arith.maxsi %22, %c0 : index
        %24 = arith.minsi %21, %c2048 : index
        %25 = arith.maxsi %24, %c0 : index
        %26 = arith.cmpi slt, %23, %c32 : index
        %27 = arith.cmpi slt, %25, %c2048 : index
        %28 = arith.ori %26, %27 : i1
        scf.if %28 {
          linalg.fill ins(%cst : f16) outs(%alloc : memref<32x2048xf16>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast[0, 0] [%23, %25] [1, 1] : memref<32x2048xf16, strided<[8192, 1], offset: ?>> to memref<?x?xf16, strided<[8192, 1], offset: ?>>
        %subview_5 = memref.subview %alloc[0, 0] [%23, %25] [1, 1] : memref<32x2048xf16> to memref<?x?xf16, strided<[2048, 1]>>
        memref.copy %subview, %subview_5 : memref<?x?xf16, strided<[8192, 1], offset: ?>> to memref<?x?xf16, strided<[2048, 1]>>
        %29 = bufferization.to_tensor %alloc restrict writable : memref<32x2048xf16>
        %30 = arith.extf %29 : tensor<32x2048xf16> to tensor<32x2048xf32>
        %31 = arith.mulf %30, %30 : tensor<32x2048xf32>
        %32 = arith.mulf %31, %30 : tensor<32x2048xf32>
        %33 = arith.mulf %32, %4 : tensor<32x2048xf32>
        %34 = arith.addf %30, %33 : tensor<32x2048xf32>
        %35 = arith.mulf %34, %3 : tensor<32x2048xf32>
        %36 = arith.mulf %30, %2 : tensor<32x2048xf32>
        %37 = math.tanh %35 : tensor<32x2048xf32>
        %38 = arith.addf %37, %1 : tensor<32x2048xf32>
        %39 = arith.mulf %36, %38 : tensor<32x2048xf32>
        %40 = arith.truncf %39 : tensor<32x2048xf32> to tensor<32x2048xf16>
        %extracted_slice = tensor.extract_slice %40[0, 0] [%23, %25] [1, 1] : tensor<32x2048xf16> to tensor<?x?xf16>
        %subview_6 = memref.subview %reinterpret_cast_4[0, 0] [%23, %25] [1, 1] : memref<32x2048xf16, strided<[8192, 1], offset: ?>> to memref<?x?xf16, strided<[8192, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?x?xf16>, memref<?x?xf16, strided<[8192, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}
