// RUN: bishengir-opt %s --l1-prefetch-ps-analysis --l1-prefetch-ps-transform | FileCheck %s --check-prefix=ACCEPT
//
// Real direct GELU TTAdapter input. Triton-Ascend owns the whole-program
// semantic decision and emits the backend contract plus selected-input marker.
// ACCEPT: func.func @_gelu_direct_56aiv
// ACCEPT-SAME: logical_programs = 56 : i32
// ACCEPT-SAME: persistent_steps = 148 : i32
// ACCEPT-SAME: physical_programs = 28 : i32
// ACCEPT-SAME: row_tasks = 2048 : i32
// ACCEPT: annotation.mark %{{.*}} {program_split.selected_input = 0 : i32}
// ACCEPT-NOT: l1_prefetch_ps.reject_reason

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @_gelu_direct_56aiv(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", ascend.memory.pointer_args_noalias, mix_mode = "aiv", parallel_mode = "simd"} {
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
