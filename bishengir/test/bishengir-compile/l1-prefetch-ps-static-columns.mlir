// Actual original SiLU forward dump at PVK e0d71f5, analyzed in AscendNPU-IR.
// No JIT source or math changes. Static columns are derived from the loop bound.
// RUN: mkdir -p %t.ir
// RUN: bishengir-compile %s --target=Ascend950PR_9589 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=true --enable-program-split-mix=true --mlir-print-ir-after=program-split-l1-materialize --save-temps=%t.ir -o %t.o 2>&1 | FileCheck %s --check-prefix=IR
// RUN: python3 %S/Inputs/l1-contract/check.py %S/Inputs/l1-contract/static-columns.json %t.o.program_split.json
// RUN: FileCheck %s --check-prefix=FINAL < %t.ir/module.hivm.opt.mlir
// RUN: sed 's/constant 8192 : i32/constant 2147483647 : i32/' %s | bishengir-compile --target=Ascend950PR_9589 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=true --enable-program-split-mix=true -o %t.bad.o 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: python3 %S/Inputs/check_l1_ordinary_fallback.py %t.bad.o nested_unproved_shape_bounds
// BAD: [program-split] reject: nested_unproved_shape_bounds
// FINAL-LABEL: func.func @_silu_fwd_kernel_mix_aic(
// FINAL: llvm.sdiv {{.*}}program_split.physical_count
// FINAL: call @load_gm_to_cbuf_2d_half_program_split(
// The committed per-consumer handoff precedes the unchanged paired drain.
// FINAL: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 2 : i64}>
// FINAL: call @l12ub_2d_to_2d_half(
// FINAL: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 0 : i64}>
// FINAL: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 18 : i64}>
// FINAL: call @l12ub_2d_to_2d_half_sub1(
// FINAL: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 16 : i64}>
// FINAL: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>]
// FINAL-LABEL: func.func @_silu_fwd_kernel_mix_aiv(
// FINAL-NOT: call @load_gm_to_
// FINAL: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>]
// FINAL-NOT: call @load_gm_to_
// IR: IR Dump After ProgramSplitL1Materialize
// IR: program_split.launch = {{.*}}static_cols = 8192 : i32
// IR-SAME: program_split.transformed
// IR: memref<32x2048xf16
// IR: hivm.hir.l12ub

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @_silu_fwd_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", ascend.memory.pointer_args_noalias, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %c8192 = arith.constant 8192 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x2048xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<32x2048xf32>) -> tensor<32x2048xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32x2048xf32>) -> tensor<32x2048xf32>
    %3 = arith.addi %arg4, %c31_i32 : i32
    %4 = arith.divsi %3, %c32_i32 : i32
    scf.for %arg11 = %arg8 to %4 step %arg5  : i32 {
      %5 = arith.muli %arg11, %c32_i32 : i32
      %6 = arith.index_cast %5 : i32 to index
      %7 = arith.muli %6, %c8192 : index
      scf.for %arg12 = %c0_i32 to %c8192_i32 step %c2048_i32  : i32 {
        %8 = arith.index_cast %arg12 : i32 to index
        %9 = arith.addi %7, %8 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 2048], strides: [8192, 1] : memref<?xf16> to memref<32x2048xf16, strided<[8192, 1], offset: ?>>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 2048], strides: [8192, 1] : memref<?xf16> to memref<32x2048xf16, strided<[8192, 1], offset: ?>>
        %alloc = memref.alloc() : memref<32x2048xf16>
        %10 = arith.addi %6, %c32 : index
        %11 = arith.index_cast %arg4 : i32 to index
        %12 = arith.maxsi %6, %11 : index
        %13 = arith.minsi %10, %12 : index
        %14 = arith.subi %13, %6 : index
        %15 = arith.addi %8, %c2048 : index
        %16 = arith.maxsi %8, %c8192 : index
        %17 = arith.minsi %15, %16 : index
        %18 = arith.subi %17, %8 : index
        %19 = arith.minsi %14, %c32 : index
        %20 = arith.maxsi %19, %c0 : index
        %21 = arith.minsi %18, %c2048 : index
        %22 = arith.maxsi %21, %c0 : index
        %23 = arith.cmpi slt, %20, %c32 : index
        %24 = arith.cmpi slt, %22, %c2048 : index
        %25 = arith.ori %23, %24 : i1
        scf.if %25 {
          linalg.fill ins(%cst : f16) outs(%alloc : memref<32x2048xf16>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast[0, 0] [%20, %22] [1, 1] : memref<32x2048xf16, strided<[8192, 1], offset: ?>> to memref<?x?xf16, strided<[8192, 1], offset: ?>>
        %subview_3 = memref.subview %alloc[0, 0] [%20, %22] [1, 1] : memref<32x2048xf16> to memref<?x?xf16, strided<[2048, 1]>>
        memref.copy %subview, %subview_3 : memref<?x?xf16, strided<[8192, 1], offset: ?>> to memref<?x?xf16, strided<[2048, 1]>>
        %26 = bufferization.to_tensor %alloc restrict writable : memref<32x2048xf16>
        %27 = arith.extf %26 : tensor<32x2048xf16> to tensor<32x2048xf32>
        %28 = arith.subf %2, %27 : tensor<32x2048xf32>
        %29 = math.exp %28 : tensor<32x2048xf32>
        %30 = arith.addf %29, %1 : tensor<32x2048xf32>
        %31 = arith.divf %1, %30 : tensor<32x2048xf32>
        %32 = arith.mulf %27, %31 : tensor<32x2048xf32>
        %33 = arith.truncf %32 : tensor<32x2048xf32> to tensor<32x2048xf16>
        %extracted_slice = tensor.extract_slice %33[0, 0] [%20, %22] [1, 1] : tensor<32x2048xf16> to tensor<?x?xf16>
        %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%20, %22] [1, 1] : memref<32x2048xf16, strided<[8192, 1], offset: ?>> to memref<?x?xf16, strided<[8192, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_4 : (tensor<?x?xf16>, memref<?x?xf16, strided<[8192, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}
