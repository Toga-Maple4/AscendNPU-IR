// Original GELU backward TTAdapter, recognized by backend analysis with the caller no-alias promise.
// RUN: sed 's/reinterpret_cast %arg4 to/reinterpret_cast %arg2 to/' %s | bishengir-compile --target=Ascend950PR_9589 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=true --enable-program-split-mix=true -o %t.bad.o 2>&1 | FileCheck %s
// RUN: python3 %S/Inputs/check_l1_ordinary_fallback.py %t.bad.o nested_unproved_store_partition
// CHECK: [program-split] reject: nested_unproved_store_partition
// RUN: mkdir -p %t.ir
// RUN: bishengir-compile %s --target=Ascend950PR_9589 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=true --enable-program-split-mix=true --mlir-print-ir-after=program-split-l1-materialize --save-temps=%t.ir -o %t.o 2>&1 | FileCheck %s --check-prefix=MATERIALIZE
// RUN: python3 %S/Inputs/l1-contract/check.py %S/Inputs/l1-contract/two-input.json %t.o.program_split.json
// RUN: FileCheck %s --check-prefix=FINAL < %t.ir/module.hivm.opt.mlir
// MATERIALIZE: program_split.launch = {
// MATERIALIZE: hivm.sync_memory_read_id = 0 : i32
// MATERIALIZE: hivm.sync_memory_read_id = 1 : i32
// MATERIALIZE: hivm.hir.store{{.*}}hivm.sync_memory_read_ids = array<i32: 0, 1>
// FINAL-LABEL: func.func @_gelu_bwd_kernel_mix_aic(
// FINAL-COUNT-8: hivm.hir.pointer_cast({{.*}}) : memref<16x2048xf16, #hivm.address_space<cbuf>>
// FINAL: call @load_gm_to_cbuf_2d_half_program_split(
// FINAL: call @load_gm_to_cbuf_2d_half_program_split(
// The committed per-consumer handoff precedes the unchanged paired drain.
// FINAL: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 2 : i64}>
// FINAL: call @l12ub_2d_to_2d_half(
// FINAL: call @l12ub_2d_to_2d_half(
// FINAL: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 0 : i64}>
// FINAL: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 18 : i64}>
// FINAL: call @l12ub_2d_to_2d_half_sub1(
// FINAL: call @l12ub_2d_to_2d_half_sub1(
// FINAL: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 16 : i64}>
// FINAL: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>]
// FINAL-LABEL: func.func @_gelu_bwd_kernel_mix_aiv(
// FINAL-NOT: call @load_gm_to_
// FINAL: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>]
// FINAL-NOT: call @load_gm_to_

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @_gelu_bwd_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
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
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [16, 2048], strides: [%10, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [16, 2048], strides: [%10, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
        %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%13], sizes: [16, 2048], strides: [%10, 1] : memref<?xf16> to memref<16x2048xf16, strided<[?, 1], offset: ?>>
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
