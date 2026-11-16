// Backend unit fixture: three selected input tiles with one common live rectangle.
// This is not a benchmark preset. 3 x 42 KiB x 2 MB slots exceeds 248 KiB UB;
// 3 x 42 KiB x 4 L1 banks fits 512 KiB. The source continuation adds
// arithmetic to this unit fixture to exercise retry above its profitability
// threshold, without changing its access proof or any benchmark preset.
// RUN: bishengir-opt %s --l1-prefetch-analyze | FileCheck %s --check-prefix=ANALYSIS
// RUN: bishengir-opt %s --l1-prefetch-analyze --l1-prefetch-normalize | FileCheck %s --check-prefix=TRANSFORM
// RUN: python3 %S/Inputs/profitability_fixture.py %s %t.eligible
// RUN: mkdir -p %t.ir
// The A5 driver owns the regbase pipeline; inspect its actual materialization and final IR.
// RUN: bishengir-compile %t.eligible --target=Ascend950PR_9589 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-multi-buffer=true --enable-program-split-mix=true --disable-ffts --mlir-print-ir-after=l1-prefetch-materialize,hivm-plan-memory-regbase --save-temps=%t.ir -o %t.external 2> %t.external.log
// RUN: python3 -c "import pathlib; print(pathlib.Path('%t.external.log').read_text().split('// -----// IR Dump After PlanMemoryRegBase (hivm-plan-memory-regbase) //----- //')[-1])" | FileCheck %s --check-prefix=LOWER
// RUN: FileCheck %s --check-prefix=MATERIALIZE < %t.external.log
// RUN: FileCheck %s --check-prefix=CONTRACT < %t.external.program_split.json
// CONTRACT: "kind":"split_mix_half_grid"
// CONTRACT-SAME: "inputs":[{"arg":2,"type":"*fp16"},{"arg":3,"type":"*fp16"},{"arg":4,"type":"*fp16"}]
// MATERIALIZE: hivm.sync_memory_read_id = 0 : i32
// MATERIALIZE: hivm.sync_memory_read_id = 1 : i32
// MATERIALIZE: hivm.sync_memory_read_id = 2 : i32
// MATERIALIZE: hivm.hir.store{{.*}}hivm.sync_memory_read_ids = array<i32: 0, 1, 2>
// LOWER-LABEL: func.func @three_inputs_mix_aic
// LOWER-COUNT-12: hivm.hir.pointer_cast({{.*}}) : memref<16x1344xf16, #hivm.address_space<cbuf>>
// LOWER: hivm.hir.l12ub
// LOWER: hivm.hir.l12ub
// LOWER: hivm.hir.l12ub
// LOWER-LABEL: func.func @three_inputs_mix_aiv
// ANALYSIS: l1_prefetch_ps.accepted
// ANALYSIS-COUNT-3: program_split.selected_input
// TRANSFORM: program_split.launch = {
// TRANSFORM-SAME: input_args = array<i32: 2, 3, 4>
// TRANSFORM-SAME: output_arg = 5
// TRANSFORM-SAME: partial_tiles = 1 : i32
// TRANSFORM: program_split.transformed
// TRANSFORM-COUNT-3: program_split.selected_input
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @three_inputs(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16>, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", ascend.memory.pointer_args_noalias, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1344 = arith.constant 1344 : index
    %c16 = arith.constant 16 : index
    %c1344_i32 = arith.constant 1344 : i32
    %c0_i32 = arith.constant 0 : i32
    %c15_i32 = arith.constant 15 : i32
    %c16_i32 = arith.constant 16 : i32
    %6 = arith.addi %arg7, %c15_i32 : i32
    %7 = arith.divsi %6, %c16_i32 : i32
    scf.for %arg15 = %arg12 to %7 step %arg9  : i32 {
      %8 = arith.muli %arg15, %c16_i32 : i32
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %arg6 : i32 to index
      %11 = arith.muli %9, %10 : index
      scf.for %arg16 = %c0_i32 to %arg8 step %c1344_i32  : i32 {
        %12 = arith.index_cast %arg16 : i32 to index
        %13 = arith.addi %11, %12 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [16, 1344], strides: [%10, 1] : memref<?xf16> to memref<16x1344xf16, strided<[?, 1], offset: ?>>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [16, 1344], strides: [%10, 1] : memref<?xf16> to memref<16x1344xf16, strided<[?, 1], offset: ?>>
        %third_view = memref.reinterpret_cast %arg4 to offset: [%13], sizes: [16, 1344], strides: [%10, 1] : memref<?xf16> to memref<16x1344xf16, strided<[?, 1], offset: ?>>
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%13], sizes: [16, 1344], strides: [%10, 1] : memref<?xf16> to memref<16x1344xf16, strided<[?, 1], offset: ?>>
        %alloc = memref.alloc() : memref<16x1344xf16>
        %14 = arith.addi %9, %c16 : index
        %15 = arith.index_cast %arg7 : i32 to index
        %16 = arith.maxsi %9, %15 : index
        %17 = arith.minsi %14, %16 : index
        %18 = arith.subi %17, %9 : index
        %19 = arith.addi %12, %c1344 : index
        %20 = arith.index_cast %arg8 : i32 to index
        %21 = arith.maxsi %12, %20 : index
        %22 = arith.minsi %19, %21 : index
        %23 = arith.subi %22, %12 : index
        %24 = arith.minsi %18, %c16 : index
        %25 = arith.maxsi %24, %c0 : index
        %26 = arith.minsi %23, %c1344 : index
        %27 = arith.maxsi %26, %c0 : index
        %28 = arith.cmpi slt, %25, %c16 : index
        %29 = arith.cmpi slt, %27, %c1344 : index
        %30 = arith.ori %28, %29 : i1
        scf.if %30 {
          linalg.fill ins(%cst : f16) outs(%alloc : memref<16x1344xf16>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast[0, 0] [%25, %27] [1, 1] : memref<16x1344xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_7 = memref.subview %alloc[0, 0] [%25, %27] [1, 1] : memref<16x1344xf16> to memref<?x?xf16, strided<[1344, 1]>>
        memref.copy %subview, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[1344, 1]>>
        %31 = bufferization.to_tensor %alloc restrict writable : memref<16x1344xf16>
        %alloc_8 = memref.alloc() : memref<16x1344xf16>
        scf.if %30 {
          linalg.fill ins(%cst : f16) outs(%alloc_8 : memref<16x1344xf16>)
        } {hivm.unlikely_condition}
        %subview_9 = memref.subview %reinterpret_cast_5[0, 0] [%25, %27] [1, 1] : memref<16x1344xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_10 = memref.subview %alloc_8[0, 0] [%25, %27] [1, 1] : memref<16x1344xf16> to memref<?x?xf16, strided<[1344, 1]>>
        memref.copy %subview_9, %subview_10 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[1344, 1]>>
        %32 = bufferization.to_tensor %alloc_8 restrict writable : memref<16x1344xf16>
        %third_alloc = memref.alloc() : memref<16x1344xf16>
        scf.if %30 {
          linalg.fill ins(%cst : f16) outs(%third_alloc : memref<16x1344xf16>)
        } {hivm.unlikely_condition}
        %third_src = memref.subview %third_view[0, 0] [%25, %27] [1, 1] : memref<16x1344xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %third_dst = memref.subview %third_alloc[0, 0] [%25, %27] [1, 1] : memref<16x1344xf16> to memref<?x?xf16, strided<[1344, 1]>>
        memref.copy %third_src, %third_dst : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[1344, 1]>>
        %third_tensor = bufferization.to_tensor %third_alloc restrict writable : memref<16x1344xf16>
        %sum = arith.addf %31, %32 : tensor<16x1344xf16>
        %53 = arith.addf %sum, %third_tensor : tensor<16x1344xf16>
        %extracted_slice = tensor.extract_slice %53[0, 0] [%25, %27] [1, 1] : tensor<16x1344xf16> to tensor<?x?xf16>
        %subview_11 = memref.subview %reinterpret_cast_6[0, 0] [%25, %27] [1, 1] : memref<16x1344xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_11 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}
