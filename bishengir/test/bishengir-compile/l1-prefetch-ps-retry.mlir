// Real direct GELU TTAdapter input, selected by backend semantic analysis.
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=true --enable-program-split-mix=true --mlir-print-ir-after=program-split-mix-blockify --mlir-print-ir-after=program-split-l1-materialize --mlir-print-ir-after=hivm-plan-memory-regbase --mlir-print-ir-after=program-split-l1-finalize-sync --mlir-print-ir-after=hivm-graph-sync-solver -o %t.split.o > %t.ir 2>&1
// RUN: FileCheck %s --check-prefix=SCHEDULE < %t.ir
// RUN: FileCheck %s --check-prefix=MATERIALIZE < %t.ir
// RUN: FileCheck %s --check-prefix=MEMORY < %t.ir
// RUN: FileCheck %s --check-prefix=SYNC < %t.ir
// RUN: FileCheck %s --check-prefix=GSS < %t.ir
// RUN: python3 %S/Inputs/l1-contract/check.py %S/Inputs/l1-contract/retry.json %t.split.o.program_split.json
// RUN: bishengir-compile %s --target=Ascend950PR_9589 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=true --enable-program-split-mix=true -o %t.sim.o
// RUN: python3 %S/Inputs/l1-contract/check.py %S/Inputs/l1-contract/retry-simulator.json %t.sim.o.program_split.json
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=false --enable-program-split-mix=true -o %t.ordinary.o
// RUN: FileCheck %s --check-prefix=ORDINARY < %t.ordinary.o.program_split.json
// SCHEDULE: hivm.func_core_type = #hivm.func_core_type<MIX>
// SCHEDULE: %[[HALF:.*]] = arith.constant {program_split.physical_count} 28 : i32
// SCHEDULE: scope.scope
// SCHEDULE: %[[SUB:.*]] = hivm.hir.get_sub_block_idx -> i64
// SCHEDULE: %[[SUB32:.*]] = arith.trunci %[[SUB]] : i64 to i32
// SCHEDULE: %[[OFFSET:.*]] = arith.muli %[[SUB32]], %[[HALF]] : i32
// SCHEDULE: %{{.*}} = arith.addi %{{.*}}, %[[OFFSET]] : i32
// SCHEDULE: program_split.whole_program_vector
// ORDINARY: "kind":"ordinary"
// MATERIALIZE: IR Dump After ProgramSplitL1Materialize
// MATERIALIZE: %[[S0_PING:.*]] = memref.alloc() : memref<16x2048xf32, #hivm.address_space<cbuf>>
// MATERIALIZE: %[[S0_PONG:.*]] = memref.alloc() : memref<16x2048xf32, #hivm.address_space<cbuf>>
// MATERIALIZE: %[[S1_PING:.*]] = memref.alloc() : memref<16x2048xf32, #hivm.address_space<cbuf>>
// MATERIALIZE: %[[S1_PONG:.*]] = memref.alloc() : memref<16x2048xf32, #hivm.address_space<cbuf>>
// Prologue: iteration zero is staged into both ping banks before the loop.
// MATERIALIZE: hivm.hir.load ins(%{{.*}}) outs(%{{.*}} : memref<16x2048xf32>) {{.*}}eviction_policy = <EvictFirst> core_type = <CUBE>
// MATERIALIZE: annotation.mark %{{.*}} keys = ["bind_buffer"] values = [%[[S0_PING]]
// MATERIALIZE: hivm.hir.load ins(%{{.*}}) outs(%{{.*}} : memref<16x2048xf32>) {{.*}}eviction_policy = <EvictFirst> core_type = <CUBE>
// MATERIALIZE: annotation.mark %{{.*}} keys = ["bind_buffer"] values = [%[[S1_PING]]
// MATERIALIZE: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// MATERIALIZE: %[[TWO:.*]] = arith.constant 2 : i32
// MATERIALIZE: %[[ZERO:.*]] = arith.constant 0 : i32
// MATERIALIZE: %[[PARITY:.*]] = arith.remsi %[[IV]], %[[TWO]] : i32
// MATERIALIZE: %[[PING:.*]] = arith.cmpi eq, %[[PARITY]], %[[ZERO]] : i32
// Current delivery and opposite-bank prefetch share one parity branch.
// Keep the bank identities and the i+1 guard exact.
// MATERIALIZE: annotation.mark %{{.*}} {hivm.sync_memory_read_id = 0 : i32, program_split.selected_input = 0 : i32}
// MATERIALIZE: %[[NEXT:.*]] = arith.addi %[[IV]], %{{.*}} : i32
// MATERIALIZE: %[[HAS_NEXT:.*]] = arith.cmpi slt, %[[NEXT]], %{{.*}} : i32
// MATERIALIZE: scf.if %[[PING]]
// MATERIALIZE: scf.if %[[HAS_NEXT]]
// MATERIALIZE: annotation.mark %{{.*}} keys = ["bind_buffer"] values = [%[[S0_PONG]]
// MATERIALIZE: annotation.mark %{{.*}} keys = ["bind_buffer"] values = [%[[S1_PONG]]
// MATERIALIZE: hivm.hir.l12ub ins(%[[S0_PING]]
// MATERIALIZE-NEXT: hivm.hir.l12ub ins(%[[S1_PING]]{{.*}}sub_blockid = true
// MATERIALIZE: } else {
// MATERIALIZE: scf.if %[[HAS_NEXT]]
// MATERIALIZE: annotation.mark %{{.*}} keys = ["bind_buffer"] values = [%[[S0_PING]]
// MATERIALIZE: annotation.mark %{{.*}} keys = ["bind_buffer"] values = [%[[S1_PING]]
// MATERIALIZE: hivm.hir.l12ub ins(%[[S0_PONG]]
// MATERIALIZE-NEXT: hivm.hir.l12ub ins(%[[S1_PONG]]{{.*}}sub_blockid = true
// MATERIALIZE: hivm.hir.store{{.*}}hivm.sync_memory_read_ids = array<i32: 0>
// MATERIALIZE-NOT: core_type = <VECTOR>
// SYNC: IR Dump After ProgramSplitL1FinalizeSync
// SYNC-LABEL: func.func @_gelu_direct_56aiv_mix_aic
// AIC owns the two-load prologue.
// SYNC: hivm.hir.load
// SYNC: hivm.hir.load
// SYNC: scf.for
// UB handback is outside the validity guard and precedes the current delivery.
// SYNC: <PIPE_V>, <PIPE_MTE1>
// SYNC: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = [[$REUSE_FLAG:[0-9]+]]
// SYNC: scf.if
// The i+1 guard precedes paired parity branches; readiness follows both branches.
// SYNC: arith.addi
// SYNC: arith.cmpi slt
// SYNC: scf.if
// SYNC: scf.if
// SYNC: hivm.hir.load
// SYNC: hivm.hir.load
// SYNC: hivm.hir.l12ub
// SYNC: hivm.hir.l12ub{{.*}}sub_blockid = true
// SYNC: } else {
// SYNC: scf.if
// SYNC: hivm.hir.load
// SYNC: hivm.hir.load
// SYNC: hivm.hir.l12ub
// SYNC: hivm.hir.l12ub{{.*}}sub_blockid = true
// SYNC: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_V>]
// SYNC-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>]
// SYNC-LABEL: func.func @_gelu_direct_56aiv_mix_aiv
// SYNC: scf.for
// SYNC: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>]
// SYNC-NOT: hivm.hir.load
// SYNC: scf.if
// SYNC: hivm.hir.store
// SYNC: <PIPE_V>, <PIPE_MTE1>
// SYNC: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = [[$REUSE_FLAG]]
// Four disjoint 128 KiB L1 banks exactly occupy the 512 KiB target budget.
// Maple4 may assign the middle banks in either address order.
// MEMORY: IR Dump After PlanMemoryRegBase
// MEMORY-DAG: %[[L1_0:.*]] = hivm.hir.pointer_cast(%c0_i64) : memref<16x2048xf32, #hivm.address_space<cbuf>>
// MEMORY-DAG: %[[L1_1:.*]] = hivm.hir.pointer_cast(%c131072_i64) : memref<16x2048xf32, #hivm.address_space<cbuf>>
// MEMORY-DAG: %[[L1_2:.*]] = hivm.hir.pointer_cast(%c262144_i64) : memref<16x2048xf32, #hivm.address_space<cbuf>>
// MEMORY-DAG: %[[L1_3:.*]] = hivm.hir.pointer_cast(%c393216_i64) : memref<16x2048xf32, #hivm.address_space<cbuf>>
// MEMORY: %[[UB_REUSED:.*]] = hivm.hir.pointer_cast(%c0_i64) : memref<16x2048xf32, #hivm.address_space<ub>>
// GraphSyncSolver owns bank protection and keeps both final-reader edges.
// GSS: IR Dump After GraphSyncSolver
// GSS: func.func @_gelu_direct_56aiv_mix_aiv
// GSS: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_MTE1>]
// GSS: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>]
// GSS: scf.for
// GSS: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_V>]
// GSS: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>]
// GSS: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
// GSS: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
// GSS: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
// GSS: hivm.hir.store
// GSS: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_MTE1>]
// GSS: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>]
// GSS: func.func @_gelu_direct_56aiv_mix_aic
// GSS: hivm.hir.load
// GSS: hivm.hir.load
// GSS: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// GSS: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// GSS: scf.for
// GSS: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_MTE1>]
// GSS: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>]
// The prologue above establishes first-bank readiness. Each parity branch
// drains its next-bank writes and both current-bank readers before advancing.
// GSS: scf.if
// GSS: hivm.hir.load
// GSS: hivm.hir.load
// GSS: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// GSS-NOT: hivm.hir.wait_flag
// GSS: hivm.hir.l12ub
// GSS: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
// GSS: hivm.hir.l12ub{{.*}}sub_blockid = true
// GSS: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// GSS: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// GSS: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// GSS: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
// GSS: } else {
// GSS: hivm.hir.load
// GSS: hivm.hir.load
// GSS: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// GSS-NOT: hivm.hir.wait_flag
// GSS: hivm.hir.l12ub
// GSS: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
// GSS: hivm.hir.l12ub{{.*}}sub_blockid = true
// GSS: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// GSS: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// GSS: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// GSS: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
// GSS: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>]
// GSS: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_MTE1>]

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
