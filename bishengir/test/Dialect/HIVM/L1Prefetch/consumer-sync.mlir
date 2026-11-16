// RUN: bishengir-opt %s --l1-prefetch-consumer-handoff | FileCheck %s
// RUN: python3 %S/Inputs/check_consumer_sync.py bishengir-opt %s
//
// The first consumer must receive both readiness events before the second
// consumer's store-completion wait. The inactive pair must balance the same
// event counts without copying UB. Local L1 bank events and drain are retained.
// CHECK-LABEL: func.func @consumer_handoff
// CHECK: scf.for
// CHECK-NOT: sync_block_wait
// CHECK: scf.if %{{.*}} {
// CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// CHECK: scf.if %{{.*}} {
// CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 2 : i64}>
// CHECK-NEXT: hivm.hir.l12ub
// CHECK-NEXT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 1 : i64}>
// CHECK-NEXT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 0 : i64}>
// CHECK-NEXT: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 18 : i64}>
// CHECK-NEXT: hivm.hir.l12ub {{.*}}sub_blockid = true
// CHECK-NEXT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 17 : i64}>
// CHECK-NEXT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 16 : i64}>
// CHECK-NEXT: hivm.hir.load
// CHECK: } else {
// CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = {{[0-9]+}} : i64}>
// CHECK-NEXT: hivm.hir.l12ub
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = {{[0-9]+}} : i64}>
// CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = {{[0-9]+}} : i64}>
// CHECK-NEXT: hivm.hir.l12ub
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = {{[0-9]+}} : i64}>
// CHECK: hivm.hir.load
// CHECK: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// CHECK-NEXT: } else {
// CHECK-NEXT: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = {{[0-9]+}} : i64}>
// CHECK-NEXT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = {{[0-9]+}} : i64}>
// CHECK-NEXT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = {{[0-9]+}} : i64}>
// CHECK-NEXT: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = {{[0-9]+}} : i64}>
// CHECK-NEXT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = {{[0-9]+}} : i64}>
// CHECK-NEXT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = {{[0-9]+}} : i64}>
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @consumer_handoff(%gm: memref<16x2048xf32, #hivm.address_space<gm>>, %ping0: memref<16x2048xf32, #hivm.address_space<cbuf>>, %pong0: memref<16x2048xf32, #hivm.address_space<cbuf>>, %ping1: memref<16x2048xf32, #hivm.address_space<cbuf>>, %pong1: memref<16x2048xf32, #hivm.address_space<cbuf>>, %ub: memref<16x2048xf32, #hivm.address_space<ub>>, %active: i1, %even: i1) attributes {program_split.transformed, hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %end = arith.constant 4 : index
  scf.for %i = %zero to %end step %one {
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
    scf.if %active {
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      scf.if %even {
        hivm.hir.l12ub ins(%ping0 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>)
        hivm.hir.l12ub ins(%ping1 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>) sub_blockid = true
        hivm.hir.load ins(%gm : memref<16x2048xf32, #hivm.address_space<gm>>) outs(%pong0 : memref<16x2048xf32, #hivm.address_space<cbuf>>)
      } else {
        hivm.hir.l12ub ins(%pong0 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>)
        hivm.hir.l12ub ins(%pong1 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>) sub_blockid = true
        hivm.hir.load ins(%gm : memref<16x2048xf32, #hivm.address_space<gm>>) outs(%ping0 : memref<16x2048xf32, #hivm.address_space<cbuf>>)
      }
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    }
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_V>] flag = 1
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
  }
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
  return
}
}
