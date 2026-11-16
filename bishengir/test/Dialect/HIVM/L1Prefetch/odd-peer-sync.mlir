// RUN: bishengir-opt %s --l1-prefetch-consumer-handoff | FileCheck %s
// grid odd and base == floor(grid/2) makes peer == grid (inactive).
// Its payload must be skipped, but its handoff remains balanced in both banks.
// CHECK-LABEL: func.func @odd_peer_handoff
// CHECK: %[[ACTIVE:.*]] = arith.cmpi slt, {{.*}} : i32
// CHECK: scf.for
// CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 18 : i64}>
// CHECK-NEXT: scf.if %[[ACTIVE]] {
// CHECK-NEXT: {{.*}}arith.divsi
// CHECK-NEXT: annotation.mark {{.*}}test_odd_peer_payload
// CHECK-NEXT: hivm.hir.l12ub {{.*}}sub_blockid = true
// CHECK-NEXT: }
// CHECK-NEXT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 16 : i64}>
// CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 18 : i64}>
// CHECK-NEXT: scf.if %[[ACTIVE]] {
// CHECK-NEXT: {{.*}}arith.divsi
// CHECK-NEXT: annotation.mark {{.*}}test_odd_peer_payload
// CHECK-NEXT: hivm.hir.l12ub {{.*}}sub_blockid = true
// CHECK-NEXT: }
// CHECK-NEXT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 16 : i64}>
// CHECK: hivm.hir.sync_block_wait
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @odd_peer_handoff(%gm: memref<16x2048xf32, #hivm.address_space<gm>>, %ping0: memref<16x2048xf32, #hivm.address_space<cbuf>>, %pong0: memref<16x2048xf32, #hivm.address_space<cbuf>>, %ping1: memref<16x2048xf32, #hivm.address_space<cbuf>>, %pong1: memref<16x2048xf32, #hivm.address_space<cbuf>>, %ub: memref<16x2048xf32, #hivm.address_space<ub>>, %active0: i1, %grid: i32, %base: i32, %even: i1) attributes {program_split.transformed, hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %end = arith.constant 4 : index
  %one_i32 = arith.constant 1 : i32
  %two_i32 = arith.constant 2 : i32
  %rounded = arith.addi %grid, %one_i32 : i32
  %half = arith.divsi %rounded, %two_i32 : i32
  %peer = arith.addi %base, %half : i32
  %active1 = arith.cmpi slt, %peer, %grid : i32
  %denom = arith.subi %grid, %peer : i32
  scf.for %i = %zero to %end step %one {
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
    scf.if %even {
      scf.if %active0 {
        hivm.hir.l12ub ins(%ping0 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>)
      }
      scf.if %active1 {
        %q = arith.divsi %peer, %denom : i32
        annotation.mark %q {effects = [], test_odd_peer_payload} : i32
        hivm.hir.l12ub ins(%ping1 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>) sub_blockid = true
      }
    } else {
      scf.if %active0 {
        hivm.hir.l12ub ins(%pong0 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>)
      }
      scf.if %active1 {
        %q = arith.divsi %peer, %denom : i32
        annotation.mark %q {effects = [], test_odd_peer_payload} : i32
        hivm.hir.l12ub ins(%pong1 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>) sub_blockid = true
      }
    }
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
  }
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
  return
}
}
