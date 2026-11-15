// RUN: bishengir-opt %s --hivm-graph-sync-solver | FileCheck %s
//
// Opposite-bank prefetch and delivery overlap within each iteration. Maple4
// drains both MTE2 writes and MTE1 reads before leaving either parity branch;
// this protects the next iteration without loop-carried outstanding events.
// CHECK: scf.for
// CHECK: scf.if
// CHECK: hivm.hir.load
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// CHECK-NOT: hivm.hir.wait_flag
// CHECK: hivm.hir.l12ub
// CHECK: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// CHECK: } else {
// CHECK: hivm.hir.load
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// CHECK-NOT: hivm.hir.wait_flag
// CHECK: hivm.hir.l12ub
// CHECK: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// CHECK: return

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
 func.func @bank_schedule(%gm: memref<16x2048xf32, #hivm.address_space<gm>>, %ping: i1) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : index
  %end = arith.constant 4 : index
  %offset = arith.constant 131072 : i64
  %a = hivm.hir.pointer_cast(%c0) : memref<16x2048xf32, #hivm.address_space<cbuf>>
  %b = hivm.hir.pointer_cast(%offset) : memref<16x2048xf32, #hivm.address_space<cbuf>>
  %ub = hivm.hir.pointer_cast(%c0) : memref<16x2048xf32, #hivm.address_space<ub>>
  scf.for %i = %zero to %end step %c1 {
   scf.if %ping {
   hivm.hir.load ins(%gm : memref<16x2048xf32, #hivm.address_space<gm>>) outs(%b : memref<16x2048xf32, #hivm.address_space<cbuf>>)
   hivm.hir.l12ub ins(%a : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>)
   } else {
   hivm.hir.load ins(%gm : memref<16x2048xf32, #hivm.address_space<gm>>) outs(%a : memref<16x2048xf32, #hivm.address_space<cbuf>>)
   hivm.hir.l12ub ins(%b : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>)
   }
  }
  return
 }
}
