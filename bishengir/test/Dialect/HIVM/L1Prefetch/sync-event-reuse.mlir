// RUN: bishengir-opt %s --l1-prefetch-balance-sync | FileCheck %s
// Numeric flags can be reused. Finalization must preserve all dependencies,
// including an unrelated V->MTE1 event sharing a flag with a guarded handoff.
// CHECK-LABEL: func.func @cube
// CHECK-COUNT-2: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_MTE1>] flag = 1
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
// CHECK: hivm.hir.l12ub
// CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_V>] flag = 3
module {
  func.func @cube(%src: memref<16x16xf16, #hivm.address_space<cbuf>>,
                  %dst: memref<16x16xf16, #hivm.address_space<ub>>, %active: i1)
      attributes {program_split.transformed, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_MTE1>] flag = 1
    scf.if %active {
      hivm.hir.l12ub ins(%src : memref<16x16xf16, #hivm.address_space<cbuf>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
    }
    scf.if %active {
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_MTE1>] flag = 1
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_V>] flag = 3
    }
    return
  }
}
