// RUN: python3 %S/Inputs/check_jagged_sync.py bishengir-opt %s
// RUN: python3 %S/Inputs/check_common_handoff.py bishengir-opt %S/consumer-sync.mlir %s
// Pure-copy handoff: per-consumer events remain outside independent delivery guards.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @jagged_handoff(%gm: memref<16x2048xf32, #hivm.address_space<gm>>, %ping0: memref<16x2048xf32, #hivm.address_space<cbuf>>, %pong0: memref<16x2048xf32, #hivm.address_space<cbuf>>, %ping1: memref<16x2048xf32, #hivm.address_space<cbuf>>, %pong1: memref<16x2048xf32, #hivm.address_space<cbuf>>, %ub: memref<16x2048xf32, #hivm.address_space<ub>>, %active0: i1, %active1: i1, %even: i1) attributes {program_split.transformed, hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %end = arith.constant 4 : index
  scf.for %i = %zero to %end step %one {
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
    scf.if %even {
      scf.if %active0 {
        hivm.hir.l12ub ins(%ping0 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>)
      }
      scf.if %active1 {
        hivm.hir.l12ub ins(%ping1 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>) sub_blockid = true
      }
    } else {
      scf.if %active0 {
        hivm.hir.l12ub ins(%pong0 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>)
      }
      scf.if %active1 {
        hivm.hir.l12ub ins(%pong1 : memref<16x2048xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<16x2048xf32, #hivm.address_space<ub>>) sub_blockid = true
      }
    }
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
  }
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
  return
}
}
