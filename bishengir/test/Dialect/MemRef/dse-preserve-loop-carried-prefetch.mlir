// RUN: bishengir-opt %s -memref-dse | FileCheck %s

// Regression test: memref-dse must NOT erase a software-pipelined GM->L1
// prefetch load whose buffer is consumed by the *next* loop iteration.
//
// The hivm.hir.load at the bottom of the loop stages the next iteration's tile
// into %buf; the hivm.hir.l12ub at the top consumes what the previous iteration
// staged (loop-carried through %buf). eraseUnusedLoad walked ops in linear
// program order: it saw the prefetch load as the last op on %buf with an unused
// result and erased it, never modelling the loop back-edge that carries the
// value to the top-of-loop read. The result was an L1 buffer read before it was
// ever written. The fix (isTargetReadWithinEnclosingLoop) keeps a write whose
// buffer is read anywhere inside the enclosing loop.

// CHECK-LABEL: func.func @loop_carried_prefetch
// CHECK:      scf.for
// The loop-carried consume stays:
// CHECK:        hivm.hir.l12ub
// The prefetch that feeds the next iteration must be preserved:
// CHECK:        hivm.hir.load
// CHECK-SAME:     outs(%{{.*}} : memref<16x2048xf16, #hivm.address_space<cbuf>>)
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @loop_carried_prefetch(%gm: memref<?xf16> {tt.tensor_kind = 0 : i32},
                                   %ubsink: memref<16x2048xf16, #hivm.address_space<ub>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %gm2d = memref.reinterpret_cast %gm to offset: [0], sizes: [16, 2048], strides: [16384, 1] : memref<?xf16> to memref<16x2048xf16, strided<[16384, 1], offset: ?>>
    %buf = memref.alloc() : memref<16x2048xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0 to %c16 step %c1 {
      hivm.hir.l12ub ins(%buf : memref<16x2048xf16, #hivm.address_space<cbuf>>) outs(%ubsink : memref<16x2048xf16, #hivm.address_space<ub>>)
      hivm.hir.load ins(%gm2d : memref<16x2048xf16, strided<[16384, 1], offset: ?>>) outs(%buf : memref<16x2048xf16, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %true : i1 eviction_policy = <EvictFirst>
    }
    return
  }
}

// -----

// Nested loops: the prefetch write lives in an INNER loop while the read that
// consumes it is carried by the OUTER loop. Checking only the nearest (inner)
// loop would miss the outer read and wrongly erase the prefetch; the fix walks
// the outermost enclosing loop.

// CHECK-LABEL: func.func @nested_loop_carried_prefetch
// CHECK:      scf.for
// CHECK:        hivm.hir.l12ub
// CHECK:        scf.for
// The inner-loop prefetch, carried by the outer loop, must survive:
// CHECK:          hivm.hir.load
// CHECK-SAME:       outs(%{{.*}} : memref<16x2048xf16, #hivm.address_space<cbuf>>)
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @nested_loop_carried_prefetch(%gm: memref<?xf16> {tt.tensor_kind = 0 : i32},
                                          %ubsink: memref<16x2048xf16, #hivm.address_space<ub>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %gm2d = memref.reinterpret_cast %gm to offset: [0], sizes: [16, 2048], strides: [16384, 1] : memref<?xf16> to memref<16x2048xf16, strided<[16384, 1], offset: ?>>
    %buf = memref.alloc() : memref<16x2048xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0 to %c16 step %c1 {
      hivm.hir.l12ub ins(%buf : memref<16x2048xf16, #hivm.address_space<cbuf>>) outs(%ubsink : memref<16x2048xf16, #hivm.address_space<ub>>)
      scf.for %j = %c0 to %c16 step %c1 {
        hivm.hir.load ins(%gm2d : memref<16x2048xf16, strided<[16384, 1], offset: ?>>) outs(%buf : memref<16x2048xf16, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %true : i1 eviction_policy = <EvictFirst>
      }
    }
    return
  }
}

// -----

// Three levels deep: the read is at an INTERMEDIATE depth (L2) and the prefetch
// write is at the DEEPEST level (L3). The L2 loop carries the write to the read.
// Only the outermost-loop search (walking all depths) catches this.

// CHECK-LABEL: func.func @triple_nested_intermediate_read
// CHECK:      scf.for
// CHECK:        scf.for
// CHECK:          hivm.hir.l12ub
// CHECK:          scf.for
// CHECK:            hivm.hir.load
// CHECK-SAME:        outs(%{{.*}} : memref<16x2048xf16, #hivm.address_space<cbuf>>)
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @triple_nested_intermediate_read(%gm: memref<?xf16> {tt.tensor_kind = 0 : i32},
                                             %ubsink: memref<16x2048xf16, #hivm.address_space<ub>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %gm2d = memref.reinterpret_cast %gm to offset: [0], sizes: [16, 2048], strides: [16384, 1] : memref<?xf16> to memref<16x2048xf16, strided<[16384, 1], offset: ?>>
    %buf = memref.alloc() : memref<16x2048xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0 to %c16 step %c1 {           // L1
      scf.for %j = %c0 to %c16 step %c1 {         // L2 -- carries %buf
        hivm.hir.l12ub ins(%buf : memref<16x2048xf16, #hivm.address_space<cbuf>>) outs(%ubsink : memref<16x2048xf16, #hivm.address_space<ub>>)
        scf.for %k = %c0 to %c16 step %c1 {       // L3 -- deepest
          hivm.hir.load ins(%gm2d : memref<16x2048xf16, strided<[16384, 1], offset: ?>>) outs(%buf : memref<16x2048xf16, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %true : i1 eviction_policy = <EvictFirst>
        }
      }
    }
    return
  }
}
