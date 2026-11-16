// RUN: bishengir-opt %s --l1-prefetch-map-programs -verify-diagnostics
// Malformed physical Flat input bypasses the source predicate proof. A moved
// value used by the scf.if condition escapes the payload, even though the if
// operation considers itself its own ancestor. Reject before sinking the slice;
// do not produce a condition whose definition is inside its own then-region.
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @condition_escape(%grid_x: i32, %grid_y: i32, %grid_z: i32,
                              %src: memref<16xf32>, %dst: memref<16xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>,
                  program_split.transformed,
                  program_split.launch = {input_args = array<i32: 3>, output_arg = 4 : i32,
                    flat_tile = 1 : i32, pair_stride = 28 : i32}} {
    %xy = arith.muli %grid_x, %grid_y : i32
    %count = arith.muli %xy, %grid_z : i32
    annotation.mark %count {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    %id = arith.trunci %block : i64 to i32
    %divisor = arith.subi %count, %id : i32
    %quotient = arith.divsi %id, %divisor : i32
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %buf = memref.alloc() : memref<16xf32>
    annotation.mark %buf {program_split.selected_input = 0 : i32} : memref<16xf32>
    // expected-error@+1 {{L1 pipeline error: unsafe resident dependency escapes the valid-program payload}}
    scf.for %i = %zero to %two step %one : i32 {
      %active = arith.cmpi slt, %quotient, %count : i32
      scf.if %active {
        hivm.hir.load ins(%src : memref<16xf32>) outs(%buf : memref<16xf32>)
      }
    }
    return
  }
}
