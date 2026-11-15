// RUN: bishengir-opt %s -program-split-mix-blockify -verify-diagnostics -split-input-file -o /dev/null

module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // expected-error@+1 {{program-split lost selected input identity}}
  func.func @two_logical_programs(%grid_x: i32, %grid_y: i32, %grid_z: i32,
                                  %src: memref<16xf32, #hivm.address_space<gm>>,
                                  %dst: memref<16xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<AIV>,
                  program_split.transformed,
                  program_split.launch = {input_args = array<i32: 3>, output_arg = 4 : i32, logical_programs = 4 : i32, physical_programs = 2 : i32, persistent_steps = 5 : i32, row_tasks = 6 : i32}} {
    %xy = arith.muli %grid_x, %grid_y : i32
    %logical_count = arith.muli %xy, %grid_z : i32
    annotation.mark %logical_count {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    %buf = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<16xf32, #hivm.address_space<gm>>) outs(%buf : memref<16xf32, #hivm.address_space<ub>>)
    %id = arith.trunci %block : i64 to i32
    %next = arith.addi %id, %grid_x : i32
    annotation.mark %next {program_split.test_result} : i32
    return
  }
}



// -----
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // expected-error@+1 {{L1 pipeline error: missing transformed launch metadata}}
  func.func @missing_launch()
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>,
                  program_split.transformed} {
    return
  }
}
// -----
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // expected-error@+1 {{L1 pipeline error: inconsistent transformed launch metadata}}
  func.func @inconsistent_grid()
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>,
                  program_split.transformed,
                  program_split.launch = {input_args = array<i32: 3>, output_arg = 4 : i32, logical_programs = 4 : i32, physical_programs = 3 : i32}} {
    return
  }
}
