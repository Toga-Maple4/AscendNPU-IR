// RUN: bishengir-opt %s -l1-prefetch-map-programs | FileCheck %s
// RUN: bishengir-opt %s -l1-prefetch-map-programs -hivm-split-mix-kernel | FileCheck %s --check-prefix=SPLIT

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @two_logical_programs(%grid_x: i32, %grid_y: i32, %grid_z: i32,
                                  %src: memref<16xf32, #hivm.address_space<gm>>,
                                  %dst: memref<16xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>,
                  program_split.transformed,
                  program_split.launch = {input_args = array<i32: 3>, output_arg = 4 : i32, logical_programs = 4 : i32, physical_programs = 2 : i32, persistent_steps = 5 : i32, row_tasks = 6 : i32}} {
    %xy = arith.muli %grid_x, %grid_y : i32
    %logical_count = arith.muli %xy, %grid_z : i32
    annotation.mark %logical_count {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    %buf = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %buf {program_split.selected_input = 0 : i32} : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<16xf32, #hivm.address_space<gm>>) outs(%buf : memref<16xf32, #hivm.address_space<ub>>)
    %id = arith.trunci %block : i64 to i32
    %next = arith.addi %id, %grid_x : i32
    annotation.mark %next {program_split.test_result} : i32
    return
  }
}

// CHECK-LABEL: func.func @two_logical_programs
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
// CHECK: annotation.mark %{{.*}} {logical_block_num}
// CHECK: %[[BLOCK:.*]] = hivm.hir.get_block_idx
// CHECK: %[[PHYSICAL:.*]] = arith.trunci %[[BLOCK]] {program_split.physical_id} : i64 to i32
// CHECK: %[[HALF:.*]] = arith.constant {program_split.physical_count} 2 : i32
// CHECK: scope.scope
// CHECK: %[[SUB:.*]] = hivm.hir.get_sub_block_idx -> i64
// CHECK: %[[SUB32:.*]] = arith.trunci %[[SUB]] : i64 to i32
// CHECK: %[[OFFSET:.*]] = arith.muli %[[SUB32]], %[[HALF]] : i32
// CHECK: %[[LOGICAL32:.*]] = arith.addi %[[PHYSICAL]], %[[OFFSET]] : i32
// CHECK: %[[LOGICAL:.*]] = arith.extsi %[[LOGICAL32]] {program_split.logical_id} : i32 to i64
// CHECK: %{{.*}} = arith.trunci %[[LOGICAL]] : i64 to i32
// CHECK: annotation.mark %{{.*}} {program_split.test_result}
// CHECK: scope.return
// CHECK: {hivm.tcore_type = #hivm.tcore_type<VECTOR>, no_inline, program_split.whole_program_vector}

// SPLIT-LABEL: func.func @two_logical_programs_mix_aic
// SPLIT-NOT: get_sub_block_idx
// SPLIT-NOT: program_split.whole_program_vector
// SPLIT-NOT: program_split.selected_input
// SPLIT-LABEL: func.func @two_logical_programs_mix_aiv
// SPLIT: hivm.hir.get_sub_block_idx
// SPLIT: program_split.selected_input
// SPLIT: program_split.whole_program_vector
