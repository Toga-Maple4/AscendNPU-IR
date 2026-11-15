// RUN: bishengir-opt %s -split-input-file -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-plan-memory -verify-diagnostics --mlir-print-ir-after-failure 2>&1 | FileCheck %s

module {
  // expected-error@+1 {{ub overflow}}
  func.func @overlapping_selected_set(%a: memref<16x2048xf16, #hivm.address_space<gm>>, %b: memref<16x2048xf16, #hivm.address_space<gm>>) {
    %x = memref.alloc() : memref<16x2048xf16, #hivm.address_space<ub>>
    %y = memref.alloc() : memref<16x2048xf16, #hivm.address_space<ub>>
    annotation.mark %x {program_split.selected_input = 0 : i32, hivm.multi_buffer = 2 : i32} : memref<16x2048xf16, #hivm.address_space<ub>>
    annotation.mark %y {program_split.selected_input = 1 : i32, hivm.multi_buffer = 2 : i32} : memref<16x2048xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%a : memref<16x2048xf16, #hivm.address_space<gm>>) outs(%x : memref<16x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%b : memref<16x2048xf16, #hivm.address_space<gm>>) outs(%y : memref<16x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%x, %y : memref<16x2048xf16, #hivm.address_space<ub>>, memref<16x2048xf16, #hivm.address_space<ub>>) outs(%x : memref<16x2048xf16, #hivm.address_space<ub>>)
    return
  }
}
// CHECK-DAG: @overlapping_selected_set{{.*}}program_split.memory_result = {aligned_slot_bits = array<i64: 524288, 524288>, capacity_bits = 2031616 : i64, kind = "proven_selected_set_capacity_excess", lower_bound_bits = 2097152 : i64, mandatory_conflict = true, selected_ids = array<i32: 0, 1>, slots = array<i32: 2, 2>, space = "UB", version = 2 : i32}

// -----

module {
  func.func @disjoint_selected_set(%a: memref<16x2048xf16, #hivm.address_space<gm>>, %b: memref<16x2048xf16, #hivm.address_space<gm>>, %o: memref<16x2048xf16, #hivm.address_space<gm>>) {
    %x = memref.alloc() : memref<16x2048xf16, #hivm.address_space<ub>>
    annotation.mark %x {program_split.selected_input = 0 : i32, hivm.multi_buffer = 2 : i32} : memref<16x2048xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%a : memref<16x2048xf16, #hivm.address_space<gm>>) outs(%x : memref<16x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%x : memref<16x2048xf16, #hivm.address_space<ub>>) outs(%o : memref<16x2048xf16, #hivm.address_space<gm>>)
    %y = memref.alloc() : memref<16x2048xf16, #hivm.address_space<ub>>
    annotation.mark %y {program_split.selected_input = 1 : i32, hivm.multi_buffer = 2 : i32} : memref<16x2048xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%b : memref<16x2048xf16, #hivm.address_space<gm>>) outs(%y : memref<16x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%y : memref<16x2048xf16, #hivm.address_space<ub>>) outs(%o : memref<16x2048xf16, #hivm.address_space<gm>>)
    return
  }
}
// CHECK-DAG: @disjoint_selected_set{{.*}}kind = "ordinary_plan_succeeded", lower_bound_bits = 1048576 : i64, mandatory_conflict = false

// -----

module {
  // expected-error@+1 {{ub overflow}}
  func.func @duplicate_selected_id(%a: memref<16x2048xf16, #hivm.address_space<gm>>, %b: memref<16x2048xf16, #hivm.address_space<gm>>) {
    %x = memref.alloc() : memref<16x2048xf16, #hivm.address_space<ub>>
    %y = memref.alloc() : memref<16x2048xf16, #hivm.address_space<ub>>
    annotation.mark %x {program_split.selected_input = 0 : i32, hivm.multi_buffer = 2 : i32} : memref<16x2048xf16, #hivm.address_space<ub>>
    annotation.mark %y {program_split.selected_input = 0 : i32, hivm.multi_buffer = 2 : i32} : memref<16x2048xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%a : memref<16x2048xf16, #hivm.address_space<gm>>) outs(%x : memref<16x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%b : memref<16x2048xf16, #hivm.address_space<gm>>) outs(%y : memref<16x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%x, %y : memref<16x2048xf16, #hivm.address_space<ub>>, memref<16x2048xf16, #hivm.address_space<ub>>) outs(%x : memref<16x2048xf16, #hivm.address_space<ub>>)
    return
  }
}
// CHECK-DAG: @duplicate_selected_id{{.*}}program_split.memory_result = {kind = "unknown_or_other_failure", selected_ids = array<i32: 0>, version = 2 : i32}

// -----

module {
  func.func @fitting_selected_set(%a: memref<8x2048xf16, #hivm.address_space<gm>>, %b: memref<8x2048xf16, #hivm.address_space<gm>>) {
    %x = memref.alloc() : memref<8x2048xf16, #hivm.address_space<ub>>
    %y = memref.alloc() : memref<8x2048xf16, #hivm.address_space<ub>>
    annotation.mark %x {program_split.selected_input = 0 : i32, hivm.multi_buffer = 2 : i32} : memref<8x2048xf16, #hivm.address_space<ub>>
    annotation.mark %y {program_split.selected_input = 1 : i32, hivm.multi_buffer = 2 : i32} : memref<8x2048xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%a : memref<8x2048xf16, #hivm.address_space<gm>>) outs(%x : memref<8x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%b : memref<8x2048xf16, #hivm.address_space<gm>>) outs(%y : memref<8x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%x, %y : memref<8x2048xf16, #hivm.address_space<ub>>, memref<8x2048xf16, #hivm.address_space<ub>>) outs(%x : memref<8x2048xf16, #hivm.address_space<ub>>)
    return
  }
}
// CHECK-DAG: @fitting_selected_set{{.*}}kind = "ordinary_plan_succeeded", lower_bound_bits = 1048576 : i64, mandatory_conflict = true

// -----

module {
  // expected-error@+1 {{ub overflow}}
  func.func @unrelated_selected_set(%a: memref<8x2048xf16, #hivm.address_space<gm>>, %b: memref<8x2048xf16, #hivm.address_space<gm>>, %other_source: memref<16x2048xf32, #hivm.address_space<gm>>) {
    %x = memref.alloc() : memref<8x2048xf16, #hivm.address_space<ub>>
    %y = memref.alloc() : memref<8x2048xf16, #hivm.address_space<ub>>
    annotation.mark %x {program_split.selected_input = 0 : i32, hivm.multi_buffer = 2 : i32} : memref<8x2048xf16, #hivm.address_space<ub>>
    annotation.mark %y {program_split.selected_input = 1 : i32, hivm.multi_buffer = 2 : i32} : memref<8x2048xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%a : memref<8x2048xf16, #hivm.address_space<gm>>) outs(%x : memref<8x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%b : memref<8x2048xf16, #hivm.address_space<gm>>) outs(%y : memref<8x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%x, %y : memref<8x2048xf16, #hivm.address_space<ub>>, memref<8x2048xf16, #hivm.address_space<ub>>) outs(%x : memref<8x2048xf16, #hivm.address_space<ub>>)
    %other = memref.alloc() : memref<16x2048xf32, #hivm.address_space<ub>>
    annotation.mark %other {hivm.multi_buffer = 2 : i32} : memref<16x2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%other_source : memref<16x2048xf32, #hivm.address_space<gm>>) outs(%other : memref<16x2048xf32, #hivm.address_space<ub>>)
    return
  }
}
// CHECK-DAG: @unrelated_selected_set{{.*}}kind = "unknown_or_other_failure", lower_bound_bits = 1048576 : i64, mandatory_conflict = true

// -----

module {
  // expected-error@+1 {{ub overflow}}
  func.func @missing_selected_id(%a: memref<16x2048xf16, #hivm.address_space<gm>>, %b: memref<16x2048xf16, #hivm.address_space<gm>>) {
    %x = memref.alloc() : memref<16x2048xf16, #hivm.address_space<ub>>
    %y = memref.alloc() : memref<16x2048xf16, #hivm.address_space<ub>>
    annotation.mark %x {program_split.selected_input = 0 : i32, hivm.multi_buffer = 2 : i32} : memref<16x2048xf16, #hivm.address_space<ub>>
    annotation.mark %y {hivm.multi_buffer = 2 : i32} : memref<16x2048xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%a : memref<16x2048xf16, #hivm.address_space<gm>>) outs(%x : memref<16x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%b : memref<16x2048xf16, #hivm.address_space<gm>>) outs(%y : memref<16x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%x, %y : memref<16x2048xf16, #hivm.address_space<ub>>, memref<16x2048xf16, #hivm.address_space<ub>>) outs(%x : memref<16x2048xf16, #hivm.address_space<ub>>)
    return
  }
}
// Only one surviving marker cannot certify the two-input retry. Its single-buffer
// result is unknown and cannot satisfy the selected-set certificate consumer.
// CHECK-DAG: @missing_selected_id{{.*}}program_split.memory_result = {{.*}}kind = "unknown_or_other_failure"

// -----

module {
  // expected-error@+1 {{ub overflow}}
  func.func @invalid_selected_id(%a: memref<16x2048xf16, #hivm.address_space<gm>>, %b: memref<16x2048xf16, #hivm.address_space<gm>>) {
    %x = memref.alloc() : memref<16x2048xf16, #hivm.address_space<ub>>
    %y = memref.alloc() : memref<16x2048xf16, #hivm.address_space<ub>>
    annotation.mark %x {program_split.selected_input = 0 : i32, hivm.multi_buffer = 2 : i32} : memref<16x2048xf16, #hivm.address_space<ub>>
    annotation.mark %y {program_split.selected_input = 2 : i32, hivm.multi_buffer = 2 : i32} : memref<16x2048xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%a : memref<16x2048xf16, #hivm.address_space<gm>>) outs(%x : memref<16x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%b : memref<16x2048xf16, #hivm.address_space<gm>>) outs(%y : memref<16x2048xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%x, %y : memref<16x2048xf16, #hivm.address_space<ub>>, memref<16x2048xf16, #hivm.address_space<ub>>) outs(%x : memref<16x2048xf16, #hivm.address_space<ub>>)
    return
  }
}
// CHECK-DAG: @invalid_selected_id{{.*}}program_split.memory_result = {kind = "unknown_or_other_failure", selected_ids = array<i32: 0>, version = 2 : i32}

// -----
module {
  // expected-error@+1 {{ub overflow}}
  func.func @three_selected_inputs(%a: memref<16x1536xf16, #hivm.address_space<gm>>) {
    %x = memref.alloc() : memref<16x1536xf16, #hivm.address_space<ub>>
    %y = memref.alloc() : memref<16x1536xf16, #hivm.address_space<ub>>
    %z = memref.alloc() : memref<16x1536xf16, #hivm.address_space<ub>>
    annotation.mark %x {program_split.selected_input = 0 : i32, hivm.multi_buffer = 2 : i32} : memref<16x1536xf16, #hivm.address_space<ub>>
    annotation.mark %y {program_split.selected_input = 1 : i32, hivm.multi_buffer = 2 : i32} : memref<16x1536xf16, #hivm.address_space<ub>>
    annotation.mark %z {program_split.selected_input = 2 : i32, hivm.multi_buffer = 2 : i32} : memref<16x1536xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%a : memref<16x1536xf16, #hivm.address_space<gm>>) outs(%x : memref<16x1536xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%a : memref<16x1536xf16, #hivm.address_space<gm>>) outs(%y : memref<16x1536xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%a : memref<16x1536xf16, #hivm.address_space<gm>>) outs(%z : memref<16x1536xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%x, %y : memref<16x1536xf16, #hivm.address_space<ub>>, memref<16x1536xf16, #hivm.address_space<ub>>) outs(%x : memref<16x1536xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%x, %z : memref<16x1536xf16, #hivm.address_space<ub>>, memref<16x1536xf16, #hivm.address_space<ub>>) outs(%x : memref<16x1536xf16, #hivm.address_space<ub>>)
    return
  }
}
// CHECK-DAG: @three_selected_inputs{{.*}}kind = "proven_selected_set_capacity_excess", lower_bound_bits = 2359296 : i64, mandatory_conflict = true, selected_ids = array<i32: 0, 1, 2>
