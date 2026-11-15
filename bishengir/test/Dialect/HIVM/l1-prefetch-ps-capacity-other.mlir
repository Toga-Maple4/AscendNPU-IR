// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -hivm-plan-memory -verify-diagnostics --mlir-print-ir-after-failure -o /dev/null 2>&1 | FileCheck %s

module {
  // expected-error@+1 {{ub overflow}}
  func.func @unrelated_excess(%src: memref<16x1024xf32, #hivm.address_space<gm>>, %other_src: memref<16x2048xf32, #hivm.address_space<gm>>) {
    %selected = memref.alloc() : memref<16x1024xf32, #hivm.address_space<ub>>
    annotation.mark %selected {program_split.selected_input = 0 : i32, hivm.multi_buffer = 2 : i32} : memref<16x1024xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<16x1024xf32, #hivm.address_space<gm>>) outs(%selected : memref<16x1024xf32, #hivm.address_space<ub>>)
    %other = memref.alloc() : memref<16x2048xf32, #hivm.address_space<ub>>
    annotation.mark %other {hivm.multi_buffer = 2 : i32} : memref<16x2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%other_src : memref<16x2048xf32, #hivm.address_space<gm>>) outs(%other : memref<16x2048xf32, #hivm.address_space<ub>>)
    return
  }
}

// CHECK: program_split.memory_result = {aligned_slot_bits = array<i64: 524288>, capacity_bits = 2031616 : i64, kind = "unknown_or_other_failure", lower_bound_bits = 1048576 : i64, mandatory_conflict = true, selected_ids = array<i32: 0>, slots = array<i32: 2>, space = "UB", version = 2 : i32}
