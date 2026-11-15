// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -hivm-plan-memory | FileCheck %s

module {
  func.func @selected_input_fits(%src: memref<16x1024xf32, #hivm.address_space<gm>>) {
    %buf = memref.alloc() : memref<16x1024xf32, #hivm.address_space<ub>>
    annotation.mark %buf {program_split.selected_input = 0 : i32, hivm.multi_buffer = 2 : i32} : memref<16x1024xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<16x1024xf32, #hivm.address_space<gm>>) outs(%buf : memref<16x1024xf32, #hivm.address_space<ub>>)
    return
  }
}

// CHECK: program_split.memory_result = {aligned_slot_bits = array<i64: 524288>, capacity_bits = 2031616 : i64, kind = "ordinary_plan_succeeded", lower_bound_bits = 1048576 : i64, mandatory_conflict = true, selected_ids = array<i32: 0>, slots = array<i32: 2>, space = "UB", version = 2 : i32}
