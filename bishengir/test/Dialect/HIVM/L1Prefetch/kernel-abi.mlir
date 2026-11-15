// RUN: bishengir-opt %s --split-input-file --l1-prefetch-ps-analysis | FileCheck %s

// The common ABI proof runs before choosing an ownership matcher. A hidden
// pointer with an incompatible type cannot be treated as a public tensor.
// CHECK-LABEL: func.func @wrong_hidden_pointer
// CHECK-SAME: l1_prefetch_ps.reject_reason = "unsupported_kernel_abi"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @wrong_hidden_pointer(%lock: memref<?xf32>, %workspace: memref<?xi8>,
      %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32, %pz: i32)
      attributes {mix_mode = "aiv", parallel_mode = "simd", ascend.memory.pointer_args_noalias} {
    return
  }
}

// -----

// CHECK-LABEL: func.func @wrong_program_id
// CHECK-SAME: l1_prefetch_ps.reject_reason = "unsupported_kernel_abi"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @wrong_program_id(%lock: memref<?xi8>, %workspace: memref<?xi8>,
      %gx: i32, %gy: i32, %gz: i32, %px: i64, %py: i32, %pz: i32)
      attributes {mix_mode = "aiv", parallel_mode = "simd", ascend.memory.pointer_args_noalias} {
    return
  }
}

// -----

// CHECK-LABEL: func.func @missing_launch_arguments
// CHECK-SAME: l1_prefetch_ps.reject_reason = "unsupported_kernel_abi"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @missing_launch_arguments(%lock: memref<?xi8>, %workspace: memref<?xi8>)
      attributes {mix_mode = "aiv", parallel_mode = "simd", ascend.memory.pointer_args_noalias} {
    return
  }
}
