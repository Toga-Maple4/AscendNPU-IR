// RUN: bishengir-opt %s --hacc-append-device-spec='target=Ascend950PR_9589' --hivm-plan-memory='mem-plan-mode=global-work-space-plan' | FileCheck %s
// RUN: bishengir-opt %s --hacc-append-device-spec='target=Ascend950PR_9589' --hivm-plan-memory-regbase='mem-plan-mode=global-work-space-plan' | FileCheck %s
// Ordinary workspace lifetimes must survive the New-L1 allocation-root helper.
// The tensor view aliases a workspace allocation, not a memref.alloc.
// CHECK-LABEL: func.func @workspace_root
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: memref_ext.alloc_workspace() from %{{.*}} offset = [%[[ZERO]]]
// CHECK: hivm.hir.load
// CHECK: hivm.hir.store
module {
  func.func @workspace_root(%workspace: memref<?xi8>, %src: tensor<16xf32>, %dst: memref<16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %buffer = memref_ext.alloc_workspace() from %workspace : from memref<?xi8> to memref<16xf32>
    %view = bufferization.to_tensor %buffer restrict writable : memref<16xf32>
    %loaded = hivm.hir.load ins(%src : tensor<16xf32>) outs(%view : tensor<16xf32>) core_type = <VECTOR> -> tensor<16xf32>
    hivm.hir.store ins(%loaded : tensor<16xf32>) outs(%dst : memref<16xf32>)
    return
  }
}
