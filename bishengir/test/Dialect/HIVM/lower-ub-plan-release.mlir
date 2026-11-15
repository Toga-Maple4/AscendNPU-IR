// RUN: bishengir-opt -hivm-lower-ub-plan-release %s | FileCheck %s

// CHECK-LABEL: func.func @dealloc_to_mark
// CHECK-NOT: memref.dealloc
// CHECK: annotation.mark {{.*}} {hivm.ub_plan_release}
func.func @dealloc_to_mark() {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<32x2048xf16, #hivm.address_space<ub>>
  memref.dealloc %alloc : memref<32x2048xf16, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @gm_dealloc_unchanged
// CHECK: memref.dealloc
func.func @gm_dealloc_unchanged() {
  %alloc = memref.alloc() : memref<32x2048xf16, #hivm.address_space<gm>>
  memref.dealloc %alloc : memref<32x2048xf16, #hivm.address_space<gm>>
  return
}
