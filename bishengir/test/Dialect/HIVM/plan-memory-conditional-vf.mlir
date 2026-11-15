// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory -split-input-file | FileCheck %s

// Conditional last uses may reuse storage only with the existing VF and alias proofs.

// -----

func.func @pointwise(%in: memref<64xf32, #hivm.address_space<ub>>, %out: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %x = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %y = arith.addf %x, %x : vector<64xf32>
  vector.transfer_write %y, %out[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @conditional_last_use(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: hivm.hir.pointer_cast(%[[ZERO]])
func.func @conditional_last_use(%cond: i1, %other: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  scf.if %cond {
    %output = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    func.call @pointwise(%input, %output) {hivm.vector_function} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
  }
  return
}

// -----

func.func @pointwise(%in: memref<64xf32, #hivm.address_space<ub>>, %out: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %x = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %y = arith.addf %x, %x : vector<64xf32>
  vector.transfer_write %y, %out[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @nested_conditional(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: hivm.hir.pointer_cast(%[[ZERO]])
func.func @nested_conditional(%cond: i1, %other: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  scf.if %cond {
    scf.if %other {
      %output = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      func.call @pointwise(%input, %output) {hivm.vector_function} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
    }
  }
  return
}

// -----

func.func @pointwise(%in: memref<64xf32, #hivm.address_space<ub>>, %out: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %x = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %y = arith.addf %x, %x : vector<64xf32>
  vector.transfer_write %y, %out[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @per_iteration_conditional(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: hivm.hir.pointer_cast(%[[ZERO]])
func.func @per_iteration_conditional(%cond: i1, %other: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %input = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    scf.if %cond {
      %output = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      func.call @pointwise(%input, %output) {hivm.vector_function} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
    }
  }
  return
}

// -----

func.func @pointwise(%in: memref<64xf32, #hivm.address_space<ub>>, %out: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %x = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %y = arith.addf %x, %x : vector<64xf32>
  vector.transfer_write %y, %out[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @live_after_conditional(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[OTHER:.*]] = arith.constant 256 : i64
// CHECK: hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: hivm.hir.pointer_cast(%[[OTHER]])
func.func @live_after_conditional(%cond: i1, %other: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  scf.if %cond {
    %output = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    func.call @pointwise(%input, %output) {hivm.vector_function} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
  }
  %v = memref.load %input[%c0] : memref<64xf32, #hivm.address_space<ub>>
  return
}

// -----

func.func @pointwise(%in: memref<64xf32, #hivm.address_space<ub>>, %out: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %x = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %y = arith.addf %x, %x : vector<64xf32>
  vector.transfer_write %y, %out[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @live_in_else(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[OTHER:.*]] = arith.constant 256 : i64
// CHECK: hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: hivm.hir.pointer_cast(%[[OTHER]])
func.func @live_in_else(%cond: i1, %other: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  scf.if %cond {
    %output = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    func.call @pointwise(%input, %output) {hivm.vector_function} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
  } else {
    %v = memref.load %input[%c0] : memref<64xf32, #hivm.address_space<ub>>
  }
  return
}

// -----

func.func @pointwise(%in: memref<64xf32, #hivm.address_space<ub>>, %out: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %x = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %y = arith.addf %x, %x : vector<64xf32>
  vector.transfer_write %y, %out[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @captured_across_iterations(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[OTHER:.*]] = arith.constant 256 : i64
// CHECK: hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: hivm.hir.pointer_cast(%[[OTHER]])
func.func @captured_across_iterations(%cond: i1, %other: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  scf.for %i = %c0 to %n step %c1 {
    scf.if %cond {
      %output = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      func.call @pointwise(%input, %output) {hivm.vector_function} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
    }
  }
  return
}

// -----

func.func @pointwise(%in: memref<64xf32, #hivm.address_space<ub>>, %out: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %x = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %y = arith.addf %x, %x : vector<64xf32>
  vector.transfer_write %y, %out[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @alias_live_after_conditional(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[OTHER:.*]] = arith.constant 256 : i64
// CHECK: hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: hivm.hir.pointer_cast(%[[OTHER]])
func.func @alias_live_after_conditional(%cond: i1, %other: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %alias = memref.subview %input[0] [64] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
  scf.if %cond {
    %output = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    func.call @pointwise(%input, %output) {hivm.vector_function} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
  }
  %v = memref.load %alias[%c0] : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
  return
}

// -----
// A last use alone is insufficient: the VF must prove in-place safety.

func.func @write_before_read(%in: memref<64xf32, #hivm.address_space<ub>>, %out: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %zeros = arith.constant dense<0.0> : vector<64xf32>
  vector.transfer_write %zeros, %out[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  %x = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %y = arith.addf %x, %x : vector<64xf32>
  vector.transfer_write %y, %out[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @callee_cannot_reuse(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[OTHER:.*]] = arith.constant 256 : i64
// CHECK: hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: hivm.hir.pointer_cast(%[[OTHER]])
func.func @callee_cannot_reuse(%cond: i1, %other: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  scf.if %cond {
    %output = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    func.call @write_before_read(%input, %output) {hivm.vector_function} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
  }
  return
}

// -----

func.func @pointwise(%in: memref<64xf32, #hivm.address_space<ub>>, %out: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %x = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %y = arith.addf %x, %x : vector<64xf32>
  vector.transfer_write %y, %out[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @yielded_input_stays_live(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[OTHER:.*]] = arith.constant 256 : i64
// CHECK: hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: hivm.hir.pointer_cast(%[[OTHER]])
func.func @yielded_input_stays_live(%cond: i1, %other: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %result = scf.if %cond -> (memref<64xf32, #hivm.address_space<ub>>) {
    %output = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    func.call @pointwise(%input, %output) {hivm.vector_function} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
    scf.yield %input : memref<64xf32, #hivm.address_space<ub>>
  } else {
    scf.yield %input : memref<64xf32, #hivm.address_space<ub>>
  }
  %v = memref.load %result[%c0] : memref<64xf32, #hivm.address_space<ub>>
  return
}
