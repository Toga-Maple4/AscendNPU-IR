// RUN: bishengir-opt %s --l1-prefetch-map-programs | FileCheck %s
// Float results and zero-result pure annotations in the program-dependent slice
// are relocation operations, not scalar addresses. Both remain supported.
// CHECK-LABEL: func.func @relocation_types
// CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<16xf32>
// CHECK: scf.for
// CHECK: %[[FLOAT:.*]] = arith.sitofp {{.*}} : i32 to f32
// CHECK: %[[VALUE:.*]] = arith.addf %[[FLOAT]], %[[FLOAT]] : f32
// CHECK: memref.subview %[[SCRATCH]]
// CHECK: scf.if
// CHECK: annotation.mark %[[VALUE]] {effects = [], test_pure_zero_result} : f32
// CHECK: test_reused_allocation
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @relocation_types(%grid_x: i32, %grid_y: i32, %grid_z: i32,
                             %src: memref<16xf32>, %dst: memref<16xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>,
                  program_split.transformed,
                  program_split.launch = {input_args = array<i32: 3>, output_arg = 4 : i32,
                    program_tile = 1 : i32, pair_stride = 28 : i32}} {
    %xy = arith.muli %grid_x, %grid_y : i32
    %count = arith.muli %xy, %grid_z : i32
    annotation.mark %count {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    %id = arith.trunci %block : i64 to i32
    %float = arith.sitofp %id : i32 to f32
    %value = arith.addf %float, %float : f32
    annotation.mark %value {effects = [], test_pure_zero_result} : f32
    %scratch = memref.alloc() : memref<16xf32>
    %index = arith.index_cast %id : i32 to index
    %view = memref.subview %scratch[%index] [1] [1] : memref<16xf32> to memref<1xf32, strided<[1], offset: ?>>
    annotation.mark %view {effects = [], test_reused_allocation} : memref<1xf32, strided<[1], offset: ?>>
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %buf = memref.alloc() : memref<16xf32>
    annotation.mark %buf {program_split.selected_input = 0 : i32} : memref<16xf32>
    scf.for %i = %zero to %two step %one : i32 {
      %active = arith.cmpi slt, %id, %count : i32
      scf.if %active {
        hivm.hir.load ins(%src : memref<16xf32>) outs(%buf : memref<16xf32>)
      }
    }
    return
  }
}
