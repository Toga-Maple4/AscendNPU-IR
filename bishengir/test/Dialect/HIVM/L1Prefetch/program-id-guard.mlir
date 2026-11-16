// RUN: bishengir-opt %s --split-input-file --l1-prefetch-map-programs --canonicalize --cse | FileCheck %s
// Valid raw IDs imply valid XYZ coordinates for positive, overflow-checked
// launch dimensions. Guard-only coordinate arithmetic must disappear, but the
// raw bound must remain: an odd-grid peer must not wrap to program zero.
// CHECK-LABEL: func.func @program_grid
// CHECK-SAME: %[[GX:arg[0-9]+]]: i32, %[[GY:arg[0-9]+]]: i32, %[[GZ:arg[0-9]+]]: i32
// CHECK: %[[XY:.*]] = arith.muli %[[GX]], %[[GY]]
// CHECK: %[[COUNT:.*]] = arith.muli %[[XY]], %[[GZ]]
// CHECK: scf.for
// CHECK-NOT: arith.remsi
// CHECK: %[[RAW:.*]] = arith.addi {{.*}} : i64
// CHECK: %[[WIDE:.*]] = arith.extsi %[[COUNT]] : i32 to i64
// CHECK: %[[ACTIVE:.*]] = arith.cmpi slt, %[[RAW]], %[[WIDE]] : i64
// CHECK-NEXT: scf.if %[[ACTIVE]]
// CHECK-NEXT: hivm.hir.load
// CHECK-NOT: arith.remsi
// CHECK: return
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @program_grid(%gx: i32, %gy: i32, %gz: i32,
                         %src: memref<16x128xf32>, %dst: memref<16xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>,
                  program_split.transformed,
                  program_split.launch = {input_args = array<i32: 3>, output_arg = 4 : i32, program_tile = 1 : i32, pair_stride = 28 : i32}} {
    %xy = arith.muli %gx, %gy : i32
    %count = arith.muli %xy, %gz : i32
    annotation.mark %count {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    %buf = memref.alloc() : memref<16x128xf32>
    annotation.mark %buf {program_split.selected_input = 0 : i32} : memref<16x128xf32>
    %id = arith.trunci %block : i64 to i32
    %x = arith.remsi %id, %gx : i32
    %yz = arith.divsi %id, %gx : i32
    %y = arith.remsi %yz, %gy : i32
    %zq = arith.divsi %yz, %gy : i32
    %z = arith.remsi %zq, %gz : i32
    %xok = arith.cmpi slt, %x, %gx : i32
    %yok = arith.cmpi slt, %y, %gy : i32
    %zok = arith.cmpi slt, %z, %gz : i32
    %xyok = arith.andi %xok, %yok : i1
    %active = arith.andi %xyok, %zok : i1
    scf.if %active {
      hivm.hir.load ins(%src : memref<16x128xf32>) outs(%buf : memref<16x128xf32>)
    }
    return
  }
}

// -----

// Flat workloads have an additional data-extent condition. It must survive
// alongside the raw-ID guard, including when the logical grid is oversized.
// CHECK-LABEL: func.func @flat_extent
// CHECK: %[[BOUND:.*]] = arith.cmpi slt, {{.*}} : i32
// CHECK: %[[RAWBOUND:.*]] = arith.cmpi slt, {{.*}} : i64
// CHECK: %[[BOTH:.*]] = arith.andi %[[RAWBOUND]], %[[BOUND]] : i1
// CHECK-NEXT: scf.if %[[BOTH]]
// CHECK-NEXT: hivm.hir.load
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @flat_extent(%grid: i32, %size: i32, %unused: i32,
                        %src: memref<128xf32>, %dst: memref<128xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>,
                  program_split.transformed,
                  program_split.launch = {input_args = array<i32: 3>, output_arg = 4 : i32, flat_tile = 1 : i32, pair_stride = 28 : i32}} {
    annotation.mark %grid {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    %buf = memref.alloc() : memref<128xf32>
    annotation.mark %buf {program_split.selected_input = 0 : i32} : memref<128xf32>
    %id = arith.trunci %block : i64 to i32
    %width = arith.constant 128 : i32
    %offset = arith.muli %id, %width : i32
    %active = arith.cmpi slt, %offset, %size : i32
    scf.if %active {
      hivm.hir.load ins(%src : memref<128xf32>) outs(%buf : memref<128xf32>)
    }
    return
  }
}
