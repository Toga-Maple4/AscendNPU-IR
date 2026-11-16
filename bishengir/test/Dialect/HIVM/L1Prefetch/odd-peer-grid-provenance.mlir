// RUN: bishengir-opt %s --l1-prefetch-map-programs | FileCheck %s
// A logical-grid product and ABI-looking arguments alone are not positive-grid
// proof. Without the normalizer's role annotation, preserve the payload guard.
// CHECK-LABEL: func.func @unverified_grid
// CHECK: scf.for
// CHECK-NOT: arith.remsi {{.*}}, %arg0
// CHECK: %[[ACTIVE:.*]] = arith.cmpi slt, {{.*}} : i64
// CHECK-NEXT: scf.if %[[ACTIVE]] {
// CHECK-NEXT: {{.*}}arith.remsi {{.*}}, %arg0 : i32
// CHECK: test_odd_peer_division
// CHECK: hivm.hir.load
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
  func.func @unverified_grid(%gx: i32, %gy: i32, %gz: i32, %input: memref<?xf32>, %output: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, program_split.transformed, program_split.launch = {input_args = array<i32: 3>, output_arg = 4 : i32, program_tile = 1 : i32, pair_stride = 28 : i32, tile_rows = 16 : i32, tile_cols = 64 : i32}} {
    %xy = arith.muli %gx, %gy : i32
    %grid = arith.muli %xy, %gz : i32
    annotation.mark %grid {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    %id = arith.trunci %block : i64 to i32
    %wrapped = arith.remsi %id, %gx : i32
    annotation.mark %wrapped {effects = [], test_grid_coordinate} : i32
    %denom = arith.subi %grid, %id : i32
    %q = arith.divsi %id, %denom {test_odd_peer_division} : i32
    %r = arith.remsi %id, %denom {test_odd_peer_remainder} : i32
    %both = arith.addi %q, %r : i32
    %qi = arith.index_cast %both : i32 to index
    %in = memref.reinterpret_cast %input to offset: [%qi], sizes: [16, 64], strides: [64, 1] : memref<?xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
    %out = memref.reinterpret_cast %output to offset: [%qi], sizes: [16, 64], strides: [64, 1] : memref<?xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
    %selected = memref.alloc() : memref<16x64xf32>
    annotation.mark %selected {program_split.selected_input = 0 : i32} : memref<16x64xf32>
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    scf.for %i = %zero to %two step %one : i32 {
      %active = arith.cmpi slt, %id, %grid : i32
      scf.if %active {
        hivm.hir.load ins(%in : memref<16x64xf32, strided<[64, 1], offset: ?>>) outs(%selected : memref<16x64xf32>) pad_mode = <PadNull> core_type = <VECTOR>
        %tensor = bufferization.to_tensor %selected restrict writable : memref<16x64xf32>
        %squared = arith.mulf %tensor, %tensor : tensor<16x64xf32>
        hivm.hir.store ins(%squared : tensor<16x64xf32>) outs(%out : memref<16x64xf32, strided<[64, 1], offset: ?>>)
      }
    }
    return
  }
}
