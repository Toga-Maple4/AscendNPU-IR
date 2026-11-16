// RUN: bishengir-opt %s --l1-prefetch-map-programs | FileCheck %s --check-prefix=MIX
// RUN: bishengir-opt %s --l1-prefetch-map-programs --l1-prefetch-materialize | FileCheck %s --check-prefix=STAGE --implicit-check-not=test_odd_peer_division --implicit-check-not=test_odd_peer_remainder --implicit-check-not=program_split.launch_grid_axis --implicit-check-not=program_split.nonempty_tile_stream
// For valid raw IDs, grid - id is positive. The inactive odd-grid peer has
// id == grid, so its division MUST stay behind the raw valid-program guard.
// This physical unit isolates relocation; it is not a new workbook workload.
// MIX-LABEL: func.func @odd_peer
// MIX-NOT: program_split.launch_grid_axis
// MIX: scf.for
// MIX: arith.remsi {{.*}}, %arg0 : i32
// MIX-NOT: test_odd_peer_division
// MIX: %[[ACTIVE:.*]] = arith.cmpi slt, {{.*}} : i64
// MIX-NEXT: scf.if %[[ACTIVE]] {
// MIX-NEXT: annotation.mark {{.*}}test_grid_coordinate
// MIX-NEXT: {{.*}}arith.divsi {{.*}}test_odd_peer_division
// MIX-NEXT: {{.*}}arith.remsi {{.*}}test_odd_peer_remainder
// MIX: hivm.hir.load
// STAGE-LABEL: func.func @odd_peer
// STAGE: %[[ACTIVE0:.*]] = arith.cmpi slt, {{.*}} : i64
// STAGE-NEXT: scf.if %[[ACTIVE0]] {
// STAGE-NEXT: {{.*}}arith.trunci
// STAGE-NEXT: {{.*}}arith.subi
// STAGE-NEXT: {{.*}}arith.divsi {{.*}}test_odd_peer_division
// STAGE-NEXT: {{.*}}arith.remsi {{.*}}test_odd_peer_remainder
// STAGE: hivm.hir.load
// STAGE: %[[ACTIVE1:.*]] = arith.cmpi slt, {{.*}} : i64
// STAGE-NEXT: scf.if %[[ACTIVE1]] {
// STAGE-NEXT: {{.*}}arith.trunci
// STAGE-NEXT: {{.*}}arith.subi
// STAGE-NEXT: {{.*}}arith.divsi {{.*}}test_odd_peer_division
// STAGE-NEXT: {{.*}}arith.remsi {{.*}}test_odd_peer_remainder
// STAGE: hivm.hir.load
// STAGE: scf.for
// STAGE: %[[VECTORACTIVE:.*]] = arith.cmpi slt, {{.*}} : i64
// STAGE: %[[ACTIVE2:.*]] = arith.cmpi slt, {{.*}} : i64
// STAGE-NEXT: scf.if %[[ACTIVE2]] {
// STAGE-NEXT: {{.*}}arith.trunci
// STAGE-NEXT: {{.*}}arith.subi
// STAGE-NEXT: {{.*}}arith.divsi {{.*}}test_odd_peer_division
// STAGE-NEXT: {{.*}}arith.remsi {{.*}}test_odd_peer_remainder
// STAGE: hivm.hir.load
// STAGE: %[[ACTIVE3:.*]] = arith.cmpi slt, {{.*}} : i64
// STAGE-NEXT: scf.if %[[ACTIVE3]] {
// STAGE-NEXT: {{.*}}arith.trunci
// STAGE-NEXT: {{.*}}arith.subi
// STAGE-NEXT: {{.*}}arith.divsi {{.*}}test_odd_peer_division
// STAGE-NEXT: {{.*}}arith.remsi {{.*}}test_odd_peer_remainder
// STAGE: hivm.hir.load
// STAGE: hivm.hir.l12ub
// STAGE: hivm.hir.l12ub {{.*}}sub_blockid = true
// STAGE: } else {
// STAGE: %[[ACTIVE4:.*]] = arith.cmpi slt, {{.*}} : i64
// STAGE-NEXT: scf.if %[[ACTIVE4]] {
// STAGE-NEXT: {{.*}}arith.trunci
// STAGE-NEXT: {{.*}}arith.subi
// STAGE-NEXT: {{.*}}arith.divsi {{.*}}test_odd_peer_division
// STAGE-NEXT: {{.*}}arith.remsi {{.*}}test_odd_peer_remainder
// STAGE: hivm.hir.load
// STAGE: %[[ACTIVE5:.*]] = arith.cmpi slt, {{.*}} : i64
// STAGE-NEXT: scf.if %[[ACTIVE5]] {
// STAGE-NEXT: {{.*}}arith.trunci
// STAGE-NEXT: {{.*}}arith.subi
// STAGE-NEXT: {{.*}}arith.divsi {{.*}}test_odd_peer_division
// STAGE-NEXT: {{.*}}arith.remsi {{.*}}test_odd_peer_remainder
// STAGE: hivm.hir.load
// STAGE: hivm.hir.l12ub
// STAGE: hivm.hir.l12ub {{.*}}sub_blockid = true
// STAGE: scf.if %[[VECTORACTIVE]] {
// STAGE-NEXT: annotation.mark {{.*}}test_grid_coordinate
// STAGE-NEXT: {{.*}}arith.divsi {{.*}}test_odd_peer_division
// STAGE-NEXT: {{.*}}arith.remsi {{.*}}test_odd_peer_remainder
// STAGE: hivm.hir.store
// STAGE: return
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
  func.func @odd_peer(%gx: i32 {program_split.launch_grid_axis = 0 : i32}, %gy: i32 {program_split.launch_grid_axis = 1 : i32}, %gz: i32 {program_split.launch_grid_axis = 2 : i32}, %input: memref<?xf32>, %output: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, program_split.transformed, program_split.launch = {input_args = array<i32: 3>, output_arg = 4 : i32, program_tile = 1 : i32, pair_stride = 28 : i32, tile_rows = 16 : i32, tile_cols = 64 : i32}} {
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
