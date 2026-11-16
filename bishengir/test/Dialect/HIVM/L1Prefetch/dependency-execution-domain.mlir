// RUN: bishengir-opt %s --split-input-file --l1-prefetch-materialize | FileCheck %s
// Physical movement regressions. No ranges are inferred from parameter roles.
// CHECK-LABEL: func.func @preserved_payload_division
// CHECK-NOT: arith.divsi
// CHECK: %[[ENTRY:.*]] = arith.cmpi slt, {{.*}}, %arg4 : i32
// CHECK-NEXT: scf.if %[[ENTRY]] {
// CHECK: %[[ACTIVE:.*]] = arith.cmpi slt
// CHECK-NEXT: scf.if %[[ACTIVE]] {
// CHECK: arith.divsi {{.*}}, %arg2 : i32
// CHECK: scf.for

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
    func.func @preserved_payload_division(%input: memref<?xf32>, %output: memref<?xf32>, %rows: i32, %columns: index, %rounds: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32, tile_rows = 16 : i32, tile_cols = 64 : i32}} {
    %physical64 = hivm.hir.get_block_idx -> i64
    %physical = arith.trunci %physical64 {program_split.physical_id} : i64 to i32
    %half = arith.constant {program_split.physical_count} 28 : i32
    scope.scope : () -> () {
      %sub64 = hivm.hir.get_sub_block_idx -> i64
      %sub = arith.trunci %sub64 : i64 to i32
      %streamOffset = arith.muli %sub, %half : i32
      %logical = arith.addi %physical, %streamOffset {program_split.logical_id} : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c16_i32 = arith.constant 16 : i32
      %c56_i32 = arith.constant 56 : i32
      scf.for %i = %c0_i32 to %rounds step %c1_i32 : i32 {
        %roundOffset = arith.muli %i, %c56_i32 : i32
        %tile = arith.addi %logical, %roundOffset : i32
        %start = arith.muli %tile, %c16_i32 : i32
        %active = arith.cmpi slt, %start, %rows : i32
        scf.if %active {
          %quotient = arith.divsi %start, %rows : i32
          %rowStart = arith.index_cast %quotient : i32 to index
          %rowLimit = arith.index_cast %rows : i32 to index
          %remaining = arith.subi %rowLimit, %rowStart : index
          %validRows = arith.minsi %remaining, %c16 : index
          %offset = arith.muli %rowStart, %columns : index
          %in = memref.reinterpret_cast %input to offset: [%offset], sizes: [16, 64], strides: [%columns, 1] : memref<?xf32> to memref<16x64xf32, strided<[?, 1], offset: ?>>
          %src = memref.subview %in[0, 0] [%validRows, %columns] [1, 1] : memref<16x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %selected = memref.alloc() : memref<16x64xf32>
          annotation.mark %selected {program_split.selected_input = 0 : i32} : memref<16x64xf32>
          hivm.hir.load ins(%src : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%selected : memref<16x64xf32>) pad_mode = <PadNull> core_type = <VECTOR>
          %tensor = bufferization.to_tensor %selected restrict writable : memref<16x64xf32>
          %squared = arith.mulf %tensor, %tensor : tensor<16x64xf32>
          %slice = tensor.extract_slice %squared[0, 0] [%validRows, %columns] [1, 1] : tensor<16x64xf32> to tensor<?x?xf32>
          %dst = memref.reinterpret_cast %output to offset: [%offset], sizes: [%validRows, %columns], strides: [%columns, 1] : memref<?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          hivm.hir.store ins(%slice : tensor<?x?xf32>) outs(%dst : memref<?x?xf32, strided<[?, 1], offset: ?>>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, no_inline, program_split.whole_program_vector}
    return
  }
}

// -----
// CHECK-LABEL: func.func @zero_trip_division
// CHECK-NOT: arith.divsi
// CHECK: %[[ENTRY:.*]] = arith.cmpi slt, {{.*}}, %arg4 : i32
// CHECK-NEXT: scf.if %[[ENTRY]] {
// CHECK: arith.divsi {{.*}}, %arg4 : i32
// CHECK: scf.for

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
    func.func @zero_trip_division(%input: memref<?xf32>, %output: memref<?xf32>, %rows: i32, %columns: index, %rounds: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32, tile_rows = 16 : i32, tile_cols = 64 : i32}} {
    %physical64 = hivm.hir.get_block_idx -> i64
    %physical = arith.trunci %physical64 {program_split.physical_id} : i64 to i32
    %half = arith.constant {program_split.physical_count} 28 : i32
    scope.scope : () -> () {
      %sub64 = hivm.hir.get_sub_block_idx -> i64
      %sub = arith.trunci %sub64 : i64 to i32
      %streamOffset = arith.muli %sub, %half : i32
      %logical = arith.addi %physical, %streamOffset {program_split.logical_id} : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c16_i32 = arith.constant 16 : i32
      %c56_i32 = arith.constant 56 : i32
      scf.for %i = %c0_i32 to %rounds step %c1_i32 : i32 {
        %roundOffset = arith.muli %i, %c56_i32 : i32
        %tile = arith.addi %logical, %roundOffset : i32
        %start = arith.muli %tile, %c16_i32 : i32
        %quotient = arith.divsi %start, %rounds : i32
        %active = arith.cmpi slt, %start, %rows : i32
        scf.if %active {
          %rowStart = arith.index_cast %quotient : i32 to index
          %rowLimit = arith.index_cast %rows : i32 to index
          %remaining = arith.subi %rowLimit, %rowStart : index
          %validRows = arith.minsi %remaining, %c16 : index
          %offset = arith.muli %rowStart, %columns : index
          %in = memref.reinterpret_cast %input to offset: [%offset], sizes: [16, 64], strides: [%columns, 1] : memref<?xf32> to memref<16x64xf32, strided<[?, 1], offset: ?>>
          %src = memref.subview %in[0, 0] [%validRows, %columns] [1, 1] : memref<16x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %selected = memref.alloc() : memref<16x64xf32>
          annotation.mark %selected {program_split.selected_input = 0 : i32} : memref<16x64xf32>
          hivm.hir.load ins(%src : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%selected : memref<16x64xf32>) pad_mode = <PadNull> core_type = <VECTOR>
          %tensor = bufferization.to_tensor %selected restrict writable : memref<16x64xf32>
          %squared = arith.mulf %tensor, %tensor : tensor<16x64xf32>
          %slice = tensor.extract_slice %squared[0, 0] [%validRows, %columns] [1, 1] : tensor<16x64xf32> to tensor<?x?xf32>
          %dst = memref.reinterpret_cast %output to offset: [%offset], sizes: [%validRows, %columns], strides: [%columns, 1] : memref<?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          hivm.hir.store ins(%slice : tensor<?x?xf32>) outs(%dst : memref<?x?xf32, strided<[?, 1], offset: ?>>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, no_inline, program_split.whole_program_vector}
    return
  }
}

// -----
// CHECK-LABEL: func.func @uncited_role_fact
// CHECK-NOT: arith.divsi
// CHECK: %[[ENTRY:.*]] = arith.cmpi slt, {{.*}}, %arg4 : i32
// CHECK-NEXT: scf.if %[[ENTRY]] {
// CHECK: arith.divsi {{.*}}, %arg4 : i32
// CHECK: scf.for

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
    func.func @uncited_role_fact(%input: memref<?xf32>, %output: memref<?xf32>, %rows: i32, %columns: index, %rounds: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32, tile_rows = 16 : i32, cols_arg = 4 : i32, tile_cols = 64 : i32}} {
    %physical64 = hivm.hir.get_block_idx -> i64
    %physical = arith.trunci %physical64 {program_split.physical_id} : i64 to i32
    %half = arith.constant {program_split.physical_count} 28 : i32
    scope.scope : () -> () {
      %sub64 = hivm.hir.get_sub_block_idx -> i64
      %sub = arith.trunci %sub64 : i64 to i32
      %streamOffset = arith.muli %sub, %half : i32
      %logical = arith.addi %physical, %streamOffset {program_split.logical_id} : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c16_i32 = arith.constant 16 : i32
      %c56_i32 = arith.constant 56 : i32
      scf.for %i = %c0_i32 to %rounds step %c1_i32 : i32 {
        %roundOffset = arith.muli %i, %c56_i32 : i32
        %tile = arith.addi %logical, %roundOffset : i32
        %start = arith.muli %tile, %c16_i32 : i32
        %quotient = arith.divsi %start, %rounds : i32
        %active = arith.cmpi slt, %start, %rows : i32
        scf.if %active {
          %rowStart = arith.index_cast %quotient : i32 to index
          %rowLimit = arith.index_cast %rows : i32 to index
          %remaining = arith.subi %rowLimit, %rowStart : index
          %validRows = arith.minsi %remaining, %c16 : index
          %offset = arith.muli %rowStart, %columns : index
          %in = memref.reinterpret_cast %input to offset: [%offset], sizes: [16, 64], strides: [%columns, 1] : memref<?xf32> to memref<16x64xf32, strided<[?, 1], offset: ?>>
          %src = memref.subview %in[0, 0] [%validRows, %columns] [1, 1] : memref<16x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %selected = memref.alloc() : memref<16x64xf32>
          annotation.mark %selected {program_split.selected_input = 0 : i32} : memref<16x64xf32>
          hivm.hir.load ins(%src : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%selected : memref<16x64xf32>) pad_mode = <PadNull> core_type = <VECTOR>
          %tensor = bufferization.to_tensor %selected restrict writable : memref<16x64xf32>
          %squared = arith.mulf %tensor, %tensor : tensor<16x64xf32>
          %slice = tensor.extract_slice %squared[0, 0] [%validRows, %columns] [1, 1] : tensor<16x64xf32> to tensor<?x?xf32>
          %dst = memref.reinterpret_cast %output to offset: [%offset], sizes: [%validRows, %columns], strides: [%columns, 1] : memref<?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          hivm.hir.store ins(%slice : tensor<?x?xf32>) outs(%dst : memref<?x?xf32, strided<[?, 1], offset: ?>>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, no_inline, program_split.whole_program_vector}
    return
  }
}

// -----
// CHECK-LABEL: func.func @unverified_grid_roles
// CHECK-NOT: arith.divsi
// CHECK: %[[ENTRY:.*]] = arith.cmpi slt, {{.*}}, %arg4 : i32
// CHECK-NEXT: scf.if %[[ENTRY]] {
// CHECK: arith.divsi {{.*}}, %arg4 : i32
// CHECK: scf.for

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
    func.func @unverified_grid_roles(%input: memref<?xf32>, %output: memref<?xf32>, %rows: i32, %columns: index, %rounds: i32) attributes {SyncBlockLockArgIdx = 0 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32, tile_rows = 16 : i32, tile_cols = 64 : i32}} {
    %physical64 = hivm.hir.get_block_idx -> i64
    %physical = arith.trunci %physical64 {program_split.physical_id} : i64 to i32
    %half = arith.constant {program_split.physical_count} 28 : i32
    scope.scope : () -> () {
      %sub64 = hivm.hir.get_sub_block_idx -> i64
      %sub = arith.trunci %sub64 : i64 to i32
      %streamOffset = arith.muli %sub, %half : i32
      %logical = arith.addi %physical, %streamOffset {program_split.logical_id} : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c16_i32 = arith.constant 16 : i32
      %c56_i32 = arith.constant 56 : i32
      scf.for %i = %c0_i32 to %rounds step %c1_i32 : i32 {
        %roundOffset = arith.muli %i, %c56_i32 : i32
        %tile = arith.addi %logical, %roundOffset : i32
        %start = arith.muli %tile, %c16_i32 : i32
        %quotient = arith.divsi %start, %rounds : i32
        %active = arith.cmpi slt, %start, %rows : i32
        scf.if %active {
          %rowStart = arith.index_cast %quotient : i32 to index
          %rowLimit = arith.index_cast %rows : i32 to index
          %remaining = arith.subi %rowLimit, %rowStart : index
          %validRows = arith.minsi %remaining, %c16 : index
          %offset = arith.muli %rowStart, %columns : index
          %in = memref.reinterpret_cast %input to offset: [%offset], sizes: [16, 64], strides: [%columns, 1] : memref<?xf32> to memref<16x64xf32, strided<[?, 1], offset: ?>>
          %src = memref.subview %in[0, 0] [%validRows, %columns] [1, 1] : memref<16x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %selected = memref.alloc() : memref<16x64xf32>
          annotation.mark %selected {program_split.selected_input = 0 : i32} : memref<16x64xf32>
          hivm.hir.load ins(%src : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%selected : memref<16x64xf32>) pad_mode = <PadNull> core_type = <VECTOR>
          %tensor = bufferization.to_tensor %selected restrict writable : memref<16x64xf32>
          %squared = arith.mulf %tensor, %tensor : tensor<16x64xf32>
          %slice = tensor.extract_slice %squared[0, 0] [%validRows, %columns] [1, 1] : tensor<16x64xf32> to tensor<?x?xf32>
          %dst = memref.reinterpret_cast %output to offset: [%offset], sizes: [%validRows, %columns], strides: [%columns, 1] : memref<?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          hivm.hir.store ins(%slice : tensor<?x?xf32>) outs(%dst : memref<?x?xf32, strided<[?, 1], offset: ?>>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, no_inline, program_split.whole_program_vector}
    return
  }
}
