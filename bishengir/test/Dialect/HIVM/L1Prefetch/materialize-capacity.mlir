// RUN: bishengir-opt %s --split-input-file --l1-prefetch-materialize --verify-diagnostics
// Both flat and rank-2 payloads use the same physical bank-capacity check.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
  // expected-error @+1 {{L1 pipeline error: ping/pong reservation exceeds target L1 capacity}}
  func.func @flat(%src: memref<65536xf32>, %dst: memref<65536xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32}} {
    %physical = arith.constant {program_split.physical_id} 0 : i32
    %count = arith.constant {program_split.physical_count} 1 : i32
    scope.scope : () -> () {
      %logical = arith.constant {program_split.logical_id} 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %true = arith.constant true
      scf.for %i = %c0 to %c1 step %c1 : i32 {
        scf.if %true {
          %input = memref.alloc() : memref<65536xf32>
          annotation.mark %input {program_split.selected_input = 0 : i32} : memref<65536xf32>
          hivm.hir.load ins(%src : memref<65536xf32>) outs(%input : memref<65536xf32>) core_type = <VECTOR>
          %value = bufferization.to_tensor %input restrict writable : memref<65536xf32>
          hivm.hir.store ins(%value : tensor<65536xf32>) outs(%dst : memref<65536xf32>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, program_split.whole_program_vector}
    return
  }
}

// -----
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
  // expected-error @+1 {{L1 pipeline error: ping/pong reservation exceeds target L1 capacity}}
  func.func @rank2(%src: memref<64x1024xf32>, %dst: memref<64x1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32}} {
    %physical = arith.constant {program_split.physical_id} 0 : i32
    %count = arith.constant {program_split.physical_count} 1 : i32
    scope.scope : () -> () {
      %logical = arith.constant {program_split.logical_id} 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %true = arith.constant true
      scf.for %i = %c0 to %c1 step %c1 : i32 {
        scf.if %true {
          %input = memref.alloc() : memref<64x1024xf32>
          annotation.mark %input {program_split.selected_input = 0 : i32} : memref<64x1024xf32>
          hivm.hir.load ins(%src : memref<64x1024xf32>) outs(%input : memref<64x1024xf32>) core_type = <VECTOR>
          %value = bufferization.to_tensor %input restrict writable : memref<64x1024xf32>
          hivm.hir.store ins(%value : tensor<64x1024xf32>) outs(%dst : memref<64x1024xf32>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, program_split.whole_program_vector}
    return
  }
}

// -----
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
  // expected-error @+1 {{L1 pipeline error: tile violates ND transfer descriptor bounds or row alignment}}
  func.func @unencodable_rows(%src: memref<4096x8xf32>, %dst: memref<4096x8xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32}} {
    %physical = arith.constant {program_split.physical_id} 0 : i32
    %count = arith.constant {program_split.physical_count} 1 : i32
    scope.scope : () -> () {
      %logical = arith.constant {program_split.logical_id} 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %true = arith.constant true
      scf.for %i = %c0 to %c1 step %c1 : i32 {
        scf.if %true {
          %input = memref.alloc() : memref<4096x8xf32>
          annotation.mark %input {program_split.selected_input = 0 : i32} : memref<4096x8xf32>
          hivm.hir.load ins(%src : memref<4096x8xf32>) outs(%input : memref<4096x8xf32>) core_type = <VECTOR>
          %value = bufferization.to_tensor %input restrict writable : memref<4096x8xf32>
          hivm.hir.store ins(%value : tensor<4096x8xf32>) outs(%dst : memref<4096x8xf32>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, program_split.whole_program_vector}
    return
  }
}

// -----
module {
  // expected-error @+1 {{L1 pipeline error: missing NPU target specification}}
  func.func @missing_target(%src: memref<1024xf32>, %dst: memref<1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32}} {
    %physical = arith.constant {program_split.physical_id} 0 : i32
    %count = arith.constant {program_split.physical_count} 1 : i32
    scope.scope : () -> () {
      %logical = arith.constant {program_split.logical_id} 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %true = arith.constant true
      scf.for %i = %c0 to %c1 step %c1 : i32 {
        scf.if %true {
          %input = memref.alloc() : memref<1024xf32>
          annotation.mark %input {program_split.selected_input = 0 : i32} : memref<1024xf32>
          hivm.hir.load ins(%src : memref<1024xf32>) outs(%input : memref<1024xf32>) core_type = <VECTOR>
          %value = bufferization.to_tensor %input restrict writable : memref<1024xf32>
          hivm.hir.store ins(%value : tensor<1024xf32>) outs(%dst : memref<1024xf32>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, program_split.whole_program_vector}
    return
  }
}


// -----
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
  // expected-error @+1 {{L1 pipeline error: missing or non-integer target L1 capacity/alignment}}
  func.func @missing_capacity(%src: memref<1024xf32>, %dst: memref<1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32}} {
    %physical = arith.constant {program_split.physical_id} 0 : i32
    %count = arith.constant {program_split.physical_count} 1 : i32
    scope.scope : () -> () {
      %logical = arith.constant {program_split.logical_id} 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %true = arith.constant true
      scf.for %i = %c0 to %c1 step %c1 : i32 {
        scf.if %true {
          %input = memref.alloc() : memref<1024xf32>
          annotation.mark %input {program_split.selected_input = 0 : i32} : memref<1024xf32>
          hivm.hir.load ins(%src : memref<1024xf32>) outs(%input : memref<1024xf32>) core_type = <VECTOR>
          %value = bufferization.to_tensor %input restrict writable : memref<1024xf32>
          hivm.hir.store ins(%value : tensor<1024xf32>) outs(%dst : memref<1024xf32>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, program_split.whole_program_vector}
    return
  }
}


// -----
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>>>} {
  // expected-error @+1 {{L1 pipeline error: missing or non-integer target L1 capacity/alignment}}
  func.func @missing_alignment(%src: memref<1024xf32>, %dst: memref<1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32}} {
    %physical = arith.constant {program_split.physical_id} 0 : i32
    %count = arith.constant {program_split.physical_count} 1 : i32
    scope.scope : () -> () {
      %logical = arith.constant {program_split.logical_id} 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %true = arith.constant true
      scf.for %i = %c0 to %c1 step %c1 : i32 {
        scf.if %true {
          %input = memref.alloc() : memref<1024xf32>
          annotation.mark %input {program_split.selected_input = 0 : i32} : memref<1024xf32>
          hivm.hir.load ins(%src : memref<1024xf32>) outs(%input : memref<1024xf32>) core_type = <VECTOR>
          %value = bufferization.to_tensor %input restrict writable : memref<1024xf32>
          hivm.hir.store ins(%value : tensor<1024xf32>) outs(%dst : memref<1024xf32>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, program_split.whole_program_vector}
    return
  }
}


// -----
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 0 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
  // expected-error @+1 {{L1 pipeline error: invalid target reservation capacity}}
  func.func @zero_capacity(%src: memref<1024xf32>, %dst: memref<1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32}} {
    %physical = arith.constant {program_split.physical_id} 0 : i32
    %count = arith.constant {program_split.physical_count} 1 : i32
    scope.scope : () -> () {
      %logical = arith.constant {program_split.logical_id} 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %true = arith.constant true
      scf.for %i = %c0 to %c1 step %c1 : i32 {
        scf.if %true {
          %input = memref.alloc() : memref<1024xf32>
          annotation.mark %input {program_split.selected_input = 0 : i32} : memref<1024xf32>
          hivm.hir.load ins(%src : memref<1024xf32>) outs(%input : memref<1024xf32>) core_type = <VECTOR>
          %value = bufferization.to_tensor %input restrict writable : memref<1024xf32>
          hivm.hir.store ins(%value : tensor<1024xf32>) outs(%dst : memref<1024xf32>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, program_split.whole_program_vector}
    return
  }
}


// -----
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 0 : i32>>>} {
  // expected-error @+1 {{L1 pipeline error: invalid target reservation capacity}}
  func.func @zero_alignment(%src: memref<1024xf32>, %dst: memref<1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32}} {
    %physical = arith.constant {program_split.physical_id} 0 : i32
    %count = arith.constant {program_split.physical_count} 1 : i32
    scope.scope : () -> () {
      %logical = arith.constant {program_split.logical_id} 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %true = arith.constant true
      scf.for %i = %c0 to %c1 step %c1 : i32 {
        scf.if %true {
          %input = memref.alloc() : memref<1024xf32>
          annotation.mark %input {program_split.selected_input = 0 : i32} : memref<1024xf32>
          hivm.hir.load ins(%src : memref<1024xf32>) outs(%input : memref<1024xf32>) core_type = <VECTOR>
          %value = bufferization.to_tensor %input restrict writable : memref<1024xf32>
          hivm.hir.store ins(%value : tensor<1024xf32>) outs(%dst : memref<1024xf32>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, program_split.whole_program_vector}
    return
  }
}


// -----
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", "unknown">>>} {
  // expected-error @+1 {{L1 pipeline error: missing or non-integer target L1 capacity/alignment}}
  func.func @noninteger_alignment(%src: memref<1024xf32>, %dst: memref<1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32}} {
    %physical = arith.constant {program_split.physical_id} 0 : i32
    %count = arith.constant {program_split.physical_count} 1 : i32
    scope.scope : () -> () {
      %logical = arith.constant {program_split.logical_id} 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %true = arith.constant true
      scf.for %i = %c0 to %c1 step %c1 : i32 {
        scf.if %true {
          %input = memref.alloc() : memref<1024xf32>
          annotation.mark %input {program_split.selected_input = 0 : i32} : memref<1024xf32>
          hivm.hir.load ins(%src : memref<1024xf32>) outs(%input : memref<1024xf32>) core_type = <VECTOR>
          %value = bufferization.to_tensor %input restrict writable : memref<1024xf32>
          hivm.hir.store ins(%value : tensor<1024xf32>) outs(%dst : memref<1024xf32>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, program_split.whole_program_vector}
    return
  }
}

// -----
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>>>} {
  // expected-error @+1 {{L1 pipeline error: missing explicit input/output roles}}
  func.func @missing_pointer_roles(%src: memref<65536xf32>, %dst: memref<65536xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {}} {
    %physical = arith.constant {program_split.physical_id} 0 : i32
    %count = arith.constant {program_split.physical_count} 1 : i32
    scope.scope : () -> () {
      %logical = arith.constant {program_split.logical_id} 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %true = arith.constant true
      scf.for %i = %c0 to %c1 step %c1 : i32 {
        scf.if %true {
          %input = memref.alloc() : memref<65536xf32>
          annotation.mark %input {program_split.selected_input = 0 : i32} : memref<65536xf32>
          hivm.hir.load ins(%src : memref<65536xf32>) outs(%input : memref<65536xf32>) core_type = <VECTOR>
          %value = bufferization.to_tensor %input restrict writable : memref<65536xf32>
          hivm.hir.store ins(%value : tensor<65536xf32>) outs(%dst : memref<65536xf32>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, program_split.whole_program_vector}
    return
  }
}
