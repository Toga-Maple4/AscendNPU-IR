// RUN: bishengir-opt %s --split-input-file --program-split-mix-blockify --verify-diagnostics -o /dev/null
// RUN: bishengir-opt %s --split-input-file --program-split-l1-materialize --verify-diagnostics -o /dev/null
// Malformed internal metadata is a compiler handoff error, not a new semantic
// acceptance policy. Both lowered consumers share the same runtime diagnostics.
// Role indices name the pristine ABI: no current-signature bound or hidden-arg
// offset is inferred here. The producer checks those on its original TTAdapter
// function, where existing access-derived roles already prove public pointers.
module {
  // expected-error@+1 {{L1 pipeline error: missing explicit input/output roles}}
  func.func @missing() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {logical_programs = 4 : i32, physical_programs = 2 : i32}} {
    return
  }
}

// -----
module {
  // expected-error@+1 {{L1 pipeline error: missing explicit input/output roles}}
  func.func @empty() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32>, output_arg = 3 : i32, logical_programs = 4 : i32, physical_programs = 2 : i32}} {
    return
  }
}

// -----
module {
  // expected-error@+1 {{L1 pipeline error: missing explicit input/output roles}}
  func.func @wrong_input_type() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i64: 2>, output_arg = 3 : i32, logical_programs = 4 : i32, physical_programs = 2 : i32}} {
    return
  }
}

// -----
module {
  // expected-error@+1 {{L1 pipeline error: output_arg must be a nonnegative i32 index}}
  func.func @wrong_output_type() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 2>, output_arg = 3 : i64, logical_programs = 4 : i32, physical_programs = 2 : i32}} {
    return
  }
}

// -----
module {
  // expected-error@+1 {{L1 pipeline error: input_args contains a negative index}}
  func.func @negative_input() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: -1>, output_arg = 3 : i32, logical_programs = 4 : i32, physical_programs = 2 : i32}} {
    return
  }
}

// -----
module {
  // expected-error@+1 {{L1 pipeline error: output_arg must be a nonnegative i32 index}}
  func.func @negative_output() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 2>, output_arg = -1 : i32, logical_programs = 4 : i32, physical_programs = 2 : i32}} {
    return
  }
}

// -----
module {
  // expected-error@+1 {{L1 pipeline error: input_args contains duplicate indices}}
  func.func @duplicate() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 2, 2>, output_arg = 3 : i32, logical_programs = 4 : i32, physical_programs = 2 : i32}} {
    return
  }
}

// -----
module {
  // expected-error@+1 {{L1 pipeline error: input/output pointer roles overlap}}
  func.func @overlap() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 2, 3>, output_arg = 3 : i32, logical_programs = 4 : i32, physical_programs = 2 : i32}} {
    return
  }
}
