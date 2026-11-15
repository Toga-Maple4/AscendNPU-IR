// RUN: bishengir-opt %s -split-input-file -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-cross-core-gss -verify-diagnostics -o /dev/null
// RUN: bishengir-opt %s -split-input-file -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-graph-sync-solver -verify-diagnostics -o /dev/null

module {
  func.func @duplicate_read_id(%x: memref<16x16xf16, #hivm.address_space<ub>>, %o: memref<16x16xf16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    annotation.mark %x {hivm.sync_memory_read_id = 0 : i32} : memref<16x16xf16, #hivm.address_space<ub>>
    // expected-error@+1 {{plural sync-memory-read IDs require unique marked buffers}}
    hivm.hir.store ins(%x : memref<16x16xf16, #hivm.address_space<ub>>) outs(%o : memref<16x16xf16, #hivm.address_space<gm>>) {hivm.sync_memory_read_ids = array<i32: 0, 0>}
    return
  }
}

// -----

module {
  func.func @missing_read_id(%x: memref<16x16xf16, #hivm.address_space<ub>>, %o: memref<16x16xf16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    annotation.mark %x {hivm.sync_memory_read_id = 0 : i32} : memref<16x16xf16, #hivm.address_space<ub>>
    // expected-error@+1 {{plural sync-memory-read IDs require unique marked buffers}}
    hivm.hir.store ins(%x : memref<16x16xf16, #hivm.address_space<ub>>) outs(%o : memref<16x16xf16, #hivm.address_space<gm>>) {hivm.sync_memory_read_ids = array<i32: 0, 1>}
    return
  }
}

// -----

module {
  func.func @duplicate_buffer_mark(%x: memref<16x16xf16, #hivm.address_space<ub>>, %o: memref<16x16xf16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    annotation.mark %x {hivm.sync_memory_read_id = 0 : i32} : memref<16x16xf16, #hivm.address_space<ub>>
    annotation.mark %x {hivm.sync_memory_read_id = 0 : i32} : memref<16x16xf16, #hivm.address_space<ub>>
    // expected-error@+1 {{plural sync-memory-read IDs require unique marked buffers}}
    hivm.hir.store ins(%x : memref<16x16xf16, #hivm.address_space<ub>>) outs(%o : memref<16x16xf16, #hivm.address_space<gm>>) {hivm.sync_memory_read_ids = array<i32: 0>}
    return
  }
}

// -----

module {
  func.func @scalar_and_plural(%x: memref<16x16xf16, #hivm.address_space<ub>>, %o: memref<16x16xf16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    // expected-error@+1 {{invalid plural sync-memory-read metadata}}
    hivm.hir.store ins(%x : memref<16x16xf16, #hivm.address_space<ub>>) outs(%o : memref<16x16xf16, #hivm.address_space<gm>>) {hivm.sync_memory_read_id = 0 : i32, hivm.sync_memory_read_ids = array<i32: 0>}
    return
  }
}

// -----

module {
  func.func @wrong_list_type(%x: memref<16x16xf16, #hivm.address_space<ub>>, %o: memref<16x16xf16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    // expected-error@+1 {{invalid plural sync-memory-read metadata}}
    hivm.hir.store ins(%x : memref<16x16xf16, #hivm.address_space<ub>>) outs(%o : memref<16x16xf16, #hivm.address_space<gm>>) {hivm.sync_memory_read_ids = array<i64: 0, 1>}
    return
  }
}

// -----

module {
  func.func @singular_missing(%x: memref<16x16xf16, #hivm.address_space<ub>>, %o: memref<16x16xf16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    // expected-error@+1 {{singular sync-memory-read ID requires a unique marked buffer}}
    hivm.hir.store ins(%x : memref<16x16xf16, #hivm.address_space<ub>>) outs(%o : memref<16x16xf16, #hivm.address_space<gm>>) {hivm.sync_memory_read_id = 0 : i32}
    return
  }
}

// -----

module {
  func.func @singular_duplicate(%x: memref<16x16xf16, #hivm.address_space<ub>>, %o: memref<16x16xf16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    annotation.mark %x {hivm.sync_memory_read_id = 0 : i32} : memref<16x16xf16, #hivm.address_space<ub>>
    annotation.mark %x {hivm.sync_memory_read_id = 0 : i32} : memref<16x16xf16, #hivm.address_space<ub>>
    // expected-error@+1 {{singular sync-memory-read ID requires a unique marked buffer}}
    hivm.hir.store ins(%x : memref<16x16xf16, #hivm.address_space<ub>>) outs(%o : memref<16x16xf16, #hivm.address_space<gm>>) {hivm.sync_memory_read_id = 0 : i32}
    return
  }
}

// -----

module {
  func.func @singular_wrong_type(%x: memref<16x16xf16, #hivm.address_space<ub>>, %o: memref<16x16xf16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    // expected-error@+1 {{invalid singular sync-memory-read metadata}}
    hivm.hir.store ins(%x : memref<16x16xf16, #hivm.address_space<ub>>) outs(%o : memref<16x16xf16, #hivm.address_space<gm>>) {hivm.sync_memory_read_id = 0 : i64}
    return
  }
}

// -----

module {
  func.func @singular_negative(%x: memref<16x16xf16, #hivm.address_space<ub>>, %o: memref<16x16xf16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    annotation.mark %x {hivm.sync_memory_read_id = 0 : i32} : memref<16x16xf16, #hivm.address_space<ub>>
    // expected-error@+1 {{singular sync-memory-read ID requires a unique marked buffer}}
    hivm.hir.store ins(%x : memref<16x16xf16, #hivm.address_space<ub>>) outs(%o : memref<16x16xf16, #hivm.address_space<gm>>) {hivm.sync_memory_read_id = -1 : i32}
    return
  }
}
