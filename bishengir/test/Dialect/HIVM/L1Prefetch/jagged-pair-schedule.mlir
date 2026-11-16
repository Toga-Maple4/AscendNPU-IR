// RUN: bishengir-opt %s --l1-prefetch-map-programs | FileCheck %s --check-prefix=STREAM
// RUN: bishengir-opt %s --l1-prefetch-map-programs --l1-prefetch-materialize | FileCheck %s --check-prefix=PIPELINE
// RUN: bishengir-opt %s --l1-prefetch-map-programs --l1-prefetch-materialize | FileCheck %s --check-prefix=ROWS
// ROWS-LABEL: func.func @triton_jagged_to_dense
// ROWS-COUNT-8: memref.subview {{.*}}[0, 0] [{{.*}}, 256] [1, 1] {{.*}} to memref<?x256xbf16, strided<[256, 1]>, #hivm.address_space<{{cbuf|ub}}>>
// Real original 1D jagged IR immediately before MIX blockification.
// All resident pairs share one normalized stream, including lookahead metadata.
// STREAM: %[[PHYSICAL:.*]] = arith.trunci {{.*}}program_split.physical_id
// STREAM: %[[HALF:.*]] = arith.divsi {{.*}}program_split.physical_count
// STREAM: %[[BASE:.*]] = arith.extsi {{.*}}program_split.logical_id
// STREAM: %[[TILES:.*]] = arith.muli
// STREAM: %[[STRIDE:.*]] = arith.constant 28 : i32
// STREAM: %[[PAD:.*]] = arith.constant 27 : i32
// STREAM: %[[LEFT:.*]] = arith.subi %[[HALF]], %[[PHYSICAL]]
// STREAM: %[[ROUNDED:.*]] = arith.addi %[[LEFT]], %[[PAD]]
// STREAM: %[[PAIRS:.*]] = arith.divsi %[[ROUNDED]], %[[STRIDE]]
// STREAM: %[[STEPS:.*]] = arith.muli %[[PAIRS]], %[[TILES]]
// STREAM: scf.for %[[FLAT:.*]] = {{.*}} to %[[STEPS]] step
// STREAM-NEXT: %[[PAIR:.*]] = arith.divsi %[[FLAT]], %[[TILES]]
// STREAM-NEXT: %[[TILE:.*]] = arith.remsi %[[FLAT]], %[[TILES]]
// STREAM-NEXT: %[[OFFSET:.*]] = arith.muli %[[PAIR]], %[[STRIDE]]
// STREAM-NEXT: %[[WIDE:.*]] = arith.extsi %[[OFFSET]]
// STREAM-NEXT: %[[LOGICAL:.*]] = arith.addi %[[BASE]], %[[WIDE]]
// STREAM-NEXT: {{.*}}arith.trunci %[[LOGICAL]]
// STREAM: memref.load {{.*}}program_split.metadata
// STREAM: arith.divsi %[[TILE]],
// STREAM-NOT: scf.for
// STREAM: hivm.hir.store
// STREAM-NOT: program_split.pair_loop
// PIPELINE: memref.alloc() {{.*}}cbuf
// PIPELINE: hivm.hir.load {{.*}}core_type = <CUBE>
// PIPELINE: scf.for %[[FLAT:.*]] =
// PIPELINE: arith.remsi %[[FLAT]], {{.*}} : i32
// PIPELINE: %[[NEXT:.*]] = arith.addi %[[FLAT]],
// PIPELINE: %[[HAS_NEXT:.*]] = arith.cmpi slt, %[[NEXT]],
// PIPELINE: scf.if %[[HAS_NEXT]]
// PIPELINE: arith.divsi %[[NEXT]],
// PIPELINE: memref.load {{.*}}program_split.metadata
// PIPELINE: hivm.hir.load {{.*}}core_type = <CUBE>
// PIPELINE: hivm.hir.l12ub
// PIPELINE: hivm.hir.l12ub {{.*}}sub_blockid = true
// PIPELINE: hivm.hir.store
// PIPELINE-NOT: program_split.pair_loop
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.disable_auto_tile_and_bind_subblock, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @triton_jagged_to_dense(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, func_dyn_memref_args = dense<[true, true, true, true, false, true, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd", program_split.launch = {input_args = array<i32: 2>, element_bits = 16 : i32, input_stride_arg = 4 : i32, jagged_copy = 1 : i32, pair_stride = 28 : i32, matrix_stride_arg = 7 : i32, offsets_arg = 3 : i32, output_arg = 5 : i32, stride_arg = 6 : i32, tile_cols = 256 : i32, tile_rows = 256 : i32}, program_split.transformed} {
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg8, %arg9 : i32
    %1 = arith.muli %0, %arg10 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.remsi %3, %arg8 : i32
    %5 = arith.index_cast %4 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%5], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    %6 = memref.load %reinterpret_cast[%c0] {program_split.metadata} : memref<1xi64, strided<[1], offset: ?>>
    %7 = arith.addi %4, %c1_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%8], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    %9 = memref.load %reinterpret_cast_0[%c0] {program_split.metadata} : memref<1xi64, strided<[1], offset: ?>>
    %10 = arith.extsi %arg4 : i32 to i64
    %11 = arith.muli %6, %10 : i64
    %12 = arith.muli %4, %arg7 : i32
    %13 = arith.minsi %arg6, %arg4 : i32
    %14 = arith.divsi %arg7, %arg6 : i32
    %15 = arith.subi %9, %6 : i64
    %16 = arith.extsi %14 : i32 to i64
    %17 = arith.minsi %16, %15 : i64
    %18 = arith.addi %13, %c255_i32 : i32
    %19 = arith.divsi %18, %c256_i32 : i32
    %20 = arith.addi %14, %c255_i32 : i32
    %21 = arith.divsi %20, %c256_i32 : i32
    %22 = arith.muli %21, %19 : i32
    scf.for %arg11 = %c0_i32 to %22 step %c1_i32  : i32 {
      %23 = arith.divsi %arg11, %19 : i32
      %24 = arith.muli %23, %c256_i32 : i32
      %25 = arith.remsi %arg11, %19 : i32
      %26 = arith.muli %25, %c256_i32 : i32
      %27 = arith.extsi %24 : i32 to i64
      %28 = arith.cmpi slt, %27, %17 : i64
      scf.if %28 {
        %29 = arith.index_cast %24 : i32 to index
        %30 = arith.index_cast %26 : i32 to index
        %31 = arith.index_cast %arg6 : i32 to index
        %32 = arith.muli %29, %31 : index
        %33 = arith.addi %32, %30 : index
        %34 = arith.index_cast %11 : i64 to index
        %35 = arith.addi %34, %33 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [256, 256], strides: [%31, 1] : memref<?xbf16> to memref<256x256xbf16, strided<[?, 1], offset: ?>>
        %alloc = memref.alloc() : memref<256x256xbf16>
        annotation.mark %alloc {program_split.selected_input = 0 : i32} : memref<256x256xbf16>
        %36 = arith.addi %29, %c256 : index
        %37 = arith.index_cast %17 : i64 to index
        %38 = arith.maxsi %29, %37 : index
        %39 = arith.minsi %36, %38 : index
        %40 = arith.subi %39, %29 : index
        %41 = arith.addi %30, %c256 : index
        %42 = arith.index_cast %13 : i32 to index
        %43 = arith.maxsi %30, %42 : index
        %44 = arith.minsi %41, %43 : index
        %45 = arith.subi %44, %30 : index
        %46 = arith.minsi %40, %c256 : index
        %47 = arith.maxsi %46, %c0 : index
        %48 = arith.minsi %45, %c256 : index
        %49 = arith.maxsi %48, %c0 : index
        %subview = memref.subview %reinterpret_cast_1[0, 0] [%47, %49] [1, 1] : memref<256x256xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
        hivm.hir.load ins(%subview : memref<?x?xbf16, strided<[?, 1], offset: ?>>) outs(%alloc : memref<256x256xbf16>) pad_mode = <PadNull> core_type = <VECTOR>
        %50 = bufferization.to_tensor %alloc restrict writable : memref<256x256xbf16>
        %51 = arith.index_cast %12 : i32 to index
        %52 = arith.addi %51, %33 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%52], sizes: [256, 256], strides: [%31, 1] : memref<?xbf16> to memref<256x256xbf16, strided<[?, 1], offset: ?>>
        %extracted_slice = tensor.extract_slice %50[0, 0] [%47, %49] [1, 1] : tensor<256x256xbf16> to tensor<?x?xbf16>
        %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%47, %49] [1, 1] : memref<256x256xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
        hivm.hir.store ins(%extracted_slice : tensor<?x?xbf16>) outs(%subview_3 : memref<?x?xbf16, strided<[?, 1], offset: ?>>)
      }
    }
    hivm.hir.set_ctrl true at ctrl[60]
    return
  }
}
