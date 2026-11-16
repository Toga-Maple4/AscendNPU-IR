// RUN: bishengir-opt %s --l1-prefetch-analyze | FileCheck %s
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=true --enable-program-split-mix=true --enable-vf-fusion=true --enable-auto-bind-sub-block=true --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-vf-merge-level=1 -o %t.o 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: python3 %S/../../../bishengir-compile/Inputs/check_l1_ordinary_fallback.py %t.o program_unmodelled_access_dependency
// REJECT: [program-split] reject: program_unmodelled_access_dependency
// REJECT: automatically set enable-auto-multi-buffer to false
// Unsupported region-valued address dependency: a named analysis rejection,
// before any transformed body, selected allocation mark, or New-L1 launch contract.
// Maple4 may still produce a valid ordinary executable after memory fallback.
// CHECK-LABEL: func.func @unsupported_address_region
// CHECK-SAME: l1_prefetch_ps.reject_reason = "program_unmodelled_access_dependency"
// CHECK-NOT: program_split.launch
// CHECK-NOT: program_split.selected_input
// CHECK-NOT: hivm.hir.load
// CHECK: scf.if
// CHECK: memref.copy
// CHECK-NOT: program_split.transformed
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unsupported_address_region(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, ascend.memory.pointer_args_noalias, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.muli %arg13, %c256_i32 : i32
    %zero = arith.constant 0 : i32
    %valid = arith.cmpi ne, %arg5, %zero : i32
    %head = scf.if %valid -> (i32) {
      %q = arith.divsi %arg14, %arg5 : i32
      scf.yield %q : i32
    } else {
      scf.yield %zero : i32
    }
    %1 = arith.muli %head, %arg6 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.index_cast %0 : i32 to index
    %4 = arith.index_cast %arg7 : i32 to index
    %5 = arith.muli %3, %4 : index
    %6 = arith.addi %2, %5 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [256, 128], strides: [%4, 1] : memref<?xbf16> to memref<256x128xbf16, strided<[?, 1], offset: ?>>
    %7 = arith.muli %arg14, %arg8 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.index_cast %arg9 : i32 to index
    %10 = arith.muli %3, %9 : index
    %11 = arith.addi %8, %10 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%11], sizes: [256, 128], strides: [%9, 1] : memref<?xbf16> to memref<256x128xbf16, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<256x128xbf16>
    %12 = arith.addi %3, %c256 : index
    %13 = arith.index_cast %arg5 : i32 to index
    %14 = arith.maxsi %3, %13 : index
    %15 = arith.minsi %12, %14 : index
    %16 = arith.subi %15, %3 : index
    %17 = arith.cmpi slt, %16, %c256 : index
    scf.if %17 {
      linalg.fill ins(%cst_0 : bf16) outs(%alloc : memref<256x128xbf16>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0, 0] [%16, 128] [1, 1] : memref<256x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
    %subview_2 = memref.subview %alloc[0, 0] [%16, 128] [1, 1] : memref<256x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
    memref.copy %subview, %subview_2 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
    %18 = bufferization.to_tensor %alloc restrict writable : memref<256x128xbf16>
    %alloc_3 = memref.alloc() : memref<256x128xbf16>
    scf.if %17 {
      linalg.fill ins(%cst_0 : bf16) outs(%alloc_3 : memref<256x128xbf16>)
    } {hivm.unlikely_condition}
    %subview_4 = memref.subview %reinterpret_cast_1[0, 0] [%16, 128] [1, 1] : memref<256x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
    %subview_5 = memref.subview %alloc_3[0, 0] [%16, 128] [1, 1] : memref<256x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
    memref.copy %subview_4, %subview_5 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
    %19 = bufferization.to_tensor %alloc_3 restrict writable : memref<256x128xbf16>
    %20 = arith.mulf %18, %19 : tensor<256x128xbf16>
    %21 = arith.extf %20 : tensor<256x128xbf16> to tensor<256x128xf32>
    %22 = tensor.empty() : tensor<256xf32>
    %23 = linalg.fill ins(%cst : f32) outs(%22 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%21 : tensor<256x128xf32>) outs(%23 : tensor<256xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %28 = arith.addf %in, %init : f32
        linalg.yield %28 : f32
      }
    %24 = arith.muli %arg14, %arg5 : i32
    %25 = arith.index_cast %24 : i32 to index
    %26 = arith.addi %25, %3 : index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%26], sizes: [256], strides: [1] : memref<?xbf16> to memref<256xbf16, strided<[1], offset: ?>>
    %27 = arith.truncf %reduced : tensor<256xf32> to tensor<256xbf16>
    %extracted_slice = tensor.extract_slice %27[0] [%16] [1] : tensor<256xbf16> to tensor<?xbf16>
    %subview_7 = memref.subview %reinterpret_cast_6[0] [%16] [1] : memref<256xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_7 : (tensor<?xbf16>, memref<?xbf16, strided<[1], offset: ?>>) -> ()
    return
  }
}
