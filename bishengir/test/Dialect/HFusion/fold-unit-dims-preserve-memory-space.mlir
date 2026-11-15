
// RUN: bishengir-opt -hfusion-fold-unit-dims -cse -canonicalize -split-input-file %s | FileCheck %s

// -----

// Verify that DropUnitDimsAllocPattern preserves memory space when folding
// a leading unit dimension from a memref with #hivm.address_space<cbuf>.

// CHECK-LABEL: fold_unit_dim_preserves_cbuf_address_space
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<256x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: memref.expand_shape %[[ALLOC]]
// CHECK-SAME: memref<256x16x16xf16, #hivm.address_space<cbuf>> into memref<1x256x16x16xf16, #hivm.address_space<cbuf>>
func.func @fold_unit_dim_preserves_cbuf_address_space() -> memref<1x256x16x16xf16, #hivm.address_space<cbuf>> {
  %alloc = memref.alloc() : memref<1x256x16x16xf16, #hivm.address_space<cbuf>>
  return %alloc : memref<1x256x16x16xf16, #hivm.address_space<cbuf>>
}

// -----

// Verify with #hivm.address_space<ub> to ensure the fix is general.

// CHECK-LABEL: fold_unit_dim_preserves_ub_address_space
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
// CHECK: memref.expand_shape %[[ALLOC]]
// CHECK-SAME: memref<64x32xf32, #hivm.address_space<ub>> into memref<1x64x32xf32, #hivm.address_space<ub>>
func.func @fold_unit_dim_preserves_ub_address_space() -> memref<1x64x32xf32, #hivm.address_space<ub>> {
  %alloc = memref.alloc() : memref<1x64x32xf32, #hivm.address_space<ub>>
  return %alloc : memref<1x64x32xf32, #hivm.address_space<ub>>
}

// -----

// Verify trailing unit dim with address space.

// CHECK-LABEL: fold_trailing_unit_dim_preserves_cbuf
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<128x16xf16, #hivm.address_space<cbuf>>
// CHECK: memref.expand_shape %[[ALLOC]]
// CHECK-SAME: #hivm.address_space<cbuf>
func.func @fold_trailing_unit_dim_preserves_cbuf() -> memref<128x16x1xf16, #hivm.address_space<cbuf>> {
  %alloc = memref.alloc() : memref<128x16x1xf16, #hivm.address_space<cbuf>>
  return %alloc : memref<128x16x1xf16, #hivm.address_space<cbuf>>
}

// -----

// L1 skeleton NZ tile: two leading unit dims folded to [16, 8].

// CHECK-LABEL: fold_double_leading_unit_dim_preserves_cbuf
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16x8xf32, #hivm.address_space<cbuf>>
// CHECK: memref.expand_shape %[[ALLOC]]
// CHECK-SAME: memref<16x8xf32, #hivm.address_space<cbuf>> into memref<1x1x16x8xf32, #hivm.address_space<cbuf>>
func.func @fold_double_leading_unit_dim_preserves_cbuf() -> memref<1x1x16x8xf32, #hivm.address_space<cbuf>> {
  %alloc = memref.alloc() : memref<1x1x16x8xf32, #hivm.address_space<cbuf>>
  return %alloc : memref<1x1x16x8xf32, #hivm.address_space<cbuf>>
}
