/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DMA/DMAUtils.h"

template <typename T>
__aicore__ __attribute__((always_inline)) void
check_inputs_of_load_gm_to_cbuf_2d_core(memref_t<__gm__ T, 2> *src,
                                        memref_t<__cbuf__ T, 2> *dst,
                                        pad_t pad_mode) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto stride0_cbuf = dst->strides[0];
  auto stride1_cbuf = dst->strides[1];
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(((isSizeAlignedToBlock<T>(stride0_cbuf) || stride0_cbuf == 1) &&
          (isSizeAlignedToBlock<T>(stride1_cbuf) || stride1_cbuf == 1)) &&
         "The dst strides[0]/strides[1] must be 1 or aligned to block.");
#endif
}

template <typename T>
__aicore__ __attribute__((always_inline)) void
load_gm_to_cbuf_2d_core(memref_t<__gm__ T, 2> *src,
                        memref_t<__cbuf__ T, 2> *dst,
                        pad_t pad_mode = pad_t::PAD_NONE) {
  if (is_no_op<2>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_load_gm_to_cbuf_2d_core(src, dst, pad_mode);

  // Keep rank-2 row geometry explicit at the exported ABI boundary. The stock
  // meta-op contains colliding inline helper definitions, so deferring this
  // split to a helper can silently select a stale multi-row implementation.
  {
    const int64_t size0 = src->sizes[0];
    const int64_t size1 = src->sizes[1];
    const int64_t stride0_gm = src->strides[0];
    const int64_t stride0_cbuf = dst->strides[0];
    for (int64_t row = 0; row < size0; ++row) {
      memref_t<__gm__ T, 1> gm_1d = {src->allocated,
                                     src->aligned,
                                     src->offset + row * stride0_gm,
                                     {size1},
                                     {1}};
      memref_t<__cbuf__ T, 1> cbuf_1d = {dst->allocated,
                                         dst->aligned,
                                         dst->offset + row * stride0_cbuf,
                                         {size1},
                                         {1}};
      load_gm_to_cbuf_1d_core_with_contiguous_last_dim<T>(&gm_1d, &cbuf_1d,
                                                          pad_mode);
    }
    return;
  }

  // deal copy memref<1x1> to memref<1x1>
  if (dst->sizes[0] == 1 && dst->sizes[1] == 1) {
    auto src_ptr = src->aligned + src->offset;
    auto dst_ptr = dst->aligned + dst->offset;
    load_gm_to_cbuf_intrin_core(src_ptr, 0, dst_ptr, 0, 1, 1 * sizeof(T), 0, 0,
                                pad_mode);
    return;
  }

  int64_t stride1_gm = src->strides[1];
  int64_t stride1_cbuf = dst->strides[1];
  if (stride1_gm == 1 && stride1_cbuf == 1) [[likely]] {
    // last dimension is contiguous
    load_gm_to_cbuf_2d_core_with_contiguous_last_dim<T>(src, dst, pad_mode);
    return;
  }

  // last dimension is not contiguous,
  // view the src (size0, size1) with stride [stride0, stride1] as viewed_src
  // (size0, size1, 1) with stride [stride0, stride1, 1], where last dimension
  // of viewed_src is contiguous
  int64_t size0 = src->sizes[0];
  int64_t size1 = src->sizes[1];
  int64_t stride0_gm = src->strides[0];
  int64_t stride0_cbuf = dst->strides[0];
  memref_t<__gm__ T, 3> gm_3d = {src->allocated,
                                 src->aligned,
                                 src->offset,
                                 {size0, size1, 1},
                                 {stride0_gm, stride1_gm, 1}};
  memref_t<__cbuf__ T, 3> cbuf_3d = {dst->allocated,
                                     dst->aligned,
                                     dst->offset,
                                     {size0, size1, 1},
                                     {stride0_cbuf, stride1_cbuf, 1}};
  load_gm_to_cbuf_3d_core_with_contiguous_last_dim<T>(&gm_3d, &cbuf_3d,
                                                      pad_mode);
}

extern "C" {
//===-------------------------------------------------------------------===//
// Load gm to cbuf, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_L1_LOAD(2, int8_t);
REGISTE_DMA_L1_LOAD(2, uint8_t);
REGISTE_DMA_L1_LOAD(2, int16_t);
REGISTE_DMA_L1_LOAD(2, uint16_t);
REGISTE_DMA_L1_LOAD(2, int32_t);
REGISTE_DMA_L1_LOAD(2, uint32_t);
REGISTE_DMA_L1_LOAD(2, int64_t);
REGISTE_DMA_L1_LOAD(2, uint64_t);
REGISTE_DMA_L1_LOAD(2, half);
REGISTE_DMA_L1_LOAD(2, float);
REGISTE_DMA_L1_LOAD(2, bfloat16_t);
}

// New-L1-only entry points. Keep every legacy helper and export above unchanged.
// Internal linkage prevents stock weak template bodies from replacing this code.
template <typename T>
static __aicore__ __attribute__((always_inline)) void
load_gm_to_cbuf_2d_program_split_core(memref_t<__gm__ T, 2> *src,
                        memref_t<__cbuf__ T, 2> *dst,
                        pad_t pad_mode = pad_t::PAD_NONE) {
  if (is_no_op<2>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_load_gm_to_cbuf_2d_core(src, dst, pad_mode);

#if defined(__DAV_C310__)
  // Keep the intrinsic at this exported ABI boundary: linking another inline
  // DMA helper can select a colliding implementation from the stock bitcode.
  // A5 align-v2 uses byte lengths and full row strides. Keep the selected
  // region unchanged: gaps in GM are skipped and L1 rows retain their layout.
  const int64_t row_bytes = src->sizes[1] * sizeof(T);
  const int64_t src_row_bytes = src->strides[0] * sizeof(T);
  const int64_t dst_row_bytes = dst->strides[0] * sizeof(T);
  constexpr int64_t max_length_or_dst_stride = (1LL << 21) - 1;
  constexpr int64_t max_src_stride = (1LL << 40) - 1;
  if (pad_mode == pad_t::PAD_NONE &&
      src->strides[1] == 1 && dst->strides[1] == 1 &&
      row_bytes > 0 && row_bytes % L1_ALIGN_BYTES == 0 &&
      src_row_bytes >= row_bytes && dst_row_bytes >= row_bytes &&
      src_row_bytes % L1_ALIGN_BYTES == 0 &&
      dst_row_bytes % L1_ALIGN_BYTES == 0 &&
      row_bytes <= max_length_or_dst_stride &&
      src_row_bytes <= max_src_stride &&
      dst_row_bytes <= max_length_or_dst_stride &&
      reinterpret_cast<uint64_t>(src->aligned + src->offset) % L1_ALIGN_BYTES == 0 &&
      reinterpret_cast<uint64_t>(dst->aligned + dst->offset) % L1_ALIGN_BYTES == 0) {
    for (int64_t row = 0; row < src->sizes[0]; row += INTRIN_MAX_BURST_CNT) {
      const int64_t remaining = src->sizes[0] - row;
      const uint16_t count = remaining < INTRIN_MAX_BURST_CNT
                                 ? remaining : INTRIN_MAX_BURST_CNT;
      INTRINSIC(copy_gm_to_cbuf_align_v2,
                dst->aligned + dst->offset + row * dst->strides[0],
                src->aligned + src->offset + row * src->strides[0],
                0, count, static_cast<uint32_t>(row_bytes), 0, 0, true, 0,
                static_cast<uint64_t>(src_row_bytes),
                static_cast<uint32_t>(dst_row_bytes));
    }
    return;
  }
#endif

  // The legacy entry point keeps its established row-wise behavior.
  load_gm_to_cbuf_2d_core<T>(src, dst, pad_mode);
}

extern "C" {
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_load_gm_to_cbuf_2d_half_program_split(
    memref_t<__gm__ half, 2> *src, memref_t<__cbuf__ half, 2> *dst,
    pad_t pad_mode) {
  load_gm_to_cbuf_2d_program_split_core(src, dst, pad_mode);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_load_gm_to_cbuf_2d_bfloat16_t_program_split(
    memref_t<__gm__ bfloat16_t, 2> *src,
    memref_t<__cbuf__ bfloat16_t, 2> *dst, pad_t pad_mode) {
  load_gm_to_cbuf_2d_program_split_core(src, dst, pad_mode);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_load_gm_to_cbuf_2d_float_program_split(
    memref_t<__gm__ float, 2> *src, memref_t<__cbuf__ float, 2> *dst,
    pad_t pad_mode) {
  load_gm_to_cbuf_2d_program_split_core(src, dst, pad_mode);
}
}
