/**
 * ND L1 -> ND UB (2d->2d) l12ub template symbols. Separate TU so we can
 * llvm-link onto meta_op.aic.c310.bc without duplicating 4d NZ->ND symbols
 * from l12ub.cpp.
 */
#include "DMA/L12UB.h"

template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_cbuf_to_ub_2d_to_2d_core(memref_t<__cbuf__ T, 2> *src,
                              memref_t<__ubuf__ T, 2> *dst,
                              bool sub_blockid = false) {
  __cbuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  int64_t size0 = src->sizes[0];
  int64_t size1 = src->sizes[1];
  int64_t stride0_src = src->strides[0];
  int64_t stride0_dst = dst->strides[0];
  const int64_t c0 = 32 / sizeof(T);

  // The 3510 L1->UB instruction describes ND tensors as one burst per row.
  // Do not flatten contiguous rank-2 tensors into one oversized burst: that
  // encoding truncates the transfer and leaves most of the private UB stale.
  uint16_t blockCount = (uint16_t)size0;
  uint16_t blockLen = (uint16_t)CEIL_DIV(size1, c0);
  uint16_t srcGap = (uint16_t)CEIL_DIV(stride0_src - size1, c0);
  uint16_t dstGap = (uint16_t)CEIL_DIV(stride0_dst - size1, c0);
  INTRINSIC(copy_cbuf_to_ubuf, (__ubuf__ void *)dst_ptr,
            (__cbuf__ void *)src_ptr, sub_blockid, blockCount, blockLen,
            srcGap, dstGap);
}

extern "C" {
REGISTER_L12UB(float, 2, 2)
REGISTER_L12UB(bfloat16_t, 2, 2)
REGISTER_L12UB(half, 2, 2)
REGISTER_L12UB(int8_t, 2, 2)
REGISTER_L12UB(uint8_t, 2, 2)
REGISTER_L12UB(int16_t, 2, 2)
REGISTER_L12UB(uint16_t, 2, 2)
REGISTER_L12UB(int32_t, 2, 2)
REGISTER_L12UB(uint32_t, 2, 2)
REGISTER_L12UB(int64_t, 2, 2)
REGISTER_L12UB(uint64_t, 2, 2)

REGISTER_L12UB_SUB1(float, 2, 2)
REGISTER_L12UB_SUB1(bfloat16_t, 2, 2)
REGISTER_L12UB_SUB1(half, 2, 2)
REGISTER_L12UB_SUB1(int8_t, 2, 2)
REGISTER_L12UB_SUB1(uint8_t, 2, 2)
REGISTER_L12UB_SUB1(int16_t, 2, 2)
REGISTER_L12UB_SUB1(uint16_t, 2, 2)
REGISTER_L12UB_SUB1(int32_t, 2, 2)
REGISTER_L12UB_SUB1(uint32_t, 2, 2)
REGISTER_L12UB_SUB1(int64_t, 2, 2)
REGISTER_L12UB_SUB1(uint64_t, 2, 2)
}
