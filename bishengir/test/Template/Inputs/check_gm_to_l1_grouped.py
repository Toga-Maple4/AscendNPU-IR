"""Execute the actual GM->L1 template with a host DMA model.

This checks addressing, chunk limits and fallback selection. Real CANN linking
and simulator execution remain required to validate generated device code.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

HEADER = r'''
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <type_traits>
struct bfloat16_t { uint16_t bits; };
#define __bf16 bfloat16_t
using half = uint16_t;
constexpr int BYTES_B64 = 8;
#define __aicore__
#define __gm__
#define __cbuf__
#define __DAV_C310__ 1
#define REGISTE_DMA_L1_LOAD(...)
#define INTRINSIC(name, ...) name(__VA_ARGS__)
constexpr int64_t L1_ALIGN_BYTES = 32;
constexpr uint16_t INTRIN_MAX_BURST_CNT = 4095;
enum class pad_t { PAD_NONE, PAD_ZERO };
static int padded_fallbacks = 0;
template<class T, int N> struct memref_t {
  T *allocated, *aligned;
  int64_t offset, sizes[N], strides[N];
};
template<int N> bool is_no_op(const int64_t *sizes) {
  for (int i=0; i<N; ++i) if (sizes[i] == 0) return true;
  return false;
}
template<class T> bool isAddress32ByteAligned(T *p) {
  return reinterpret_cast<uintptr_t>(p) % 32 == 0;
}
struct Copy { int count, len; uintptr_t src, dst; };
static std::vector<Copy> copies;
template<class T> void copy_gm_to_cbuf_align_v2(T *dst, T *src, int,
    uint16_t count, uint32_t len, int lp, int rp, bool, int,
    uint64_t src_stride, uint32_t dst_stride) {
  assert(lp == 0 && rp == 0);
  assert(count > 0 && count <= INTRIN_MAX_BURST_CNT);
  copies.push_back({count, static_cast<int>(len),
                    reinterpret_cast<uintptr_t>(src),
                    reinterpret_cast<uintptr_t>(dst)});
  for (int i=0; i<count; ++i)
    std::memcpy(reinterpret_cast<char*>(dst) + i*dst_stride,
                reinterpret_cast<char*>(src) + i*src_stride, len);
}
template<class T> void load_gm_to_cbuf_1d_core_with_contiguous_last_dim(
    memref_t<T,1> *src, memref_t<T,1> *dst, pad_t pad) {
  if (pad != pad_t::PAD_NONE) ++padded_fallbacks;
  const auto bytes = src->sizes[0]*sizeof(T);
  copy_gm_to_cbuf_align_v2(dst->aligned+dst->offset,src->aligned+src->offset,
                         0,1,bytes,0,0,true,0,bytes,bytes);
}
template<class... T> void load_gm_to_cbuf_intrin_core(T...) {}
template<class T, class... Args> void load_gm_to_cbuf_2d_core_with_contiguous_last_dim(Args...) {}
template<class T, class... Args> void load_gm_to_cbuf_3d_core_with_contiguous_last_dim(Args...) {}
'''
HARNESS = r'''
#include <cstdlib>
#include "template.cpp"
template<class T> void check(int64_t rows, int64_t cols, int64_t src_stride,
    int64_t dst_stride, int offset, std::vector<int> counts, pad_t pad = pad_t::PAD_NONE) {
  const size_t src_n = offset + rows*src_stride + 32;
  const size_t dst_n = offset + rows*dst_stride + 32;
  T *src=static_cast<T*>(std::aligned_alloc(32, (src_n*sizeof(T)+31)/32*32));
  T *dst=static_cast<T*>(std::aligned_alloc(32, (dst_n*sizeof(T)+31)/32*32));
  assert(src && dst);
  for (size_t i=0;i<src_n;++i) src[i]=static_cast<T>((i*17+i/37)%251);
  std::fill(dst,dst+dst_n,static_cast<T>(-1));
  memref_t<T,2> s={src,src,offset,{rows,cols},{src_stride,1}};
  memref_t<T,2> d={dst,dst,offset,{rows,cols},{dst_stride,1}};
  copies.clear();
  load_gm_to_cbuf_2d_program_split_core(&s,&d,pad);
  assert(copies.size()==counts.size());
  int64_t copied_rows = 0;
  for (size_t i=0;i<counts.size();++i) {
    assert(copies[i].count==counts[i]);
    assert(copies[i].len==cols*sizeof(T));
    assert(copies[i].src == reinterpret_cast<uintptr_t>(src + offset + copied_rows*src_stride));
    assert(copies[i].dst == reinterpret_cast<uintptr_t>(dst + offset + copied_rows*dst_stride));
    // The grouped path requires aligned L1 bases, including every later chunk.
    // GM bases are byte-aligned by the intrinsic contract, not necessarily 32B.
    if ((offset*sizeof(T)) % 32 == 0 && (dst_stride*sizeof(T)) % 32 == 0)
      assert(copies[i].dst % 32 == 0);
    copied_rows += copies[i].count;
  }
  assert(copied_rows == rows);
  for (int64_t row=0;row<rows;++row) {
    for (int64_t col=0;col<cols;++col)
      assert(dst[offset+row*dst_stride+col]==src[offset+row*src_stride+col]);
    for (int64_t col=cols;col<dst_stride;++col)
      assert(dst[offset+row*dst_stride+col]==static_cast<T>(-1));
  }
  for (int i=0;i<offset;++i) assert(dst[i]==static_cast<T>(-1));
  std::free(src);std::free(dst);
}
int main() {
  check<float>(16,2048,8192,2048,0,{16}); // recovered FP32
  check<uint16_t>(16,4096,16384,4096,0,{16}); // original FP16
  check<float>(3,16,48,24,8,{3}); // nonzero offset and gaps in both spaces
  check<float>(4096,8,16,8,0,{4095,1}); // burst-count chunk plus tail
  check<float>(2,5,8,8,0,{1,1}); // unaligned row length
  check<float>(2,8,13,8,0,{2}); // GM row pitch is byte-granular
  check<float>(16,2048,8195,2048,0,{16}); // recovered partial FP32, full-width tile
  check<float>(16,2048,16387,2048,0,{16}); // original partial FP32, full-width tile
  check<uint16_t>(16,4096,8195,4096,0,{16}); // partial 16-bit storage
  check<float>(4096,8,13,8,0,{4095,1}); // unaligned pitch across a burst chunk
  assert(copies[0].src % 32 == 0 && copies[1].src % 32 == 12);
  check<float>(2,3,8195,2048,0,{1,1}); // partial columns retain row fallback
  check<float>(2,8,16,8,1,{1,1}); // unaligned base address
  check<float>(2,8,(1<<19)+8,8,0,{2}); // source stride exceeds 21 bits
  check<float>(2,8,8,1<<19,0,{1,1}); // destination stride exceeds 21 bits
  check<float>(2,8,16,8,0,{1,1},pad_t::PAD_ZERO);
  assert(padded_fallbacks == 2); // padding must use the padding-aware helper
  check<float>(0,8,16,8,0,{}); // no-op stays a no-op
}
'''
with tempfile.TemporaryDirectory(prefix="gm-l1-template-") as tmp:
    root = pathlib.Path(tmp)
    (root / "DMA").mkdir()
    (root / "DMA/DMAUtils.h").write_text(HEADER)
    (root / "template.cpp").write_text(pathlib.Path(sys.argv[1]).read_text())
    (root / "test.cpp").write_text(HARNESS)
    compiler = sys.argv[2] if len(sys.argv) > 2 else (shutil.which("clang++") or shutil.which("g++"))
    if not compiler:
        raise SystemExit("C++ compiler required for GM->L1 template regression")
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(root),
                    str(root / "test.cpp"), "-o", str(root / "test")], check=True)
    subprocess.run([str(root / "test")], check=True)
print("GM->L1 grouped-copy addressing and fallback checks passed")
