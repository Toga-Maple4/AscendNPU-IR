// Dtype-only variant of the real original-GELU TTAdapter input. Keep its FP32
// arithmetic, bounds and 128-KiB allocation; only storage changes to BF16.
// RUN: sed 's/f16/bf16/g' %S/l1-prefetch-ps-nested.mlir > %t.input.mlir
// RUN: mkdir -p %t.ir
// RUN: bishengir-compile %t.input.mlir --target=Ascend950PR_9589 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=true --enable-program-split-mix=true --save-temps=%t.ir -o %t.o
// RUN: python3 %S/Inputs/l1-contract/check.py %S/Inputs/l1-contract/bf16.json %t.o.program_split.json
// RUN: FileCheck %s --check-prefix=FINAL < %t.ir/module.hivm.opt.mlir
// With MB disabled, enabling L1 must still leave the kernel ordinary.
// RUN: bishengir-compile %t.input.mlir --target=Ascend950PR_9589 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=false --enable-program-split-mix=true -o %t.ordinary.o
// RUN: FileCheck %s --check-prefix=ORDINARY < %t.ordinary.o.program_split.json
// A smaller BF16 tile fits ordinary MB and must not acquire an L1 retry.
// RUN: sed 's/2048/1024/g' %t.input.mlir > %t.fits.mlir
// RUN: bishengir-compile %t.fits.mlir --target=Ascend950PR_9589 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=true --enable-program-split-mix=true -o %t.fits.o
// RUN: FileCheck %s --check-prefix=ORDINARY < %t.fits.o.program_split.json
// ORDINARY: "kind":"ordinary"
// FINAL-LABEL: func.func @_gelu_fwd_kernel_mix_aic(
// FINAL: call @load_gm_to_cbuf_2d_bfloat16_t_program_split(
// FINAL: call @l12ub_2d_to_2d_bfloat16_t(
// FINAL: call @l12ub_2d_to_2d_bfloat16_t_sub1(
// FINAL-LABEL: func.func @_gelu_fwd_kernel_mix_aiv_outlined_vf_0(
// FINAL-NOT: hivm.vcmp
// FINAL-NOT: hivm.vsel
// FINAL: return
