// Unmodified recovered FP32 forward IR: the stride remains a runtime argument.
// RUN: mkdir -p %t.gelu %t.silu
// RUN: bishengir-compile %S/../Dialect/HIVM/L1Prefetch/Inputs/recovered_gelu.mlir --target=Ascend950PR_9589 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=true --enable-program-split-mix=true --save-temps=%t.gelu -o %t.gelu.o
// RUN: python3 %S/Inputs/l1-contract/check.py %S/Inputs/l1-contract/recovered.json %t.gelu.o.program_split.json
// RUN: FileCheck %s --check-prefix=FINAL < %t.gelu/module.hivm.opt.mlir
// RUN: bishengir-compile %S/../Dialect/HIVM/L1Prefetch/Inputs/recovered_silu.mlir --target=Ascend950PR_9589 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=true --enable-program-split-mix=true --save-temps=%t.silu -o %t.silu.o
// RUN: python3 %S/Inputs/l1-contract/check.py %S/Inputs/l1-contract/recovered-silu.json %t.silu.o.program_split.json
// RUN: FileCheck %s --check-prefix=FINAL < %t.silu/module.hivm.opt.mlir
// FINAL: func.func @{{.*}}_mix_aic(
// FINAL-SAME: program_split.launch = {cols_arg = 6 : i32, element_bits = 32 : i32, input_args = array<i32: 2>, output_arg = 3 : i32, partial_tiles = 1 : i32, rows_arg = 5 : i32, stride_arg = 4 : i32, tile_cols = 2048 : i32, tile_rows = 16 : i32}
// FINAL: call @load_gm_to_cbuf_2d_float_program_split(
// FINAL: call @l12ub_2d_to_2d_float(
// FINAL: call @l12ub_2d_to_2d_float_sub1(
// FINAL: func.func @{{.*}}_mix_aiv(
// FINAL-NOT: call @load_gm_to_
