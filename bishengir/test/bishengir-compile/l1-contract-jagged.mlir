// RUN: bishengir-compile %S/../Dialect/HIVM/L1Prefetch/l1-prefetch-ps-jagged.mlir --target=Ascend950PR_9579 --enable-lib-call-no-inline=false --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-blockify-loop=true --enable-auto-multi-buffer=true --enable-program-split-mix=true --enable-vf-fusion=false -o %t.o 2>&1 | FileCheck %s
// CHECK: "compute_score":0
// CHECK: [program-split] filtered: compute_score_below_40
// The unchanged pure-copy fixture no longer produces a split executable.
// Maple4 preserves ordinary memory fallback after the profitability filter.
// RUN: python3 %S/Inputs/check_l1_ordinary_fallback.py %t.o ordinary_compiled
