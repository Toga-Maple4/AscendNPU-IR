// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950PR_9589 --l1-prefetch-materialize -o %t.materialized
// RUN: FileCheck %s --check-prefix=L1 < %t.materialized
// RUN: not bishengir-opt %t.materialized --one-shot-bufferize='allow-unknown-ops=true bufferize-function-boundaries=true' --canonicalize --bind-buffer --hivm-infer-mem-scope --hivm-plan-memory -o %t.planned 2> %t.plan.log
// RUN: FileCheck %s --check-prefix=UB < %t.plan.log
// RUN: not test -e %t.planned
// RUN: not bishengir-compile %s --target=Ascend950PR_9589 --enable-hivm-compile=true --enable-program-split-mix=true -o %t.compile 2> %t.compile.log
// RUN: FileCheck %s --check-prefix=UB < %t.compile.log
// RUN: not test -e %t.compile
// RUN: not test -e %t.compile.o
// RUN: not test -e %t.compile.program_split.json
// RUN: not bishengir-compile %s --target=not-a-target -o %t.invalid.o 2> %t.invalid-compile.log
// RUN: FileCheck %s --check-prefix=INVALID < %t.invalid-compile.log
// RUN: not test -e %t.invalid.o
// RUN: not test -e %t.invalid.o.program_split.json
// RUN: not bishengir-opt %s --hacc-append-device-spec=target=not-a-target -o %t.invalid.mlir 2> %t.invalid-opt.log
// RUN: FileCheck %s --check-prefix=INVALID < %t.invalid-opt.log
// RUN: not test -e %t.invalid.mlir
//
// Physical-pass regression, not a benchmark or an accepted frontend kernel.
// This fixture starts at the materializer boundary. The selected 4 KiB tile
// needs four aligned L1 banks (16 KiB), so the exact shared reservation query
// succeeds against the ordinary Ascend950 512 KiB L1 specification. That only
// establishes L1 feasibility: unrelated scratch stays live across the selected computation. Its 248 KiB
// allocation fits UB alone, but the selected 4 KiB UB buffer makes simultaneous
// lifetimes exceed the real 248 KiB capacity. Standard bufferization, binding and
// memory-scope inference feed the authoritative UB planner; the failed pipeline
// must not emit a successful output or accepted compiler result sidecar.
// This does not claim semantic candidate acceptance or capacity-triggered retry.
//
// L1-LABEL: func.func @l1_fits_ub_fails
// L1: memref.alloc() : memref<63488xf32, #hivm.address_space<ub>>
// L1-COUNT-4: memref.alloc() : memref<1x1024xf32, #hivm.address_space<cbuf>>
// L1: hivm.hir.l12ub
// UB: error: ub overflow, requires 2064384 bits while 2031616 bits available
// INVALID: Cannot find option named 'not-a-target'
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @l1_fits_ub_fails(%src: memref<1024xf32>, %dst: memref<1024xf32>, %extra: memref<63488xf32, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, program_split.transformed, program_split.launch = {input_args = array<i32: 0>, output_arg = 1 : i32}} {
    %physical = arith.constant {program_split.physical_id} 0 : i32
    %count = arith.constant {program_split.physical_count} 1 : i32
    %scratch = memref.alloc() : memref<63488xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%extra : memref<63488xf32, #hivm.address_space<gm>>) outs(%scratch : memref<63488xf32, #hivm.address_space<ub>>)
    scope.scope : () -> () {
      %logical = arith.constant {program_split.logical_id} 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %true = arith.constant true
      scf.for %i = %c0 to %c1 step %c1 : i32 {
        scf.if %true {
          %input = memref.alloc() : memref<1024xf32>
          annotation.mark %input {program_split.selected_input = 0 : i32} : memref<1024xf32>
          hivm.hir.load ins(%src : memref<1024xf32>) outs(%input : memref<1024xf32>) core_type = <VECTOR>
          %value = bufferization.to_tensor %input restrict writable : memref<1024xf32>
          hivm.hir.store ins(%value : tensor<1024xf32>) outs(%dst : memref<1024xf32>)
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, program_split.whole_program_vector}
    hivm.hir.store ins(%scratch : memref<63488xf32, #hivm.address_space<ub>>) outs(%extra : memref<63488xf32, #hivm.address_space<gm>>)
    return
  }
}
