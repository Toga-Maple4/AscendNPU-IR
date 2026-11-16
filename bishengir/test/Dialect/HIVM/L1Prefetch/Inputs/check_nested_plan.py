"""Real-source plan tests and zero-mutation negative controls (no device runs)."""
import re
import subprocess
import sys
from pathlib import Path

opt, forward, backward = sys.argv[1:]
source = Path(forward).read_text()

def run(text, analyze=False):
    args = [opt]
    if analyze:
        args += ["--l1-prefetch-analyze"]
    return subprocess.check_output(args, input=text, text=True)

def reject(name, text, reason):
    ordinary = run(text)
    actual = run(text, True)
    marker = 'l1_prefetch_ps.reject_reason = "' + reason + '", '
    assert marker in actual, (name, actual)
    assert actual.replace(marker, "") == ordinary, name + ": rejection mutated IR"
    print("PASS", name, reason)

actual = run(source, True)
assert "l1_prefetch_ps.reject_reason" not in actual, actual
assert "l1_prefetch_ps.accepted" in actual and "l1_prefetch_ps.proof" in actual
assert actual.count("program_split.selected_input") == 1
# Strip only analysis annotations, canonicalize again to eliminate SSA name and
# empty-attribute formatting differences, then compare the entire executable IR.
unmarked = re.sub(r'^.*annotation.mark.*program_split.selected_input.*\n', '', actual, flags=re.M)
# Facts contain only scalar references/literals, never an expression graph.
unmarked = re.sub(r'l1_prefetch_ps\.(?:facts|value_roles) = \{(?:[^{}]|\{[^{}]*\})*\},? ?', '', unmarked)
unmarked = re.sub(r'l1_prefetch_ps\.\w+(?: = "[^"]*")?,? ?', '', unmarked)
unmarked = unmarked.replace(', }', '}').replace(' attributes {}', '').replace(' {}', '')
assert run(unmarked) == run(source), "analysis changed executable IR"
# Serialization followed by the registered transform also exercises a fresh
# allocation of IR objects: the integrity stamp must not depend on C++ pointers.
transformed = subprocess.check_output([opt, "--l1-prefetch-normalize"], input=actual, text=True)
assert "program_split.transformed" in transformed and "program_split.launch" in transformed
assert "l1_prefetch_ps.accepted" not in transformed
assert "l1_prefetch_ps.proof" not in transformed
assert "partial_tiles = 1 : i32" in transformed
assert "hivm.hir.load" in transformed
assert "tensor.extract_slice" in transformed and "memref.subview" in transformed
again = run(actual, True)
assert again == actual, "reanalysis inherited or duplicated old selection"
print("PASS analysis-only, reanalysis, serialized handoff and transformation")

reject("backward_without_contract", Path(backward).read_text(), "missing_pointer_noalias_contract")
# Pure arithmetic is admitted through properties, not an operation-name list.
assert "l1_prefetch_ps.accepted" in run(source.replace("math.tanh", "math.sqrt"), True)
reject("argument_type", source.replace("%arg11: i32", "%arg11: i64"),
       "unsupported_kernel_abi")
reject("pointer_type", source.replace("f16", "f8E5M2"),
       "nested_unsupported_pointer_type")
reject("tile_loop_mismatch", source.replace("32x2048", "16x2048").replace("[32, 2048]", "[16, 2048]"),
       "nested_unproved_shape_bounds")
reject("loop", source.replace("= %arg9 to", "= %arg10 to"),
       "nested_unsupported_loop")
reject("shape_bound", source.replace("arith.divsi %5, %c32_i32", "arith.divsi %5, %c2048_i32"),
       "nested_unproved_shape_bounds")
reject("store_alias", source.replace("reinterpret_cast %arg3 to", "reinterpret_cast %arg2 to"),
       "nested_unproved_store_partition")
reject("mask", source.replace("arith.maxsi %8, %13", "arith.maxsi %8, %8"),
       "nested_unproved_mask")
reject("padding", source.replace("0.000000e+00 : f16", "1.000000e+00 : f16"),
       "nested_unproved_padding")
reject("extra_write", source.replace("        memref.copy", "        linalg.fill ins(%cst : f16) outs(%alloc : memref<32x2048xf16>)\n        memref.copy"),
       "nested_unproved_selected_input")
reject("preexisting_plan", source.replace('mix_mode = "aiv"', 'program_split.plan_version = 4 : i32, mix_mode = "aiv"'),
       "preexisting_program_split_plan")

# Even an elementwise use must not observe arbitrary invalid bytes through an
# integer conversion. Rejection must happen before any accepted-plan annotation.
reject("padding_integer_observer", source.replace(
    "%30 = arith.extf %29 : tensor<32x2048xf16> to tensor<32x2048xf32>",
    "%bits = arith.bitcast %29 : tensor<32x2048xf16> to tensor<32x2048xi16>\n"
    "        %30 = arith.sitofp %bits : tensor<32x2048xi16> to tensor<32x2048xf32>"),
    "nested_unproved_padding_independence")

# BF16 is a single-input capability; it retains the padding proof and bounds.
bf16 = run(source.replace("f16", "bf16"), True)
assert "l1_prefetch_ps.accepted" in bf16 and "l1_prefetch_ps.reject_reason" not in bf16
