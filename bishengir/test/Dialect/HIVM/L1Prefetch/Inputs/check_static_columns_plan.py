"""Real static-column source, plan-only acceptance and fail-closed mutations."""
import re
import subprocess
import sys
from pathlib import Path

opt, fixture = sys.argv[1:]
source = Path(fixture).read_text()

def run(text, analyze=False):
    return subprocess.check_output([opt] + (["--l1-prefetch-analyze"] if analyze else []), input=text, text=True)

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
again = run(actual, True)
assert again == actual, "reanalysis inherited or duplicated old selection"
print("PASS analysis-only, reanalysis, serialized handoff and transformation")

def reject(old, new, reason):
    assert old in source, old
    changed = source.replace(old, new)
    actual = run(changed, True)
    marker = f'l1_prefetch_ps.reject_reason = "{reason}", '
    assert marker in actual, actual
    assert actual.replace(marker, '') == run(changed), "rejection mutated IR"

for cols in (0, -8192, 2147483647):
    reject('constant 8192 : i32', f'constant {cols} : i32', 'nested_unproved_shape_bounds')
reject('constant 8192 : i32', 'constant 10240 : i32', 'nested_unproved_store_partition')
reject('arith.maxsi %8, %c8192', 'arith.maxsi %8, %c2048', 'nested_unproved_mask')
reject('0.000000e+00 : f16', '1.000000e+00 : f16', 'nested_unproved_padding')
reject('reinterpret_cast %arg3 to', 'reinterpret_cast %arg2 to', 'nested_unproved_store_partition')
# Same geometry, newly admitted pure arithmetic.
assert "l1_prefetch_ps.accepted" in run(source.replace("math.exp", "math.sqrt"), True)
reject('= %arg8 to', '= %arg9 to', 'nested_unsupported_loop')
print('PASS original static-column plan and negative controls')

# Matched partial-column masks are now preserved, not erased under a full-tile premise.
tail = source.replace('constant 8192 : i32', 'constant 8191 : i32').replace(
    'constant 8192 : index', 'constant 8191 : index').replace('strides: [8192, 1]', 'strides: [8191, 1]').replace('strided<[8192, 1]', 'strided<[8191, 1]')
accepted = run(tail, True)
assert 'l1_prefetch_ps.accepted' in accepted, accepted
normalized = subprocess.check_output([opt, '--l1-prefetch-normalize'], input=accepted, text=True)
assert 'partial_tiles = 1 : i32' in normalized
assert 'static_cols = 8191 : i32' in normalized
assert 'tensor.extract_slice' in normalized and 'memref.subview' in normalized
