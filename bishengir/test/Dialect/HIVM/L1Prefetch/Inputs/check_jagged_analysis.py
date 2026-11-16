"""Original jagged IR: semantic selection and zero-mutation rejections."""
from pathlib import Path
import re
import subprocess
import sys

opt, path = sys.argv[1:]
source = Path(path).read_text()

def run(text, analyze=False):
    return subprocess.check_output([opt] + (["--l1-prefetch-analyze"] if analyze else []), input=text, text=True)

def reject(name, old, new, reason):
    assert old in source, name
    text = source.replace(old, new)
    actual = run(text, True)
    marker = 'l1_prefetch_ps.reject_reason = "' + reason + '", '
    assert marker in actual, (name, actual)
    assert run(actual.replace(marker, "")) == run(text), name + ': executable mutation'
    print('PASS', name)

actual = run(source, True)
assert 'l1_prefetch_ps.accepted' in actual, actual
assert actual.count('program_split.selected_input') == 1
unmarked = re.sub(r'^.*annotation.mark.*program_split.selected_input.*\n', '', actual, flags=re.M)
# Facts contain only scalar references/literals, never an expression graph.
unmarked = re.sub(r'l1_prefetch_ps\.(?:facts|value_roles) = \{(?:[^{}]|\{[^{}]*\})*\},? ?', '', unmarked)
unmarked = re.sub(r'l1_prefetch_ps\.\w+(?: = "[^"]*")?,? ?', '', unmarked)
unmarked = unmarked.replace(', }', '}').replace(' attributes {}', '').replace(' {}', '')
assert run(unmarked) == run(source), 'analysis mutated executable IR'
assert run(actual, True) == actual, 'reanalysis changed selection'
print('PASS original selection, zero executable mutation, reanalysis')
reject('noalias', 'ascend.memory.pointer_args_noalias, ', '', 'missing_pointer_noalias_contract')
reject('dtype', 'bf16', 'f64', 'jagged_unsupported_selected_tile')
reject('offset index', 'arith.addi %arg11, %c1_i32', 'arith.addi %arg11, %c256_i32', 'jagged_unproved_offsets')
reject('loop start', 'to %4 step', 'to %1 step', 'jagged_unproved_loop')
reject('row recurrence', 'arith.addi %arg15, %c256', 'arith.addi %arg15, %c1', 'jagged_unproved_loop')
reject('column bound', 'arith.minsi %arg6, %arg4', 'arith.maxsi %arg6, %arg4', 'jagged_unproved_bounds')
reject('row mask', 'arith.maxsi %arg15, %21', 'arith.maxsi %arg15, %arg15', 'jagged_unproved_bounds')
reject('output ownership', 'reinterpret_cast %arg5 to', 'reinterpret_cast %arg2 to', 'jagged_unproved_address')
reject('metadata write', '    return', '    memref.store %1, %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>\n    return', 'jagged_unsupported_memory_or_control_flow')

reject('preexisting plan', 'mix_mode = "aiv"',
       'program_split.plan_version = 4 : i32, mix_mode = "aiv"',
       'preexisting_program_split_plan')

# The transform consumes a clone-stable proof and retires it before lowering.
transformed = subprocess.check_output(
    [opt, "--l1-prefetch-normalize"], input=actual, text=True)
assert 'program_split.transformed' in transformed
assert 'jagged_copy = 1 : i32' in transformed
assert transformed.count('program_split.metadata') == 2
assert transformed.count('scf.for') == 1
assert 'hivm.hir.load' in transformed
assert 'l1_prefetch_ps.' not in transformed
print('PASS serialized proof, uniform normalization and metadata selection')

for name, text in [
    ('missing proof', source),
    ('stale executable', actual.replace('arith.minsi', 'arith.maxsi', 1)),
    ('stale marker', actual.replace('l1_prefetch_ps.rows', 'l1_prefetch_ps.wrong_rows')),
]:
    result = subprocess.run([opt, "--l1-prefetch-normalize"],
                            input=text, text=True, capture_output=True)
    assert result.returncode != 0, name
    assert 'missing, stale or inconsistent accepted analysis metadata' in result.stderr, result.stderr
    print('PASS', name)

# Padding after the payload copy would overwrite valid values on partial tiles.
# Such an ordering must never inherit the accepted pure-copy semantics.
fill = re.search(r"        scf.if .*?\n        }[^\n]*", source, re.S).group(0)
copy = next(line for line in source.splitlines() if "memref.copy" in line)
reordered = source.replace(fill + "\n", "").replace(copy, copy + "\n" + fill)
result = run(reordered, True)
marker = 'l1_prefetch_ps.reject_reason = "jagged_unproved_copy_order", '
assert marker in result
assert run(result.replace(marker, "")) == run(reordered)
print('PASS padding/copy ordering')

# Physical pairing uses the generated target specification, not a host guess.
for target, cores in [('Ascend950PR_9579', 28), ('Ascend950PR_9589', 32)]:
    accepted = run(source.replace('Ascend950PR_9579', target), True)
    normalized = subprocess.check_output([opt, '--l1-prefetch-normalize'],
                                         input=accepted, text=True)
    assert f'pair_stride = {cores} : i32' in normalized
    print('PASS target pair stride', target, cores)
