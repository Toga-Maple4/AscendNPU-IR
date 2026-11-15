"""Real two-input proof, reanalysis, nonmutation, and stale-handoff coverage."""
import re
import subprocess
import sys
from pathlib import Path
opt, backward = sys.argv[1:]
backward_source = Path(backward).read_text()
def contract(text):
    return text.replace('mix_mode = "aiv"', 'ascend.memory.pointer_args_noalias, mix_mode = "aiv"')
def run(text, *passes):
    return subprocess.check_output([opt, *passes], input=text, text=True)
def two_input(text):
    return run(contract(text), "--l1-prefetch-ps-analysis")
bwd = two_input(backward_source)
assert "l1_prefetch_ps.accepted" in bwd and "l1_prefetch_ps.proof" in bwd, bwd
assert "plan_version" not in bwd
assert bwd.count("program_split.selected_input") == 2
for id in (0, 1):
    assert f"program_split.selected_input = {id} : i32" in bwd
assert run(bwd, "--l1-prefetch-ps-analysis") == bwd
unmarked = re.sub(r'^.*annotation.mark.*program_split.selected_input.*\n', '', bwd, flags=re.M)
unmarked = re.sub(r'l1_prefetch_ps\.\w+(?: = "[^"]*")?,? ?', '', unmarked)
unmarked = unmarked.replace(', }', '}').replace(' attributes {}', '').replace(' {}', '')
assert run(unmarked) == run(contract(backward_source)), "analysis changed executable IR"
transformed = run(bwd, "--l1-prefetch-ps-transform")
for field in ("input_args = array<i32: 2, 3>", "output_arg = 4", "stride_arg = 5", "element_bits = 16"):
    assert field in transformed, field
assert "plan_version" not in transformed and "l1_prefetch_ps.proof" not in transformed
for text in (contract(backward_source), bwd.replace('program_split.selected_input = 1', 'program_split.selected_input = 0'),
             bwd.replace('arith.mulf', 'arith.addf', 1), bwd.replace('Ascend950PR_9589', 'Ascend950PR_9579')):
    failed = subprocess.run([opt, "--l1-prefetch-ps-transform"], input=text, text=True, capture_output=True)
    assert failed.returncode and "L1 pipeline error:" in failed.stderr, failed.stderr
for name, text, reason in [
    ("stride_mismatch", backward_source.replace("strides: [%10, 1]", "strides: [%9, 1]", 1), "nested_unproved_store_partition"),
    ("pointer_alias", backward_source.replace("reinterpret_cast %arg3 to", "reinterpret_cast %arg2 to"), "nested_unproved_selected_input"),
    ("output_alias", backward_source.replace("reinterpret_cast %arg4 to", "reinterpret_cast %arg2 to"), "nested_unproved_store_partition"),
    ("padding", backward_source.replace("0.000000e+00 : f16", "1.000000e+00 : f16"), "nested_unproved_padding"),
    ("tile_loop_mismatch", backward_source.replace("16x2048", "17x2048").replace("[16, 2048]", "[17, 2048]"), "nested_unproved_shape_bounds"),
]:
    actual = two_input(text)
    marker = 'l1_prefetch_ps.reject_reason = "' + reason + '", '
    assert marker in actual, (name, actual)
    ordinary = run(contract(text))
    assert actual.replace(marker, "") == ordinary, name + ": rejection mutated IR"
print("PASS backward analysis and negative controls")

# Types and byte sizes are derived, not fixed to the original FP16 fixture.
# Shape changes update both tiles and loop/address constants coherently.
for dtype in ("f16", "bf16", "f32"):
    for columns in (512, 1024, 2048):
        source = backward_source.replace("f16", dtype).replace("2048", str(columns))
        if dtype == "f32":
            # The FP32 source has no widening/narrowing conversions.
            for match in list(re.finditer(r"^\s*(%\w+) = arith\.(?:extf|truncf) (%\w+) : (tensor<[^>]+>) to \3\s*$", source, re.M)):
                source = source.replace(match.group(0), "")
                source = re.sub(re.escape(match.group(1)) + r"(?![\w])", match.group(2), source)
        accepted = two_input(source)
        assert "l1_prefetch_ps.accepted" in accepted, (dtype, columns, accepted)
        transformed = run(accepted, "--l1-prefetch-ps-transform")
        assert f"element_bits = {32 if dtype == 'f32' else 16}" in transformed
        assert f"tile_cols = {columns}" in transformed
# Larger aligned tiles are semantically legal; the physical planner owns capacity.
assert "l1_prefetch_ps.accepted" in two_input(backward_source.replace("2048", "4096"))
print("PASS generic two-input dtype and tile geometry")
