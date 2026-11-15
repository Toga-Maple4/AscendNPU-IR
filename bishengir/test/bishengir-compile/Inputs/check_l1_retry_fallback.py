"""Exercise retry ordering and stale-plan rejection with the unchanged GELU input."""
import json
from pathlib import Path
import subprocess
import sys

compiler, fixture, work = sys.argv[1:]
work = Path(work)
work.mkdir(parents=True, exist_ok=True)
source = Path(fixture).read_text()
options = ["--target=Ascend950PR_9579", "--enable-lib-call-no-inline=false",
           "--enable-hfusion-compile=true", "--enable-hivm-compile=true",
           "--enable-triton-kernel-compile=true", "--enable-auto-blockify-loop=true",
           "--enable-auto-multi-buffer=true", "--enable-program-split-mix=true"]

def compile_case(name, text):
    path = work / (name + ".mlir")
    output = work / (name + ".o")
    path.write_text(text)
    result = subprocess.run([compiler, str(path), *options, "-o", str(output)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (work / (name + ".log")).write_text(result.stdout)
    return result, output, Path(str(output) + ".program_split.json")

# Normal eligibility: New-L1 gets its opportunity before disabling MB.
result, output, sidecar = compile_case("eligible", source)
assert result.returncode == 0, result.stdout
assert output.exists()
accepted = json.loads(sidecar.read_text())
assert accepted["kind"] == "split_mix_half_grid", accepted
first = result.stdout.index("trying ordinary MB before memory fallback")
second = result.stdout.index("trying New-L1 after selected-capacity proof")
assert first < second
assert "set enable-auto-multi-buffer to false" not in result.stdout

# Remove only the semantic premise. Work, dtype, tile and grid stay unchanged.
assert source.count("ascend.memory.pointer_args_noalias,") == 1
result, output, sidecar = compile_case(
    "ineligible", source.replace("ascend.memory.pointer_args_noalias,", ""))
assert result.returncode == 0, result.stdout
assert output.exists()
ordinary = json.loads(sidecar.read_text())
assert ordinary["kind"] == "ordinary", ordinary
assert ordinary["reason"] == "missing_pointer_noalias_contract", ordinary
assert ordinary["compile_options"]["enable_auto_multi_buffer"] is False, ordinary
assert "trying New-L1 after selected-capacity proof" not in result.stdout
assert "set enable-auto-multi-buffer to false" in result.stdout

# Invalid source-supplied plan is an error, never an ordinary fallback.
result, output, sidecar = compile_case(
    "spoofed", source.replace("ascend.memory.pointer_args_noalias,",
        "ascend.memory.pointer_args_noalias, program_split.plan_version = 1 : i32,"))
assert result.returncode != 0
assert "preexisting_program_split_plan" in result.stdout
assert "set enable-auto-multi-buffer to false" not in result.stdout
assert not output.exists()
assert not sidecar.exists()
print("New-L1 retry ordering, ordinary fallback and invalid-plan rejection passed")
