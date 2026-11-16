"""Check the private accepted-facts handoff, without executing a kernel."""
from pathlib import Path
import re
import subprocess
import sys

opt, root = sys.argv[1], Path(sys.argv[2])

def run(source, *passes, ok=True):
    result = subprocess.run([opt, *passes], input=source, text=True, capture_output=True)
    assert (result.returncode == 0) == ok, result.stderr
    if not ok:
        assert "missing, stale or inconsistent accepted analysis metadata" in result.stderr
        assert "program_split.transformed" not in result.stdout
    return result.stdout

for path in ("role-derivation.mlir", "l1-prefetch-ps-nested.mlir",
             "l1-prefetch-ps-jagged.mlir", "l1-prefetch-ps-flat.mlir",
             "program-tiles.mlir"):
    source = (root / path).read_text().split("// -----")[0]
    accepted = run(source, "--l1-prefetch-analyze")
    assert "l1_prefetch_ps.facts" in accepted, path
    assert run(accepted, "--l1-prefetch-analyze") == accepted, path
    transformed = run(accepted, "--l1-prefetch-normalize")
    assert "program_split.launch" in transformed
    assert "l1_prefetch_ps." not in transformed
    # No scalar annotation.mark uses are introduced by the record.
    assert accepted.count("annotation.mark") == accepted.count("program_split.selected_input")
    variants = [accepted.replace("version = 1 : i32", "version = 9 : i32", 1),
                accepted.replace("l1_prefetch_ps.facts", "missing_facts", 1),
                accepted.replace("output = {arg =", "output = {unknown =", 1),
                accepted.replace("schedule =", "wrong_schedule =", 1)]
    if "l1_prefetch_ps.value_roles" in accepted:
        variants.append(accepted.replace("l1_prefetch_ps.value_roles", "missing_roles", 1))
    for modified in variants:
        assert modified != accepted
        run(modified, "--l1-prefetch-normalize", ok=False)
print("PASS accepted facts: all five schedules, reanalysis, retirement, stale handoffs")
