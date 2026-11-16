"""Each count source must reach transformation; no acceptance-only positives."""
from pathlib import Path
import itertools
import subprocess
import sys
opt, fixture = sys.argv[1:]
source = Path(fixture).read_text()
# Keep original shapes and computation. Only the unit's count bindings vary.
for dynamic in itertools.product((False, True), repeat=3):
    text = source
    uses = (("%4, %n", "%4, %c56_i32"),
            ("to %steps", "to %c148_i32"),
            ("slt, %7, %tasks", "slt, %7, %c2048_i32"))
    for flag, (old, new) in zip(dynamic, uses):
        if not flag:
            text = text.replace(old, new)
    p = subprocess.run([opt, "--l1-prefetch-analyze", "--l1-prefetch-normalize"],
                       input=text, text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    assert "program_split.launch" in p.stdout
    assert ("persistent_runtime = 1" in p.stdout) == any(dynamic)
for old, new in (("%4, %n", "%4, %arg4"),  # verified grid X
                 ("to %steps", "to %tasks"),  # shared scalar roles
                 ("%4, %n", "%4, %tasks")):
    p = subprocess.run([opt, "--l1-prefetch-analyze", "--l1-prefetch-normalize"],
                       input=source.replace(old, new), text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    assert "persistent_runtime" in p.stdout
for old, new, reason in (("%4, %n", "%4, %arg7", "unsupported_logical_grid"),
                         ("to %steps", "to %c0_i32", "unsupported_persistent_loop"),
                         ("slt, %7, %tasks", "slt, %7, %c0_i32", "unproved_worker_disjointness")):
    p = subprocess.run([opt, "--l1-prefetch-analyze"],
                       input=source.replace(old, new), text=True, capture_output=True)
    assert p.returncode == 0 and reason in p.stdout, p.stderr + p.stdout
    assert "l1_prefetch_ps.facts" not in p.stdout
print("PASS independent/shared/grid counts and invalid count sources")
