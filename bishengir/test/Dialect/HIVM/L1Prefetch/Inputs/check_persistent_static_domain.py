"""Static Persistent arithmetic boundaries; no allocation or device execution."""
from pathlib import Path
import subprocess
import sys

opt, fixture = sys.argv[1:]
source = Path(fixture).read_text()
max32, max64 = (1 << 31) - 1, (1 << 63) - 1
bm, bn, byte_width = 16, 2048, 4


def check(n, steps, tasks, stride=8192, reason=None):
    columns = stride // bn
    text = source.replace("to %steps", "to %c148_i32").replace(
        "%4, %n", "%4, %c56_i32").replace(
        "slt, %7, %tasks", "slt, %7, %static_tasks")
    text = text.replace("%c8192 = arith.constant 8192 : index", f"""%c8192 = arith.constant {stride} : index
    %static_tasks = arith.constant {tasks} : i32""")
    for old, new in (("constant 56 : i32", f"constant {n} : i32"),
                     ("constant 148 : i32", f"constant {steps} : i32"),
                     ("constant 4 : i32", f"constant {columns} : i32"),
                     ("[8192, 1]", f"[{stride}, 1]")):
        text = text.replace(old, new)
    passes = ["--l1-prefetch-analyze"]
    if reason is None:
        passes.append("--l1-prefetch-normalize")
    result = subprocess.run([opt, *passes], input=text, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    if reason:
        assert reason in result.stdout, result.stdout
        assert "l1_prefetch_ps.facts" not in result.stdout
        assert "program_split.selected_input" not in result.stdout
    else:
        assert "program_split.launch" in result.stdout, result.stdout
        assert "persistent_runtime =" not in result.stdout


# Bound q + N * ((S-1)/C), even for iterations whose payload is inactive.
n, columns = 56, 4
last_round = (max32 - (n - 1)) // n
last_steps = (last_round + 1) * columns
check(n, last_steps, 1)
check(n, last_steps + 1, 1, reason="persistent_row_task_overflow")
# Bound the i32 row-start product before index_cast. T bounds live row tasks.
last = max32 // bm
check(2, 2 * (last + 1), last + 1)
check(2, 2 * (last + 1) + 1, last + 2,
      reason="persistent_row_offset_overflow")
# The i64 byte tail must fit, not merely the first element of the tile.
stride = (max32 // bn) * bn
last_row = (max64 // byte_width - (bn - 1)) // stride
safe_tasks = (last_row + 1) // bm
check(134217728, 1, safe_tasks, stride)
check(134217728, 1, safe_tasks + 1, stride,
      "persistent_byte_offset_overflow")
print("PASS fixed Persistent: three last-safe/first-overflow boundaries")
