"""Check shared per-consumer handoff, empty participation and rejection."""
from pathlib import Path
import re
import subprocess
import sys

opt, path = sys.argv[1:]
source = Path(path).read_text()

def run(text, sync=False):
    return subprocess.run([opt] + (["--l1-prefetch-consumer-handoff"] if sync else []),
                          input=text, text=True, capture_output=True)

ordinary = run(source)
checked = run(source, True)
assert ordinary.returncode == checked.returncode == 0, checked.stderr
lines = checked.stdout.splitlines()
waits = [i for i, line in enumerate(lines) if 'hivm.WAIT.INTRA.BLOCKI.mode' in line]
assert len(waits) == 4  # Two consumers in each parity branch.
for index, position in enumerate(waits):
    stream = index % 2
    group = lines[position:position + 5]
    assert f'sync_id = {2 + 16 * stream} :' in group[0], group
    assert 'scf.if' in group[1] and 'l12ub' in group[2], group
    assert ('sub_blockid = true' in group[2]) == bool(stream), group
    assert group[3].strip() == '}', group  # READY is outside the validity guard.
    assert 'hivm.SET.INTRA.BLOCKI.mode' in group[4], group
    assert f'sync_id = {16 * stream} :' in group[4], group
    assert len(group[0]) - len(group[0].lstrip()) == len(group[4]) - len(group[4].lstrip())
assert checked.stdout.count('hivm.hir.sync_block_wait') == 1  # Final paired drain.
assert 'hivm.hir.sync_block_set' not in checked.stdout
assert 'PIPE_V' not in checked.stdout  # No invented vector-arithmetic dependency.

# Functions outside the accepted automatic-L1 path are unchanged.
unmarked = source.replace('program_split.transformed, ', '')
assert run(unmarked).stdout == run(unmarked, True).stdout
for name, old, new, reason in [
    ("missing release", "hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2", "", "missing paired release/readiness events"),
    ("wrong ready pipe", "<PIPE_MTE1>, <PIPE_MTE3>", "<PIPE_MTE1>, <PIPE_MTE2>", "invalid readiness events or ordering"),
    ("wrong consumer", "sub_blockid = true", "sub_blockid = false", "deliveries must group both consumers"),
    ("invalid release ID", "flag = 2", "flag = 16", "invalid release event or ordering"),
    ("invalid ready ID", "flag = 0", "flag = 16", "invalid readiness events or ordering"),
]:
    result = run(source.replace(old, new, 1), True)
    assert result.returncode != 0 and reason in result.stderr, (name, result.stderr)
    print("PASS", name)
print("PASS per-consumer handoff, unconditional empty participation, final drain")
