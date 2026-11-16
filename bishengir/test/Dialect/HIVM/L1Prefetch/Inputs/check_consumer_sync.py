"""Check two-input handoff grouping and fail-closed handling of stale schedules."""
from pathlib import Path
import re
import subprocess
import sys

opt, fixture = sys.argv[1:]
source = Path(fixture).read_text()

def run(text, *passes):
    return subprocess.run([opt, *passes], input=text, text=True, capture_output=True)

# Two inputs must both arrive before their consumer is published. The reduced
# schedule uses the same operand twice to exercise ordering, not memory layout.
two = re.sub(r'^(\s*hivm.hir.l12ub[^\n]*)$', r'\1\n\1', source, flags=re.M)
result = run(two, '--l1-prefetch-consumer-handoff')
assert result.returncode == 0, result.stderr
lines = result.stdout.splitlines()
groups = 0
for i, line in enumerate(lines):
    if 'hivm.WAIT.INTRA.BLOCKI.mode' in line and 'l12ub' in lines[i + 1]:
        groups += 1
        assert 'l12ub' in lines[i + 2], lines[i:i + 5]
        assert 'hivm.SET.INTRA.BLOCKI.mode' in lines[i + 3], lines[i:i + 5]

assert groups == 4, groups  # Two consumers in each parity branch.

# Ordinary/legacy functions are untouched.
ordinary = source.replace('program_split.transformed, ', '')
plain = run(ordinary)
unchanged = run(ordinary, '--l1-prefetch-consumer-handoff')
assert unchanged.returncode == 0 and unchanged.stdout == plain.stdout

for bad in (
    source.replace('sub_blockid = true', 'sub_blockid = false'),
    source.replace('flag = 2', 'flag = 16'),
    source.replace('<PIPE_V>] flag = 1', '<PIPE_MTE3>] flag = 1'),
    source.replace('hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2', ''),
    re.sub(r'^\s*hivm.hir.sync_block_set[^\n]*', '', source, flags=re.M),
):
    failed = run(bad, '--l1-prefetch-consumer-handoff')
    assert failed.returncode != 0 and 'L1 consumer sync:' in failed.stderr, failed.stderr
print('PASS: two-input ordering, ordinary path, and malformed schedule rejection')
