"""Accepted analysis survives cloning/serialization; stale proof is a pipeline error."""
from pathlib import Path
import re
import subprocess
import sys
opt, *fixtures = sys.argv[1:]
def run(text, *passes):
    return subprocess.run([opt, *passes], input=text, text=True, capture_output=True)
for fixture in fixtures:
    source = Path(fixture).read_text()
    accepted = run(source, '--l1-prefetch-ps-analysis')
    assert accepted.returncode == 0 and 'l1_prefetch_ps.accepted' in accepted.stdout, accepted.stderr + accepted.stdout
    normalized = run(accepted.stdout, '--l1-prefetch-ps-transform')
    assert normalized.returncode == 0, normalized.stderr
    assert 'program_split.launch' in normalized.stdout
    assert 'plan_version' not in normalized.stdout
    assert 'l1_prefetch_ps.proof' not in normalized.stdout
    for text in (source,
                 accepted.stdout.replace('Ascend950PR_9589', 'Ascend950PR_9579'),
                 accepted.stdout.replace('l1_prefetch_ps.accepted, ', ''),
                 re.sub(r'l1_prefetch_ps.proof = "[^"]+"', 'l1_prefetch_ps.proof = "stale"', accepted.stdout),
                 re.sub(r'^.*annotation.mark.*program_split.selected_input.*\n', '', accepted.stdout, flags=re.M),
                 accepted.stdout.replace('arith.mulf', 'arith.addf', 1)):
        failed = run(text, '--l1-prefetch-ps-transform')
        assert failed.returncode != 0 and 'L1 pipeline error:' in failed.stderr, failed.stderr
    # Fresh analysis can replace the old proof, and preserves changed arithmetic.
    changed = accepted.stdout.replace('arith.mulf', 'arith.addf', 1)
    fresh = run(changed, '--l1-prefetch-ps-analysis', '--l1-prefetch-ps-transform')
    assert fresh.returncode == 0, fresh.stderr
    if 'recovered_' in fixture:
        assert 'element_bits = 32 : i32' in normalized.stdout
        assert 'stride_arg = 4 : i32' in normalized.stdout
        # A different stride SSA value in the output view must not be approved.
        raw = run(source).stdout
        views = re.findall(r'(memref.reinterpret_cast %arg3[^\n]+)', raw)
        assert len(views) == 1
        stride = re.search(r'strides: \[(%\w+), 1\]', views[0]).group(1)
        # A well-typed mutation uses an existing index-valued offset instead.
        offset = re.search(r'offset: \[(%\w+)\]', views[0]).group(1)
        wrong = views[0].replace('strides: [' + stride, 'strides: [' + offset)
        rejection = run(raw.replace(views[0], wrong), '--l1-prefetch-ps-analysis')
        assert rejection.returncode == 0 and 'nested_unproved_store_partition' in rejection.stdout, rejection.stderr
    print('PASS', Path(fixture).name)
