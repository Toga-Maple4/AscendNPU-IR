"""Exercise retry enforcement separately from semantic/lowering unit passes."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

compiler, source = sys.argv[1], Path(sys.argv[2]).read_text()
cheap = source.replace('%16 = math.exp %15', '%16 = arith.mulf %15, %15')
cheap = cheap.replace('%18 = arith.divf %1, %17', '%18 = arith.mulf %1, %17')
boundary = cheap.replace('%22 = arith.addf %18, %21', '%22 = arith.addf %18, %19')
below = boundary.replace('%22 = arith.addf %18, %19', '%22 = arith.addf %18, %14')
unknown = source.replace('%12 = arith.extf %11',
    '%half_tanh = math.tanh %11 : tensor<32768xf16>\n    %12 = arith.extf %half_tanh')
options = ['--target=Ascend950PR_9579', '--enable-lib-call-no-inline=false',
    '--enable-hfusion-compile=true', '--enable-hivm-compile=true',
    '--enable-triton-kernel-compile=true', '--enable-auto-blockify-loop=true',
    '--enable-auto-multi-buffer=true', '--enable-program-split-mix=true']
for name, text, score, kind in [('below', below, 36, 'ordinary'),
    ('equal', boundary, 40, 'split_mix_half_grid'),
    ('unknown', unknown, None, 'split_mix_half_grid'),
    ('ordinary', below.replace('32768', '1024'), 36, 'ordinary')]:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        input = root / 'input.mlir'
        input.write_text(text)
        output = root / 'kernel'
        proc = subprocess.run([compiler, str(input), *options, '-o', str(output)],
                              capture_output=True, text=True)
        prefix = '[program-split-profitability] '
        records = [json.loads(line[len(prefix):]) for line in proc.stderr.splitlines() if line.startswith(prefix)]
        assert len(records) == 1 and records[0]['compute_score'] == score, (name, proc.stderr)
        result = root / 'kernel.program_split.json'
        if kind is None:
            assert proc.returncode != 0 and not result.exists() and not (root / 'kernel.o').exists(), (name, proc.stderr)
            assert '[program-split] filtered: compute_score_below_40' in proc.stderr
        else:
            assert proc.returncode == 0 and (root / 'kernel.o').exists(), (name, proc.stderr)
            data = json.loads(result.read_text())
            assert data['kind'] == kind and data['profitability']['compute_score'] == score, data
            if name == 'below':
                assert '[program-split] filtered: compute_score_below_40' in proc.stderr
                assert data['compile_options']['enable_auto_multi_buffer'] is False
                assert 'launch' not in data and 'guard' not in data
            else:
                assert '[program-split] filtered:' not in proc.stderr
print('strict threshold, unknown retry, and ordinary success verified')
