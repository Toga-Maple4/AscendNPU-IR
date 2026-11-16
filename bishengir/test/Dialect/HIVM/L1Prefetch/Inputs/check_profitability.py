"""Profitability facts, strict boundary, unknowns and semantic noninterference."""
import json
from pathlib import Path
import re
import subprocess
import sys

opt, directory = sys.argv[1], Path(sys.argv[2])
flat = (directory / 'l1-prefetch-ps-flat.mlir').read_text()
nested = (directory / 'Inputs/recovered_silu.mlir').read_text()
prefix = '[program-split-profitability] '


def analyze(source, prediction, reason=None):
    command = [opt, '--l1-prefetch-analyze']
    p = subprocess.run(command, input=source, text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    records = [json.loads(s[len(prefix):]) for s in p.stderr.splitlines()
               if s.startswith(prefix)]
    assert len(records) == 1, p.stderr
    record = records[0]
    assert record['version'] == 2 and record['threshold'] == 40, record
    assert record['decision'] == prediction, record
    if reason:
        assert record['reason'] == reason, record
    assert 'l1_prefetch_ps.accepted' in p.stdout, p.stdout
    # Reanalysis is stable and the diagnostic adds no IR/contract attributes.
    again = subprocess.run(command, input=p.stdout, text=True, capture_output=True)
    assert again.returncode == 0 and again.stdout == p.stdout, again.stderr
    assert 'profitability' not in p.stdout
    return record


record = analyze(flat, 'passed')
assert record['compute_score'] == 72 and record['inputs'] == 2, record
renamed = analyze(flat.replace('@_silu_backward', '@unrelated_kernel'),
                  'passed')
assert {k: v for k, v in record.items() if k != 'function'} == {
    k: v for k, v in renamed.items() if k != 'function'}
# Tile size is normalized out; these are unit variants, not benchmark presets.
for size in (1024, 3072, 65536):
    assert analyze(flat.replace('32768', str(size)),
                   'passed')['compute_score'] == 72
# The explicit FP32->BF16 conversion proxy preserves the corresponding FP16
# score. It must not imply general native BF16 arithmetic support.
assert analyze(flat.replace('f16', 'bf16'),
               'passed')['compute_score'] == 72
# FP32 tanh expands to primitive work costing 13 versus exp's 4. Each unit
# contributes four score points in this two-input mixed-precision fixture.
assert analyze(flat.replace('math.exp', 'math.tanh'),
               'passed')['compute_score'] == 72 + (13 - 4) * 4
# Do not extrapolate that FP32 estimate to a native FP16 tanh operation.
native_half = flat.replace('%12 = arith.extf %11',
                           '%half_tanh = math.tanh %11 : tensor<32768xf16>\n'
                           '    %12 = arith.extf %half_tanh')
analyze(native_half, 'unknown', 'unknown_operation_cost')
dead = flat.replace('%16 = math.exp %15',
                    '%unused = math.tanh %15 : tensor<32768xf32>\n'
                    '    %16 = math.exp %15')
assert analyze(dead, 'passed')['compute_score'] == 72
# Score-only boundary: exp/div -> mul removes 24 points. Removing two
# reachable unit operations leaves exactly 40; one further unit leaves 36.
cheap = flat.replace('%16 = math.exp %15', '%16 = arith.mulf %15, %15')
cheap = cheap.replace('%18 = arith.divf %1, %17', '%18 = arith.mulf %1, %17')
assert analyze(cheap, 'passed')['compute_score'] == 48
boundary = cheap.replace('%22 = arith.addf %18, %21', '%22 = arith.addf %18, %19')
assert analyze(boundary, 'passed')['compute_score'] == 40
below = boundary.replace('%22 = arith.addf %18, %19', '%22 = arith.addf %18, %14')
assert analyze(below, 'filtered', 'compute_score_below_40')['compute_score'] == 36
# Geometry, alignment and input count no longer contribute separate decisions.
n = analyze(nested, 'passed')
assert analyze(nested.replace('tt.divisibility = 16 : i32, ', ''),
               'passed')['compute_score'] == n['compute_score']
rowwise = nested.replace('arith.index_cast %arg4 : i32 to index',
                         'arith.constant 8193 : index')
assert analyze(rowwise, 'passed')['compute_score'] == n['compute_score']
# Metadata-dependent trip counts do not hide a proved zero-arithmetic payload.
# The same rule must recognize an ordinary flat copy, not a jagged family tag.
jagged = (directory / 'l1-prefetch-ps-jagged.mlir').read_text()
for source in (jagged, jagged.replace('@triton_jagged_to_dense', '@unrelated_copy'),
               jagged.replace('bf16', 'f32')):
    result = analyze(source, 'filtered',
                     'compute_score_below_40')
    assert result['compute_score'] == 0, result
copy = re.sub(r'    %12 = arith.extf.*?(?=    %reinterpret_cast_7)', '', flat,
              flags=re.S)
copy = re.sub(r'    %24 = arith.truncf[^\n]*\n', '', copy)
copy = copy.replace('tensor.extract_slice %24[', 'tensor.extract_slice %11[')
assert analyze(copy, 'filtered',
               'compute_score_below_40')['compute_score'] == 0
scaled = copy.replace('%extracted_slice =',
                      '%scaled = arith.mulf %11, %11 : tensor<32768xf16>\n'
                      '    %extracted_slice =')
scaled = scaled.replace('tensor.extract_slice %11[', 'tensor.extract_slice %scaled[')
assert analyze(scaled, 'filtered', 'compute_score_below_40')['compute_score'] > 0
analyze(scaled.replace('f16', 'bf16'), 'unknown', 'unknown_operation_cost')
# A module without the New-L1 pass emits no predictor diagnostic.
ordinary = subprocess.run([opt], input=flat, text=True, capture_output=True)
assert ordinary.returncode == 0 and prefix not in ordinary.stderr
print('profitability policy checks passed')
