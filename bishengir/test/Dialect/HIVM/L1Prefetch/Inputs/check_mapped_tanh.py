"""Maple4 tanh representation compatibility and proof-integrity checks."""
from pathlib import Path
import re
import subprocess
import sys

opt, fixture = sys.argv[1:]
native = Path(fixture).read_text()
decl = 'func.func private @__hmf_tanhf(f32) -> f32 attributes {llvm.readnone}'
mapped = native.replace('  func.func @_gelu', '  ' + decl + '\n  func.func @_gelu', 1)
mapped = mapped.replace('math.tanh %35 : tensor<32x2048xf32>',
    'linalg.map { func.call {callee = @__hmf_tanhf} } '
    'ins(%35 : tensor<32x2048xf32>) outs(%35 : tensor<32x2048xf32>)')

def run(source, *passes, succeeds=True):
    p = subprocess.run([opt, *passes], input=source, text=True, capture_output=True)
    assert (p.returncode == 0) == succeeds, p.stderr
    return p.stdout if succeeds else p.stderr

analysis = '--l1-prefetch-analyze'
transform = '--l1-prefetch-normalize'
a = run(mapped, analysis)
assert a.count('l1_prefetch_ps.accepted') == 1 and 'reject_reason' not in a
assert 'attributes {llvm.readnone}' in a  # declaration receives no kernel proof
profit = subprocess.run([opt, analysis], input=mapped, text=True, capture_output=True)
assert profit.returncode == 0
assert profit.stderr.count('[program-split-profitability]') == 1
assert 'unknown_operation_cost' in profit.stderr
assert '"compute_score":null' in profit.stderr.replace(' ', '')
assert 'l1_prefetch_ps.facts' in a
print('PASS mapped tanh profitability remains unknown; only kernel is reported')
assert run(a, analysis) == a
transformed = run(a, transform)
assert 'program_split.transformed' in transformed
assert transformed.count('@__hmf_tanhf') == 2
assert 'partial_tiles = 1 : i32' in transformed
native_transformed = run(run(native, analysis), transform)
# Representation does not alter launch facts or selected input roles.
contract = r'program_split.launch = (\{[^\n]+\})'
assert re.search(contract, transformed).group(1) == re.search(contract, native_transformed).group(1)
print('PASS mapped tanh, partial tiles, serialization, reanalysis and launch equivalence')

for name, text in {
    'unknown_readnone': mapped.replace('__hmf_tanhf', '__hmf_unknown'),
    'prefix_is_not_identity': mapped.replace('__hmf_tanhf', '__hmf_tanhf_unknown'),
    'wrong_scalar_signature': mapped.replace('f32', 'f64'),
    'fp16_scalar_signature': mapped.replace(
        decl, decl.replace('f32', 'f16')).replace(
        '%37 = linalg.map { func.call {callee = @__hmf_tanhf} } '
        'ins(%35 : tensor<32x2048xf32>) outs(%35 : tensor<32x2048xf32>)',
        '%tanh_input_f16 = arith.truncf %35 : tensor<32x2048xf32> to tensor<32x2048xf16>\n'
        '        %tanh_result_f16 = linalg.map { func.call {callee = @__hmf_tanhf} } '
        'ins(%tanh_input_f16 : tensor<32x2048xf16>) outs(%tanh_input_f16 : tensor<32x2048xf16>)\n'
        '        %37 = arith.extf %tanh_result_f16 : tensor<32x2048xf16> to tensor<32x2048xf32>'),
    'missing_purity': mapped.replace(' attributes {llvm.readnone}', ''),
    'defined_helper': mapped.replace(decl, 'func.func private @__hmf_tanhf(%x: f32) -> f32 { return %x : f32 }'),
    'missing_noalias': mapped.replace('ascend.memory.pointer_args_noalias, ', ''),
    'second_entry': mapped.replace(decl, decl + '\n func.func @second() { return }'),
}.items():
    result = run(text, analysis)
    assert 'l1_prefetch_ps.accepted' not in result and 'reject_reason' in result, name
    print('PASS rejection:', name)

# Parsed map body contains exactly one call and yield. Reject extra scalar
# computation/captured values, even when the callee itself is recognized.
extra = mapped.replace(
    'linalg.map { func.call {callee = @__hmf_tanhf} } '
    'ins(%35 : tensor<32x2048xf32>) outs(%35 : tensor<32x2048xf32>)',
    'linalg.map ins(%35 : tensor<32x2048xf32>) outs(%35 : tensor<32x2048xf32>) '
    '(%x: f32) { %extra = arith.constant 0.0 : f32 '
    '%y = func.call @__hmf_tanhf(%x) : (f32) -> f32 linalg.yield %y : f32 }')
assert 'l1_prefetch_ps.accepted' not in run(extra, analysis)
print('PASS extra map operation rejected')
for name, text in {
    'captured_scalar': extra.replace('%extra = arith.constant 0.0 : f32 ', '').replace('@__hmf_tanhf(%x)', '@__hmf_tanhf(%cst_0)'),
    'yield_bypasses_call': extra.replace('%extra = arith.constant 0.0 : f32 ', '').replace('linalg.yield %y', 'linalg.yield %x'),
}.items():
    assert 'l1_prefetch_ps.accepted' not in run(text, analysis), name
    print('PASS rejection:', name)


for name, text in {
    'changed_declaration_mapping_attributes': a.replace(
        'attributes {llvm.readnone}',
        'attributes {llvm.readnone, integration.mapping_context = "changed"}'),
    'changed_accepted_facts': a.replace('l1_prefetch_ps.facts = {',
        'l1_prefetch_ps.facts = {unexpected = 1 : i32, ', 1),
    'changed_declaration_purity': a.replace('attributes {llvm.readnone}', ''),
    'changed_callee': a.replace('__hmf_tanhf', '__hmf_unknown'),
    'changed_declaration_body': a.replace(decl, 'func.func private @__hmf_tanhf(%x: f32) -> f32 { return %x : f32 }'),
}.items():
    assert text != a, name
    assert 'L1 pipeline error' in run(text, transform, succeeds=False), name
    print('PASS stale proof:', name)
