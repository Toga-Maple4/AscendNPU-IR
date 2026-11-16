"""Exact spellings, their negative twins, and complete compile/contract checks."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

opt, compiler, persistent, flat = sys.argv[1:]
source = Path(persistent).read_text()
flat_source = Path(flat).read_text().replace('Ascend950PR_9579', 'Ascend950PR_9589')

def analyze(text, accepted=True, reason=None):
    p = subprocess.run([opt, '--l1-prefetch-analyze', '--l1-prefetch-normalize']
                       if accepted else [opt, '--l1-prefetch-analyze'],
                       input=text, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert ('program_split.transformed' in p.stdout) == accepted, p.stdout
    if reason:
        assert reason in p.stdout, p.stdout
    return p.stdout

flat_alt = flat_source.replace('%3 = arith.muli %arg9, %c32768_i32 : i32', '''%c16384_i32 = arith.constant 16384 : i32
    %half = arith.muli %c16384_i32, %arg9 : i32
    %3 = arith.addi %half, %half : i32''')
# Static descriptor facts may use constant SSA operands too.
views = source.replace('%c8192 = arith.constant 8192 : index', '''%c8192 = arith.constant 8192 : index
    %bm = arith.constant 16 : index
    %bn = arith.constant 2048 : index
    %one = arith.constant 1 : index''').replace('sizes: [16, 2048], strides: [8192, 1]', 'sizes: [%bm, %bn], strides: [%c8192, %one]')
same_iv = source.replace('%4 = arith.divsi %arg10', '%same_iv = arith.addi %arg10, %c0_i32 : i32\n      %4 = arith.divsi %same_iv')
fixed = source.replace('to %steps', 'to %c148_i32').replace('slt, %7, %tasks', 'slt, %7, %c2048_i32').replace('%4, %n', '%4, %c56_i32').replace('%n, %round_again', '%c56_i32, %round_again')
positives = [('fixed', fixed, 'fixed_programs'), ('same_iv', same_iv, 'persistent_tiles'), ('persistent', source, 'persistent_tiles'),
             ('views', views, 'persistent_tiles'), ('flat', flat_alt, 'flat_tiles')]
with tempfile.TemporaryDirectory() as directory:
    for name, text, kind in positives:
        analyze(text)
        path = Path(directory) / (name + '.mlir')
        path.write_text(text)
        output = Path(directory) / name
        p = subprocess.run([compiler, str(path), '--target=Ascend950PR_9589',
             '--enable-lib-call-no-inline=false', '--enable-hfusion-compile=true',
             '--enable-hivm-compile=true', '--enable-triton-kernel-compile=true',
             '--enable-auto-blockify-loop=true', '--enable-auto-multi-buffer=true',
             '--enable-program-split-mix=true', '-o', str(output)], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-5000:]
        contract = json.loads(Path(str(output)+'.program_split.json').read_text())
        assert contract['guard']['kind'] == kind, contract

# Linear is insufficient: coefficients, signedness, widths and stride must agree.
for text, reason in [
    (source.replace('%7 = arith.subi %sum1, %c1_i32', '%7 = arith.addi %sum1, %c1_i32'), 'unproved_worker_disjointness'),
    (source.replace('%row_again = arith.muli %n, %round_again', '%row_again = arith.muli %c16_i32, %round_again'), 'unproved_store_partition'),
    (source.replace('%column_again = arith.remsi', '%column_again = arith.remui'), 'unproved_store_partition'),
    (source.replace('%round_again = arith.divsi', '%round_again = arith.divui'), 'unproved_store_partition'),
    (source.replace('%row_index = arith.index_cast', '%row_index = arith.index_castui'), 'unproved_store_partition'),
    (source.replace('%row_index = arith.index_cast %row_scaled : i32 to index', '''%short = arith.trunci %row_scaled : i32 to i16
        %wide = arith.extsi %short : i16 to i32
        %row_index = arith.index_cast %wide : i32 to index'''), 'unproved_store_partition'),
    (source.replace('%output_offset = arith.addi %column_index, %row_offset', '%output_offset = arith.subi %column_index, %row_offset'), 'unproved_store_partition'),
    (flat_alt.replace('%3 = arith.addi %half, %half', '%3 = arith.subi %half, %half'), 'flat_unproved_program_ownership'),
]:
    analyze(text, False, reason)
print('PASS exact equivalence, contract/codegen and eight negative twins')
