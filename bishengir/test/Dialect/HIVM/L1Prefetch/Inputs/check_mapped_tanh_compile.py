"""Compile Maple4's mapped tanh through the New-L1 retry and sidecar writer."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

compiler, fixture = sys.argv[1:]
source = Path(fixture).read_text()
source = source.replace('  func.func @_gelu',
    '  func.func private @__hmf_tanhf(f32) -> f32 attributes {llvm.readnone}\n'
    '  func.func @_gelu', 1)
source = source.replace('math.tanh %35 : tensor<32x2048xf32>',
    'linalg.map { func.call {callee = @__hmf_tanhf} } '
    'ins(%35 : tensor<32x2048xf32>) outs(%35 : tensor<32x2048xf32>)')
with TemporaryDirectory() as tmp:
    root = Path(tmp)
    input = root / 'mapped.mlir'
    input.write_text(source)
    output = root / 'kernel'
    result = subprocess.run([compiler, str(input), '--target=Ascend950PR_9589',
        '--enable-lib-call-no-inline=false', '--enable-hfusion-compile=true',
        '--enable-hivm-compile=true', '--enable-triton-kernel-compile=true',
        '--enable-auto-multi-buffer=true', '--enable-program-split-mix=true',
        '--disable-ffts', '-o', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (root / 'kernel.o').is_file()
    sidecar = json.loads((root / 'kernel.program_split.json').read_text())
    assert sidecar['kind'] == 'split_mix_half_grid', sidecar
    assert sidecar['guard']['kind'] == 'tiles_2d', sidecar
    assert sidecar['guard']['tile_mode'] == 'partial', sidecar
    assert sidecar['profitability']['compute_score'] is None, sidecar
    assert sidecar['profitability']['reason'] == 'unknown_operation_cost', sidecar
print('PASS mapped tanh full compilation and partial-tile launch contract')
