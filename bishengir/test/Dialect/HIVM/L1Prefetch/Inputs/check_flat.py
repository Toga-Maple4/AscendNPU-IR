"""Prove flat eligibility/ownership without name, shape or ABI allowlists."""
import re,subprocess,sys
from pathlib import Path
source=Path(sys.argv[1]).read_text()
opt=sys.argv[2]
def analyze(text, reason=None):
    p=subprocess.run([opt,'--l1-prefetch-analyze'],input=text,text=True,capture_output=True)
    assert p.returncode==0,p.stderr
    if reason:
        assert f'l1_prefetch_ps.reject_reason = "{reason}"' in p.stdout,p.stdout
        assert 'program_split.selected_input' not in p.stdout,p.stdout
    else:
        assert 'l1_prefetch_ps.accepted' in p.stdout,p.stdout
        q=subprocess.run([opt,'--l1-prefetch-normalize'],input=p.stdout,text=True,capture_output=True)
        assert q.returncode==0,q.stderr
        assert 'flat_tile = 1' in q.stdout and 'pair_stride = 28' in q.stdout,q.stdout
        assert q.stdout.count('hivm.hir.load')==2,q.stdout
        assert 'l1_prefetch_ps.' not in q.stdout,q.stdout
        return q.stdout
analyze(source)
for size in (1024,3072,8192,16384,65536):
    analyze(source.replace('32768',str(size)))
analyze(source.replace('@_silu_backward','@unrelated_elementwise'))
analyze(source.replace('f16','bf16'))
# Reorder declarations only: the unchanged accesses now refer to different ABI
# positions. This exercises derived roles, not a cosmetic SSA rename.
declarations=re.search(r'(%arg2: .*?), (%arg3: .*?), (%arg4: .*?), %arg5:',source)
assert declarations
permuted=source.replace(declarations[0],
    f'{declarations[2]}, {declarations[3]}, {declarations[1]}, %arg5:',1)
result=analyze(permuted)
assert 'output_arg = 4' in result and 'input_args = array<i32: 2, 3>' in result,result
analyze(source.replace('ascend.memory.pointer_args_noalias, ',''),'missing_pointer_noalias_contract')
analyze(source.replace('arith.muli %arg9,','arith.muli %arg10,'),'flat_unproved_program_ownership')
analyze(source.replace('strides: [1]', 'strides: [2]').replace('strided<[1], offset:', 'strided<[2], offset:'),'flat_unproved_store_partition')
# Integer conversion cannot consume undefined padding: unlike pure floating
# elementwise arithmetic, its padded lanes are outside the established proof.
tensor=re.search(r'(%[\w]+) = bufferization.to_tensor[^\n]+',source)
assert tensor
unsafe=source.replace(tensor[0],tensor[0] + '\n    %unsafe = arith.fptosi ' + tensor[1] +
    ' : tensor<32768xf16> to tensor<32768xi32>',1)
analyze(unsafe,'flat_observable_or_unproved_padding')
print('flat semantic/normalization checks passed')
