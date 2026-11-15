"""Common protocol across guard shapes, input counts and solver dependencies."""
from pathlib import Path
import re
import subprocess
import sys

opt, outer_path, guarded_path = sys.argv[1:]
outer, guarded = (Path(p).read_text() for p in (outer_path, guarded_path))


def run(source, *extra):
    return subprocess.run([opt, '--program-split-l1-consumer-sync', *extra],
                          input=source, text=True, capture_output=True)


def native_events(text):
    return re.findall(r'hivm\.(WAIT|SET)\.INTRA\.BLOCKI.mode.*sync_id = (\d+)', text)


def prefetch_ahead(source):
    # Exercise the same handoff refinement with a guarded next-input load
    # before current delivery, for both outer and per-consumer work guards.
    source = re.sub(r'^//[^\n]*\n', '', source, flags=re.M)
    source = re.sub(r'^\s*hivm.hir.load[^\n]*', '', source, flags=re.M)
    source = source.replace('%even: i1)', '%even: i1, %more: i1)')
    def load(bank):
        return ('\n      scf.if %more {\n'
                '        hivm.hir.load ins(%gm : memref<16x2048xf32, #hivm.address_space<gm>>) '
                f'outs(%{bank}0 : memref<16x2048xf32, #hivm.address_space<cbuf>>)\n      }}')
    source = source.replace('scf.if %even {', 'scf.if %even {' + load('pong'))
    return source.replace('} else {', '} else {' + load('ping'))


def bounded_views(source):
    # A dynamically bounded L1/UB view must keep the same per-consumer event
    # protocol, including the guard-false (empty consumer) case.
    source = source.replace('%even: i1)', '%even: i1, %liveRows: index)')
    serial = iter(range(4))
    def replace(match):
        indent, src, dst, sub = match.groups()
        number = next(serial)
        lines = []
        for role, value, space in (('src', src, 'cbuf'), ('dst', dst, 'ub')):
            lines.append(f'{indent}%{role}{number} = memref.subview {value}[0, 0] '
                         f'[%liveRows, 2048] [1, 1] : memref<16x2048xf32, #hivm.address_space<{space}>> '
                         f'to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<{space}>>')
        lines.append(f'{indent}hivm.hir.l12ub ins(%src{number} : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<cbuf>>) '
                     f'outs(%dst{number} : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>){sub}')
        return '\n'.join(lines)
    return re.sub(r'^( +)hivm.hir.l12ub ins\((%[^ :]+) : memref<16x2048xf32, #hivm.address_space<cbuf>>\) '
                  r'outs\((%[^ :]+) : memref<16x2048xf32, #hivm.address_space<ub>>\)([^\n]*)$',
                  replace, source, flags=re.M)


for source, groups in ((outer, 6), (guarded, 4),
                       (prefetch_ahead(outer), 6), (prefetch_ahead(guarded), 4),
                       (bounded_views(outer), 6), (bounded_views(guarded), 4)):
    for inputs in (1, 2):
        for pipes in (('PIPE_V',), ('PIPE_MTE3',), ('PIPE_V', 'PIPE_MTE3'), ('PIPE_MTE3', 'PIPE_V')):
            text = re.sub(r'^\s*hivm.hir.sync_block_set[^\n]*', '', source, flags=re.M)
            signals = '\n'.join(f'    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <{pipe}>] flag = {flag}'
                                for pipe, flag in zip(pipes, (0, 1)))
            text = text.replace('  }\n  hivm.hir.sync_block_wait', signals + '\n  }\n  hivm.hir.sync_block_wait')
            if inputs == 2:
                text = re.sub(r'^(\s*hivm.hir.l12ub[^\n]*)$', r'\1\n\1', text, flags=re.M)
            result = run(text)
            assert result.returncode == 0, result.stderr
            expected = []
            for group in range(groups):
                offset = (group % 2) * 16
                expected += [('WAIT', str(2 + offset))]
                expected += [('SET', str(flag + offset)) for flag in range(len(pipes))]
            assert native_events(result.stdout) == expected, (inputs, pipes, result.stdout)
            if '%more' in source:
                # Each next-bank prefetch must still precede its first UB wait.
                # A wait hoisted above prefetch would lose the intended overlap.
                branches = result.stdout.split('hivm.hir.load')[1:]
                assert len(branches) == 2
                for branch in branches:
                    assert branch.index('hivm.WAIT.INTRA.BLOCKI.mode') < branch.index('hivm.hir.l12ub')
                assert result.stdout.index('hivm.hir.load') < result.stdout.index('hivm.WAIT.INTRA.BLOCKI.mode')
            assert result.stdout.count('hivm.hir.l12ub') == 4 * inputs
            assert result.stdout.count('hivm.hir.sync_block_wait') == 1  # Original drain.
            # Both selected inputs must precede publication, including guarded copies.
            segments = re.split(r'(?=\s*"hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode)', result.stdout)
            for segment in segments:
                if 'hivm.hir.l12ub' in segment:
                    before_ready = segment.split('hivm.intr.hivm.SET.INTRA.BLOCKI.mode')[0]
                    assert before_ready.count('hivm.hir.l12ub') == inputs
            # A launch family tag cannot select a different handoff algorithm.
            tagged = text.replace('program_split.transformed, ',
                                  'program_split.transformed, program_split.launch = {jagged_copy = 1 : i32}, ')
            tagged_result = run(tagged)
            assert tagged_result.returncode == 0, tagged_result.stderr
            assert native_events(tagged_result.stdout) == expected

# Guarded two-input payload must finish in full before the single ready event.
two_guarded = re.sub(r'^(\s*hivm.hir.l12ub[^\n]*)$', r'\1\n\1', guarded, flags=re.M)
r = run(two_guarded)
for segment in r.stdout.split('hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode')[1:]:
    before_ready = segment.split('hivm.intr.hivm.SET.INTRA.BLOCKI.mode')[0]
    assert before_ready.count('hivm.hir.l12ub') == 2

bad_inputs = [
    # Duplicate readiness endpoint and conflicting endpoint IDs.
    outer.replace('<PIPE_V>] flag = 1', '<PIPE_MTE3>] flag = 1'),
    outer.replace('<PIPE_V>] flag = 1', '<PIPE_V>] flag = 0'),
    outer.replace('<PIPE_V>] flag = 1', '<PIPE_V>] flag = 2'),
    # Bank branches must agree about the number and ownership of inputs.
    re.sub(r'^(\s*hivm.hir.l12ub[^\n]*)$', r'\1\n\1', outer, count=1, flags=re.M),
    guarded.replace('sub_blockid = true', 'sub_blockid = false', 1),
    # Distinct payload guards for two inputs of one consumer are unsupported.
    re.sub(r'^(      scf.if %active[01] \{\n.*?^      \})', r'\1\n\1', guarded,
           flags=re.M | re.S),
    # Unexpected nesting cannot make conditional handshakes silently disappear.
    guarded.replace('      scf.if %active0 {', '      scf.if %active0 {\n        scf.if %active1 {', 1)
           .replace('      }\n      scf.if %active1', '        }\n      }\n      scf.if %active1', 1),
]
for text in bad_inputs:
    failed = run(text)
    assert failed.returncode != 0 and 'L1 consumer sync:' in failed.stderr, failed.stderr

# A later malformed loop must not leave the earlier valid loop rewritten.
start = guarded.index('  scf.for %i')
end = guarded.index('  hivm.hir.sync_block_wait', guarded.index('\n  }', start))
body = guarded[start:end]
second = body.replace('sub_blockid = true', 'sub_blockid = false', 1)
failed = run(guarded[:end] + second + guarded[end:], '--mlir-print-ir-after-failure')
assert failed.returncode != 0 and 'L1 consumer sync:' in failed.stderr, failed.stderr
assert 'IR Dump After' in failed.stderr, failed.stderr
assert 'hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode' not in failed.stderr, failed.stderr
print('PASS: shared guard/input/dependency matrix, endpoint validation and atomic rejection')
