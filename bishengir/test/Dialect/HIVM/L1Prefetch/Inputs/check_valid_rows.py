"""Selected-load row bounds must preserve padding and consumer identity."""
from pathlib import Path
import re
import subprocess
import sys

opt, fixture = sys.argv[1:]
source = Path(fixture).read_text()


def run(text, expected_copies=4):
    result = subprocess.run([opt, '--l1-prefetch-materialize', '--canonicalize',
                             '--mlir-print-unique-ssa-ids'],
                            input=text, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    copies = [line for line in result.stdout.splitlines() if 'hivm.hir.l12ub' in line]
    assert len(copies) == expected_copies
    assert sum('sub_blockid = true' in line for line in copies) == expected_copies // 2
    return result.stdout, copies


# An explicit value/first-element pad initializes rows beyond the source. Those
# rows must still be delivered, independently of the current consumer's uses.
for mode in ('PadValue', 'PadFirstElem'):
    padded = source.replace('%c0 = arith.constant 0 : index',
                            '%padding = arith.constant -1.0 : f32\n      %c0 = arith.constant 0 : index')
    replacement = f'pad_mode = <{mode}>'
    if mode == 'PadValue':
        replacement += ' pad_value = %padding : f32'
    padded = padded.replace('pad_mode = <PadNull>', replacement)
    text, copies = run(padded)
    assert all('memref<16x64xf32,' in line and 'memref<?x64xf32' not in line
               for line in copies), text
    assert f'pad_mode = <{mode}>' in text

# Statically bounded sources follow the same rule; full rows keep direct copies.
for rows in (1, 7, 16):
    static = source.replace('%validRows = arith.minsi %remaining, %c16 : index',
                            f'%validRows = arith.constant {rows} : index')
    text, copies = run(static)
    if rows == 16:
        assert all('memref<16x64xf32,' in line for line in copies), text
    else:
        bounded = [line for line in text.splitlines()
                   if 'memref.subview' in line and f'[{rows}, 64]' in line
                   and '#hivm.address_space<' in line]
        assert len(bounded) == 8, text

# A non-jagged partial source uses each consumer's logical id. In this fixture,
# consumer 1 starts 28 tiles later; it must not reuse consumer 0's row bound.
text, copies = run(source)
views = re.findall(r'memref.subview[^\n]*\[0, 0\] \[(%[a-zA-Z0-9_]+), 64\] \[1, 1\][^\n]*cbuf', text)
assert len(views) == 4 and views[0] != views[1] and views[2] != views[3], views
# Empty input: the original inactive predicate removes every zero-row payload.
empty = source.replace('%active = arith.cmpi slt, %start, %rows : i32',
                       '%active = arith.constant false')
empty = empty.replace('%validRows = arith.minsi %remaining, %c16 : index',
                      '%validRows = arith.constant 0 : index')
text, _ = run(empty, expected_copies=0)
assert 'hivm.hir.load' not in text and 'hivm.hir.store' not in text

# Exercise independent selected inputs through the same materializer. This is
# physical-stage coverage, not an extension of the current full-tile two-input
# semantic analysis. Both row bounds are positive under the original predicate;
# only their intersection is observed by the output slice.
two = source.replace('input_args = array<i32: 0>',
                     'input_args = array<i32: 0, 5>')
two = two.replace('%rounds: i32)', '%rounds: i32, %input2: memref<?xf32>)')
two = two.replace('%tensor = bufferization.to_tensor', """%seven = arith.constant 7 : index
          %secondRows = arith.minsi %validRows, %seven : index
          %in2 = memref.reinterpret_cast %input2 to offset: [%offset], sizes: [16, 64], strides: [%columns, 1] : memref<?xf32> to memref<16x64xf32, strided<[?, 1], offset: ?>>
          %src2 = memref.subview %in2[0, 0] [%secondRows, %columns] [1, 1] : memref<16x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %selected2 = memref.alloc() : memref<16x64xf32>
          annotation.mark %selected2 {program_split.selected_input = 1 : i32} : memref<16x64xf32>
          hivm.hir.load ins(%src2 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%selected2 : memref<16x64xf32>) pad_mode = <PadNull> core_type = <VECTOR>
          %tensor2 = bufferization.to_tensor %selected2 restrict writable : memref<16x64xf32>
          %tensor = bufferization.to_tensor""")
two = two.replace('arith.mulf %tensor, %tensor', 'arith.mulf %tensor, %tensor2')
two = two.replace('%squared[0, 0] [%validRows, %columns]',
                  '%squared[0, 0] [%secondRows, %columns]')
two = two.replace('sizes: [%validRows, %columns]', 'sizes: [%secondRows, %columns]')
text, copies = run(two, expected_copies=8)
views = re.findall(r'memref.subview[^\n]*\[0, 0\] \[(%[a-zA-Z0-9_]+), 64\] \[1, 1\][^\n]*cbuf', text)
assert len(views) == 8
assert all(views[i] != views[i + 1] for i in range(0, 8, 2)), views
print('PASS: independent inputs/consumers, zero/static/dynamic rows, full rows, required padding')
