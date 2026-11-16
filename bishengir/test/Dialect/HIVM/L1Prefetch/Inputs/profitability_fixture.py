"""Add arithmetic to a unit fixture to exercise the accepted contract serializer.

The original fixture still tests filtering. This does not change any workload
preset, access pattern, tile, or semantic selection.
"""
from pathlib import Path
import re
import sys
source = Path(sys.argv[1]).read_text()
match = re.search(r"(%extracted_slice = tensor.extract_slice )(%\w+)(\[.*? : )(tensor<[^>]+>)", source)
assert match
ops = ""
previous = match[2]
for i in range(4):
    name = f"%profitability_work{i}"
    ops += f"{name} = math.exp {previous} : {match[4]}\n        "
    previous = name
source = source[:match.start()] + ops + source[match.start():].replace(match[0], match[1] + previous + match[3] + match[4], 1)
Path(sys.argv[2]).write_text(source)
