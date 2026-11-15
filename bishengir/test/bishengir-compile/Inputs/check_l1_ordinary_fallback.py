"""Check a named New-L1 rejection retains a real ordinary executable."""
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
assert output.is_file() and output.stat().st_size > 0, output
contract = json.loads(Path(str(output) + ".program_split.json").read_text())
assert contract["kind"] == "ordinary", contract
assert contract["reason"] == sys.argv[2], contract
assert contract["compile_options"]["enable_auto_multi_buffer"] is False, contract
assert "inputs" not in contract and "launch" not in contract, contract
