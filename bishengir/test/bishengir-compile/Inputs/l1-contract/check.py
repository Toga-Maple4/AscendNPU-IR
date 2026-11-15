"""Compare complete sidecar facts; JSON printing order is not the interface."""
import json
from pathlib import Path
import sys

expected, actual = (json.loads(Path(p).read_text()) for p in sys.argv[1:])
assert actual == expected, ("contract mismatch", expected, actual)
