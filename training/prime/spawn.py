"""Start prime-rl training as a *spawned* call on the deployed app, so it survives the local client.

`modal run --detach` keeps a run alive only while the launching client lives; when this sandbox's
background processes died, Modal cancelled both runs (05:26 and 08:50 UTC). Spawned calls on a
deployed app run independently of any local process.

  set -a; . "care package/.env"; set +a
  modal deploy training/prime/modal_prime.py
  python training/prime/spawn.py <config.toml> <run_name> [extra_args]
"""

import sys
from pathlib import Path

import modal

config, run_name = sys.argv[1], sys.argv[2]
extra = sys.argv[3] if len(sys.argv) > 3 else ""
fn = modal.Function.from_name("tpch-opd-prime", "train1")
call = fn.spawn(Path(config).read_text(), run_name, extra)
print(run_name, call.object_id)
