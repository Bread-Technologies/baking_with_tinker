"""Patch prime-rl's OPD algorithm so teacher scoring overlaps instead of running one episode at a time.

The orchestrator's main loop awaits train_sink.add(episode) per finished episode, and OPD's
finalize_episode awaits the teacher prefill-score inside it. With a remote teacher (~3 s per call
to Tinker) that serializes ~128 calls per step (~6 min). Here finalize_episode only *starts* the
scoring task; finalize_group (which runs before the group can enter a batch) awaits the tasks for
that group's episodes. Reference logprobs are therefore all assigned before any batch ships.

Applied at image build time: python opd_concurrent_scoring.py /app/src/prime_rl/orchestrator/algo/opd.py
"""

import sys
from pathlib import Path

PATCH = '''

# --- concurrent teacher scoring (patched; see training/prime/patches/opd_concurrent_scoring.py) ---
_orig_finalize_episode = OPDAlgorithm.finalize_episode
_orig_finalize_group = OPDAlgorithm.finalize_group


async def _finalize_episode_async(self, episode):
    tasks = self.__dict__.setdefault("_scoring_tasks", {})
    tasks[id(episode)] = asyncio.create_task(_orig_finalize_episode(self, episode))


async def _finalize_group_awaiting(self, episodes):
    tasks = self.__dict__.setdefault("_scoring_tasks", {})
    pending = [tasks.pop(id(e)) for e in episodes if id(e) in tasks]
    if pending:
        await asyncio.gather(*pending)
    await _orig_finalize_group(self, episodes)


OPDAlgorithm.finalize_episode = _finalize_episode_async
OPDAlgorithm.finalize_group = _finalize_group_awaiting
'''


def main():
    path = Path(sys.argv[1])
    src = path.read_text()
    if "concurrent teacher scoring (patched" in src:
        print("already patched")
        return
    assert "class OPDAlgorithm" in src and "import asyncio" in src, "unexpected opd.py layout"
    path.write_text(src + PATCH)
    print("patched", path)


if __name__ == "__main__":
    main()
