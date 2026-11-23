#!/usr/bin/env python3
"""Run frozen MAHRM test only after all ten training seeds have completed."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CODE_SHA256 = "f4c89fcee604261045f52f19af11a5a8651bbb4af4f8225d286880a7f1ca6509"
PROTOCOL_SHA256 = "0695b8305806c0e12bc031728f6e5c1af34f5983d024762cba90f6727d574e0e"
TEST_SHA = {"easy": "1c3efbe50a2fbcafe9e542e99f11e02eba50af79ae7a030eecf7db6bbbb6f7c2",
            "hard": "866590f0112b8a56cab99b4f8155793773131848f4805ab97591923868edbecd"}


def sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    root = repo / "experiments/mahrm_direct_20260924"
    code = root / "code/mahrm_drama.py"
    if sha(code) != CODE_SHA256 or sha(root / "protocols/protocol_audit.json") != PROTOCOL_SHA256:
        raise RuntimeError("Code or protocol audit changed")
    for difficulty, cap, budget in (("easy", 300, 3000), ("hard", 500, 1200)):
        if sha(root / f"protocols/{difficulty}_test.json") != TEST_SHA[difficulty]:
            raise RuntimeError("Test manifest changed")
        for seed in (42, 43, 44, 45, 46):
            record = root / f"records/{difficulty}_seed{seed}.json"
            while True:
                if record.exists():
                    training = json.loads(record.read_text(encoding="utf-8"))
                    if training["status"] == "completed":
                        break
                    if training["status"] == "failed":
                        raise RuntimeError(f"Training failed: {record}")
                time.sleep(15)
            if training["trained_episodes"] != budget:
                raise RuntimeError(f"Training budget mismatch: {record}")
            checkpoint = root / f"checkpoints/{difficulty}_seed{seed}.pkl"
            if sha(checkpoint) != training["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    print("All ten seeds completed; opening previously frozen test manifests", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(repo / "src/sacc-drama/DRAMA"), str(root / "code")))
    for difficulty, cap in (("easy", 300), ("hard", 500)):
        for seed in (42, 43, 44, 45, 46):
            result = root / f"eval_records/{difficulty}_seed{seed}.json"
            if result.exists():
                raise RuntimeError(f"Refusing duplicate test: {result}")
            command = [sys.executable, str(code), "eval", "--difficulty", difficulty,
                       "--seed", str(seed), "--max-steps", str(cap),
                       "--test-starts", str(root / f"protocols/{difficulty}_test.json"),
                       "--checkpoint", str(root / f"checkpoints/{difficulty}_seed{seed}.pkl"),
                       "--record", str(result)]
            log = root / f"logs/eval_{difficulty}_seed{seed}.log"
            with log.open("x", encoding="utf-8") as handle:
                outcome = subprocess.run(command, cwd=repo, env=env, stdout=handle,
                                         stderr=subprocess.STDOUT, check=False)
            if outcome.returncode != 0:
                raise RuntimeError(f"Evaluation failed: {log}")
            print(f"completed {difficulty} seed {seed}: {sha(result)}", flush=True)


if __name__ == "__main__":
    main()

