#!/usr/bin/env python3
"""Freeze six established checkpoints on MAHRM's fresh Table 6 holdout."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


TEST_SHA = {
    "easy": "1c3efbe50a2fbcafe9e542e99f11e02eba50af79ae7a030eecf7db6bbbb6f7c2",
    "hard": "866590f0112b8a56cab99b4f8155793773131848f4805ab97591923868edbecd",
}


def digest(path: Path) -> str:
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
    for difficulty in ("easy", "hard"):
        original = repo / f"experiments/specs/original_submission_v8_{difficulty}_all_methods_5seed_v1.json"
        test = repo / f"experiments/mahrm_direct_20260924/protocols/{difficulty}_test.json"
        if digest(test) != TEST_SHA[difficulty]:
            raise RuntimeError(f"Frozen holdout changed: {test}")
        manifest = json.loads(test.read_text(encoding="utf-8"))
        spec = json.loads(original.read_text(encoding="utf-8"))
        spec["name"] = f"table6_mahrm_t035_{difficulty}_holdout_v1"
        spec["reviewer_items"] = ["MAHRM_direct_comparison", "Table6_common_t035"]
        spec["evaluation_seed"] = 2026092403 if difficulty == "easy" else 2026092404
        spec["policy_mode"] = "tempered"
        spec["temperature"] = 0.35
        spec["protocols"] = [{"name": "random", "path": str(test.relative_to(repo)),
                              "file_sha256": digest(test), "start_set_id": manifest["start_set_id"],
                              "count": manifest["count"]}]
        spec["holdout_protocol_audit"] = str(
            (repo / "experiments/mahrm_direct_20260924/protocols/protocol_audit.json").relative_to(repo))
        spec["holdout_protocol_audit_sha256"] = digest(repo / spec["holdout_protocol_audit"])
        spec["source_spec_sha256"] = digest(original)
        output = repo / f"experiments/mahrm_direct_20260924/specs/table6_mahrm_t035_{difficulty}_holdout_v1.json"
        if output.exists():
            raise RuntimeError(f"Refusing to overwrite spec: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"spec": str(output), "sha256": digest(output), "methods": spec["expected_methods"]}))


if __name__ == "__main__":
    main()

