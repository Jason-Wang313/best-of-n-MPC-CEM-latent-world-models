from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


BANNED_PATTERNS = [
    re.compile(r"\bcem always hurts\b", re.IGNORECASE),
    re.compile(r"\bcem always helps\b", re.IGNORECASE),
    re.compile(r"\balways outperforms\b", re.IGNORECASE),
    re.compile(r"\balways fails\b", re.IGNORECASE),
    re.compile(r"\bguarantees improved real return\b", re.IGNORECASE),
    re.compile(r"\bsolves model exploitation\b", re.IGNORECASE),
]


def load_claims(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise ValueError("claim file must contain a top-level claims list")
    return claims


def audit_claims(claims: list[dict], repo_root: Path) -> list[str]:
    errors: list[str] = []
    allowed = {"supported", "partial", "unsupported"}
    for claim in claims:
        claim_id = claim.get("id", "<missing id>")
        status = claim.get("status")
        if status not in allowed:
            errors.append(f"{claim_id}: status must be one of {sorted(allowed)}")
        evidence = claim.get("evidence", [])
        if status in {"supported", "partial"} and not evidence:
            errors.append(f"{claim_id}: {status} claim has no evidence files")
        if status == "unsupported" and evidence:
            errors.append(f"{claim_id}: unsupported claim should not cite evidence as if proven")
        for rel in evidence:
            target = repo_root / rel
            if not target.exists():
                errors.append(f"{claim_id}: missing evidence file {rel}")
    return errors


def audit_language(repo_root: Path) -> list[str]:
    errors: list[str] = []
    for path in list((repo_root / "docs").glob("*.md")) + list((repo_root / "paper").glob("*.md")) + [repo_root / "README.md"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in BANNED_PATTERNS:
            match = pattern.search(text)
            if match:
                errors.append(f"{path.relative_to(repo_root)}: banned universal language `{match.group(0)}`")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--claim-file", type=Path, default=Path("docs/claims.json"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    claim_file = args.claim_file
    if not claim_file.is_absolute():
        claim_file = repo_root / claim_file

    errors = []
    errors.extend(audit_claims(load_claims(claim_file), repo_root))
    errors.extend(audit_language(repo_root))
    if errors:
        print("Claim audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Claim audit passed.")


if __name__ == "__main__":
    main()
