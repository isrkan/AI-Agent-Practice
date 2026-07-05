#!/usr/bin/env python3
"""
parse_git_log.py
================
Categorize a list of commit subjects into Keep a Changelog sections.

This is a bundled skill *script*: a deterministic helper the agent runs rather
than reasons through token by token. It reads commit subjects from stdin (one per
line, in the `git log --pretty=%s` format) and prints a JSON object mapping each
changelog section to its list of human-readable entries.

Usage:
    git log v1.3.0..HEAD --pretty=%s | python scripts/parse_git_log.py
    python scripts/parse_git_log.py < commits.txt

Output (stdout):
    {"Added": ["Pagination to the search API"], "Fixed": ["Crash on empty query"], ...}

Exit codes:
    0 - parsed successfully
    1 - no commits were provided on stdin
"""

import json
import re
import sys

# Conventional Commit type -> changelog section. Types mapped to None are dropped
# because they are not user-facing (tooling, tests, docs, style).
TYPE_TO_SECTION = {
    "feat": "Added",
    "fix": "Fixed",
    "perf": "Changed",
    "refactor": "Changed",
    "revert": "Removed",
    "security": "Security",
    "deprecate": "Deprecated",
    "docs": None,
    "test": None,
    "style": None,
    "chore": None,
    "ci": None,
    "build": None,
}

# Matches "type(scope)!: subject" capturing type, the optional "!", and the subject.
COMMIT_RE = re.compile(r"^(?P<type>\w+)(?:\([^)]*\))?(?P<breaking>!)?:\s*(?P<subject>.+)$")


def humanize(subject: str) -> str:
    """Turn a commit subject into a past-tense, user-facing changelog line."""
    subject = subject.strip().rstrip(".")
    if not subject:
        return subject
    # Capitalize the first word; leave the rest as the author wrote it.
    return subject[0].upper() + subject[1:]


def categorize(lines):
    """Group commit subjects into changelog sections.

    Args:
        lines: Iterable of raw commit subject strings.

    Returns:
        Dict mapping section name -> list of humanized entry strings. Only
        non-empty sections are included.
    """
    sections = {}
    for raw in lines:
        raw = raw.strip()
        if not raw or raw.lower().startswith("merge "):
            continue

        match = COMMIT_RE.match(raw)
        if match:
            commit_type = match.group("type").lower()
            section = TYPE_TO_SECTION.get(commit_type, "Changed")
            subject = match.group("subject")
            breaking = bool(match.group("breaking"))
        else:
            # No Conventional Commit prefix: fall back to "Changed".
            section, subject, breaking = "Changed", raw, False

        if section is None:  # Dropped, non-user-facing type.
            continue

        entry = humanize(subject)
        if breaking:
            entry = f"**Breaking:** {entry}"
        sections.setdefault(section, []).append(entry)

    return sections


def main():
    lines = sys.stdin.read().splitlines()
    if not any(line.strip() for line in lines):
        print("error: no commit subjects provided on stdin", file=sys.stderr)
        sys.exit(1)

    sections = categorize(lines)
    json.dump(sections, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
