---
name: changelog-generator
description: Generates a Keep a Changelog formatted CHANGELOG entry from a release's git commit history, grouping changes into Added, Changed, Fixed, and Removed
version: 1.1.0
allowed-tools:
  - read_file
  - list_directory
  - run_git_log
  - write_file
tags:
  - engineering
  - documentation
  - release-management
author: developer-experience-team
license: MIT
---

# Changelog Generator

Turns the raw commit history of a release into a clean, human-readable CHANGELOG entry that follows the Keep a Changelog convention. The goal is a changelog written for humans reading the release notes, not a dump of git log.

## When to Use This Skill

**Activate when:**
- The user asks to "write a changelog", "draft release notes", or "summarize commits for a release"
- A release or tag is being prepared and the commits since the last tag need to be summarized
- The user provides a commit range and wants it grouped by change type

**Do NOT activate when:**
- The user wants a single conventional commit message for staged changes (use `git-commit-message`)
- The user wants a prose blog-style announcement rather than a structured changelog
- There is no commit history available yet (ask the user to provide a commit range first)

## Required Context

Before proceeding, confirm you have:
- [ ] The target version number for this release (e.g., `1.4.0`)
- [ ] The commit range or the previous tag to diff against (e.g., `v1.3.0..HEAD`)
- [ ] The release date (default to today if not given)
- [ ] Repository read access so commits can be listed

If the version number or commit range is missing, ask the user before continuing.

## Workflow

### Phase 1: Collect commits
1. Use `run_git_log` to list commit subjects for the requested range.
2. If the range is empty, stop and tell the user there are no changes to report.

### Phase 2: Categorize
3. Group each commit into a Keep a Changelog section using its Conventional Commit type. Refer to `./references/keep-a-changelog.md` for the type-to-section
   mapping and the meaning of each section.
4. Drop noise commits (`chore`, `ci`, `build`, merge commits) unless they are user-visible.

### Phase 3: Rewrite for humans
5. Rewrite each entry as a short, past-tense, user-facing sentence. Strip the `type(scope):` prefix and the commit hash.
6. Collapse duplicates and order entries within a section by impact.

### Phase 4: Render
7. Fill in `./assets/changelog-template.md`, replacing every `{{placeholder}}`.
8. Use `write_file` to append the rendered entry to the top of `CHANGELOG.md`.

## Decision Rules

**If a commit has no Conventional Commit type:**
- Infer the section from the subject; if unclear, place it under `Changed`.

**If a change is a breaking change (`!` or `BREAKING CHANGE`):**
- Keep it in its section but prefix the line with `**Breaking:**`.

**If the range contains more than ~50 commits:**
- Summarize related entries into themes instead of listing every commit.

## Constraints

- Do not invent changes that are not present in the commit history.
- Do not include internal-only commits (tooling, CI) in user-facing sections.
- Do not rewrite or amend existing CHANGELOG entries — only prepend the new one.
- Keep each entry to one line; move detail into the commit body, not the changelog.

## Output Format

Produce a single CHANGELOG entry that follows `./assets/changelog-template.md`:

- A version header: `## [VERSION] - DATE`
- One `###` subsection per non-empty category (Added, Changed, Fixed, Removed)
- Bulleted, past-tense, user-facing lines under each subsection

## Resources

- `./references/keep-a-changelog.md` — Used in Phase 2: section definitions and the Conventional-Commit-to-section mapping
- `./assets/changelog-template.md` — Used in Phase 4: the output template to populate
- `./scripts/parse_git_log.py` — Used in Phase 2 when the commit list is large: categorizes commit subjects into sections as JSON