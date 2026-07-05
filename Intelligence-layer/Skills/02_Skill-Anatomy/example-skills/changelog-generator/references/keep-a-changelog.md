# Keep a Changelog — Reference

This reference is loaded on demand during **Phase 2 (Categorize)**. It defines the changelog sections and maps Conventional Commit types to them. It is intentionally kept out of `SKILL.md` so it only enters the context window when categorization actually happens.

## The sections (in order)

A changelog entry lists changes under these headings, in this order. Omit any section that has no entries.

| Section       | Meaning                                                        |
|---------------|----------------------------------------------------------------|
| `Added`       | New features visible to users                                  |
| `Changed`     | Changes in existing behavior                                   |
| `Deprecated`  | Soon-to-be removed features                                    |
| `Removed`     | Features removed in this release                               |
| `Fixed`       | Bug fixes                                                      |
| `Security`    | Vulnerability fixes                                            |

## Conventional Commit → section mapping

| Commit type            | Changelog section | Notes                                    |
|------------------------|-------------------|------------------------------------------|
| `feat`                 | Added             | A new user-facing capability             |
| `fix`                  | Fixed             | A bug fix                                 |
| `perf`                 | Changed           | Performance improvement                  |
| `refactor`             | Changed           | Only if user-visible; otherwise drop     |
| `revert`               | Removed / Changed | Depends on what was reverted             |
| `security` / `fix(sec)`| Security          | Vulnerability remediation                |
| `deprecate`            | Deprecated        | Announce upcoming removal                |
| `docs`, `test`, `style`| (drop)            | Not user-facing                          |
| `chore`, `ci`, `build` | (drop)            | Tooling; exclude unless user-visible     |

## Breaking changes

A commit is breaking if its type carries a `!` (e.g., `feat!:`) or its body contains `BREAKING CHANGE:`. Keep the entry in its normal section, but prefix the line with `**Breaking:**` so it stands out to readers scanning the release.

## Writing style for entries

- Past tense, user-facing: "Added dark mode to settings", not "feat: add dark mode".
- One line per entry; drop the commit hash and the `type(scope):` prefix.
- Describe the effect on the user, not the implementation detail.
