---
name: github-sync
description: >
  Handles all Git/GitHub push and pull workflows from within Claude Code.
  Use this skill whenever the user wants to: push local changes to GitHub,
  pull or sync from a remote, resolve merge conflicts, stage and commit with
  proper messages, set upstream tracking, force-push safely, rebase before
  push, or do anything involving `git push`, `git pull`, `git fetch`, or
  `git remote`. Also trigger for phrases like "sync my repo", "push my
  changes", "pull latest", "update my branch", "open a PR", "commit and push",
  "create a pull request", or "resolve conflicts". Do NOT wait for the user to
  say "git" explicitly — intent-based phrases are sufficient to trigger this skill.
---

# GitHub Sync Skill

Handles the full push/pull lifecycle for Git repositories in Claude Code.
Covers: status checks, conflict detection, staged commits, push/pull strategy
selection, PR creation, and upstream configuration.

---

## Phase 1: Situational Awareness

Before any push or pull, always run a state snapshot:

```bash
git status --short
git remote -v
git branch -vv          # shows tracking info and divergence
git log --oneline -5    # recent commit history
```

Derive the working state from this output:

| Condition | Action |
|---|---|
| Untracked / modified files | Stage before push |
| Behind remote | Pull/rebase first |
| Ahead of remote | Push (standard case) |
| Diverged | Resolve divergence strategy |
| No upstream set | Set upstream on first push |
| Dirty working tree + pull needed | Stash → pull → unstash |

---

## Phase 2: Pull Strategy

### Default Pull (fast-forward or merge)
```bash
git pull origin <branch>
```
Use when: collaborating on a shared branch, merge commits are acceptable.

### Rebase Pull (preferred for clean history)
```bash
git pull --rebase origin <branch>
```
Use when: user wants linear history or is working on a feature branch solo.

### Stash-Pull-Pop (dirty working tree)
```bash
git stash
git pull --rebase origin <branch>
git stash pop
```
Use when: `git status` shows uncommitted changes and a pull is needed.

### Fetch Only (inspect before merging)
```bash
git fetch origin
git diff HEAD origin/<branch>   # inspect delta before committing to merge
```
Use when: user wants to review remote changes before applying.

---

## Phase 3: Conflict Detection and Resolution

After a pull or rebase, check for conflicts:

```bash
git status --short | grep -E "^(UU|AA|DD|UA|AU|DU|UD)"
```

If conflicts exist:
1. Show the conflicted files clearly to the user.
2. Open each file and identify the `<<<<<<`, `=======`, `>>>>>>>` markers.
3. Present the conflict in a structured diff view — show both `HEAD` and incoming changes side by side.
4. Ask the user which version to keep, or whether to merge manually.
5. After resolution:
```bash
git add <resolved-file>
git rebase --continue    # if rebase
# OR
git commit               # if merge
```

**Never auto-resolve conflicts silently.** Always surface them to the user.

---

## Phase 4: Stage and Commit

### Stage Changes

```bash
# Stage all tracked + untracked
git add -A

# Stage only modified tracked files
git add -u

# Stage specific files
git add <file1> <file2>

# Interactive staging (hunk-level)
git add -p
```

Prefer `-u` for commits to avoid accidentally staging unrelated files.
Use `git diff --cached` to verify what's staged before committing.

### Commit Message Format

Follow **Conventional Commits** by default:

```
<type>(<scope>): <short summary>

[optional body — what changed and why, not how]

[optional footer: BREAKING CHANGE, closes #issue]
```

**Types:**
- `feat` — new feature
- `fix` — bug fix
- `refactor` — code restructuring, no behavior change
- `chore` — build/tooling/config changes
- `docs` — documentation only
- `test` — test additions/modifications
- `perf` — performance improvement

**Examples:**
```
feat(auth): add OAuth2 GitHub login
fix(api): correct null handling in user endpoint
chore(ci): update GitHub Actions runner to ubuntu-22.04
```

If no scope is obvious, omit the parentheses: `feat: add dark mode`.

---

## Phase 5: Push

### Standard Push

```bash
git push origin <branch>
```

### First Push (set upstream)

```bash
git push -u origin <branch>
```

Always use `-u` on the first push to a new branch. Verify with `git branch -vv` after.

### Force Push (with safety guard)

**Never use `git push --force` on `main` or `master`.** Use the lease variant:

```bash
git push --force-with-lease origin <branch>
```

`--force-with-lease` fails if someone else has pushed to the branch since your last fetch, preventing accidental overwrites. Only suggest force-push when:
- User has rebased a feature branch
- User needs to amend the last commit on their own branch

Before force-pushing, always confirm with the user: "This will rewrite remote history on `<branch>`. Are you sure?"

### Push with Tags

```bash
git push origin --tags
```

---

## Phase 6: Upstream Tracking

Check if upstream is configured:
```bash
git branch -vv
```

If output shows `[origin/<branch>]` → tracking is set. If blank → set it:
```bash
git branch --set-upstream-to=origin/<branch> <branch>
```

---

## Phase 7: PR Creation (GitHub CLI)

If `gh` is available and the user wants to open a PR:

```bash
# Check if gh is installed
gh --version

# Create PR interactively
gh pr create --title "<title>" --body "<body>" --base main --head <branch>

# Create PR and open in browser
gh pr create --web
```

If `gh` is not installed, provide the GitHub URL pattern:
```
https://github.com/<owner>/<repo>/compare/<base>...<head>
```

---

## Phase 8: Safety Checklist

Run through this before any push to `main`, `master`, or `release/*`:

```bash
# 1. Confirm you're not on a protected branch by accident
git branch --show-current

# 2. Check for any accidentally staged secrets
git diff --cached | grep -iE "(api_key|secret|password|token|private_key)"

# 3. Run tests if a test script is available
if [ -f "Makefile" ]; then make test; fi
if [ -f "package.json" ]; then npm test -- --passWithNoTests; fi

# 4. Ensure no debug artifacts
git diff --cached | grep -E "(console\.log|debugger|pdb\.set_trace|breakpoint\(\))"
```

Flag any hits from step 2 or 4 to the user before proceeding.

---

## Canonical Workflows

### Workflow A: Push new feature branch

```bash
git checkout -b feat/my-feature
# ... make changes ...
git add -u
git diff --cached                     # verify staged content
git commit -m "feat(scope): summary"
git push -u origin feat/my-feature
gh pr create --base main --title "feat(scope): summary"
```

### Workflow B: Sync with main before merging

```bash
git fetch origin
git rebase origin/main
# resolve any conflicts
git push --force-with-lease origin feat/my-feature
```

### Workflow C: Hotfix to main

```bash
git checkout main
git pull --rebase origin main
# make fix
git add -u
git commit -m "fix(scope): patch critical bug"
git push origin main
```

---

## Error Reference

| Error | Cause | Fix |
|---|---|---|
| `rejected - non-fast-forward` | Remote is ahead of local | `git pull --rebase` then retry push |
| `src refspec does not match any` | Branch name mismatch | Check `git branch` for exact name |
| `Permission denied (publickey)` | SSH key not registered | Check `~/.ssh/` or switch to HTTPS remote |
| `Updates were rejected - force-with-lease` | Someone else pushed since your last fetch | `git fetch` + rebase + retry |
| `CONFLICT (content)` | Merge/rebase conflict | Follow Phase 3 protocol |
| `fatal: no upstream configured` | First push without -u | Use `git push -u origin <branch>` |

---

## Notes

- Always prefer `--rebase` over default merge-pull for feature branches to keep history linear.
- Never auto-commit without showing the user the staged diff first unless explicitly told to do so.
- Default branch is `main`; fall back to `master` only if `git remote show origin` indicates otherwise.
- For monorepos, check if `git sparse-checkout` is active before staging with `-A`.
