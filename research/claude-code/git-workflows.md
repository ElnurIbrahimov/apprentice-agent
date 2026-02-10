# Claude Code — Git Workflows

**Tags:** claude-code, git, workflows, ci-cd
**Created:** 2025
**Category:** claude-code

---

## Commit Workflow

### How Claude Code Creates Commits
1. Runs `git status` to see changes
2. Runs `git diff` to analyze staged/unstaged changes
3. Checks `git log` for commit message style
4. Drafts a descriptive commit message
5. Stages specific files (avoids `git add -A`)
6. Commits with HEREDOC format for proper formatting
7. Verifies with `git status` after commit

### Commit Message Format
```bash
git commit -m "$(cat <<'EOF'
Short summary of changes (imperative mood)

- Detail 1
- Detail 2

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

### Best Practices
- Never use `--no-verify` (respects pre-commit hooks)
- Never amend unless explicitly asked
- Stage specific files, not `git add .`
- Never commit `.env`, credentials, or secrets
- If hook fails: fix issue, re-stage, NEW commit (don't amend)

---

## Pull Request Workflow

### How Claude Code Creates PRs
1. Check `git status` and `git diff`
2. Check if branch tracks remote, needs pushing
3. Analyze ALL commits since branching (not just latest)
4. Draft PR title (<70 chars) and body
5. Push with `-u` flag if needed
6. Create PR via `gh pr create`

### PR Format
```bash
gh pr create --title "Add user authentication" --body "$(cat <<'EOF'
## Summary
- Added JWT-based auth middleware
- Created login/register endpoints
- Added password hashing with bcrypt

## Test plan
- [ ] Run auth test suite
- [ ] Test login with valid/invalid credentials
- [ ] Verify JWT token expiration

Generated with Claude Code
EOF
)"
```

---

## Branch Management

### Common Patterns
```bash
# Create feature branch
git checkout -b feature/user-auth

# Work across branches
git stash && git checkout main && git pull && git checkout -

# Merge with main
git fetch origin && git merge origin/main
```

### Git Worktrees (Parallel Work)
```bash
# Create worktree for parallel feature
git worktree add ../project-feature feature/new-thing

# Work on both simultaneously
# main in /project/
# feature in /project-feature/

# Clean up when done
git worktree remove ../project-feature
```

---

## CI/CD Integration

### Using Claude Code in Pipelines
```bash
# Headless mode for CI
claude -p "run tests and fix any failures" \
  --output-format json \
  --max-turns 10 \
  --permission-mode bypassPermissions

# Code review in CI
claude -p "review the changes in this PR for bugs and security issues" \
  --output-format json
```

### GitHub Actions Example
```yaml
- name: AI Code Review
  run: |
    claude -p "Review changes since ${{ github.event.pull_request.base.sha }}" \
      --output-format json > review.json
```

---

## Safety Rules
- NEVER force push to main/master
- NEVER use `git reset --hard` without user confirmation
- NEVER use `-i` flag (interactive mode not supported)
- NEVER skip hooks with `--no-verify`
- NEVER update git config
- Always create NEW commits, don't amend previous ones
- Always use `gh` CLI for GitHub operations (PRs, issues, etc.)
