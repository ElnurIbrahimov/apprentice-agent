---
id: skill_2db881ff9a5d
name: web-design-guidelines
version: 1.0
category: coding
tags: ['imported', 'coding']
trigger_patterns:
  - "web-design-guidelines"
  - "review ui code web interface guidelines compliance"
  - "asked review my ui"
  - "check accessibility"
  - "audit design"
  - "review ux"
  - "check my site against best practices"
success_count: 0
failure_count: 0
total_uses: 0
last_used: null
created_at: 2026-04-13T13:02:46.394212
updated_at: 2026-04-13T13:02:46.394212
---

# web-design-guidelines

## Description

Review UI code for Web Interface Guidelines compliance. Use when asked to "review my UI", "check accessibility", "audit design", "review UX", or "check my site against best practices".

## Procedure

_Source: `D:\Aura\aura_skills\vercel-labs-agent--web-design-guidelines\SKILL.md`_

This skill was imported from a Claude/OpenCode SKILL.md file. To invoke its underlying CLI scripts (if any), look for a `scripts/` sibling folder next to the source path above.

# Web Interface Guidelines

Review files for compliance with Web Interface Guidelines.

## How It Works

1. Fetch the latest guidelines from the source URL below
2. Read the specified files (or prompt user for files/pattern)
3. Check against all rules in the fetched guidelines
4. Output findings in the terse `file:line` format

## Guidelines Source

Fetch fresh guidelines before each review:

```
https://raw.githubusercontent.com/vercel-labs/web-interface-guidelines/main/command.md
```

Use WebFetch to retrieve the latest rules. The fetched content contains all the rules and output format instructions.

## Usage

When a user provides a file or pattern argument:
1. Fetch guidelines from the source URL above
2. Read the specified files
3. Apply all rules from the fetched guidelines
4. Output findings using the format specified in the guidelines

If no files specified, ask the user which files to review.

