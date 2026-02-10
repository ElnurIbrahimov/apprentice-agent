# Claude Code — Prompt Engineering Techniques

**Tags:** claude-code, prompting, techniques, best-practices
**Created:** 2025
**Category:** claude-code

---

## Claude-Specific Prompting Patterns

### 1. XML Tags for Structure
Claude responds exceptionally well to XML-tagged instructions:
```xml
<instructions>
You are a code reviewer. Focus on security issues.
</instructions>

<code>
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}'"
    ...
</code>

<output_format>
List each issue with severity (HIGH/MEDIUM/LOW) and fix suggestion.
</output_format>
```

### 2. Chain-of-Thought
Explicitly ask Claude to think step by step:
```
Analyze this error. Think through it step by step:
1. What is the error message saying?
2. What could cause this?
3. What's the most likely root cause?
4. What's the fix?
```

### 3. Few-Shot Examples
Provide examples of desired input/output:
```
Convert these function names to snake_case:

Example: getUserName -> get_user_name
Example: setMaxRetries -> set_max_retries

Now convert: fetchAllDataFromAPI
```

### 4. Role Assignment
```
You are a senior Python developer with 15 years of experience.
Review this code for performance issues, focusing on:
- O(n^2) or worse algorithms
- Unnecessary memory allocations
- Missing caching opportunities
```

### 5. Constraint Setting
```
Requirements:
- Use only Python standard library (no pip packages)
- Must work on Python 3.8+
- Max 50 lines of code
- Include type hints
- Include docstrings
```

### 6. Output Format Control
```
Respond in exactly this format:
DIAGNOSIS: <one sentence>
ROOT_CAUSE: <one sentence>
FIX: <code block>
TESTING: <how to verify the fix>
```

---

## Claude Code-Specific Prompting

### Be Specific About Files
Bad: "Fix the bug"
Good: "Fix the authentication bug in src/auth/login.py where users can't log in with special characters in passwords"

### Reference Files with @
Use `@filename` to pull file content into context:
```
Look at @src/auth/login.py and @src/auth/tests/test_login.py
The login function fails when passwords contain quotes. Fix it and update the tests.
```

### Multi-Step Tasks
Break complex tasks into explicit steps:
```
1. First, read all files in src/api/routes/
2. List all endpoints that don't have error handling
3. Add try/except blocks with proper HTTP error responses
4. Run the tests to make sure nothing broke
```

### CLAUDE.md as Persistent Prompt
Put repeating instructions in CLAUDE.md:
```markdown
# Code Style
- Always use type hints
- Use dataclasses, not dicts, for structured data
- Prefer f-strings over .format()
- Max line length: 100 characters

# Testing
- Every new function needs a test
- Use pytest, not unittest
- Mock external services
```

---

## Anti-Patterns to Avoid

### Don't Be Vague
Bad: "Make it better"
Good: "Reduce the time complexity from O(n^2) to O(n log n)"

### Don't Over-Constrain
Bad: "Write exactly 47 lines of code using only map/filter/reduce"
Good: "Write concise, functional-style code"

### Don't Ask for Everything at Once
Bad: "Build a complete REST API with auth, database, tests, docs, CI/CD, and deployment"
Good: "Set up the database models for users and sessions. Use SQLAlchemy with PostgreSQL."

### Don't Repeat What's in CLAUDE.md
If it's in CLAUDE.md, Claude already knows. Don't waste tokens repeating it.

---

## Prompting for Different Tasks

| Task | Prompt Pattern |
|------|---------------|
| Bug fix | "Error: [msg]. In [file]. Expected: [X]. Got: [Y]. Fix it." |
| New feature | "Add [feature] to [file]. It should [behavior]. Follow the pattern in [similar file]." |
| Refactor | "Refactor [file] to [goal]. Keep the same public API. Don't change tests." |
| Code review | "Review [file] for [concerns]. Rate severity. Suggest fixes." |
| Research | "Search for [topic]. Save findings to research/[category]/[name].md" |
| Testing | "Write tests for [file]. Cover: [cases]. Use pytest." |
