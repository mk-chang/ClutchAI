# Claude Skills & Coding Best Practices

Reference guide for working with Claude Code on ClutchAI. Covers the available slash commands, workflow skills, and coding standards Claude follows in this project.

---

## Slash Commands (Project-Local)

These live in `.claude/commands/` and are specific to ClutchAI.

| Command | Purpose |
|---------|---------|
| `/startup` | Load all context files, summarize state, and orient Claude at the start of a session |
| `/wrapup` | Write session file, update context/, run tests, commit, and push at session end |

Run `/startup` at the beginning of every session. Run `/wrapup` before closing.

---

## Superpowers Skills

Global skills installed via the superpowers plugin. Invoke with the `Skill` tool or by typing the skill name. These govern *how* Claude approaches tasks — they override default behavior.

### Workflow / Process

| Skill | When to Use |
|-------|-------------|
| `superpowers:brainstorming` | Before any creative work — new features, components, behavior changes. Explores intent and design *before* implementation. |
| `superpowers:writing-plans` | When you have a spec or requirements for a multi-step task, before touching code. |
| `superpowers:executing-plans` | When running a written plan in a separate session with review checkpoints. |
| `superpowers:subagent-driven-development` | When executing plans with independent tasks in the current session (parallelizes work via subagents). |
| `superpowers:dispatching-parallel-agents` | When 2+ tasks are independent and can run simultaneously. |

### Implementation Quality

| Skill | When to Use |
|-------|-------------|
| `superpowers:test-driven-development` | Before writing any implementation code. Write the failing test first, always. |
| `superpowers:systematic-debugging` | When encountering any bug, test failure, or unexpected behavior. Diagnose before proposing fixes. |
| `superpowers:verification-before-completion` | Before claiming work is done. Run verification commands and confirm output — no success claims without evidence. |

### Code Review

| Skill | When to Use |
|-------|-------------|
| `superpowers:requesting-code-review` | When completing tasks or implementing major features. |
| `superpowers:receiving-code-review` | When receiving review feedback. Requires technical verification, not blind agreement. |
| `superpowers:finishing-a-development-branch` | When implementation is complete and tests pass — guides merge/PR/cleanup decisions. |

### Infrastructure

| Skill | When to Use |
|-------|-------------|
| `superpowers:using-git-worktrees` | Before feature work that needs isolation from current workspace. |
| `superpowers:writing-skills` | When creating or editing Claude skills — applies TDD to skill documentation. |
| `superpowers:using-superpowers` | Loaded automatically at session start to establish skill discovery rules. |

### Utility Skills

| Skill | What It Does |
|-------|-------------|
| `update-config` | Modify Claude Code settings.json — permissions, hooks, env vars, automated behaviors. |
| `keybindings-help` | Customize keyboard shortcuts in `~/.claude/keybindings.json`. |
| `fewer-permission-prompts` | Scans transcripts and adds allowlist entries to reduce permission prompts. |
| `loop` | Run a command on a recurring interval (e.g., `/loop 5m /wrapup`). |
| `schedule` | Create scheduled remote agents that run on a cron schedule. |
| `claude-api` | Build and debug Anthropic SDK apps; handles prompt caching, model versions, tool use. |
| `run` | Launch and drive the app to confirm a change works in the real UI. |
| `verify` | Run the app and observe behavior to confirm a fix or feature actually works. |
| `simplify` | Review changed code for reuse and quality, then fix issues found. |
| `init` | Initialize a new CLAUDE.md file with codebase documentation. |
| `review` | Review a pull request. |
| `security-review` | Security review of pending changes on the current branch. |

---

## Skill Priority Rules

When multiple skills could apply, use this order:

1. **Process skills first** — brainstorming, debugging, TDD. These determine *how* to approach the task.
2. **Implementation skills second** — domain-specific guidance for execution.

Examples:
- "Let's build X" → brainstorming first, then TDD + implementation skills
- "Fix this bug" → systematic-debugging first, then domain skills
- "This task has independent subtasks" → dispatching-parallel-agents

The rule: if there's even a 1% chance a skill applies, invoke it before doing anything else.

---

## Coding Standards (from CLAUDE.md)

These apply to all code written in this project:

**Minimalism**
- Write minimal, readable code — no unnecessary abstractions, features, or complexity beyond what's asked
- Don't add features "for later" — three similar lines beats a premature abstraction
- No half-finished implementations

**Comments**
- No comments by default
- Only comment when the *why* is genuinely non-obvious: hidden constraints, subtle invariants, specific workaround for a known bug
- Never explain *what* the code does — well-named identifiers do that

**Error Handling**
- Don't add error handling for impossible cases
- Validate only at system boundaries (user input, external APIs)
- Trust internal code and framework guarantees

**Security**
- No command injection, XSS, SQL injection, or OWASP Top 10 violations
- Immediately fix any insecure code noticed during implementation

**Testing**
- Always verify implementations with tests or examples (TDD is the standard)
- Run `pytest` before committing

**Scope**
- Don't refactor, abstract, or clean up beyond what the task requires
- A bug fix doesn't need surrounding cleanup
- No backwards-compatibility shims for removed code — just delete it

**Communication**
- Ask for clarification if unclear
- Verify with the user before making architectural or logic changes

---

## Skill Invocation Rules

These rules from `superpowers:using-superpowers` govern when Claude must use skills:

- Skills must be invoked **before any response** — even clarifying questions
- If a skill applies, using it is mandatory, not optional
- Skills override default Claude behavior (but user instructions in CLAUDE.md override skills)

Red flags that mean Claude is rationalizing away skill use:
- "This is just a simple question"
- "I need more context first"
- "This doesn't need a formal skill"
- "I know what that means" — knowing the concept ≠ using the skill
