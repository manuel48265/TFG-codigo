# AGENTS.md

Working instructions for any AI agent (Claude Code, Codex, etc.) in this repo.
This is the single source of truth: `CLAUDE.md` only imports this file, don't duplicate it.

## What this repo is

Code repo supporting the TFG (thesis). It's intentionally empty (clean restart,
see `git log`): don't assume any prior structure exists (Python package, research
pipeline, etc.) unless you actually see it in the current working tree.

It's independent from `~/Escritorio/TFG/codigo` (where the literature review lives,
`Estado_del_arte/`). There's no sync between the two folders — don't copy or link content from one to the other unless explicitly asked.

## Conventions

- Commit messages in Spanish, short, imperative mood (e.g. "Cambio de criterio").
- Before assuming a structure/tooling convention (Python, uv, tests, etc.),
  check what's actually in the repo — don't invent it out of thin air.
- If you add new working instructions, edit them here (not in `CLAUDE.md`) so
  they apply equally to any agent.

## General working rules

Behavior guidelines to reduce common LLM problems. Apply them together with
any project-specific instructions added below as this repo grows.

**Tradeoff:** these guidelines favor caution over speed. Use judgment on trivial tasks.

### 1. Think before coding

**Don't assume. Don't hide confusion. Lay out alternatives and their consequences.**

Before implementing:
- State your assumptions explicitly. If in doubt, ask.
- If multiple interpretations exist, present them; don't silently pick one.
- If a simpler approach exists, say so.
- If something is unclear, stop, name what's confusing, and ask.

### 2. Simplicity first

**The minimum code that solves the problem. Nothing speculative.**

- No features beyond what was requested.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't asked for.
- No error handling for impossible scenarios.
- If you wrote 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer call this overcomplicated?" If yes, simplify it.

### 3. Surgical changes

**Modify only what's strictly necessary. Clean up only what you touched.**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Respect existing style, even if you'd do it differently.
- If you spot unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions your own change orphaned; leave
  pre-existing dead code alone unless asked.

Litmus test: every changed line should trace back to the user's request.

### 4. Goal-oriented execution

**Define success criteria. Iterate and verify in a loop until confirmed.**

Turn tasks into verifiable goals ("fix the bug" → "write a test that
reproduces it and make it pass"). For multi-step tasks, lay out a short plan
with a verification step per item.

## Graphify

This project has (or will have) a knowledge graph at `graphify-out/` with god
nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill/instructions
before doing anything else.

- For codebase questions, first run `graphify query "<question>"` when
  `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for
  relationships and `graphify explain "<concept>"` for focused concepts —
  these return a scoped subgraph, usually much smaller than
  `GRAPH_REPORT.md` or raw grep output.
- A dirty `graphify-out/` after hooks or incremental updates is expected and
  not a reason to skip graphify. Only skip it if the task is about stale or
  incorrect graph output, or the user explicitly says not to use it.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead
  of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or
  when query/path/explain don't surface enough context.
- After modifying code, run `graphify update .` to keep the graph current
  (AST-only, no API cost).
