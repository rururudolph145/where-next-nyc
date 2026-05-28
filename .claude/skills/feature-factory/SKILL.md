---
name: feature-factory
description: |
  Orchestrate a full backend feature build using the six-agent chain:
  codebase-researcher → story-writer → (human approval) → spec-writer
  → (human approval) → backend-builder → test-verifier →
  implementation-validator → (loop on critical findings) →
  (human PR review) → PBI tracker update.

  TRIGGER when the user asks to "build a feature", "ship a feature",
  "implement a feature", "run the feature factory", "run the full chain",
  or "run the feature pipeline". Also trigger on "/feature-factory".

  DO NOT TRIGGER for single-agent tasks (e.g., "just write a story",
  "just run the validator") — invoke that agent directly instead.
---

# Feature Factory

End-to-end orchestration of six subagents to take a feature from rough
idea to merge-ready implementation, with two human approval gates.
Frontend work is out of scope today (the frontend-builder agent does
not yet exist) — this skill builds backend-only features. If the brief
contains frontend changes, surface them to the user as deferred work.

## When to Use

Use this skill when the user wants to build a backend feature
end-to-end and expects the chain to run in sequence. The user must
be available for two approval gates (story and brief) and one final
PR review.

Avoid when:
- The user wants only one agent (e.g., just a story, just validation).
  Call that agent directly.
- There is no rough feature description to start from. Ask for one
  first.
- The user wants to skip approval gates. Tell them this skill always
  pauses for approval — they can call agents directly to bypass.
- The feature is frontend-only. The chain has no frontend builder
  today; stop and ask the user how they want to proceed.

## Inputs Required

Before starting the chain, confirm you have:
1. A rough feature description from the user (one sentence is enough).
2. Confirmation the user is available for approval gates.

If either is missing, ask before invoking any agent.

## Chain Overview

```
┌─────────────────────┐
│ codebase-researcher │  ─► exploration findings
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   story-writer      │  ─► user story + acceptance criteria
└──────────┬──────────┘
           ▼
       ┌───────┐
       │ HUMAN │  approve / changes / reject
       │ GATE  │  (loop on "changes")
       └───┬───┘
           ▼
┌─────────────────────┐
│   spec-writer       │  ─► technical brief
└──────────┬──────────┘
           ▼
       ┌───────┐
       │ HUMAN │  approve / changes / reject
       │ GATE  │  (loop on "changes")
       └───┬───┘
           ▼
┌─────────────────────┐
│  backend-builder    │  ─► backend code + unit tests
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   test-verifier     │  ─► acceptance tests
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│implementation-      │  ─► gap report (critical / important / minor)
│ validator           │
└──────────┬──────────┘
           ▼
    ┌──────────────┐
    │ Critical     │  yes ─► route to backend-builder ─► test-verifier
    │ findings?    │         ─► validator (max 3 loops)
    └──────┬───────┘
           ▼ no
       ┌───────┐
       │ HUMAN │  final PR review
       │ GATE  │
       └───┬───┘
           ▼
┌─────────────────────┐
│  PBI tracker        │  ─► mark shipped, link commit, commit + push
│  update             │
└─────────────────────┘
```

## Workflow

### Step 1 — codebase-researcher
Invoke the `codebase-researcher` agent. Pass the user's rough feature
description. The agent returns a read-only summary of the relevant
code area (files, patterns, constraints).

**On failure** (agent reports inputs too thin): ask the user for
the missing context, then re-invoke. Do not proceed without findings.

### Step 2 — story-writer
Invoke the `story-writer` agent. Pass:
- The user's rough feature description.
- The codebase-researcher's findings (verbatim, do not summarise).

The agent returns a user story with acceptance criteria, edge cases,
and out-of-scope items.

### Step 3 — HUMAN GATE: approve the story
Show the user the full story document. Use `AskUserQuestion` with
exactly these three options:

- **Approve** — continue to spec-writer.
- **Request changes** — collect specific feedback as free text via
  the question's "Other" path, then re-invoke `story-writer` with:
  - The original rough description.
  - The codebase-researcher findings.
  - The previous story (so the agent can iterate, not start over).
  - The user's feedback as new input.

  Then loop back to Step 3 with the new story.
- **Reject** — stop the chain. Produce a closing summary listing:
  - What the codebase-researcher found.
  - Why the story was rejected (the user's reason, if given).
  - No story is saved. Tell the user they can restart with a
    different framing.

Keep looping the "Request changes" branch until the user picks
Approve or Reject.

### Step 4 — spec-writer
Invoke the `spec-writer` agent. Pass:
- The approved user story (verbatim).
- The codebase-researcher's findings.
- A pointer to CLAUDE.md and `.claude/rules/`.

The agent returns a technical brief with: Summary, Data Model,
Process Flow, API Changes, Frontend Changes, Tests Required,
Risks & Open Questions, Files That Will Change.

**Frontend handling**: if the brief's "Frontend Changes" section is
anything other than "None.", surface it to the user before the
approval gate with a note: "Frontend work is deferred — this chain
builds only the backend half. The frontend changes will be listed
as out-of-scope at PR review time."

### Step 5 — HUMAN GATE: approve the brief
Show the user the full brief. Use `AskUserQuestion` with exactly
these three options:

- **Approve** — continue to backend-builder.
- **Request changes** — collect specific feedback as free text via
  the question's "Other" path, then re-invoke `spec-writer` with:
  - The approved story.
  - The codebase-researcher findings.
  - The previous brief.
  - The user's feedback as new input.

  Then loop back to Step 5 with the new brief.
- **Reject** — stop the chain. Keep the approved story committed
  to memory and tell the user: "The story is approved; you can
  resume later with a different technical approach by re-invoking
  spec-writer directly."

Keep looping the "Request changes" branch until the user picks
Approve or Reject.

### Step 6 — backend-builder
Invoke the `backend-builder` agent. Pass:
- The approved technical brief.
- The codebase-researcher's findings.
- A pointer to CLAUDE.md and `.claude/rules/`.

The agent implements backend code + unit tests, runs ruff/mypy/pytest,
and returns: Files Changed, Patterns Reused, Tests Added, Gate Results,
Suggested CLAUDE.md Additions, Dependency Needs.

**On gate failure** (anything red): report the failure to the user
verbatim and ask whether to retry, route to a fixer agent, or stop.
Do not silently proceed.

**On dependency block**: if the agent reports a Dependency Needs
entry, surface it to the user and pause. The skill does not add
dependencies — the user decides.

### Step 7 — test-verifier
Invoke the `test-verifier` agent. Pass:
- The approved user story.
- The approved technical brief.
- The backend-builder's summary.

The agent writes acceptance tests, runs them, and returns: Files
Changed, Acceptance Criteria Coverage table, Test Results, Criteria
That Could Not Be Covered (if any), Suggested Test Conventions.

**If "Criteria That Could Not Be Covered" is non-empty**: do NOT
proceed to the validator. Report the uncoverable criteria to the
user and ask whether to route back to backend-builder.

### Step 8 — implementation-validator
Invoke the `implementation-validator` agent. Pass:
- The approved user story.
- The approved technical brief.
- The test-verifier's report.
- The list of files changed across Steps 6–7.

The agent returns: Scope of This Review, Critical findings, Important
findings, Minor findings, Recommended Next Agent.

### Step 9 — Critical-finding loop
Examine the validator's "Critical" section.

- **No critical findings**: continue to Step 10.
- **Critical findings exist**: route based on which files they touch:
  - Backend findings → re-invoke `backend-builder` with the findings
    as new input.
  - Test-only findings → re-invoke `test-verifier`.

  After the fixer agent reports done, re-run `test-verifier`
  (Step 7) and `implementation-validator` (Step 8). Loop until
  critical findings is empty, OR a hard limit of **three loop
  iterations** is hit — at which point stop and escalate to the
  user with the latest validator report.

### Step 10 — HUMAN GATE: final PR review
Produce a consolidated summary with:
- The approved story (one-line: "As a... I want... so that...").
- The approved brief (Summary section only).
- Files Changed across all builders.
- Acceptance Criteria Coverage table from test-verifier.
- Validator's Important and Minor findings (Critical should be empty
  by this point).
- Gate Results from the latest builder run.
- Any deferred frontend work surfaced at Step 4.

Use `AskUserQuestion` with two options:
- **Open PR** — continue to Step 11 (tracker update). After the
  tracker is updated, hand off to the user (or to `/commit` +
  `gh pr create` if the user has those commands).
- **More work needed** — collect free-text feedback and ask which
  agent should pick it up.

### Step 11 — PBI tracker update

The user has repeatedly pointed out that the tracker drifts when this
chain ends at Step 10. This step is **mandatory** when the user
selects "Open PR" at Step 10 and there is a matching PBI in the
tracker. Skip only when there is no PBI for the feature (e.g., a
one-off bugfix).

Do all of this without asking the user — it is a routine bookkeeping
step:

1. **Find the PBI entry.** Look in
   `implementation_checklist/pbi_tracker.md` for the PBI the feature
   implements. If you cannot find one, ask the user which PBI this
   covers (or whether to skip the tracker update entirely).
2. **Mark the PBI shipped.** In both places the PBI appears:
   - The Activity-section checklist row: change `- [ ] **PBI N — …**`
     to `- [x] **PBI N — …** — *shipped in `<short-sha>`*`. If you
     added meaningful context during the build (e.g. follow-up
     commits, new test counts that differ from the original acceptance
     criteria), append a one-sentence note.
   - The Priority Order table row: change `| [ ]` to `| [x]`, wrap
     the PBI ID and rationale in `~~…~~` strikethrough, and append
     `— *shipped in `<short-sha>`*` to the rationale cell.
3. **Handle gitignored trackers.** Many repos (this one included)
   gitignore `implementation_checklist/`. Use `git add -f
   implementation_checklist/pbi_tracker.md` — plain `git add` will
   silently skip it.
4. **Commit and push the tracker.** Use a single dedicated commit:

   ```
   📝 docs(tracker): mark PBI N shipped (<short-sha>)
   ```

   followed by a one-line body describing what landed (file count,
   test count). Push to the same branch the feature was built on.

5. **Report back.** Once pushed, tell the user one line:
   `Tracker updated and pushed (commit <short-sha-of-tracker-commit>).`

**Do not** modify the tracker mid-chain (e.g. while the validator is
still looping) — only after the user picks "Open PR" at Step 10.

**Do not** create new PBI entries during this step. If the work
revealed missing tracker entries (e.g. follow-up tickets from the
validator's Important findings), surface them to the user at Step 10
and ask whether to add them; do not silently invent PBIs.

## Output Discipline

Throughout the chain:
- After every agent invocation, briefly summarise what the agent
  returned (one or two sentences) before moving to the next step.
  This keeps the user oriented.
- Do NOT paraphrase agent outputs into your own words at approval
  gates — show the user the full document.
- Persist these artefacts in memory (one short note per artefact)
  so the user can refer back:
  - Approved story (Step 3 outcome)
  - Approved brief (Step 5 outcome)
  - Validator's final report (Step 8 / Step 9 final pass)

## Failure Modes & Recovery

- **Agent disagrees with itself across runs.** If story-writer or
  spec-writer produces contradictory output across "Request changes"
  iterations, surface the contradiction to the user and ask which
  version to keep. Do not silently pick one.
- **Validator finds a Critical gap the builder cannot fix.** After
  three loop iterations, stop and escalate. Do not loop forever.
- **Test-verifier reports uncoverable criteria.** Treat as a Critical
  gap and route back to backend-builder (Step 9 logic).
- **User goes idle at an approval gate.** Do not assume — wait. The
  chain is paused.

## Anti-Patterns

- **Skipping a gate because the user "seems busy".** Always ask.
- **Auto-approving "looks fine" outputs.** The skill never approves
  on the user's behalf.
- **Editing production files directly inside this skill.** This skill
  only orchestrates agents — it does not Read/Edit/Write production
  code. The one exception is the PBI tracker update at Step 11, which
  is a bookkeeping edit, not a production-code change.
- **Running agents in parallel that should be sequential.** The chain
  is strictly sequential. Parallel branches break the
  story→brief→build dependency.
- **Calling agents the user did not ask for.** This chain is exactly
  six agents. Do not introduce others (including frontend-builder —
  which does not exist yet).
- **Implementing frontend changes inline.** Frontend work is deferred;
  surface it as out-of-scope at Step 4 and at PR review.

## References
- `.claude/agents/codebase-researcher.md`
- `.claude/agents/story-writer.md`
- `.claude/agents/spec-writer.md`
- `.claude/agents/backend-builder.md`
- `.claude/agents/test-verifier.md`
- `.claude/agents/implementation-validator.md`
- `CLAUDE.md` and `.claude/rules/*.md`
