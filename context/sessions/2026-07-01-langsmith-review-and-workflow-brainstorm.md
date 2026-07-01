# Session: LangSmith Conversation Review + Multi-Agent Workflow Gap Brainstorm

## What happened

Two parts: (1) reviewed the last 2 real ClutchAI conversations in LangSmith and assessed whether the multi-agent workflow made sense, (2) started `/superpowers:brainstorming` to design fixes for 3 gaps found during that review. Session was paused mid-investigation at user request before a design doc was written — **no code was changed this session**.

## Part 1: LangSmith review

**Finding the project:** `clutch_ai` didn't exist as a LangSmith project name. The user had renamed a project to `clutch-ai` (hyphen, not underscore) — it's the same `project_id` (`a657cf10-b81e-41cd-a1dd-e46078d11fc4`) that used to show up as `pr-roasted-deliberation-80`. Searching `list_projects` with `project_name="clutch"` found it; exact `"clutch_ai"` did not match.

**Conversations reviewed:**
- 2026-07-01 00:50–00:52 — "best available free agents" → "which fit my team's needs" (2-turn thread, trace ids `019f1b27-...` and `019f1b29-...`)
- 2026-06-30 03:32 — "best waiver wire pickups" (1-turn, trace id `019f1695-...`)

**Verdict:** workflow is broadly sound (Supervisor → research agents → Analysis Agent → response; roster/league context correctly injected into the supervisor's system prompt so follow-up questions don't need re-fetching), but 3 real gaps surfaced:

1. **League-key rediscovery waste** — Yahoo Fantasy Agent burns ~25K tokens / ~19s per call rediscovering the user's league via `get_current_user` → `get_all_yahoo_fantasy_game_keys` → `get_user_leagues_by_game_key` → (free-agent/roster tool), even though the league_key is already sitting in the supervisor's `USER CONTEXT` block.
2. **Inconsistent Analysis Agent routing** — nearly identical queries ("best free agents" vs "best waiver wire pickups") took different paths: one called `call_analysis_agent` to synthesize a recommendation, the other answered directly off raw Yahoo tool output. No rule in the prompt explains the difference.
3. **Stats agent never wired into recommendations** — confirmed via `fetch_runs` with `filter=eq(name, "call_statistic_agent")` on the free-agent trace: **zero matches**. The new `PlayerStatsDbTool` (get_recent_form, get_schedule_density, get_season_trends, query_stats_db — added in commits `c8cd001`, `ebb19e8`, `07b5b5c`) isn't influencing free-agent recommendations at all; the Analysis Agent reasons only from Yahoo ownership % + LLM background knowledge.

## Part 2: Brainstorming — code investigation findings

Ran two Explore-agent passes plus direct reads to ground the design. Key facts, with file:line citations:

**`agents/multi_agent/base_agent.py`** — `BaseAgent.__init__` (lines 37-46) already accepts `user_context: Optional[str] = None`, and `_enhance_system_prompt()` (lines 191-205) is called universally for every subclass to inject `=== USER CONTEXT ===` into that agent's own system prompt. **This means Gap 1's fix requires zero new plumbing** — the mechanism already exists, it's just not wired up for two of the four sub-agents.

**`agents/multi_agent/multi_agent_system.py`** (lines 150-184) — of the 4 sub-agents constructed here, only `FantasyAnalystAgent` (line 178-184) and `SupervisorAgent` (line 187-196) are given `user_context=user_context`. `YahooFantasyAgent` (150-157) and `StatisticAgent` (159-166) are NOT — so neither can see the already-known league_key/team_key/roster in its own system prompt.

**`agents/multi_agent/supervisor.py`** (lines 110-209) — all 4 delegating tools (`call_yahoo_fantasy_agent`, `call_statistic_agent`, `call_news_agent`, `call_analysis_agent`) accept only free-text `query: str` (plus `research_data: str` for analysis). No structured params like `league_key` are ever threaded through, even though the supervisor's own system prompt already has it.

**`agents/multi_agent/analyst_agent.py`** — Analysis Agent's only tool is `search_knowledge_base` (RAG). It has **no ability to call other agents at all** — it's a single-shot LLM call over the `research_data` string the supervisor hands it (`invoke(query, research_data, **kwargs)`, lines 119-147). This confirms wiring in stats data must happen supervisor-side (call `call_statistic_agent` before `call_analysis_agent`), not by giving the Analysis Agent its own tools — that would be a much bigger architecture change.

**`config/multiagent_config.yaml`** (lines 4-34, supervisor system_prompt) — confirmed no distinction between "waiver wire" and "free agent" style queries anywhere in the routing instructions. The Gap 2 inconsistency is pure LLM/prompt-vagueness variance (temp=0 but a genuinely ambiguous instruction), not a coded rule.

**Branch divergence discovered (unresolved):** `agents/tools/yahoo_api.py` on the current branch (`feature/player_db`) has **no `get_waiver_wire_players` tool** — grepped the whole repo, zero matches. But the LangSmith trace (running against deployed/staging code) clearly executed a `get_waiver_wire_players` tool call. Root cause: `agents/tools/waiver_wire.py` exists on `staging` and `feature/waiver_wire` branches but is **absent from `feature/player_db` and `main`**. The deployed system is running code this branch doesn't have yet.

This matters for the Gap 1 design: `yahoo_api.py`'s existing league-level tools (`get_league_players`, `get_league_standings`, etc.) call `self.query.<method>()` with **zero params** — they rely on the `yfpy` `YahooFantasySportsQuery` object's internal `league_key` state already being set, and no setter was found (looks constructor-only). So even with `user_context` passed into `YahooFantasyAgent`, it's not yet confirmed whether the actual free-agent/waiver tool (in the not-yet-read `waiver_wire.py`) can accept a pre-known `league_key` to skip the discovery chain, or whether it has the same self.query-must-be-primed constraint. **This was the open thread when the session was paused** — next step is reading `agents/tools/waiver_wire.py` on `staging` (`git show staging:agents/tools/waiver_wire.py`) before finalizing the Gap 1 design.

## Clarifying questions asked & answered (via brainstorming skill)

1. **Scope**: bundle all 3 gaps into one design spec (not split) — chosen because they share the same supervisor.py / multiagent_config.yaml surface area.
2. **Gap 2 policy**: keep LLM discretion, just clarify the prompt wording (not a structural/code-level guarantee that analysis always runs).
3. **Gap 3 location**: supervisor-side fix — supervisor calls `call_statistic_agent` before `call_analysis_agent` for recommendation-type queries. Explicitly NOT giving the Analysis Agent its own tool-calling ability (bigger, riskier change, not needed).
4. **Gap 1 approach**: pass `user_context` into `YahooFantasyAgent` at construction (reuse existing `BaseAgent` plumbing — same pattern as `FantasyAnalystAgent`/`SupervisorAgent` already use), rather than adding a structured `league_key` param to the `call_yahoo_fantasy_agent` tool schema.
5. **Verification**: automated `pytest` coverage only — no live LangSmith re-verification pass planned.

## State at pause

Task list (in-session tracker, not persisted elsewhere):
- #1 Explore project context — completed
- #2 Ask clarifying questions — all 4 answered, functionally done
- #3 Propose 2-3 approaches per gap — not started (approaches were effectively pre-selected via the clarifying questions above, so this may just need a short recap, not fresh proposals)
- #4 Present design sections and get approval — not started
- #5 Write design doc, self-review, get spec approval — not started

**No design doc has been written yet.** No files were created or modified by this session's work (the `docs/superpowers/plans/2026-06-29-player-stats-db.md` and `docs/superpowers/specs/2026-06-29-player-stats-db-design.md` untracked files predate this session — from an earlier player-stats-db brainstorm).

## Next session should

1. Run `git show staging:agents/tools/waiver_wire.py` (or check out `feature/waiver_wire`) to see if the waiver/free-agent tool can accept a pre-known `league_key`, or has the same self.query-priming constraint as `yahoo_api.py`'s league-level tools.
2. Recap the 3 chosen approaches (already decided, see above) as the "Propose approaches" step.
3. Present the design in sections (architecture / data flow / testing) per the brainstorming skill, get approval.
4. Write the spec to `docs/superpowers/specs/2026-07-01-multi-agent-workflow-fixes-design.md`, self-review, commit, get user's review sign-off.
5. Then invoke `writing-plans` per the skill's terminal step.
