# Agent Patterns & Notes

## Agent Hierarchy

```
Supervisor (gpt-4o, temp=0)
├── Yahoo Fantasy Agent (gpt-4o-mini, temp=0)
├── Statistic Agent (gpt-4o-mini, temp=0)
├── News Agent (gpt-4o-mini, temp=0)
└── Analysis Agent (gpt-4o, temp=0)
```

All agents share a `max_tokens: 150000` limit (leaves room for response within 200k TPM).

## Supervisor Routing Logic

- **Direct response** (no agent calls): greetings, meta-questions about the system
- **Yahoo Fantasy Agent**: league data, rosters, standings, matchups, ownership, transactions
- **Statistic Agent**: NBA player/team stats, game logs, career stats, splits
- **News Agent**: injury updates, RSS feeds, dynasty rankings, knowledge base search
- **Multiple agents**: supervisor can call several research agents in parallel before calling Analysis

The supervisor calls `get_current_datetime` first for date-sensitive queries.

## Analysis Agent

- Uses `search_knowledge_base` tool to pull analyst insights from vectorstore
- Applies recency bias: more recent knowledge base content is weighted higher
- Always provides: recommendation + reasoning + supporting evidence + caveats

## Tool Files

| File | Tools |
|------|-------|
| `agents/tools/yahoo_api.py` | 45 Yahoo Fantasy tools |
| `agents/tools/nba_api.py` | 16 NBA API tools |
| `agents/tools/rotowire_rss.py` | RSS news feed |
| `agents/tools/fantasy_news.py` | Fantasy news scraping |
| `agents/tools/dynasty_ranking.py` | Hashtag Basketball dynasty rankings |
| `agents/tools/player_value.py` | Player value tools |
| `agents/tools/basic.py` | Utility tools (datetime, etc.) |
| `agents/tools/base.py` | Base tool class — extend this for new tools |

## Known Patterns / Lessons Learned

_Update this section as you work on the project._

- **`user_context` plumbing exists but is only half-wired.** `BaseAgent.__init__` (base_agent.py:37-46) accepts `user_context` and `_enhance_system_prompt()` (lines 191-205) injects it into any subclass's system prompt automatically. `multi_agent_system.py` (lines 150-196) only passes `user_context` to `SupervisorAgent` and `FantasyAnalystAgent` — `YahooFantasyAgent` and `StatisticAgent` are constructed without it, so they can't see the already-known league_key/team_key/roster and must rediscover league identity from scratch via tool calls on every research request.
- **Analysis Agent cannot call other agents.** `analyst_agent.py`'s only tool is `search_knowledge_base` (RAG). It's a single-shot `invoke(query, research_data, **kwargs)` LLM call over whatever `research_data` string the supervisor assembles — it has no way to pull additional data (e.g. NBA stats) mid-analysis. New data sources must be fetched by the supervisor *before* calling `call_analysis_agent`, not by adding tools to the analysis agent itself.
- **Supervisor's delegating tools only accept free-text `query: str`** (supervisor.py:110-209) — no structured params (e.g. `league_key`) are ever threaded through, even though the supervisor's own system prompt already has that data via `user_context`.
- **Supervisor routing prompt has no query-type-specific rules** (config/multiagent_config.yaml:4-34) — e.g. "waiver wire" vs "free agent" style queries are not distinguished, so whether `call_analysis_agent` gets invoked for a given research-requiring query is currently LLM variance, not a coded rule.
