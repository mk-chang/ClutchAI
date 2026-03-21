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
