import json
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy import text

from agents.tools.base import ClutchAITool
from data.postgres.connection import PostgresConnection
from logger import get_logger

logger = get_logger(__name__)


def _is_safe_sql(sql: str) -> bool:
    return sql.strip().lower().startswith('select')


class PlayerStatsDbTool(ClutchAITool):
    """DB-backed tools for StatsAgent: recent form, schedule density, season trends, raw SQL."""

    def __init__(self, connection: Optional[PostgresConnection] = None, debug: bool = False):
        super().__init__(debug=debug)
        self.connection = connection or PostgresConnection()

    def get_all_tools(self) -> list:
        connection = self.connection

        @tool
        def get_recent_form(player_id: int, n_games: int = 10) -> str:
            """
            Get a player's recent form from the last N games (default 10).
            Returns per-game averages (PTS, REB, AST, STL, BLK, TOV, FG%, 3P%, FT%)
            for the last N games vs the season average. Use for hot/cold streak detection.

            Args:
                player_id: NBA player ID
                n_games: Number of recent games to average (default 10)
            """
            sql = text("""
                WITH recent_stats AS (
                    SELECT
                        COUNT(*)                           AS games,
                        ROUND(AVG(pts)::numeric, 1)       AS avg_pts,
                        ROUND(AVG(reb)::numeric, 1)       AS avg_reb,
                        ROUND(AVG(ast)::numeric, 1)       AS avg_ast,
                        ROUND(AVG(stl)::numeric, 1)       AS avg_stl,
                        ROUND(AVG(blk)::numeric, 1)       AS avg_blk,
                        ROUND(AVG("to")::numeric, 1)      AS avg_tov,
                        ROUND(AVG(fg_pct)::numeric, 3)    AS avg_fg_pct,
                        ROUND(AVG(fg3_pct)::numeric, 3)   AS avg_3p_pct,
                        ROUND(AVG(ft_pct)::numeric, 3)    AS avg_ft_pct
                    FROM (
                        SELECT * FROM player_game_logs
                        WHERE player_id = :player_id
                        ORDER BY game_date DESC
                        LIMIT :n_games
                    ) r
                ),
                season_stats AS (
                    SELECT
                        ROUND(AVG(pts)::numeric, 1)       AS avg_pts,
                        ROUND(AVG(reb)::numeric, 1)       AS avg_reb,
                        ROUND(AVG(ast)::numeric, 1)       AS avg_ast,
                        ROUND(AVG(stl)::numeric, 1)       AS avg_stl,
                        ROUND(AVG(blk)::numeric, 1)       AS avg_blk,
                        ROUND(AVG("to")::numeric, 1)      AS avg_tov,
                        ROUND(AVG(fg_pct)::numeric, 3)    AS avg_fg_pct,
                        ROUND(AVG(fg3_pct)::numeric, 3)   AS avg_3p_pct,
                        ROUND(AVG(ft_pct)::numeric, 3)    AS avg_ft_pct
                    FROM player_game_logs
                    WHERE player_id = :player_id
                )
                SELECT
                    rs.games                AS recent_games,
                    rs.avg_pts              AS recent_pts,
                    rs.avg_reb              AS recent_reb,
                    rs.avg_ast              AS recent_ast,
                    rs.avg_stl              AS recent_stl,
                    rs.avg_blk              AS recent_blk,
                    rs.avg_tov              AS recent_tov,
                    rs.avg_fg_pct           AS recent_fg_pct,
                    rs.avg_3p_pct           AS recent_3p_pct,
                    rs.avg_ft_pct           AS recent_ft_pct,
                    ss.avg_pts              AS season_pts,
                    ss.avg_reb              AS season_reb,
                    ss.avg_ast              AS season_ast,
                    ss.avg_stl              AS season_stl,
                    ss.avg_blk              AS season_blk,
                    ss.avg_tov              AS season_tov,
                    ss.avg_fg_pct           AS season_fg_pct,
                    ss.avg_3p_pct           AS season_3p_pct,
                    ss.avg_ft_pct           AS season_ft_pct
                FROM recent_stats rs, season_stats ss
            """)
            try:
                with connection.get_engine().connect() as conn:
                    result = conn.execute(sql, {'player_id': player_id, 'n_games': n_games})
                    row = result.fetchone()
                    if not row or row[0] == 0:
                        return json.dumps({'error': f'No game logs found for player_id={player_id}'})
                    return json.dumps(dict(zip(result.keys(), row)), default=str)
            except Exception as e:
                logger.error(f"get_recent_form error: {e}")
                return json.dumps({'error': str(e)})

        @tool
        def get_schedule_density(team_abbreviation: str, days: int = 7) -> str:
            """
            Get upcoming games for a team in the next N days (default 7).
            Returns game list (date, opponent, home/away) plus total count.
            Use for streaming pickup recommendations and start/sit decisions.

            Args:
                team_abbreviation: NBA team abbreviation (e.g., 'LAL', 'BOS')
                days: Number of days to look ahead (default 7)
            """
            sql = text("""
                SELECT game_date, opponent_abbr, home_away
                FROM team_schedules
                WHERE team_abbreviation = :team_abbr
                  AND game_date >= CURRENT_DATE
                  AND game_date < CURRENT_DATE + :days * INTERVAL '1 day'
                  AND NOT postponed
                ORDER BY game_date
            """)
            try:
                with connection.get_engine().connect() as conn:
                    result = conn.execute(sql, {'team_abbr': team_abbreviation, 'days': days})
                    games = [
                        {'date': str(row[0]), 'opponent': row[1], 'home_away': row[2]}
                        for row in result
                    ]
                return json.dumps({
                    'team': team_abbreviation,
                    'days_ahead': days,
                    'game_count': len(games),
                    'games': games,
                }, default=str)
            except Exception as e:
                logger.error(f"get_schedule_density error: {e}")
                return json.dumps({'error': str(e)})

        @tool
        def get_season_trends(player_id: int) -> str:
            """
            Get a player's monthly stat trends over the current season.
            Returns per-game averages by calendar month showing production trajectory.
            Use for buy-low/sell-high analysis and identifying improving/declining players.

            Args:
                player_id: NBA player ID
            """
            sql = text("""
                SELECT
                    TO_CHAR(game_date, 'YYYY-MM')      AS month,
                    COUNT(*)                            AS games,
                    ROUND(AVG(pts)::numeric, 1)        AS pts,
                    ROUND(AVG(reb)::numeric, 1)        AS reb,
                    ROUND(AVG(ast)::numeric, 1)        AS ast,
                    ROUND(AVG(stl)::numeric, 1)        AS stl,
                    ROUND(AVG(blk)::numeric, 1)        AS blk,
                    ROUND(AVG("to")::numeric, 1)       AS tov,
                    ROUND(AVG(fg_pct)::numeric, 3)     AS fg_pct,
                    ROUND(AVG(fg3_pct)::numeric, 3)    AS fg3_pct,
                    ROUND(AVG(ft_pct)::numeric, 3)     AS ft_pct
                FROM player_game_logs
                WHERE player_id = :player_id
                GROUP BY TO_CHAR(game_date, 'YYYY-MM')
                ORDER BY month
            """)
            try:
                with connection.get_engine().connect() as conn:
                    result = conn.execute(sql, {'player_id': player_id})
                    rows = result.fetchall()
                    if not rows:
                        return json.dumps({'error': f'No game logs found for player_id={player_id}'})
                    keys = result.keys()
                    return json.dumps(
                        {'player_id': player_id, 'monthly_trends': [dict(zip(keys, r)) for r in rows]},
                        default=str,
                    )
            except Exception as e:
                logger.error(f"get_season_trends error: {e}")
                return json.dumps({'error': str(e)})

        @tool
        def query_stats_db(sql: str) -> str:
            """
            Execute a read-only SELECT query against the player stats database.

            Available tables:
            - player_game_logs(player_id, game_id, game_date, season, team_abbreviation,
                                min, pts, reb, ast, stl, blk, "to", fgm, fga, fg_pct,
                                fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, plus_minus)
            - team_schedules(team_id, team_abbreviation, game_id, game_date, season,
                              home_away, opponent_abbr, postponed)
            - opponent_defense_rankings(team_id, team_abbreviation, season,
                                         rank_pts, rank_reb, rank_ast, rank_stl, rank_blk,
                                         rank_to, rank_fg_pct, rank_3p_pct)
                                         Rank 1=best defense (hardest), 30=worst (easiest matchup).
            - bball_monsters_player_stats_pg / _total / _p36 — season aggregates with z-scores.

            Only SELECT statements are allowed.

            Args:
                sql: A SQL SELECT statement
            """
            if not _is_safe_sql(sql):
                return json.dumps({'error': 'Only SELECT statements are allowed.'})
            try:
                with connection.get_engine().connect() as conn:
                    result = conn.execute(text(sql))
                    rows = result.fetchall()
                    return json.dumps([dict(zip(result.keys(), r)) for r in rows], default=str)
            except Exception as e:
                logger.error(f"query_stats_db error: {e}")
                return json.dumps({'error': str(e)})

        return [get_recent_form, get_schedule_density, get_season_trends, query_stats_db]
