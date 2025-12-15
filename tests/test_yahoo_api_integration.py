"""
Integration tests for YahooFantasyTool that test actual API connections.

These tests require:
- YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET environment variables set
- Real Yahoo Fantasy Sports API access
- Valid OAuth tokens

Run with: pytest -m integration tests/test_yahoo_api_integration.py
"""

import os
import yaml
from pathlib import Path
import pytest
from yfpy.query import YahooFantasySportsQuery

from agents.tools.yahoo_api import YahooFantasyTool


# Skip all integration tests if Yahoo credentials are not set
pytestmark = pytest.mark.skipif(
    not os.environ.get('YAHOO_CLIENT_ID') or not os.environ.get('YAHOO_CLIENT_SECRET'),
    reason="YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET environment variables not set. Integration tests require real API credentials."
)


def _load_test_config():
    """Load test configuration from test_config.yaml file."""
    config_path = Path(__file__).parent / 'test_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        return config.get('yahoo_api', {})
    return {}


@pytest.mark.integration
class TestYahooFantasyToolIntegration:
    """Integration tests for YahooFantasyTool with real API calls."""
    
    # Load test configuration from YAML file
    _test_config = _load_test_config()
    
    # Test team_id (KATmandu Climbers from league 58930)
    # Configured in tests/test_config.yaml
    TEST_TEAM_ID = _test_config.get('team_id', '6')
    
    # Test week (week 1)
    # Configured in tests/test_config.yaml
    TEST_WEEK = int(_test_config.get('week', '1'))
    
    # Test player_key (example player key format: game_id.p.player_id)
    # Configured in tests/test_config.yaml
    TEST_PLAYER_KEY = _test_config.get('player_key', '466.p.4235')
    
    # Points league ID for testing get_team_stats_by_week (only works in points leagues)
    # Configured in tests/test_config.yaml
    POINTS_LEAGUE_ID = _test_config.get('points_league_id', '229522')
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for integration tests."""
        # Initialize Yahoo Fantasy query with environment variables
        self.query = YahooFantasySportsQuery(
            league_id=int(os.environ.get('YAHOO_LEAGUE_ID', '58930')),
            game_code="nba",
            game_id=466,
            yahoo_consumer_key=os.environ.get('YAHOO_CLIENT_ID'),
            yahoo_consumer_secret=os.environ.get('YAHOO_CLIENT_SECRET'),
            env_var_fallback=True,
            save_token_data_to_env_file=True,
        )
        self.tool = YahooFantasyTool(query=self.query)
        self.tools = self.tool.get_all_tools()
        
        # Setup points league query for tests that require it (e.g., get_team_stats_by_week)
        self.points_league_query = YahooFantasySportsQuery(
            league_id=int(self.POINTS_LEAGUE_ID),
            game_code="nba",
            game_id=466,
            yahoo_consumer_key=os.environ.get('YAHOO_CLIENT_ID'),
            yahoo_consumer_secret=os.environ.get('YAHOO_CLIENT_SECRET'),
            env_var_fallback=True,
            save_token_data_to_env_file=True,
        )
        self.points_league_tool = YahooFantasyTool(query=self.points_league_query)
        self.points_league_tools = self.points_league_tool.get_all_tools()
        
        yield
    
    @pytest.mark.integration
    def test_get_all_yahoo_fantasy_game_keys_tool_integration(self, save_test_output):
        """Test get_all_yahoo_fantasy_game_keys tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get all game keys.
        Sample output saved to: tests/test_outputs/test_get_all_yahoo_fantasy_game_keys_tool_integration_get_all_yahoo_fantasy_game_keys.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_all_yahoo_fantasy_game_keys')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_all_yahoo_fantasy_game_keys", result)
        
        # Verify the result
        assert "All Yahoo Fantasy game keys" in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_game_key_by_season_tool_integration(self, save_test_output):
        """Test get_game_key_by_season tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get game key by season.
        Sample output saved to: tests/test_outputs/test_get_game_key_by_season_tool_integration_get_game_key_by_season.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_game_key_by_season')
        
        # Test with recent season
        test_season = "2023"
        
        # Call real tool
        result = get_tool.invoke({"season": test_season})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_game_key_by_season", result)
        
        # Verify the result
        assert f'Game key for season {test_season}' in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_current_game_info_tool_integration(self, save_test_output):
        """Test get_current_game_info tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get current game info.
        Sample output saved to: tests/test_outputs/test_get_current_game_info_tool_integration_get_current_game_info.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_current_game_info')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_current_game_info", result)
        
        # Verify the result
        assert "Current game info" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_current_game_metadata_tool_integration(self, save_test_output):
        """Test get_current_game_metadata tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get current game metadata.
        Sample output saved to: tests/test_outputs/test_get_current_game_metadata_tool_integration_get_current_game_metadata.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_current_game_metadata')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_current_game_metadata", result)
        
        # Verify the result
        assert "Current game metadata" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_game_info_by_game_id_tool_integration(self, save_test_output):
        """Test get_game_info_by_game_id tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get game info by game ID.
        Sample output saved to: tests/test_outputs/test_get_game_info_by_game_id_tool_integration_get_game_info_by_game_id.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_game_info_by_game_id')
        
        # Use NBA game ID
        test_game_id = 466
        
        # Call real tool
        result = get_tool.invoke({"game_id": test_game_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_game_info_by_game_id", result)
        
        # Verify the result
        assert f'Game info for game ID {test_game_id}' in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_game_metadata_by_game_id_tool_integration(self, save_test_output):
        """Test get_game_metadata_by_game_id tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get game metadata by game ID.
        Sample output saved to: tests/test_outputs/test_get_game_metadata_by_game_id_tool_integration_get_game_metadata_by_game_id.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_game_metadata_by_game_id')
        
        # Use NBA game ID
        test_game_id = 466
        
        # Call real tool
        result = get_tool.invoke({"game_id": test_game_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_game_metadata_by_game_id", result)
        
        # Verify the result
        assert f'Game metadata for game ID {test_game_id}' in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_game_weeks_by_game_id_tool_integration(self, save_test_output):
        """Test get_game_weeks_by_game_id tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get game weeks by game ID.
        Sample output saved to: tests/test_outputs/test_get_game_weeks_by_game_id_tool_integration_get_game_weeks_by_game_id.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_game_weeks_by_game_id')
        
        # Use NBA game ID
        test_game_id = 466
        
        # Call real tool
        result = get_tool.invoke({"game_id": test_game_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_game_weeks_by_game_id", result)
        
        # Verify the result
        assert f'Game weeks for game ID {test_game_id}' in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_game_stat_categories_by_game_id_tool_integration(self, save_test_output):
        """Test get_game_stat_categories_by_game_id tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get game stat categories by game ID.
        Sample output saved to: tests/test_outputs/test_get_game_stat_categories_by_game_id_tool_integration_get_game_stat_categories_by_game_id.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_game_stat_categories_by_game_id')
        
        # Use NBA game ID
        test_game_id = 466
        
        # Call real tool
        result = get_tool.invoke({"game_id": test_game_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_game_stat_categories_by_game_id", result)
        
        # Verify the result
        assert f'Stat categories for game ID {test_game_id}' in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_game_position_types_by_game_id_tool_integration(self, save_test_output):
        """Test get_game_position_types_by_game_id tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get game position types by game ID.
        Sample output saved to: tests/test_outputs/test_get_game_position_types_by_game_id_tool_integration_get_game_position_types_by_game_id.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_game_position_types_by_game_id')
        
        # Use NBA game ID
        test_game_id = 466
        
        # Call real tool
        result = get_tool.invoke({"game_id": test_game_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_game_position_types_by_game_id", result)
        
        # Verify the result
        assert f'Position types for game ID {test_game_id}' in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_game_roster_positions_by_game_id_tool_integration(self, save_test_output):
        """Test get_game_roster_positions_by_game_id tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get game roster positions by game ID.
        Sample output saved to: tests/test_outputs/test_get_game_roster_positions_by_game_id_tool_integration_get_game_roster_positions_by_game_id.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_game_roster_positions_by_game_id')
        
        # Use NBA game ID
        test_game_id = 466
        
        # Call real tool
        result = get_tool.invoke({"game_id": test_game_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_game_roster_positions_by_game_id", result)
        
        # Verify the result
        assert f'Roster positions for game ID {test_game_id}' in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_current_user_tool_integration(self, save_test_output):
        """Test get_current_user tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get current user info.
        Sample output saved to: tests/test_outputs/test_get_current_user_tool_integration_get_current_user.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_current_user')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_current_user", result)
        
        # Verify the result
        assert "Current user info" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_user_games_tool_integration(self, save_test_output):
        """Test get_user_games tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get user games.
        Sample output saved to: tests/test_outputs/test_get_user_games_tool_integration_get_user_games.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_user_games')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_user_games", result)
        
        # Verify the result
        assert "User games" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_user_leagues_by_game_key_tool_integration(self, save_test_output):
        """Test get_user_leagues_by_game_key tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get user leagues by game key.
        Sample output saved to: tests/test_outputs/test_get_user_leagues_by_game_key_tool_integration_get_user_leagues_by_game_key.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_user_leagues_by_game_key')
        
        # Use NBA game key (466 is NBA game ID)
        test_game_key = "466"
        
        # Call real tool
        result = get_tool.invoke({"game_key": test_game_key})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_user_leagues_by_game_key", result)
        
        # Verify the result
        assert f'User leagues for game key {test_game_key}' in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_user_teams_tool_integration(self, save_test_output):
        """Test get_user_teams tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get user teams.
        Sample output saved to: tests/test_outputs/test_get_user_teams_tool_integration_get_user_teams.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_user_teams')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_user_teams", result)
        
        # Verify the result
        assert "User teams" in result or "teams" in result.lower()
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_league_key_tool_integration(self, save_test_output):
        """Test get_league_key tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get league key.
        Sample output saved to: tests/test_outputs/test_get_league_key_tool_integration_get_league_key.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_league_key')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_league_key", result)
        
        # Verify the result
        assert "League key" in result
        assert "Failed" not in result
        assert len(result) > 10  # Should have some content
    
    @pytest.mark.integration
    def test_get_league_info_tool_integration(self, save_test_output):
        """Test get_league_info tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get league info.
        Sample output saved to: tests/test_outputs/test_get_league_info_tool_integration_get_league_info.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_league_info')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_league_info", result)
        
        # Verify the result
        assert "League info" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_league_metadata_tool_integration(self, save_test_output):
        """Test get_league_metadata tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get league metadata.
        Sample output saved to: tests/test_outputs/test_get_league_metadata_tool_integration_get_league_metadata.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_league_metadata')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_league_metadata", result)
        
        # Verify the result
        assert "League metadata" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_league_settings_tool_integration(self, save_test_output):
        """Test get_league_settings tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get league settings.
        Sample output saved to: tests/test_outputs/test_get_league_settings_tool_integration_get_league_settings.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_league_settings')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_league_settings", result)
        
        # Verify the result
        assert "League settings" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_league_standings_tool_integration(self, save_test_output):
        """Test get_league_standings tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get league standings.
        Sample output saved to: tests/test_outputs/test_get_league_standings_tool_integration_get_league_standings.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_league_standings')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_league_standings", result)
        
        # Verify the result
        assert "League standings" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_league_teams_tool_integration(self, save_test_output):
        """Test get_league_teams tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get league teams.
        Sample output saved to: tests/test_outputs/test_get_league_teams_tool_integration_get_league_teams.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_league_teams')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_league_teams", result)
        
        # Verify the result
        assert "League teams" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_league_players_tool_integration(self, save_test_output):
        """Test get_league_players tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get league players.
        Sample output saved to: tests/test_outputs/test_get_league_players_tool_integration_get_league_players.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_league_players')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_league_players", result)
        
        # Verify the result
        assert "League players" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_league_draft_results_tool_integration(self, save_test_output):
        """Test get_league_draft_results tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get league draft results.
        Sample output saved to: tests/test_outputs/test_get_league_draft_results_tool_integration_get_league_draft_results.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_league_draft_results')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_league_draft_results", result)
        
        # Verify the result
        assert "League draft results" in result or "draft" in result.lower()
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_league_transactions_tool_integration(self, save_test_output):
        """Test get_league_transactions tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get league transactions.
        Sample output saved to: tests/test_outputs/test_get_league_transactions_tool_integration_get_league_transactions.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_league_transactions')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_league_transactions", result)
        
        # Verify the result
        assert "League transactions" in result or "transactions" in result.lower()
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_league_scoreboard_by_week_tool_integration(self, save_test_output):
        """Test get_league_scoreboard_by_week tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get league scoreboard by week.
        Sample output saved to: tests/test_outputs/test_get_league_scoreboard_by_week_tool_integration_get_league_scoreboard_by_week.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_league_scoreboard_by_week')
        
        # Test with a recent week (week 1)
        test_week = self.TEST_WEEK
        
        # Call real tool
        result = get_tool.invoke({"week": test_week})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_league_scoreboard_by_week", result)
        
        # Verify the result
        assert f'League scoreboard for week {test_week}' in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_league_matchups_by_week_tool_integration(self, save_test_output):
        """Test get_league_matchups_by_week tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get league matchups by week.
        Sample output saved to: tests/test_outputs/test_get_league_matchups_by_week_tool_integration_get_league_matchups_by_week.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_league_matchups_by_week')
        
        # Test with a recent week (week 1)
        test_week = self.TEST_WEEK
        
        # Call real tool
        result = get_tool.invoke({"week": test_week})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_league_matchups_by_week", result)
        
        # Verify the result
        assert f'League matchups for week {test_week}' in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_team_info_tool_integration(self, save_test_output):
        """Test get_team_info tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get team info.
        Sample output saved to: tests/test_outputs/test_get_team_info_tool_integration_get_team_info.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_info')
        
        # First get league teams to find a valid team ID
        get_teams_tool = next(t for t in self.tools if t.name == 'get_league_teams')
        teams_result = get_teams_tool.invoke({})
        
        # Use team_id 1 as a common default (may need adjustment based on actual league)
        test_team_id = self.TEST_TEAM_ID
        
        # Call real tool
        result = get_tool.invoke({"team_key_or_id": test_team_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_info", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_team_metadata_tool_integration(self, save_test_output):
        """Test get_team_metadata tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get team metadata.
        Sample output saved to: tests/test_outputs/test_get_team_metadata_tool_integration_get_team_metadata.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_metadata')
        
        # Use team_id 1 as a common default
        test_team_id = self.TEST_TEAM_ID
        
        # Call real tool
        result = get_tool.invoke({"team_key_or_id": test_team_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_metadata", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_team_stats_tool_integration(self, save_test_output):
        """Test get_team_stats tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get team stats.
        Sample output saved to: tests/test_outputs/test_get_team_stats_tool_integration_get_team_stats.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_stats')
        
        # Use team_id 1 as a common default
        test_team_id = self.TEST_TEAM_ID
        
        # Call real tool
        result = get_tool.invoke({"team_key_or_id": test_team_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_stats", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_team_stats_by_week_tool_integration(self, save_test_output):
        """Test get_team_stats_by_week tool with real Yahoo API.
        
        This test uses a POINTS league (not categories) because get_team_stats_by_week
        only works with points leagues. The tool requires team_projected_points which
        is only available in points leagues.
        Sample output saved to: tests/test_outputs/test_get_team_stats_by_week_tool_integration_get_team_stats_by_week.txt
        """
        # Use the points league query from setup
        get_tool = next(t for t in self.points_league_tools if t.name == 'get_team_stats_by_week')
        
        # Use Team 4: 'Trust the Tiramisu' from the points league
        test_team_id = "4"
        test_week = self.TEST_WEEK
        
        # Call real tool with test week
        result = get_tool.invoke({"team_key_or_id": test_team_id, "week": test_week})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_stats_by_week", result)
        
        # Verify the result (should succeed, not fail)
        assert "Failed" not in result
        assert "Missing field" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_team_standings_tool_integration(self, save_test_output):
        """Test get_team_standings tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get team standings.
        Sample output saved to: tests/test_outputs/test_get_team_standings_tool_integration_get_team_standings.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_standings')
        
        # Use team_id 1 as a common default
        test_team_id = self.TEST_TEAM_ID
        
        # Call real tool
        result = get_tool.invoke({"team_key_or_id": test_team_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_standings", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_team_roster_by_week_tool_integration(self, save_test_output):
        """Test get_team_roster_by_week tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get team roster by week.
        Sample output saved to: tests/test_outputs/test_get_team_roster_by_week_tool_integration_get_team_roster_by_week.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_roster_by_week')
        
        # Use team_id 1 and week 1
        test_team_id = self.TEST_TEAM_ID
        test_week = self.TEST_WEEK
        
        # Call real tool
        result = get_tool.invoke({"team_key_or_id": test_team_id, "week": test_week})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_roster_by_week", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_team_roster_player_info_by_week_tool_integration(self, save_test_output):
        """Test get_team_roster_player_info_by_week tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get team roster player info by week.
        Sample output saved to: tests/test_outputs/test_get_team_roster_player_info_by_week_tool_integration_get_team_roster_player_info_by_week.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_roster_player_info_by_week')
        
        # Use team_id 1 and week 1
        test_team_id = self.TEST_TEAM_ID
        test_week = self.TEST_WEEK
        
        # Call real tool
        result = get_tool.invoke({"team_key_or_id": test_team_id, "week": test_week})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_roster_player_info_by_week", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_team_roster_player_info_by_date_tool_integration(self, save_test_output):
        """Test get_team_roster_player_info_by_date tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get team roster player info by date.
        Sample output saved to: tests/test_outputs/test_get_team_roster_player_info_by_date_tool_integration_get_team_roster_player_info_by_date.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_roster_player_info_by_date')
        
        # Use team_id 6 and a date from the current season (2025-10-26 corresponds to week 1)
        test_team_id = self.TEST_TEAM_ID
        test_date = "2025-10-26"
        
        # Call real tool
        result = get_tool.invoke({"team_key_or_id": test_team_id, "date": test_date})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_roster_player_info_by_date", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_team_roster_player_stats_tool_integration(self, save_test_output):
        """Test get_team_roster_player_stats tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get team roster player stats.
        Sample output saved to: tests/test_outputs/test_get_team_roster_player_stats_tool_integration_get_team_roster_player_stats.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_roster_player_stats')
        
        # Use team_id 1 as a common default
        test_team_id = self.TEST_TEAM_ID
        
        # Call real tool
        result = get_tool.invoke({"team_key_or_id": test_team_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_roster_player_stats", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_team_roster_player_stats_by_week_tool_integration(self, save_test_output):
        """Test get_team_roster_player_stats_by_week tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get team roster player stats by week.
        Sample output saved to: tests/test_outputs/test_get_team_roster_player_stats_by_week_tool_integration_get_team_roster_player_stats_by_week.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_roster_player_stats_by_week')
        
        # Use team_id 1 and week 1
        test_team_id = self.TEST_TEAM_ID
        test_week = self.TEST_WEEK
        
        # Call real tool
        result = get_tool.invoke({"team_key_or_id": test_team_id, "week": test_week})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_roster_player_stats_by_week", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_team_draft_results_tool_integration(self, save_test_output):
        """Test get_team_draft_results tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get team draft results.
        Sample output saved to: tests/test_outputs/test_get_team_draft_results_tool_integration_get_team_draft_results.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_draft_results')
        
        # Use team_id 1 as a common default
        test_team_id = self.TEST_TEAM_ID
        
        # Call real tool
        result = get_tool.invoke({"team_key_or_id": test_team_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_draft_results", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_team_matchups_tool_integration(self, save_test_output):
        """Test get_team_matchups tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get team matchups.
        Sample output saved to: tests/test_outputs/test_get_team_matchups_tool_integration_get_team_matchups.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_matchups')
        
        # Use team_id 1 as a common default
        test_team_id = self.TEST_TEAM_ID
        
        # Call real tool
        result = get_tool.invoke({"team_key_or_id": test_team_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_matchups", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_player_stats_for_season_tool_integration(self, save_test_output):
        """Test get_player_stats_for_season tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get player stats for season.
        Sample output saved to: tests/test_outputs/test_get_player_stats_for_season_tool_integration_get_player_stats_for_season.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_player_stats_for_season')
        
        # Use a sample player key (format: game_id.p.player_id)
        # This is a placeholder - in real usage, you'd get this from league data
        test_player_key = self.TEST_PLAYER_KEY
        
        # Call real tool
        result = get_tool.invoke({"player_key": test_player_key})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_player_stats_for_season", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_player_stats_by_week_tool_integration(self, save_test_output):
        """Test get_player_stats_by_week tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get player stats by week.
        Sample output saved to: tests/test_outputs/test_get_player_stats_by_week_tool_integration_get_player_stats_by_week.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_player_stats_by_week')
        
        # Use a sample player key and week 1
        test_player_key = self.TEST_PLAYER_KEY
        test_week = self.TEST_WEEK
        
        # Call real tool
        result = get_tool.invoke({"player_key": test_player_key, "week": test_week})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_player_stats_by_week", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_player_stats_by_date_tool_integration(self, save_test_output):
        """Test get_player_stats_by_date tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get player stats by date.
        Sample output saved to: tests/test_outputs/test_get_player_stats_by_date_tool_integration_get_player_stats_by_date.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_player_stats_by_date')
        
        # Use a sample player key and a recent date
        test_player_key = self.TEST_PLAYER_KEY
        test_date = "2024-01-15"
        
        # Call real tool
        result = get_tool.invoke({"player_key": test_player_key, "date": test_date})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_player_stats_by_date", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_player_ownership_tool_integration(self, save_test_output):
        """Test get_player_ownership tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get player ownership.
        Sample output saved to: tests/test_outputs/test_get_player_ownership_tool_integration_get_player_ownership.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_player_ownership')
        
        # Use a sample player key
        test_player_key = self.TEST_PLAYER_KEY
        
        # Call real tool
        result = get_tool.invoke({"player_key": test_player_key})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_player_ownership", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_player_percent_owned_by_week_tool_integration(self, save_test_output):
        """Test get_player_percent_owned_by_week tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get player percent owned by week.
        Sample output saved to: tests/test_outputs/test_get_player_percent_owned_by_week_tool_integration_get_player_percent_owned_by_week.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_player_percent_owned_by_week')
        
        # Use a sample player key and week 1
        test_player_key = self.TEST_PLAYER_KEY
        test_week = self.TEST_WEEK
        
        # Call real tool
        result = get_tool.invoke({"player_key": test_player_key, "week": test_week})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_player_percent_owned_by_week", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_player_draft_analysis_tool_integration(self, save_test_output):
        """Test get_player_draft_analysis tool with real Yahoo API.
        
        This test actually calls the Yahoo Fantasy API to get player draft analysis.
        Sample output saved to: tests/test_outputs/test_get_player_draft_analysis_tool_integration_get_player_draft_analysis.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_player_draft_analysis')
        
        # Use a sample player key
        test_player_key = self.TEST_PLAYER_KEY
        
        # Call real tool
        result = get_tool.invoke({"player_key": test_player_key})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_player_draft_analysis", result)
        
        # Verify the result - should succeed, not fail
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content

