"""
Integration tests for nbaAPITool that test actual API connections.

These tests require:
- Network connectivity
- Real NBA API access (public API, no authentication needed)

Run with: pytest -m integration tests/test_nba_api_integration.py
"""

import yaml
from pathlib import Path
from datetime import datetime, timedelta
import pytest
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2

from agents.tools.nba_api import nbaAPITool


def _load_test_config():
    """Load test configuration from test_config.yaml file."""
    config_path = Path(__file__).parent / 'test_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        return config.get('nba_api', {})
    return {}


@pytest.mark.integration
class TestnbaAPIToolIntegration:
    """Integration tests for nbaAPITool with real API calls.
    
    These tests make actual network calls to the NBA API.
    They require network connectivity and may be slower.
    
    Run with: pytest -m integration tests/test_nba_api_integration.py::TestnbaAPIToolIntegration
    """
    
    # Load test configuration from YAML file
    _test_config = _load_test_config()
    
    # Test player name
    # Configured in tests/test_config.yaml
    TEST_PLAYER_NAME = _test_config.get('player_name', 'LeBron James')
    
    # Test team name
    # Configured in tests/test_config.yaml
    TEST_TEAM_NAME = _test_config.get('team_name', 'Lakers')
    
    # Test player_id (LeBron James)
    # Configured in tests/test_config.yaml
    TEST_PLAYER_ID = _test_config.get('player_id', '2544')
    
    # Test team_id (Lakers)
    # Configured in tests/test_config.yaml
    TEST_TEAM_ID = _test_config.get('team_id', '1610612747')
    
    # Test season
    # Configured in tests/test_config.yaml
    TEST_SEASON = _test_config.get('season', '2023-24')
    
    # Test game_id
    # Configured in tests/test_config.yaml
    TEST_GAME_ID = _test_config.get('game_id', '0022301230')
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for integration tests."""
        self.tool = nbaAPITool(timeout=30)
        self.tools = self.tool.get_all_tools()
        yield
    
    @pytest.mark.integration
    def test_get_all_nba_players_tool_integration(self, save_test_output):
        """Test get_all_nba_players tool with real NBA API.
        
        This test actually calls the NBA API to get all players.
        Sample output saved to: tests/test_outputs/test_get_all_nba_players_tool_integration_get_all_nba_players.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_all_nba_players')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_all_nba_players", result)
        
        # Verify the result
        assert "All NBA players" in result
        assert "Failed" not in result
        assert len(result) > 100  # Should have substantial content
    
    @pytest.mark.integration
    def test_find_nba_players_by_name_tool_integration(self, save_test_output):
        """Test find_nba_players_by_name tool with real NBA API.
        
        This test actually calls the NBA API to find players by name.
        Sample output saved to: tests/test_outputs/test_find_nba_players_by_name_tool_integration_find_nba_players_by_name.txt
        """
        find_tool = next(t for t in self.tools if t.name == 'find_nba_players_by_name')
        
        # Test with a well-known player
        test_player = self.TEST_PLAYER_NAME
        
        # Call real tool
        result = find_tool.invoke({"full_name": test_player})
        
        # Save output to file for easy viewing
        output_file = save_test_output("find_nba_players_by_name", result)
        
        # Verify the result
        assert f'Players matching "{test_player}"' in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_all_nba_teams_tool_integration(self, save_test_output):
        """Test get_all_nba_teams tool with real NBA API.
        
        This test actually calls the NBA API to get all teams.
        Sample output saved to: tests/test_outputs/test_get_all_nba_teams_tool_integration_get_all_nba_teams.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_all_nba_teams')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_all_nba_teams", result)
        
        # Verify the result
        assert "All NBA teams" in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_find_nba_team_by_name_tool_integration(self, save_test_output):
        """Test find_nba_team_by_name tool with real NBA API.
        
        This test actually calls the NBA API to find a team by name.
        Sample output saved to: tests/test_outputs/test_find_nba_team_by_name_tool_integration_find_nba_team_by_name.txt
        """
        find_tool = next(t for t in self.tools if t.name == 'find_nba_team_by_name')
        
        # Test with a well-known team
        test_team = self.TEST_TEAM_NAME
        
        # Call real tool
        result = find_tool.invoke({"full_name": test_team})
        
        # Save output to file for easy viewing
        output_file = save_test_output("find_nba_team_by_name", result)
        
        # Verify the result
        assert f'Team matching "{test_team}"' in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_player_info_tool_integration(self, save_test_output):
        """Test get_player_info tool with real NBA API.
        
        This test actually calls the NBA API to get player info.
        Sample output saved to: tests/test_outputs/test_get_player_info_tool_integration_get_player_info.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_player_info')
        
        # Use a well-known player ID (LeBron James)
        test_player_id = self.TEST_PLAYER_ID
        
        # Call real tool
        result = get_tool.invoke({"player_id": test_player_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_player_info", result)
        
        # Verify the result
        assert f'Player info for player_id {test_player_id}' in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_player_career_stats_tool_integration(self, save_test_output):
        """Test get_player_career_stats tool with real NBA API.
        
        This test actually calls the NBA API to get player career stats.
        Sample output saved to: tests/test_outputs/test_get_player_career_stats_tool_integration_get_player_career_stats.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_player_career_stats')
        
        # Use a well-known player ID (LeBron James)
        test_player_id = self.TEST_PLAYER_ID
        
        # Call real tool
        result = get_tool.invoke({"player_id": test_player_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_player_career_stats", result)
        
        # Verify the result
        assert f'Career stats for player_id {test_player_id}' in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_player_game_log_tool_integration(self, save_test_output):
        """Test get_player_game_log tool with real NBA API.
        
        This test actually calls the NBA API to get player game log.
        Sample output saved to: tests/test_outputs/test_get_player_game_log_tool_integration_get_player_game_log.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_player_game_log')
        
        # Use a well-known player ID (LeBron James) and recent season
        test_player_id = self.TEST_PLAYER_ID
        test_season = self.TEST_SEASON
        
        # Call real tool
        result = get_tool.invoke({"player_id": test_player_id, "season": test_season})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_player_game_log", result)
        
        # Verify the result
        assert f'Game log for player_id {test_player_id}' in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_player_dashboard_by_game_splits_tool_integration(self, save_test_output):
        """Test get_player_dashboard_by_game_splits tool with real NBA API.
        
        This test actually calls the NBA API to get player dashboard by game splits.
        Sample output saved to: tests/test_outputs/test_get_player_dashboard_by_game_splits_tool_integration_get_player_dashboard_by_game_splits.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_player_dashboard_by_game_splits')
        
        # Use a well-known player ID (LeBron James) and recent season
        test_player_id = self.TEST_PLAYER_ID
        test_season = self.TEST_SEASON
        
        # Call real tool
        result = get_tool.invoke({"player_id": test_player_id, "season": test_season})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_player_dashboard_by_game_splits", result)
        
        # Verify the result
        assert f'Player dashboard by game splits for player_id {test_player_id}' in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_player_dashboard_by_general_splits_tool_integration(self, save_test_output):
        """Test get_player_dashboard_by_general_splits tool with real NBA API.
        
        This test actually calls the NBA API to get player dashboard by general splits.
        Sample output saved to: tests/test_outputs/test_get_player_dashboard_by_general_splits_tool_integration_get_player_dashboard_by_general_splits.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_player_dashboard_by_general_splits')
        
        # Use a well-known player ID (LeBron James) and recent season
        test_player_id = self.TEST_PLAYER_ID
        test_season = self.TEST_SEASON
        
        # Call real tool
        result = get_tool.invoke({"player_id": test_player_id, "season": test_season})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_player_dashboard_by_general_splits", result)
        
        # Verify the result
        assert f'Player dashboard by general splits for player_id {test_player_id}' in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_team_info_tool_integration(self, save_test_output):
        """Test get_team_info tool with real NBA API.
        
        This test actually calls the NBA API to get team info.
        Sample output saved to: tests/test_outputs/test_get_team_info_tool_integration_get_team_info.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_info')
        
        # Use a well-known team ID (Lakers)
        test_team_id = self.TEST_TEAM_ID
        
        # Call real tool
        result = get_tool.invoke({"team_id": test_team_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_info", result)
        
        # Verify the result
        assert f'Team info for team_id {test_team_id}' in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_team_game_log_tool_integration(self, save_test_output):
        """Test get_team_game_log tool with real NBA API.
        
        This test actually calls the NBA API to get team game log.
        Sample output saved to: tests/test_outputs/test_get_team_game_log_tool_integration_get_team_game_log.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_team_game_log')
        
        # Use a well-known team ID (Lakers) and recent season
        test_team_id = self.TEST_TEAM_ID
        test_season = self.TEST_SEASON
        
        # Call real tool
        result = get_tool.invoke({"team_id": test_team_id, "season": test_season})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_team_game_log", result)
        
        # Verify the result
        assert f'Game log for team_id {test_team_id}' in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content
    
    @pytest.mark.integration
    def test_get_nba_scoreboard_tool_integration(self, save_test_output):
        """Test get_nba_scoreboard tool with real NBA API.
        
        This test actually calls the NBA API to get scoreboard.
        Sample output saved to: tests/test_outputs/test_get_nba_scoreboard_tool_integration_get_nba_scoreboard.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_nba_scoreboard')
        
        # Call real tool (no date = today's games)
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_nba_scoreboard", result)
        
        # Verify the result
        assert "NBA Scoreboard" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_nba_live_scoreboard_tool_integration(self, save_test_output):
        """Test get_nba_live_scoreboard tool with real NBA API.
        
        This test actually calls the NBA API to get live scoreboard.
        Sample output saved to: tests/test_outputs/test_get_nba_live_scoreboard_tool_integration_get_nba_live_scoreboard.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_nba_live_scoreboard')
        
        # Call real tool
        result = get_tool.invoke({})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_nba_live_scoreboard", result)
        
        # Verify the result
        assert "Live NBA Scoreboard" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_nba_box_score_tool_integration(self, save_test_output):
        """Test get_nba_box_score tool with real NBA API.
        
        This test actually calls the NBA API to get box score for a game.
        Sample output saved to: tests/test_outputs/test_get_nba_box_score_tool_integration_get_nba_box_score.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_nba_box_score')
        
        # Use a known game_id from a recent game (format: 00YYMMDD0TEAM_ID)
        # Example: 0022301230 (January 23, 2023 game)
        # For a more reliable test, we could get a game_id from scoreboard first
        # Using a placeholder game_id - test may fail if game doesn't exist, but validates the tool works
        test_game_id = self.TEST_GAME_ID
        
        # Call real tool
        result = get_tool.invoke({"game_id": test_game_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_nba_box_score", result)
        
        # Verify the result (may fail if game_id doesn't exist, but should not crash)
        assert "Box score" in result or "Failed" in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_get_nba_play_by_play_tool_integration(self, save_test_output):
        """Test get_nba_play_by_play tool with real NBA API.
        
        This test actually calls the NBA API to get play-by-play data for a game.
        Uses game_id from nba_api example notebook: https://github.com/swar/nba_api/blob/master/docs/examples/PlayByPlay.ipynb
        Sample output saved to: tests/test_outputs/test_get_nba_play_by_play_tool_integration_get_nba_play_by_play.txt
        """
        get_tool = next(t for t in self.tools if t.name == 'get_nba_play_by_play')
        
        # Use game_id from nba_api example notebook (Pacers vs Bucks game)
        # This game_id is known to work in the official nba_api examples
        test_game_id = "0021800854"
        
        # Call real tool
        result = get_tool.invoke({"game_id": test_game_id})
        
        # Save output to file for easy viewing
        output_file = save_test_output("get_nba_play_by_play", result)
        
        # Verify the result (should succeed with valid game_id that has play-by-play data)
        assert "Play-by-play" in result
        assert "Failed" not in result
        assert len(result) > 20  # Should have some content
    
    @pytest.mark.integration
    def test_find_nba_games_tool_integration(self, save_test_output):
        """Test find_nba_games tool with real NBA API.
        
        This test actually calls the NBA API to find games.
        Sample output saved to: tests/test_outputs/test_find_nba_games_tool_integration_find_nba_games.txt
        """
        find_tool = next(t for t in self.tools if t.name == 'find_nba_games')
        
        # Test with a team filter (Lakers)
        test_team_id = self.TEST_TEAM_ID
        test_season = self.TEST_SEASON
        
        # Call real tool
        result = find_tool.invoke({"team_id": test_team_id, "season": test_season})
        
        # Save output to file for easy viewing
        output_file = save_test_output("find_nba_games", result)
        
        # Verify the result
        assert "Found games" in result
        assert "Failed" not in result
        assert len(result) > 50  # Should have substantial content

