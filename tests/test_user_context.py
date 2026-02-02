"""
Integration test for user context gathering.

This test verifies that the UserContextGatherer class (from agents.multi_agent.user_context)
correctly:
- Fetches current date/time
- Finds the user's team by name
- Retrieves team information (ID, key)
- Retrieves league information (size, scoring type, metadata)
- Retrieves current week
- Retrieves team standings
- Retrieves team roster

The test uses MultiAgentSystem._gather_user_context() which uses the UserContextGatherer class.

These tests require:
- YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET environment variables set
- Real Yahoo Fantasy Sports API access
- Valid OAuth tokens
- YAHOO_LEAGUE_ID environment variable (or defaults to 58930)

Run with: pytest -m integration tests/test_user_context.py
"""

import os
import yaml
from pathlib import Path
import pytest
from datetime import datetime

from agents.multi_agent import MultiAgentSystem


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
class TestUserContext:
    """Integration tests for user context gathering."""
    
    # Load test configuration from YAML file
    _test_config = _load_test_config()
    
    # Test team name (KATmandu Climbers from league 58930)
    TEST_TEAM_NAME = "KATmandu Climbers"
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for integration tests."""
        # Initialize MultiAgentSystem with environment variables
        self.system = MultiAgentSystem(
            yahoo_league_id=int(os.environ.get('YAHOO_LEAGUE_ID', '58930')),
            yahoo_client_id=os.environ.get('YAHOO_CLIENT_ID'),
            yahoo_client_secret=os.environ.get('YAHOO_CLIENT_SECRET'),
            openai_api_key=os.environ.get('OPENAI_API_KEY', 'test_key'),  # Required but not used for context gathering
            team_name=self.TEST_TEAM_NAME,
            debug=True,  # Enable debug to see context in logs
        )
        yield
    
    @pytest.mark.integration
    def test_gather_user_context(self, save_test_output):
        """Test UserContextGatherer class with real Yahoo API.
        
        This test verifies that:
        1. User context is successfully gathered
        2. Context includes current date/time
        3. Context includes team information (name, ID, key)
        4. Context includes league information (size, scoring type, metadata)
        5. If current week is available, context includes current week, team standings, and roster
        6. If current week is not available (off-season), standings and roster are not included
        
        The test uses MultiAgentSystem._gather_user_context() which uses the UserContextGatherer
        class from agents.multi_agent.user_context.
        
        Sample output saved to: tests/test_outputs/test_user_context/test_gather_user_context_user_context.txt
        """
        # Call the method to gather user context (uses UserContextGatherer class)
        user_context = self.system._gather_user_context()
        
        # Verify context is not empty
        assert user_context, "User context should not be empty"
        assert len(user_context) > 0, "User context should have content"
        
        # Save output for inspection
        save_test_output("user_context", user_context)
        
        # Verify context includes expected sections (always present)
        assert "=== CURRENT DATE/TIME ===" in user_context, "Context should include current date/time"
        assert "=== USER TEAM INFO ===" in user_context, "Context should include user team info"
        assert "=== LEAGUE INFO ===" in user_context, "Context should include league info"
        
        # Verify current date/time is present
        assert "Current Date/Time:" in user_context, "Context should include current date/time value"
        
        # Verify team information is present
        assert f"Team Name: {self.TEST_TEAM_NAME}" in user_context, f"Context should include team name: {self.TEST_TEAM_NAME}"
        assert "Team ID:" in user_context, "Context should include team ID"
        assert "Team Key:" in user_context, "Context should include team key"
        
        # Verify league information is present
        assert "League Size:" in user_context, "Context should include league size"
        assert "League Scoring Type:" in user_context, "Context should include league scoring type"
        assert "League Metadata:" in user_context, "Context should include league metadata"
        
        # Verify league metadata fields
        assert "League Name:" in user_context or "League Key:" in user_context, "Context should include league name or key"
        
        # Check if current week is available
        has_current_week = "Current Week:" in user_context
        
        if has_current_week:
            # If current week is present, verify it's a number
            import re
            week_match = re.search(r'Current Week: (\d+)', user_context)
            if week_match:
                week_num = int(week_match.group(1))
                assert week_num > 0, "Current week should be a positive number"
            
            # Verify team standings are present (only when current week exists)
            assert "Team Standings:" in user_context, "Context should include team standings when current week is available"
            assert "Rank:" in user_context, "Context should include team rank when current week is available"
            
            # Verify roster information is present (only when current week exists)
            assert "Current Roster" in user_context, "Context should include current roster when current week is available"
        else:
            # During off-season, standings and roster should not be explicitly present
            # or should explicitly state unavailability
            assert "Current Week: Not available" in user_context, "Current Week status should be present"
            assert "Team Standings:" in user_context, "Standings should always be attempted"
            assert "Current Roster: Not available" in user_context, "Roster should state unavailability when current week is unavailable"
        
        # Print context for manual inspection
        print("\n" + "=" * 80)
        print("USER CONTEXT OUTPUT:")
        print("=" * 80)
        print(user_context)
        print("=" * 80 + "\n")
