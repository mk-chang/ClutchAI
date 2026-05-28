import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, call

POOL_SIZE = 150

def _make_pool_df(n=160):
    """Create a fake player pool DataFrame with realistic variance."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'player_id':   range(n),
        'season':      ['2025-26'] * n,
        'min':         rng.uniform(15, 38, n),
        'pts':         rng.uniform(5, 32, n),
        'reb':         rng.uniform(2, 14, n),
        'ast':         rng.uniform(1, 12, n),
        'stl':         rng.uniform(0.3, 2.5, n),
        'blk':         rng.uniform(0.1, 3.0, n),
        'tov':         rng.uniform(0.8, 4.5, n),
        'fg3m':        rng.uniform(0, 4.5, n),
        'fg_pct':      rng.uniform(0.38, 0.65, n),
        'fga':         rng.uniform(3, 20, n),
        'ft_pct':      rng.uniform(0.60, 0.95, n),
        'fta':         rng.uniform(0.5, 9, n),
    })


def _make_calculator():
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_pg = MagicMock()
    mock_pg.get_engine.return_value = mock_engine
    from data.postgres.player_value import PlayerValueCalculator
    return PlayerValueCalculator(mock_pg), mock_conn


# --- z-score functions ---

def test_z_score_of_mean_is_zero():
    from data.postgres.player_value import z_score
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = z_score(s)
    assert abs(result[2]) < 1e-10  # middle value (mean) ≈ 0

def test_z_score_negated_flips_sign():
    from data.postgres.player_value import z_score
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    pos = z_score(s)
    neg = z_score(s, negate=True)
    pd.testing.assert_series_equal(pos, -neg)

def test_z_score_volume_weighted_fg():
    from data.postgres.player_value import z_score_volume_weighted
    pct   = pd.Series([0.50, 0.45, 0.55])
    vol   = pd.Series([10.0, 5.0, 20.0])
    result = z_score_volume_weighted(pct, vol)
    assert isinstance(result, pd.Series)
    assert len(result) == 3


# --- pool selection ---

def test_pool_is_top_150_by_min():
    from data.postgres.player_value import select_pool
    df = _make_pool_df(n=200)
    pool = select_pool(df, size=150)
    assert len(pool) == 150
    assert pool['min'].min() >= df.nlargest(150, 'min')['min'].min()


# --- calculate ---

def test_calculate_writes_z_scores_back_to_pg():
    calc, mock_conn = _make_calculator()
    df = _make_pool_df(n=160)
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [tuple(row) for _, row in df.iterrows()]
    mock_result.keys.return_value = list(df.columns)
    mock_conn.execute.return_value = mock_result

    with patch.object(calc, '_read_table', return_value=df):
        with patch.object(calc, '_write_values') as mock_write:
            calc.calculate('2025-26', table='pg')
            mock_write.assert_called_once()
            written_df = mock_write.call_args[0][0]
            assert 'z_pts' in written_df.columns
            assert 'z_fg' in written_df.columns
            assert 'z_3ptm' in written_df.columns
            assert 'rv' in written_df.columns

def test_calculate_rv_is_sum_of_nine_z_scores():
    from data.postgres.player_value import PlayerValueCalculator
    calc = PlayerValueCalculator(MagicMock())
    df = _make_pool_df(n=160)
    result = calc._compute_values(df, include_pv=True)
    z_cols = ['z_pts', 'z_reb', 'z_ast', 'z_stl', 'z_blk',
              'z_3ptm', 'z_tov', 'z_fg', 'z_ft']
    expected_rv = result[z_cols].sum(axis=1)
    pd.testing.assert_series_equal(result['rv'], expected_rv, check_names=False)

def test_calculate_three_v_equals_z_3ptm():
    from data.postgres.player_value import PlayerValueCalculator
    calc = PlayerValueCalculator(MagicMock())
    df = _make_pool_df(n=160)
    result = calc._compute_values(df, include_pv=True)
    pd.testing.assert_series_equal(result['three_v'], result['z_3ptm'], check_names=False)

def test_calculate_pv_replacement_is_150th():
    from data.postgres.player_value import PlayerValueCalculator
    calc = PlayerValueCalculator(MagicMock())
    df = _make_pool_df(n=160)
    result = calc._compute_values(df, include_pv=True)
    # Player ranked exactly 150th by raw_pv should have pv ≈ 0
    raw_pv = (df['pts'] * 1.0 + df['reb'] * 1.2 + df['ast'] * 1.5
              + df['stl'] * 3.0 + df['blk'] * 3.0 + df['tov'] * -1.0)
    replacement = raw_pv.nlargest(150).iloc[-1]
    assert abs((raw_pv - replacement).iloc[0] - result['pv'].iloc[0]) < 1e-6
