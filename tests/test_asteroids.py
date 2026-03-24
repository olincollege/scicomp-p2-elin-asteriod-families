"""
Unit tests for asteroid family detection project.

Tests cover:
    - HCM clustering behavior
    - Completeness calculation
    - Purity calculation
    - File parsing functions

Run with:
    pytest tests/
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
from src.main import hcm_kdtree, compute_completeness, compute_purity
from src.parsing import load_asteroid_file, load_family_file

# ---------
# FIXTURES
# ---------

@pytest.fixture
def small_asteroid_df():
    """
    Create a small synthetic dataset of asteroid proper elements.

    Designed so that:
        - First two asteroids are very close (should cluster together)
        - Last two asteroids are very close (second cluster)

    Returns:
        pd.DataFrame: small dataset for clustering tests
    """
    return pd.DataFrame({
        "a_AU": [2.5, 2.5001, 3.0, 3.0001],
        "e": [0.1, 0.1001, 0.2, 0.2001],
        "sin_I": [0.05, 0.0501, 0.1, 0.1001],
        "n_deg_per_yr": [100, 100, 80, 80],
    })


@pytest.fixture
def merged_df():
    """
    Create a small merged dataset for completeness and purity testing.

    Structure:
        - Cluster 0 --> pure family 10
        - Cluster 1 --> mixed families (20 and 30)
        - Noise point (-1) included

    Returns:
        pd.DataFrame: synthetic merged dataset
    """
    return pd.DataFrame({
        "cluster": [0, 0, 1, 1, 1, -1],
        "family1": [10, 10, 20, 20, 30, 10],
    })


# ----------
# HCM TESTS
# ----------

def test_hcm_returns_series(small_asteroid_df):
    """
    Test that HCM returns a pandas Series with correct length.
    """
    labels = hcm_kdtree(small_asteroid_df, cutoff=1000, min_cluster_size=1)

    assert isinstance(labels, pd.Series)
    assert len(labels) == len(small_asteroid_df)


def test_hcm_detects_clusters(small_asteroid_df):
    """
    Test that clustering identifies at least one non-noise cluster.
    """
    labels = hcm_kdtree(small_asteroid_df, cutoff=1000, min_cluster_size=1)

    # At least one asteroid should not be labeled as noise (-1)
    assert (labels != -1).any()


def test_hcm_respects_min_cluster_size(small_asteroid_df):
    """
    Test that clusters smaller than min_cluster_size are labeled as noise.
    """
    labels = hcm_kdtree(small_asteroid_df, cutoff=1000, min_cluster_size=10)

    assert (labels == -1).all()


# -------------------
# COMPLETENESS TESTS
# -------------------

def test_compute_completeness_basic(merged_df):
    """
    Test completeness calculation on a simple dataset.

    Ensures:
        - Output contains expected columns
        - Known completeness value is correct
    """
    results = compute_completeness(merged_df)

    # Verify required output columns exist
    assert set(["family", "best_cluster", "completeness", "size"]
               ).issubset(results.columns)

    fam10 = results[results["family"] == 10].iloc[0]
    assert fam10["completeness"] == pytest.approx(2/3)


def test_compute_completeness_handles_noise():
    """
    Test that families mapped only to noise clusters have zero completeness.
    """
    df = pd.DataFrame({
        "cluster": [-1, -1, -1],
        "family1": [1, 1, 1],
    })

    results = compute_completeness(df)

    # No cluster captures any members --> completeness = 0
    assert results.iloc[0]["completeness"] == 0.0


# -------------
# PURITY TESTS
# -------------

def test_compute_purity_basic(merged_df):
    """
    Test purity calculation for a simple dataset.

    Ensures:
        - Output structure is correct
        - Pure cluster is identified correctly
    """
    results = compute_purity(merged_df)

    # Verify expected columns
    assert set(["cluster", "best_family", "purity", "size"]).issubset(results.columns)

    # Cluster 0 contains only family 10 --> purity = 1
    cluster0 = results[results["cluster"] == 0].iloc[0]
    assert cluster0["purity"] == pytest.approx(1.0)


def test_compute_purity_mixed_cluster():
    """
    Test purity for a cluster containing multiple families.

    Expected:
        purity < 1 since cluster is mixed.
    """
    df = pd.DataFrame({
        "cluster": [0, 0, 0],
        "family1": [1, 1, 2],
    })

    results = compute_purity(df)

    purity = results.iloc[0]["purity"]

    # Two out of three belong to dominant family --> purity = 2/3
    assert purity < 1.0
    assert purity == pytest.approx(2/3)


def test_compute_purity_ignores_noise():
    """
    Test that noise cluster (-1) is excluded from purity results.
    """
    df = pd.DataFrame({
        "cluster": [-1, -1, 0],
        "family1": [1, 1, 1],
    })

    results = compute_purity(df)

    # Noise cluster should not appear
    assert (-1 not in results["cluster"].values)


# --------------
# PARSING TESTS
# --------------

def test_load_asteroid_file(tmp_path):
    """
    Test parsing of asteroid proper elements file.

    Ensures:
        - File is correctly read into DataFrame
        - Expected columns exist
        - Row count matches input
    """
    test_file = tmp_path / "test_asteroids.txt"

    # Create small mock file
    test_file.write_text(
        "A1 10 2.5 0.1 0.05 100 0 0 0 0\n"
        "A2 11 2.6 0.2 0.06 101 0 0 0 0\n"
    )

    df = load_asteroid_file(test_file)

    assert not df.empty
    assert "a_AU" in df.columns
    assert len(df) == 2


def test_load_family_file(tmp_path):
    """
    Test parsing of asteroid family classification file.

    Ensures:
        - Data is loaded correctly
        - Key columns exist
        - String columns are properly typed
    """
    test_file = tmp_path / "test_families.txt"

    # Create small mock file
    test_file.write_text(
        "A1 10 1 100 5 A2 0 0 A3 RES\n"
        "A2 11 1 200 6 A1 0 0 A3 RES\n"
    )

    df = load_family_file(test_file)

    assert not df.empty
    assert "family1" in df.columns

    # Ensure ast_name is treated as string (important for merging)
    assert df["ast_name"].dtype.name == "string"
