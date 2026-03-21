""" 
Scientific Computing Project 2: Asteroid Families

Elin O'Neill

March 24th, 2026

Identifying Asteroid Groups/Families using the Hierarchical Clustering Method
(HCM). 

"""

# notes to self
    # site papers (zappala)
    # look over project elements doc before submission
    # update readme!!!!
    # tests!
    # do at least one refactoring commit



# Proper elements (remain stable!)
    # a = proper semi-major axis
    # e = proper eccentricity
    # sin(i) = sine of proper inclination


# n = mean motion 
# d = distance 


# HCM
# 1. Choose a seed asteroid
# 2. Find neighbors within cutoff velocity
# 3. Add them to the family
# 4. Repeat until cluster stops growing

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import parsing


def run_hcm_kdtree(df, cutoff=50.0, min_cluster_size=20):
    """
    Run the Hierarchical Clustering Method (HCM) using a KDTree.

    Args:
        df (pandas.DataFrame): DataFrame containing asteroid proper elements.
            Must include columns: 'a_AU', 'e', 'sin_I', 'n_deg_per_yr'.
        cutoff (float): Velocity cutoff threshold for clustering (m/s).
        min_cluster_size (int): Minimum number of members required to form a family.

    Returns:
        pandas.Series: Cluster labels for each asteroid (-1 for noise).
    """

    N = len(df)

    # Extract arrays
    a = df["a_AU"].values
    e = df["e"].values
    sini = df["sin_I"].values

    # Mean motion (convert to rad/s)
    n = np.deg2rad(df["n_deg_per_yr"].values) / (365.25 * 24 * 3600)

    # Normalize distances to approximate HCM metric
    X = np.column_stack([
        np.sqrt(5/4) * (a - np.mean(a)) / np.mean(a),
        np.sqrt(2) * e,
        np.sqrt(2) * sini
    ])

    tree = KDTree(X)

    visited = np.zeros(N, dtype=bool)
    labels = -np.ones(N, dtype=int)

    cluster_id = 0

    ### TODO: make radius a calculated estimate???
    radius = 0.0015   # PLAY AROUND WITH THIS VALUE. higher val = higher connectivity and longer runtime

    for i in range(N):
        if i % 1000 == 0:
            print(f"Processing {i}/{N}")

        if visited[i]:
            continue

        stack = [i]
        cluster_members = []
        visited[i] = True

        while stack:
            current = stack.pop()
            cluster_members.append(current)

            # FAST neighbor query
            neighbors = tree.query_radius(
                X[current].reshape(1, -1),
                r=radius
            )[0]

            for j in neighbors:
                if visited[j]:
                    continue

                # Compute exact HCM distance (important!)
                da = (a[current] - a[j]) / a[current]
                de = e[current] - e[j]
                dsini = sini[current] - sini[j]

                d = n[current] * a[current] * np.sqrt(
                    (5/4)*da*da + 2*de*de + 2*dsini*dsini
                )

                if d < cutoff:
                    visited[j] = True
                    stack.append(j)

        # Assign cluster if large enough
        if len(cluster_members) >= min_cluster_size:
            for idx in cluster_members:
                labels[idx] = cluster_id
            cluster_id += 1

    return pd.Series(labels, index=df.index)

def compute_completeness(merged):
    """
    Compute completeness for each reference asteroid family.

    Args:
        merged (pd.DataFrame): DataFrame with 'cluster' and 'family1'

    Returns:
        pd.DataFrame: completeness results per family
    """

    results = []

    # Loop over real families
    for fam in merged["family1"].unique():
        if fam == 0 or pd.isna(fam):
            continue

        subset = merged[merged["family1"] == fam]
        total = len(subset)

        if total == 0:
            continue

        cluster_counts = subset["cluster"].value_counts()

        # Remove noise (-1)
        cluster_counts = cluster_counts[cluster_counts.index != -1]

        if len(cluster_counts) == 0:
            results.append({
                "family": fam,
                "best_cluster": None,
                "completeness": 0.0,
                "size": total
            })
            continue

        best_cluster = cluster_counts.idxmax()
        correct = cluster_counts.max()
        completeness = correct / total

        results.append({
            "family": fam,
            "best_cluster": best_cluster,
            "completeness": completeness,
            "size": total
        })

    return pd.DataFrame(results).sort_values(
        "completeness", ascending=False
    )

def compute_purity(merged):
    """
    Compute how much of a cluster belongs to a single
    reference family (purity) for each detected cluster.

    Args:
        merged (pd.DataFrame): DataFrame with 'cluster' and 'family1'

    Returns:
        pd.DataFrame: purity results per cluster
    """

    results = []

    for cluster in merged["cluster"].unique():
        if cluster == -1:
            continue

        subset = merged[merged["cluster"] == cluster]

        # Remove asteroids with no known family
        subset = subset.dropna(subset=["family1"])

        if len(subset) == 0:
            continue  # nothing to evaluate

        fam_counts = subset["family1"].value_counts()

        if len(fam_counts) == 0:
            continue  # extra safety

        best_family = fam_counts.idxmax()
        correct = fam_counts.max()
        total = len(subset)

        purity = correct / total

        results.append({
            "cluster": cluster,
            "best_family": best_family,
            "purity": purity,
            "size": total
        })

    return pd.DataFrame(results).sort_values("purity", ascending=False)


if __name__ == "__main__":
    """
    Run the HCM clustering pipeline using KDTree.

    Loads data, runs clustering, saves results, compares completeness.
    """

    # Load datasets
    elements = parsing.load_asteroid_file(
        "asteroid-data/synthetic_proper_elements.txt"
    )

    families_ref = parsing.load_family_file(
        "asteroid-data/indiv_ast_fam_membership.txt"
    )

    print("datasets parsed")

    # Run KDTree HCM
    elements["cluster"] = run_hcm_kdtree(
        elements,
        cutoff=50.0,
        min_cluster_size=20
    )

    print("clustering complete")

    print("\nTop detected clusters:")
    print(elements["cluster"].value_counts().head(10))

    merged = elements.merge(
        families_ref,
        left_on="name",
        right_on="ast_name",
        how="left"
    )

    print("\nMerged dataset preview:")
    print(merged.head())

    elements.to_csv("hcm_results.csv", index=False)
    merged.to_csv("hcm_with_reference.csv", index=False)


    # check for completeness
    results = compute_completeness(merged)

    print("\nTop families by completeness:")
    print(results.head(10))

    good_families = results[results["completeness"] >= 0.95]

    print(f"\nFamilies ≥95% completeness: {len(good_families)}")

    top8 = results[results["completeness"] >= 0.95].head(8)

    print("\nTop 8 families meeting benchmark:")
    print(top8)

    print(elements["cluster"].value_counts().head(10))

    print("\nFamilies with ZERO detection:")
    print(results[results["best_cluster"].isna()].head())


    # test merge purity
    purity_results = compute_purity(merged)

    combined = results.merge(
        purity_results,
        left_on="best_cluster",
        right_on="cluster",
        how="left"
    )

    top50 = combined.sort_values(
        "completeness", ascending=False
    ).head(50)

    print("\nTop 50 families by completeness (with purity):")
    print(top50[["family", "best_cluster", "completeness", "purity"]])
