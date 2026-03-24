""" 
Scientific Computing Project 2: Asteroid Families

Elin O'Neill

March 24th, 2026

Identifying Asteroid Groups/Families using the Hierarchical Clustering Method
(HCM) and a K-D Tree for more efficient neighbor searches. 

This implementation follows the standard HCM approach:
    1. Select a seed asteroid
    2. Find nearby asteroids within a velocity cutoff
    3. Add them to a cluster (family)
    4. Continually merge nearby asteroids/clusters with one another
    5. Expand until no new members are found

"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import parsing


def hcm_kdtree(df, cutoff=50.0, min_cluster_size=20):
    """
    Perform asteroid family detection using the Hierarchical Clustering Method
    (HCM) and a K-D tree to accelerate neighbor searches.

    Args:
        df (pandas.DataFrame): DataFrame containing asteroid proper elements.
            Required columns:
                - 'a_AU' : semi-major axis (AU)
                - 'e' : eccentricity
                - 'sin_I' : sine of inclination (deg)
                - 'n_deg_per_yr' : mean motion (deg/year)
        cutoff (float): Velocity cutoff threshold for clustering (m/s).
        min_cluster_size (int): Minimum number of members required to form a
            family. Smaller clusters are labeled as noise.

    Returns:
        pandas.Series: Cluster labels for each asteroid
            (0, 1, 2, 3,... for detected families and -1 for noise).
    """

    N = len(df)

    # Extract proper elements as NumPy arrays
    a = df["a_AU"].values
    e = df["e"].values
    sini = df["sin_I"].values

    # Mean motion (convert from deg/year to rad/s)
    n = np.deg2rad(df["n_deg_per_yr"].values) / (365.25 * 24 * 3600)

    # Normalize distances to approximate HCM metric
    X = np.column_stack([
        np.sqrt(5/4) * (a - np.mean(a)) / np.mean(a),
        np.sqrt(2) * e,
        np.sqrt(2) * sini
    ])

    # Build K-D Tree: O(log N) time complexity for neighbor lookup!
    tree = KDTree(X)

    visited = np.zeros(N, dtype=bool)
    labels = -np.ones(N, dtype=int) # Initialize all labels as noise (-1)

    cluster_id = 0

    #K-D Tree radius for search
    # Larger radius: slower but more complete search; risks over-connecting
    # Smaller radius: faster but risks missing valid links
    radius = 0.0015 # value chosen for good completeness, purity, and runtime

    # Main Clustering Loop
    for i in range(N):
        # Progress tracking in terminal
        if i % 1000 == 0:
            print(f"Processing {i}/{N}")

        # Skip if already assigned to a cluster
        if visited[i]:
            continue

        # Start a new cluster from seed asteroid i
        stack = [i]
        cluster_members = []
        visited[i] = True

        # Depth-first search to grow cluster
        while stack:
            current = stack.pop()
            cluster_members.append(current)

            # Fast K-D Tree neighbor query
            neighbors = tree.query_radius(
                X[current].reshape(1, -1),
                r=radius
            )[0]

            for j in neighbors:
                if visited[j]:
                    continue

                # Compute exact HCM velocity distance
                d_a = (a[current] - a[j]) / a[current]
                d_e = e[current] - e[j]
                d_sini = sini[current] - sini[j]

                d = n[current] * a[current] * np.sqrt(
                    (5/4)*d_a*d_a + 2*d_e*d_e + 2*d_sini*d_sini
                )

                # Only accept if within cutoff
                if d < cutoff:
                    visited[j] = True
                    stack.append(j)

        # Assign cluster label if large enough
        if len(cluster_members) >= min_cluster_size:
            for idx in cluster_members:
                labels[idx] = cluster_id
            cluster_id += 1

    return pd.Series(labels, index=df.index)


def compute_completeness(merged):
    """
    Compute completeness of detected clusters relative to known families.

    Determined by calculating the fraction of how much of the real family was
    recovered.

    Args:
        merged (pd.DataFrame): DataFrame containing:
            - 'cluster' : detected cluster labels
            - 'family1' : reference (true) family labels

    Returns:
        pd.DataFrame: results per family Columns:
            - family : reference family ID
            - best_cluster : cluster capturing most members
            - completeness : fraction recovered
            - size : total family size
    """

    results = []

    # Loop over real families
    for fam in merged["family1"].unique():
        # Skip background/unlabeled objects
        if fam == 0 or pd.isna(fam):
            continue

        subset = merged[merged["family1"] == fam]
        total = len(subset)

        if total == 0:
            continue

        # Count how family members are distributed across clusters
        cluster_counts = subset["cluster"].value_counts()

        # Ignore noise cluster (-1)
        cluster_counts = cluster_counts[cluster_counts.index != -1]

        # If no cluster captured any members
        if len(cluster_counts) == 0:
            results.append({
                "family": fam,
                "best_cluster": None,
                "completeness": 0.0,
                "size": total
            })
            continue

        # Best cluster = one with most family members
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
    Compute purity of detected clusters.

    Purity measures:
        "How clean is this cluster?"
        (i.e., how much it is dominated by a single true family)

    For each detected cluster:
        - Find most common reference family
        - Compute fraction of cluster belonging to that family

    Args:
        merged (pd.DataFrame): DataFrame containing: cluster' and 'family1'

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - cluster : detected cluster ID
            - best_family : dominant true family
            - purity : fraction of cluster from that family
            - size : cluster size
    """

    results = []

    for cluster in merged["cluster"].unique():
        # Skip noise
        if cluster == -1:
            continue

        subset = merged[merged["cluster"] == cluster]

        # Remove asteroids with no known family
        subset = subset.dropna(subset=["family1"])

        if len(subset) == 0:
            continue 

        fam_counts = subset["family1"].value_counts()

        if len(fam_counts) == 0:
            continue

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
    Execute full asteroid family detection pipeline.

    Steps:
        1. Load asteroid proper elements
        2. Run HCM clustering
        3. Merge with reference family labels
        4. Evaluate completeness and purity
        5. Save results to csv files
        6. Print most relevant data
    """

    # Load datasets
    elements = parsing.load_asteroid_file(
        "asteroid-data/synthetic_proper_elements.txt"
    )

    families_ref = parsing.load_family_file(
        "asteroid-data/indiv_ast_fam_membership.txt"
    )

    # Run KDTree HCM
    elements["cluster"] = hcm_kdtree(
        elements,
        cutoff=50.0,
        min_cluster_size=20
    )

    # Merge with reference
    merged = elements.merge(
        families_ref,
        left_on="name",
        right_on="ast_name",
        how="left"
    )

    elements.to_csv("results/hcm_results.csv", index=False)
    merged.to_csv("results/hcm_with_reference.csv", index=False)

    # Check for completeness
    results = compute_completeness(merged)
    good_families = results[results["completeness"] >= 0.95]

    print(f"\nFamilies ≥95% completeness: {len(good_families)}")

    # Test Merge Purity
    purity_results = compute_purity(merged)

    # Combine completeness + purity results
    combined = results.merge(
        purity_results,
        left_on="best_cluster",
        right_on="cluster",
        how="left"
    )

    top8_comp_pure = combined[
        (combined["purity"] >= 0.9) & 
        (combined["completeness"].notna())
    ].sort_values("completeness", ascending=False).head(8)

    print("\nTop 8 families by completeness with ≥0.9 purity:")
    print(top8_comp_pure[["family", "best_cluster", "completeness", "purity"]])
