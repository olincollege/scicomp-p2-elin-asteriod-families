""" 
Parsing Functions
"""

import pandas as pd


def load_asteroid_file(filepath):
    """
    Parse the synthetic proper elements asteroid file into a pandas DataFrame.

    Args:
        filepath (str): Path to the synthetic proper elements asteroid file.

    Returns:
        pandas.DataFrame: a DataFrame containing asteroid proper elements data.
    """

    columns = [
        "name",
        "mag",
        "a_AU",
        "e",
        "sin_I",
        "n_deg_per_yr",
        "g_arcsec_per_yr",
        "s_arcsec_per_yr",
        "LCEx1E6",
        "My"
    ]

    df = pd.read_csv(
        filepath,
        comment="%",   # ignore metadata lines
        sep=r"\s+",    # split on arbitrary whitespace
        names=columns,
        engine="python"   # ensures regex separator works
    )

    return df


if __name__ == "__main__":
    file_path = "asteroid-data/synthetic_proper_elements.txt"
    df = load_asteroid_file(file_path)

    print(df.head())
    print(df.info())


def load_family_file(filepath):
    """
    Parse the asteroid family classification file into a pandas DataFrame.

    Args:
        filepath (str): Path to the asteroid family file.

    Returns:
        pandas.DataFrame: a DataFrame containing asteroid family data.
    """

    columns = [
        "ast_name",
        "Hmag",
        "status",
        "family1",
        "dv_fam1",
        "near1",
        "family2",
        "dv_fam2",
        "near2",
        "rescod"
    ]

    df = pd.read_csv(
        filepath,
        comment="%",
        sep=r"\s+",
        names=columns,
        engine="python"
    )

    # Ensure string columns are strings
    df["ast_name"] = df["ast_name"].astype("string")
    df["rescod"] = df["rescod"].astype("string")

    # Automatically downcast numeric columns
    for col in df.select_dtypes(include=["int", "float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


if __name__ == "__main__":
    file_path = "asteroid-data/indiv_ast_fam_membership.txt"
    df = load_family_file(file_path)

    print(df.head())
    print(df.info())
