# Scientific Computing Project 2: Asteroid Families

### Elin O'Neill

### March 24, 2026

### Overview

This project identifies asteroid families using the Hierarchical Clustering
Method (HCM) and K-D Trees. The goal is to identify at least 8 asteroid families
to 95% or higher completion. This project is implemented in Python.

### Instructions

1.  Clone the repository to your local machine
2.  Navigate to your cloned repository using
    `cd scicomp-p2-elin-asteroid-families`
3.  Install requirements using `pip install -r requirements.txt`
4.  Since the datasets may be too large to download directly from GitHub, as
    needed, download the following files from
    [ AstDyS-2](https://newton.spacedys.com/astdys2/index.php?pc=5) and place
    them in the `asteroid-data` folder:
    - "Numbered and multiopposition asteroids"; rename it to
      `synthetic_proper_elements.txt`
    - "Individual asteroid family membership"; rename it to
      `indiv_ast_fam_membership.txt`
5.  Run the algorithm from your terminal using `python src/main.py`
6.  Wait for the algorithm to complete (status shown in terminal)
7.  View top results printed in the terminal
8.  Optionally, view the files `results/hcm_results.csv` and
    `results/hcm_with_reference.csv` to view the complete results
9.  Optionally, run unit tests with `pytest tests/`

### Proper Orbital Elements

The variables used to group the asteroids into families.

- a: proper semi-major axis
- e: proper eccentricity
- sin(i): sine of proper inclination

### Adjustable Parameter

- radius: the radius of the K-D Tree search. The radius controls how far the
  algorithm looks for nearby asteroids when forming clusters. Increasing the
  radius improves connectivity and cluster growth but may lead to merging of
  distinct families, while decreasing it reduces false connections at the cost
  of fragmenting real families. The current radius value of 0.0015 was chosen
  because of its results producing high purity and completeness while limiting
  the runtime of the algorithm.

### Tests

To run the provided unit tests, run `pytest tests/` in your terminal.

### Results and Benchmark

{---------------------------TODO---------------}

### References

“Groups of Asteroids Probably of Common Origin” (1919) by Kiyotsugu Hirayama.

"Characterising the efficiency of the hierarchical clustering method" (2026) by
Andrew Marshall-Lee et al.

"Asteroid Families. I. Identification By Hierarchical Clustering and Reliability
Assessment" (1990) by Vincenzo Zappalà and Alberto Cellino
