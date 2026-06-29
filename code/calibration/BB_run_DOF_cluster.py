from model_DOF_clustering_5 import ObjectiveWrapper, load_ko_and_expectation_data

ko_df, expectations_df = load_ko_and_expectation_data()
wrapped_objective = ObjectiveWrapper(ko_df, expectations_df)

from blackbox import minimize
import numpy as np
import os

n_families = 5
bounds = [(0.01, 1.0)] * n_families

os.environ["BLACKBOX_SERIAL"] = "1"

result = minimize(
    wrapped_objective,
    domain=bounds,
    batch=48,
    budget=5760,
)

print("\n✅ Optimization complete")