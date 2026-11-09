
import csv
import NetfluxODE_5_params
import numpy as np 
import NetfluxODE_5
from scipy.integrate import ode 
import pandas as pd
import families_excel as families

# Load family map
n_families = 5

def load_family_id_vector(path="interactions_with_families.csv"):
    df = pd.read_csv(path)
    if 'family' in df.columns:
        col = 'family'
    elif 'Family' in df.columns:
        col = 'Family'
    else:
        raise ValueError("❌ 'family' column not found in the CSV.")
    
    # Drop rows with missing family values
    if df[col].isna().any():
        print("⚠️ Warning: NaN values found in family column. Dropping them.")
    
    clean_family_col = df[col].dropna().astype(int).tolist()
    return clean_family_col


family_id_vector = load_family_id_vector("interactions_with_families.csv")


# Load model parameters
[speciesNames, tau, ymax, y0, w_full, n_default, EC50_default] = NetfluxODE_5_params.loadParams()
wlen = len(w_full)

bas = 0.1
y0 = bas * np.ones(len(y0))

input_list = ['mechanical', 'infusionTGFB', 'AngII', 'infusionIL6', 'FGF', 'PDGF','ET1', 'TNFa', 'IL1']
input_indices = {inp: speciesNames.index(inp) for inp in input_list if inp in speciesNames}



# --- Top-level logging wrapper class (multiprocessing safe) ---
# Backward-compat alias: both names resolve to the same class so either
# `from model_DOF_clustering_5 import LoggingWrapper` or
# `from model_DOF_clustering_5 import LossLogger` works.
class LoggingWrapper:
    def __init__(self, func, log_file="loss_log_DOF_cluster.csv"):
        self.func = func
        self.log_file = log_file

    def __call__(self, params):
        params = np.array(params).flatten()   #NEW flatten
        try:
            loss = self.func(params)
        except Exception as e:
            print(f"⚠️ Error evaluating parameters: {e}")
            loss = 250  # Optional: Penalize if evaluation fails

        # Write to CSV
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(params) + [loss])
        return loss


# Alias for compatibility with newer launcher scripts
LossLogger = LoggingWrapper
def build_weight_vector(params, family_id_vector, fixed_input_weight=0.25, n_fixed_inputs=9):
    w_vec = np.array([params[family_id] for family_id in family_id_vector])
    w_vec[:n_fixed_inputs] = fixed_input_weight
    return w_vec


class ObjectiveWrapper:
    """Stage-2 objective: 5 cluster weights, with the Stage-1 scalars
    (EC50, n, b, p) held fixed. Default scalar values are read from
    stage1_best_point.csv if present; otherwise fall back to the paper's
    previously-reported Set B values. The caller can also pass scalars
    explicitly via the constructor.
    """

    _FALLBACK_SCALARS = {
        # Previous (buggy-model) Set B values — retained only as fallback
        # if stage1_best_point.csv is not available yet. They should be
        # overridden by the fresh Stage-1 fit before Stage 2 is run.
        "EC50": 0.587798,
        "n":    1.395697,
        "b":    0.283201,
        "p":    0.653814,
    }

    def __init__(self, ko_df, expectations_df, alpha=1, beta=1,
                 EC50=None, n=None, b=None, p=None):
        self.ko_df = ko_df
        self.expectations_df = expectations_df
        self.alpha = alpha
        self.beta = beta

        # Try to load Stage-1 output if none were passed explicitly
        if any(x is None for x in (EC50, n, b, p)):
            import os
            if os.path.exists("stage1_best_point.csv"):
                stage1 = np.loadtxt("stage1_best_point.csv", delimiter=",", skiprows=1)
                EC50, n, b, p = stage1[0], stage1[1], stage1[2], stage1[3]
                print(f"[ObjectiveWrapper] loaded Stage-1 scalars from stage1_best_point.csv: "
                      f"EC50={EC50:.4f}, n={n:.4f}, b={b:.4f}, p={p:.4f}")
            else:
                S = self._FALLBACK_SCALARS
                EC50, n, b, p = S["EC50"], S["n"], S["b"], S["p"]
                print(f"[ObjectiveWrapper] WARNING: stage1_best_point.csv not found, "
                      f"using fallback Set B scalars (previous buggy-model fit).")

        self.EC50 = float(EC50)
        self.n    = float(n)
        self.b    = float(b)
        self.p    = float(p)

    def __call__(self, params):
        params = np.array(params).flatten()

        if len(params) != n_families:
            print(f"❌ Expected {n_families} weights, got {len(params)}")
            return 999

        if self.b + self.p > 1:
            return 999

        EC50_vec = np.full(wlen, self.EC50)
        n_vec    = np.full(wlen, self.n)

        w_vec = build_weight_vector(params, family_id_vector)

        loss = objective_function(
            (EC50_vec, n_vec, self.b, self.p, w_vec),
            self.ko_df, self.expectations_df, self.alpha, self.beta,
        )
        return loss


def load_ko_and_expectation_data():
    # Load KO data
    ko_data = pd.read_excel('KO_data_condensed2.xlsx', sheet_name=None)
    ko_df = ko_data['Sheet1']
    if 'Unnamed: 0' in ko_df.columns:
        ko_df = ko_df.drop(columns=['Unnamed: 0'])

    # Load expectation data
    excel_path = "input_output_relations2.xlsx"
    input_output_expectations = pd.read_excel(excel_path, index_col=0).iloc[:, :11]

    # Set the "specie" column as the index and drop the old index row (row 1 was a header-like row)
    input_output_expectations_clean = input_output_expectations.copy()
    input_output_expectations_clean.index = input_output_expectations_clean['specie']
    input_output_expectations_clean.drop(columns=['specie'], inplace=True)

    # Drop any header-like rows (such as the first row which is usually dummy if index is misaligned)
    input_output_expectations_clean = input_output_expectations_clean[1:]

    df=input_output_expectations_clean

    return ko_df, df


run_model_counter = 0



# ---- MODEL RUN FUNCTION ----
def run_model_real(EC50_vec, n_vec, input_vector, w_family, family_id_vector, ymax_override=None, w_override=None):
    y0_run = y0.copy()
    y0_run[:len(input_vector)] = input_vector
    ymax_to_use = ymax_override if ymax_override is not None else ymax

    if w_override is not None:
        w = w_override.copy()
    else:
        w = build_weight_vector(w_family, family_id_vector)

    r = ode(NetfluxODE_5.ODEfunc).set_integrator(
        'vode', method='adams', order=10, rtol=1e-5, atol=1e-6, with_jacobian=False)
    
    r.set_initial_value(y0_run, 0).set_f_params(tau, ymax_to_use, w, n_vec, EC50_vec)

    sim_results = []
    tspan = [0, 600]  # FIX (reviewer concern #2): extended from 60 h to 600 h to reach steady state
    dt = tspan[1] / 6000.

    while r.successful() and r.t <= tspan[1]:
        r.integrate(r.t + dt)
        sim_results.append(r.y.copy())
    

    return sim_results[-1]



def compute_KO_mismatch(EC50_scalar, n_scalar, b_scalar, w, ko_df, family_id_vector):
    total, correct = 0, 0
    species_index = {name: i for i, name in enumerate(speciesNames)}

    # Reset the KO mismatch log at the start of each call so the file always
    # contains exactly one pass over the dataset (no append-mode accumulation).
    with open("KO_mismatch_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ko_condition", "target_species", "expected_behavior",
                         "model_behavior", "label", "delta"])

    # Build baseline input vector (without p)
    baseline_input = y0.copy()
    for name, idx in input_indices.items():
        baseline_input[idx] = b_scalar

    # Create EC50 and n as vectors
    EC50_vec = np.ones(wlen) * EC50_scalar
    n_vec = np.ones(wlen) * n_scalar

    # Baseline simulation
    baseline_ss = run_model_real(EC50_vec, n_vec, baseline_input, w, family_id_vector, w_override=w)

    ko_conditions = sorted([col for col in ko_df.columns if col in speciesNames and col != 'specie'])

    for ko_condition in ko_conditions:
        if ko_condition not in speciesNames:
            print(f"Skipping KO condition: {ko_condition} (not in speciesNames)")
            continue

        # KO by zeroing ymax and y0
        # FIX (reviewer concern #2): set inputs to b_scalar to match the
        # baseline comparison. Previously, KO used y0.copy() with inputs
        # still at 0.1, while the baseline simulation used b_scalar
        # (~0.28) for the inputs. The delta therefore mixed the KO effect
        # with the input-level drop and inflated the KO agreement score.
        ko_ymax = ymax.copy()
        ko_y0 = y0.copy()
        for _name, _idx in input_indices.items():
            ko_y0[_idx] = b_scalar
        ko_ymax[speciesNames.index(ko_condition)] = 0
        ko_y0[speciesNames.index(ko_condition)] = 0

        # KO simulation
        ko_ss = run_model_real(EC50_vec, n_vec, ko_y0, w, family_id_vector, ymax_override=ko_ymax, w_override=w)

        if np.isnan(ko_ss).any() or np.isnan(baseline_ss).any():
            print(f"⚠️ NaN in simulation output under KO of {ko_condition}, skipping")
            continue

        delta = ko_ss - baseline_ss

        for idx, row in ko_df.iterrows():
            target_species = row['specie']
            expected_behavior = row[ko_condition]

            if pd.isna(expected_behavior):
                continue

            if isinstance(expected_behavior, str):
                expected_behavior = expected_behavior.strip().lower()

            expected_behavior = str(expected_behavior).strip().lower()
            if expected_behavior not in ["increase", "decrease", "no change"]:
                continue

            if target_species not in species_index:
                print(f"⚠️ Target species '{target_species}' not in species_index")
                continue

            change = delta[species_index[target_species]]

            threshold = 0.05  # FIX (reviewer #1): harmonized with stim threshold
            if change > threshold:
                model_behavior = "increase"
            elif change < -threshold:
                model_behavior = "decrease"
            else:
                model_behavior = "no change"

            total += 1
            if model_behavior == expected_behavior:
                correct += 1
                label = "match"
            else:
                label = "not match"
            print(f"KO {ko_condition} -> {target_species}: expected {expected_behavior}, model {model_behavior}, comparison w/ data {label}, change={change:.4f}")
            
            # Save the information to a CSV file
            with open("KO_mismatch_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ko_condition, target_species, expected_behavior, model_behavior, label, change])

    match_percent = 100 * correct / total if total else 0
    mismatch = 100 - match_percent
    print(f"KO Mismatch: {mismatch:.2f}% (correct={correct}, total={total})")
    return mismatch


def compute_input_stimulation_mismatch(EC50_scalar, n_scalar, b_scalar, p_scalar, w, expectations_df, family_id_vector):
    species_index = {name: i for i, name in enumerate(speciesNames)}

    # Reset the input-stimulation mismatch log at the start of each call so the
    # file always contains exactly one pass over the dataset (no append-mode
    # accumulation).
    with open("input_stimulation_mismatch_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_name", "target_node", "model_behavior",
                         "label", "correct_so_far", "total_so_far", "delta"])

    # Expand EC50 and n to vectors
    EC50_vec = np.ones(wlen) * EC50_scalar
    n_vec = np.ones(wlen) * n_scalar

    # Build baseline input vector
    baseline_input = y0.copy()
    for name, idx in input_indices.items():
        baseline_input[idx] = b_scalar
    baseline_ss = run_model_real(EC50_vec, n_vec, baseline_input, w, family_id_vector, w_override=w)

    total, correct = 0, 0

    for input_name in input_list:
        perturbed_input = baseline_input.copy()
        perturbed_input[input_indices[input_name]] = b_scalar + p_scalar
        perturbed_ss = run_model_real(EC50_vec, n_vec, perturbed_input, w, family_id_vector, w_override=w)

        delta = perturbed_ss - baseline_ss

        for node in expectations_df.index:
            if node not in species_index:
                continue

            expected = expectations_df.loc[node, input_name]
            if not isinstance(expected, str) or expected.lower() not in ["increase", "decrease", "no change"]:
                continue

            expected = expected.lower()
            change = delta[species_index[node]]

            threshold = 0.05
            if change > threshold:
                model_behavior = "increase"
            elif change < -threshold:
                model_behavior = "decrease"
            else:
                model_behavior = "no change"

            total += 1
            if model_behavior == expected:
                correct += 1
                label = "match"
            else:
                label = "not match"

            print(f"Input {input_name} -> Node {node}: effect on node {model_behavior}, compared w/ data {label}, (correct so far={correct}, total={total}), change={change:.4f}")
            
            # Save the information to a CSV file
            with open("input_stimulation_mismatch_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([input_name, node, model_behavior, label, correct, total, change]) # print(f"Input {input_name}: mismatch so far {100 * (1 - correct/total):.2f}% (correct={correct}, total={total})")
    match_percent = 100 * correct / total if total else 0            
    mismatch = 100 - match_percent
    return mismatch

penalty_weight_B = 10

# Define the global objective function
def objective_function(params, ko_df, expectations_df, alpha=1, beta=1):

    EC50_vec, n_vec, b_scalar, p_scalar, w = params

    # Sanity check for B with soft penalty
    penalty_B = 0  # initialize

    try:
        B = (EC50_vec[0] ** n_vec[0] - 1) / (2 * EC50_vec[0] ** n_vec[0] - 1)

        if np.isnan(B) or np.isinf(B):
            print("⚠️ Invalid B (nan or inf):", B)
            return 245

        if B <= 1:
            penalty_B = penalty_weight_B * (1 - B)**2  # quadratic soft penalty when B <= 1

    except Exception as e:
        print("❌ Exception in B check:", e)
        return 245


    # Compute mismatches
    mismatch_input = compute_input_stimulation_mismatch(
    EC50_vec[0], n_vec[0], b_scalar, p_scalar, w, expectations_df, family_id_vector
    )
    mismatch_KO = compute_KO_mismatch(
        EC50_vec[0], n_vec[0], b_scalar, w, ko_df, family_id_vector
    )


    total_mismatch = alpha * mismatch_input + beta * mismatch_KO + penalty_B


    return total_mismatch


