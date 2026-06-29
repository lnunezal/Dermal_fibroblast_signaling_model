# %% [markdown]
# ### STILL STRIP BIAXIAL as before but now we will calibrate for mapping function parameters so that it matches Tina's IGA data.

# %%
## For wunning the NETFLUXODE_5 model to update growth
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import ode 
from scipy.optimize import minimize
import itertools
import pandas as pd
import csv
import seaborn as sns
from model_DOF_clustering_5 import objective_function, load_ko_and_expectation_data, wlen, family_id_vector, build_weight_vector, run_model_real

import NetfluxODE_5
import NetfluxODE_5_sensitivity
import NetfluxODE_5_params
[speciesNames, tau, ymax, y0, w, n, EC50] = NetfluxODE_5_params.loadParams()

# Read the CSV file
loss_log_df = pd.read_csv("loss_log_DOF_cluster.csv")
# Find the row with the smallest value in the last column
min_loss_row = loss_log_df.loc[loss_log_df.iloc[:, -1].idxmin()]
# Extract the vector using all columns from the first to the penultimate column
vector = min_loss_row.iloc[:-1].values
# Convert the vector to a numpy array
w_vec = build_weight_vector(vector, family_id_vector)

bas=0.1
y0=bas*np.ones(len(y0))

EC50v = 0.568305675
nv    = 1.301942649
bv    = 0.291697537
pv    = 0.507453875

# EC50v = 0.587798
# nv    = 1.395697
# bv    = 0.283201
# pv    = 0.653814

EC50=EC50v*np.ones(len(EC50))  #all
n=nv*np.ones(len(n)) #all

# %%
AngII = 0 
AT1R = 1 
AGT = 2 
ACE = 3 
NOX = 4 
ROS = 5 
ET1 = 6 
ETAR = 7 
DAG = 8 
PKC = 9 
AC = 10 
cAMP = 11 
PKA = 12 
CREB = 13 
CBP = 14 
TGFB = 15 
TGFB1R = 16 
smad3 = 17 
smad7 = 18 
latentTGFB = 19 
PDGF = 20 
PDGFR = 21 
cGMP = 22 
PKG = 23 
mechanical = 24 
B1int = 25 
Rho = 26 
ROCK = 27 
Ca = 28 
IL6 = 29 
gp130 = 30 
STAT = 31 
IL1 = 32 
IL1RI = 33 
TNFa = 34 
TNFaR = 35 
NFKB = 36 
PI3K = 37 
Akt = 38 
p38 = 39 
TRAF = 40 
ASK1 = 41 
MKK3 = 42 
PP1 = 43 
JNK = 44 
abl = 45 
Rac1 = 46 
MEKK1 = 47 
MKK4 = 48 
ERK = 49 
Ras = 50 
Raf = 51 
MEK1 = 52 
FAK = 53 
epac = 54 
Factin = 55 
FA = 56 
migration = 57 
cmyc = 58 
CTGF = 59 
proliferation = 60 
SRF = 61 
aSMA = 62 
AP1 = 63 
TIMP1 = 64 
TIMP2 = 65 
PAI1 = 66 
proMMP14 = 67 
proMMP1 = 68 
proMMP2 = 69 
proMMP9 = 70 
MMP1 = 71 
MMP2 = 72 
MMP9 = 73 
MMP14 = 74 
fibronectin = 75 
periostin = 76 
CImRNA = 77 
CIIImRNA = 78 
CI = 79 
CIII = 80 
FGF = 81 
FGFR1 = 82 
Shp2 = 83 
FRS2 = 84 
Src = 85 
cortactin = 86 
Grb2Sos = 87 
NWASP = 88 
Arp23 = 89 
TEAD = 90 
YAP = 91 
LATS12 = 92 
piezo1 = 93 
infusionTGFB = 94 
infusionIL6 = 95 


biochinputs = [mechanical, FGF, infusionIL6, IL1, TNFa, PDGF, infusionTGFB, AngII, ET1]
for idx in biochinputs:
    y0[idx] = bv
# %%
## Functions to load experimental data and filter it as needed

params = {
    "a": 1.0, "b": 0.0,   # CI → growth in G1
    # "e": 1.0, "f": 0.0,   # CI → growth in G2
    "c": 3.33, "d": -3.33,   # theta_e → signaling
    # "g": 1.0, "h": 0.0,   # theta_e → signaling
    "tau_g": 0.5,
    # "tau_s": 0.5,
    # "tau_w": 1.0,
    # "tau_k": 1.0,
    # "gamma_k": 1.0
}

def load_growth_csv(csv_path):
    # same format your labmate used: index_col=0, header=None
    df = pd.read_csv(csv_path, index_col=0, header=None)
    # rows are: theta, theta_g, theta_p, time
    theta   = df.loc["theta"].astype(float).values
    theta_g = df.loc["theta_g"].astype(float).values
    theta_p = df.loc["theta_p"].astype(float).values
    time_h  = df.loc["time"].astype(float).values
    return time_h, theta, theta_g, theta_p

def build_observed_dataset(path_iso, path_g1, path_g2):

    # Load CSVs
    t_iso, theta_iso, thetag_iso, thetap_iso = load_growth_csv(path_iso)
    t_g1,  theta_g1,  thetag_g1,  thetap_g1  = load_growth_csv(path_g1)
    t_g2,  theta_g2,  thetag_g2,  thetap_g2  = load_growth_csv(path_g2)

    # if len(t_iso) != len(t_g1) or len(t_iso) != len(t_g2):

    # Observed growth (outputs)
    G1g_obs = thetag_g1
    G2g_obs = thetag_g2
    thetag_obs = thetag_iso
    G1p_obs = thetap_g1
    G2p_obs = thetap_g2

    meanlamG1p = np.mean(G1p_obs)

    # Input stretch (IMPORTANT for Option A)
    G1_in = theta_g1
    G2_in = theta_g2

    # Derive elastic def
    G1e_der = G1_in*G1p_obs/G1g_obs
    G2e_der = G2_in*G2p_obs/G2g_obs
    thetae_iso = theta_iso*thetap_iso/thetag_iso 


    data = {
        # isotropic (optional)
        "time_iso": t_iso,
        "thetag_iso": thetag_obs,
        "thetap_iso": thetap_iso,
        "theta_iso": theta_iso,
        "thetae_iso": thetae_iso, # Option A: elastic def = stretch * growth

        # G1 direction
        "time_g1": t_g1,
        # "G1g_all": G1g_obs,
        # "G1_in_all": G1_in,
        # "G1p_all": G1p_obs,
        # "G1e_all": G1e_der,
        "G1g_all": G1g_obs*meanlamG1p/G1p_obs,
        "G1_in_all": G1_in*meanlamG1p/G1p_obs,
        "G1p_all": G1p_obs*meanlamG1p/G1p_obs,
        "G1e_all": G1e_der*meanlamG1p/G1p_obs,

        # G2 direction
        "time_g2": t_g2,
        "G2g_all": G2g_obs,
        "G2_in_all": G2_in,
        "G2p_all": G2p_obs,
        "G2e_all": G2e_der,
    }

    return data


# %%
# Function to summarize data by unique time points
def summarize_by_time(time_all, values_all):
    t_unique = np.unique(time_all)
    mean = np.array([values_all[time_all == t].mean() for t in t_unique])
    std  = np.array([values_all[time_all == t].std(ddof=1) for t in t_unique])
    return t_unique, mean, std


###Filter data
def filter_growth_below_one(data, tol=1e-6):
    data2 = data.copy()

    # G1
    m1 = np.asarray(data2["G1g_all"]) >= (1.0 - tol)
    data2["time_g1"] = np.asarray(data2["time_g1"])[m1]
    data2["G1g_all"] = np.asarray(data2["G1g_all"])[m1]
    data2["G1_in_all"] = np.asarray(data2["G1_in_all"])[m1]
    data2["G1p_all"] = np.asarray(data2["G1p_all"])[m1]
    data2["G1e_all"] = np.asarray(data2["G1e_all"])[m1]

    # Sort G1 data points by increasing G1p_all order
    sort_idx_g1 = np.argsort(data2["G1p_all"])
    data2["time_g1"] = data2["time_g1"][sort_idx_g1]
    data2["G1g_all"] = data2["G1g_all"][sort_idx_g1]
    data2["G1_in_all"] = data2["G1_in_all"][sort_idx_g1]
    data2["G1p_all"] = data2["G1p_all"][sort_idx_g1]
    data2["G1e_all"] = data2["G1e_all"][sort_idx_g1]

    # G2
    m2 = np.asarray(data2["G2g_all"]) >= (1.0 - tol)
    data2["time_g2"] = np.asarray(data2["time_g2"])[m2]
    data2["G2g_all"] = np.asarray(data2["G2g_all"])[m2]
    data2["G2_in_all"] = np.asarray(data2["G2_in_all"])[m2]
    data2["G2p_all"] = np.asarray(data2["G2p_all"])[m2]
    data2["G2e_all"] = np.asarray(data2["G2e_all"])[m2]

    # Sort G2 data points by increasing G2p_all order
    # data2["time_g2"] = data2["time_g2"][sort_idx_g2]
    # data2["G2g_all"] = data2["G2g_all"][sort_idx_g2]
    # data2["G2_in_all"] = data2["G2_in_all"][sort_idx_g2]
    # data2["G2p_all"] = data2["G2p_all"][sort_idx_g2]
    # data2["G2e_all"] = data2["G2e_all"][sort_idx_g2]

    #theta
    m3 = np.asarray(data2["thetag_iso"]) >= (-1.0 - tol)
    data2["time_iso"] = np.asarray(data2["time_iso"])[m3]
    data2["thetag_iso"] = np.asarray(data2["thetag_iso"])[m3]
    data2["thetap_iso"] = np.asarray(data2["thetap_iso"])[m3]
    data2["theta_iso"] = np.asarray(data2["theta_iso"])[m3]
    data2["thetae_iso"] = np.asarray(data2["thetae_iso"])[m3]

    return data2

data1 = build_observed_dataset(
    "Iso_data_4_pigs.csv",
    "Iso_data_4_pigs_G1.csv",
    "Iso_data_4_pigs_G2.csv",
)
data = filter_growth_below_one(data1, tol=1e-6)

def mech_to_signal1(lam, c, d):
    z = c * lam + d
    return z



def CI_to_phi_a(CI_dot, a, b):
    z = a * CI_dot + b
    return z

# %%
# Function to compute the derivative after changing mechanical input
def deriv_after_mech_change(y_current, t_current, mech_new, tau, ymax, w, n, EC50):  #this is to get the dydt after changing mechanical
    y_tmp = np.array(y_current, copy=True)
    y_tmp[mechanical] = mech_new
    dydt_post = NetfluxODE_5.ODEfunc(t_current, y_tmp, tau, ymax, w, n, EC50)
    return dydt_post



# %%
def update_Fg(
    theta, thetag, thetap, phiplus, dt,
    solver, params
):
    """
    Update growth, orientation, and dispersion for ONE time step.
    Assumes F is the EFFECTIVE total deformation (i.e. includes prestretch if any).
    """
    # ------------------------------------------------------------------
    # 1. Current growth tensor and elastic deformation
    # ------------------------------------------------------------------
    #F*Fp=Fe*Fg


    thetae = theta*thetap/thetag

    # Elastic stretches along current fiber directions

    # ------------------------------------------------------------------
    # 2. Principal elastic stretches (sorted!)
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # 3. Elastic & growth stretches in GLOBAL G1/G2 (diagnostics)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 4. Mechanobiology → collagen production signals
    # ------------------------------------------------------------------
    H_theta = mech_to_signal1(thetae, params["c"], params["d"])

    # Copy solver states (IMPORTANT)
    y_current = solver.y.copy()

    y_current[mechanical] = H_theta

    dydt = deriv_after_mech_change(
        y_current, solver.t, H_theta, tau, ymax, w, n, EC50
    )
        # y_currents, solvers.t, H_thetas, tau, ymax, w, n, EC50
    # )

    CI_dot = dydt[CI]

    phidot_plus = CI_to_phi_a(CI_dot, params["a"], params["b"])

    # Advance signaling ODEs
    solver.set_initial_value(y_current, solver.t)
    solver.integrate(solver.t + dt)

    # ------------------------------------------------------------------
    # 5. Growth update (gain retained as in your model)
    # ------------------------------------------------------------------
    GAIN = 1.0
    tau_g = max(params["tau_g"], 1e-3)
    thetag_new = thetag + GAIN * dt * phidot_plus / tau_g * (thetae - 1.0) 

    # ------------------------------------------------------------------
    # 6. Dispersion (kappa) update
    # ------------------------------------------------------------------

    phiplus_new = phiplus + dt * phidot_plus


    # ------------------------------------------------------------------
    # 7. Reorientation toward principal elastic direction
    # ------------------------------------------------------------------




    # # Normalize (safety)

    # ------------------------------------------------------------------
    # 8. Growth stretches in G1/G2 (POST-update, correct!)
    # ------------------------------------------------------------------
    


    # ------------------------------------------------------------------
    return (
        thetag_new,
        phiplus_new,
        thetae,
        phidot_plus
    )



# %%
def run_simulation_constant_F(
    params,
    theta, thetap,
    T_end,
    dt=0.1,
    prestretch_side="right",   # "right" matches F*Fp = FeFg
):
    """
    Run ONE experiment with constant measured deformation F (lam1, lam2)
    but apply labmate kinematics: F_eff = F * Fp (default).
    """

    n_steps = int(T_end / dt) + 1

    # reset solver + y0 for each run
    y0_local = np.copy(y0)
    y0_local[mechanical] = bv

    solver = ode(NetfluxODE_5.ODEfunc).set_integrator(
        "vode", method="adams", order=10, rtol=0, atol=1e-6, with_jacobian=False
    )
    solver.set_initial_value(y0_local, 0).set_f_params(tau, ymax, w_vec, n, EC50)

    #     "vode", method="adams", order=10, rtol=0, atol=1e-6, with_jacobian=False
    # )

    # initial conditions
    thetag = 1.0
    phiplus = 0.1

    # storage
    thetag_arr = np.zeros(n_steps)
    phiplus_arr = np.zeros(n_steps)

    t_arr = np.arange(n_steps) * dt

    # initial values
    # lamag_arr[0] = lamag
    # lamsg_arr[0] = lamsg
    thetag_arr[0] = thetag
    # kappa_arr[0] = kappa
    # a0_arr[0] = a0
    # s0_arr[0] = s0

    # measured deformation (what you currently call "input total deformation")
    F_meas = theta * np.eye(2)  # isotropic for now; can be made anisotropic if needed

    # prestretch tensor
    Fp = thetap * np.eye(2)  # isotropic for now; can be made anisotropic if needed

    # effective total deformation used by decomposition inside update_Fg
    # if prestretch_side.lower() == "right":
    # elif prestretch_side.lower() == "left":
    # else:
    #     raise ValueError("prestretch_side must be 'right' or 'left'")

    # time loop
    for i in range(n_steps - 1):
        # lamag, lamsg, lamgG1, lamgG2, laeG1, laeG2, kappa, a0, s0, phiplus, \
        # thetae, thetag, eigval, eigvec, phidotplus = update_Fg(
        #     F_eff, lamag, lamsg, a0, s0, kappa, phiplus, dt,
        # )
        thetag, phiplus, thetae_new, phidotplus_new = update_Fg(theta, thetag, thetap, phiplus, dt, solver=solver, params=params)
        

        # lamag_arr[i+1] = lamag
        # lamsg_arr[i+1] = lamsg
        thetag_arr[i+1] = thetag
        # kappa_arr[i+1] = kappa
        # a0_arr[i+1] = a0
        # s0_arr[i+1] = s0
        # lam_g_G1_arr[i+1] = lamgG1
        # lam_g_G2_arr[i+1] = lamgG2
        phiplus_arr[i+1] = phiplus

    return (
        t_arr,
        # lamag_arr,
        # lamsg_arr,
        # lam_g_G1_arr,
        # lam_g_G2_arr,
        thetag_arr,
        # kappa_arr,
        # a0_arr,
        # s0_arr
    )



##### Loss function considering all 8 parameters

## representative 10 points of each timepoint:
idx_24  = np.where(data["time_g1"] == 24)[0][:10]
idx_72  = np.where(data["time_g1"] == 72)[0][:10]
idx_168 = np.where(data["time_g1"] == 168)[0][:10]

selected_idx = np.concatenate([idx_24, idx_72, idx_168])

def loss_function(theta_vec, data, dt=0.1):
    # unpack parameters
    params["a"], params["b"], params["c"], params["d"], params["tau_g"]= theta_vec

    t_iso = data["time_g1"]
    loss = 0.0

    # loop over time points (independent experiments)
    for i in selected_idx:  #len(t_iso)

        # mean input stretch at this time
        theta_exp=data["G1_in_all"][i]
        thetap_exp=data["G1p_all"][i]
        thetag_exp=data["G1g_all"][i]
        thetae_exp=data["G1e_all"][i]

        t_exp=t_iso[i]

        # run model for this experiment
        results = run_simulation_constant_F(
            params,
            theta_exp, thetap_exp,
            T_end=t_exp,
            dt=dt
        )

        thetag_model = results[1][-1]

        loss += (thetag_model - thetag_exp)**2

    return loss

loss_history = []
theta_history = []
best_loss = np.inf
best_theta = None

csv_filename = "optimization_log_G1.csv"

# Write header
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["iteration", "loss", "a", "b", "c", "d", "tau"])
    
iteration_counter = {"iter": 0}  # mutable container

def callback_loss(xk):
    global best_loss, best_theta

    # Compute loss at current iterate
    current_loss = loss_function(xk, data)

    # Store history (as before)
    loss_history.append(current_loss)
    theta_history.append(xk.copy())

    # Track best-so-far
    if current_loss < best_loss:
        best_loss = current_loss
        best_theta = xk.copy()

    # Current iteration number
    it = len(loss_history)

    # Print progress (same style as before)
    print(
        f"Iter {it:3d} | "
        f"loss = {current_loss:.6e} | "
        f"a={xk[0]: .3e}, b={xk[1]: .3e}, "
        f"c={xk[2]: .3e}, d={xk[3]: .3e}, "
        f"tau={xk[4]: .3e}"
    )

    # Append to CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            it,
            current_loss,
            xk[0], xk[1], xk[2], xk[3], xk[4]
        ])
