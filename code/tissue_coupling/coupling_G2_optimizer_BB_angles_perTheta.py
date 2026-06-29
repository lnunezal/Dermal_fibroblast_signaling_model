# %% [markdown]
# ### STILL STRIP BIAXIAL as before but now we will calibrate for mapping function parameters so that it matches Tina's IGA data.

# %%
## For wunning the NETFLUXODE_5 model to update growth
import numpy as np
from scipy.integrate import ode 
from scipy.optimize import minimize
import itertools
import pandas as pd
import csv
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

def make_netflux_solver(y0_init=None, t0=0.0):
    """
    Create and initialize one Netflux solver.
    """
    if y0_init is None:
        y0_init = np.copy(y0)

    solver = ode(NetfluxODE_5.ODEfunc).set_integrator(
        "vode",
        method="adams",
        order=10,
        rtol=0,
        atol=1e-6,
        with_jacobian=False
    )
    solver.set_initial_value(y0_init, t0).set_f_params(tau, ymax, w_vec, n, EC50)
    return solver

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

# --------------------------------------------------
# 1. Read optimization log
# --------------------------------------------------
csv_path3 = "optimization_log_G1_1st_run.csv"
df1 = pd.read_csv(csv_path3)

# --------------------------------------------------
# 2. Find row with minimum loss
# --------------------------------------------------
# (Assumes column name is 'loss' — change if needed)
best_row = df1.loc[df1['loss'].idxmin()]


# --------------------------------------------------
# 3. Extract optimized parameters
# --------------------------------------------------
a_opt = best_row['a']
b_opt = best_row['b']
c_opt = best_row['c']
d_opt = best_row['d']
tau_opt = best_row['tau']

# --------------------------------------------------
# 4. Update params dictionary
# --------------------------------------------------
params = {
    "a": a_opt,
    "b": b_opt,
    "c": c_opt,
    "d": d_opt,
    "tau_g": tau_opt
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
def deriv_after_mech_change(y_current, t_current, mech_new, tau, ymax, ww, n, EC50):  #this is to get the dydt after changing mechanical
    y_tmp = np.array(y_current, copy=True)
    y_tmp[mechanical] = mech_new
    dydt_post = NetfluxODE_5.ODEfunc(t_current, y_tmp, tau, ymax, ww, n, EC50)
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
        y_current, solver.t, H_theta, tau, ymax, w_vec, n, EC50
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

def update_Fg_theta_local_signal(
    lam1, lamp1, lamg_vec, phiplus_vec, dt,
    solvers, params,
    lam2=1.0, lamp2=1.0,
    prestretch_side="right",
):
    """
    Directional growth update with ONE Netflux solver per angle.

    For each direction theta_i:
      - compute lambda(theta_i)
      - compute lambda_e(theta_i)
      - convert lambda_e(theta_i) to mechanical signaling input
      - evaluate Netflux for that angle only
      - compute phidot_plus(theta_i)
      - update lambda_g(theta_i)

    Inputs
    ------
    lam1, lam2 : measured total stretches
    lamp1, lamp2 : prestretches
    lamg_vec : current directional growth stretch, shape (n_theta,)
    phiplus_vec : current collagen-state-like variable per angle, shape (n_theta,)
    solvers : list of Netflux solvers, one per angle
    """

    # ------------------------------------------
    # 1. Build effective deformation tensor
    # ------------------------------------------
    F_meas = np.array([[lam1, 0.0],
                       [0.0,  lam2]])

    Fp = np.array([[lamp1, 0.0],
                   [0.0,   lamp2]])

    if prestretch_side.lower() == "right":
        F_eff = F_meas @ Fp
    elif prestretch_side.lower() == "left":
        F_eff = Fp @ F_meas
    else:
        raise ValueError("prestretch_side must be 'right' or 'left'")

    C = F_eff.T @ F_eff

    # ------------------------------------------
    # 2. Angular discretization
    # ------------------------------------------
    n_theta = len(lamg_vec)
    if len(solvers) != n_theta:
        raise ValueError("Number of solvers must match len(lamg_vec)")
    if len(phiplus_vec) != n_theta:
        raise ValueError("phiplus_vec must have same length as lamg_vec")

    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)

    lam_vec = np.zeros(n_theta)
    lame_vec = np.zeros(n_theta)
    phidot_vec = np.zeros(n_theta)
    lamg_new = lamg_vec.copy()
    phiplus_new = phiplus_vec.copy()

    tau_g = max(params["tau_g"], 1e-3)
    GAIN = 1.0

    # ------------------------------------------
    # 3. Compute lambda(theta) and lambda_e(theta)
    # ------------------------------------------
    for i, th in enumerate(thetas):
        a_th = np.array([np.cos(th), np.sin(th)])
        lam_th = np.sqrt(a_th @ C @ a_th)
        lam_vec[i] = lam_th
        lame_vec[i] = lam_th / max(lamg_vec[i], 1e-12)

    # ------------------------------------------
    # 4. Evaluate signaling independently for each angle
    # ------------------------------------------
    for i in range(n_theta):
        solver_i = solvers[i]

        # local mechanical input from local elastic stretch
        H_i = mech_to_signal1(lame_vec[i], params["c"], params["d"])

        # copy current biological state for this angle
        y_current = solver_i.y.copy()
        y_current[mechanical] = H_i

        # evaluate derivative under updated mechanical input
        dydt_i = deriv_after_mech_change(
            y_current, solver_i.t, H_i, tau, ymax, w_vec, n, EC50
        )
        CI_dot_i = dydt_i[CI]

        # local collagen/growth driver
        phidot_i = CI_to_phi_a(CI_dot_i, params["a"], params["b"])
        phidot_vec[i] = phidot_i

        # advance solver for this angle
        solver_i.set_initial_value(y_current, solver_i.t)
        solver_i.integrate(solver_i.t + dt)

        # update local growth
        lamg_new[i] = lamg_vec[i] + GAIN * dt * phidot_i / tau_g * (lame_vec[i] - 1.0)

        # update local collagen-state variable
        phiplus_new[i] = phiplus_vec[i] + dt * phidot_i

    return lamg_new, phiplus_new, lam_vec, lame_vec, phidot_vec

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

def run_simulation_constant_F_theta_local_signal(
    params,
    lam1, lamp1,
    T_end,
    dt=0.1,
    n_theta=10,
    lam2=1.0,
    lamp2=1.0,
    prestretch_side="right",
):
    """
    Uniaxial/strip-biaxial deformation with:
      - directional growth stored over theta
      - one signaling solver per theta
      - local signaling update in each direction

    Returns
    -------
    t_arr
    lamg_G1_arr
    lamg_arr
    lam_arr
    lame_arr
    phidot_arr
    phiplus_arr
    """

    n_steps = int(T_end / dt) + 1

    # ------------------------------------------
    # 1. Initialize one solver per angle
    # ------------------------------------------
    y0_local = np.copy(y0)
    y0_local[mechanical] = bv

    solvers = []
    for _ in range(n_theta):
        solvers.append(make_netflux_solver(y0_init=np.copy(y0_local), t0=0.0))

    # ------------------------------------------
    # 2. Initialize directional states
    # ------------------------------------------
    lamg_vec = np.ones(n_theta)
    phiplus_vec = 0.1 * np.ones(n_theta)

    # ------------------------------------------
    # 3. Storage
    # ------------------------------------------
    t_arr = np.arange(n_steps) * dt

    lamg_arr = np.zeros((n_steps, n_theta))
    lam_arr = np.zeros((n_steps, n_theta))
    lame_arr = np.zeros((n_steps, n_theta))
    phidot_arr = np.zeros((n_steps, n_theta))
    phiplus_arr = np.zeros((n_steps, n_theta))

    lamg_arr[0, :] = lamg_vec
    phiplus_arr[0, :] = phiplus_vec

    # ------------------------------------------
    # 4. Time loop
    # ------------------------------------------
    for k in range(n_steps - 1):
        lamg_vec, phiplus_vec, lam_vec, lame_vec, phidot_vec = update_Fg_theta_local_signal(
            lam1=lam1,
            lamp1=lamp1,
            lamg_vec=lamg_vec,
            phiplus_vec=phiplus_vec,
            dt=dt,
            solvers=solvers,
            params=params,
            lam2=lam2,
            lamp2=lamp2,
            prestretch_side=prestretch_side,
        )

        lamg_arr[k + 1, :] = lamg_vec
        lam_arr[k + 1, :] = lam_vec
        lame_arr[k + 1, :] = lame_vec
        phidot_arr[k + 1, :] = phidot_vec
        phiplus_arr[k + 1, :] = phiplus_vec

    lamg_G1_arr = lamg_arr[:, 0]

    return t_arr, lamg_G1_arr, lamg_arr, lam_arr, lame_arr, phidot_arr, phiplus_arr


loss_history = []
theta_history = []
best_loss = np.inf
best_theta = None

csv_filename = "optimization_log_G1_BB.csv"


class LoggingWrapper:
    def __init__(self, func, data, dt=0.1, log_file=csv_filename):
        self.func = func
        self.data = data
        self.dt = dt
        self.log_file = log_file

        # Write header once (optional but recommended)
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "c", "d", "tau_g", "loss"])

    def __call__(self, theta_vec):
        theta_vec = np.array(theta_vec).flatten()

        try:
            loss = self.func(theta_vec, self.data, self.dt)
        except Exception as e:
            print(f"⚠️ Error evaluating parameters: {e}")
            loss = 1e6   # strong penalty if model crashes

        # Append row
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(theta_vec) + [loss])

        return loss

