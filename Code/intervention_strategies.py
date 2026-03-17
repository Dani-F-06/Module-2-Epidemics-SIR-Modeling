import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# LOAD DATA
# -----------------------------
data_3 = pd.read_csv(r'Data/mystery_virus_daily_active_counts_RELEASE#3.csv', parse_dates=['date'], header=0, index_col=None
)

x_data_actual = data_3['day'].values.astype(float)
y_data_actual = data_3['active reported daily cases'].values.astype(float)

# -----------------------------
# INITIAL CONDITIONS
# -----------------------------
S0 = 38857   # VT student population
E0 = 1 # We assume 1 exposed student at the start of the epidemic
I0 = 1 # We assume 1 infectious student at the start of the epidemic
R0 = 0 # We assume no recovered students at the start of the epidemic
N = int(S0) + int(E0) + int(I0) + int(R0) # Total population

# ============================================================
# SEIR EULER FUNCTION
# ============================================================
def euler_seir(timepoints, N, S0, E0, I0, R0, beta, sigma, gamma):
    """
    Simulate SEIR model using Euler's method.
    """
    dt = timepoints[1] - timepoints[0]

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    for i in range(len(timepoints) - 1):
        dSdt = -beta * S[i] * I[i] / N
        dEdt = beta * S[i] * I[i] / N - sigma * E[i]
        dIdt = sigma * E[i] - gamma * I[i]
        dRdt = gamma * I[i]

        S[i + 1] = S[i] + dSdt * dt
        E[i + 1] = E[i] + dEdt * dt
        I[i + 1] = I[i] + dIdt * dt
        R[i + 1] = R[i] + dRdt * dt

    return S, E, I, R

# generative ai was used to help write and debug the following code, however the final code was manually reviewed and edited by a human to ensure correctness and clarity. The generative ai was used as a tool to assist in the coding process, but the final decisions on code structure, variable naming, and logic were made by a human programmer.
# ============================================================
# FIT PARAMETERS TO FULL DATASET
# ============================================================
def sse(model, data):
    """
    Sum of squared errors between model output and real data.
    """
    return np.sum((model - data) ** 2)

# Fit over the full 120 days so the model sees:
# growth + peak + decline
timepoints_fit = np.arange(1, 121)
y_fit = y_data_actual

# First-pass grid search ranges
beta_values = np.linspace(0.05, 0.20, 20)
sigma_values = np.linspace(0.10, 0.50, 20)
gamma_values = np.linspace(0.05, 0.20, 20)

best_sse = np.inf
best_beta = None
best_sigma = None
best_gamma = None

for beta in beta_values:
    for sigma in sigma_values:
        for gamma in gamma_values:
            S_tmp, E_tmp, I_tmp, R_tmp = euler_seir(
                timepoints_fit, N, S0, E0, I0, R0,
                beta, sigma, gamma
            )

            current_sse = sse(I_tmp, y_fit)

            if current_sse < best_sse:
                best_sse = current_sse
                best_beta = beta
                best_sigma = sigma
                best_gamma = gamma

print("First-pass fit:")
print("best_beta =", best_beta)
print("best_sigma =", best_sigma)
print("best_gamma =", best_gamma)
print("best_sse =", best_sse)

#refinement around the first-pass best values
beta_values_refined = np.linspace(max(0.001, best_beta - 0.03), best_beta + 0.03, 20)
sigma_values_refined = np.linspace(max(0.001, best_sigma - 0.08), best_sigma + 0.08, 20)
gamma_values_refined = np.linspace(max(0.001, best_gamma - 0.03), best_gamma + 0.03, 20)

best_sse_refined = np.inf
best_beta_refined = None
best_sigma_refined = None
best_gamma_refined = None

for beta in beta_values_refined:
    for sigma in sigma_values_refined:
        for gamma in gamma_values_refined:
            S_tmp, E_tmp, I_tmp, R_tmp = euler_seir(
                timepoints_fit, N, S0, E0, I0, R0,
                beta, sigma, gamma
            )

            current_sse = sse(I_tmp, y_fit)

            if current_sse < best_sse_refined:
                best_sse_refined = current_sse
                best_beta_refined = beta
                best_sigma_refined = sigma
                best_gamma_refined = gamma

best_beta = best_beta_refined
best_sigma = best_sigma_refined
best_gamma = best_gamma_refined
best_sse = best_sse_refined

print("\nRefined fit:")
print("best_beta =", best_beta)
print("best_sigma =", best_sigma)
print("best_gamma =", best_gamma)
print("best_sse =", best_sse)

# ============================================================
# FITTED BASELINE MODEL
# ============================================================
S_fit, E_fit, I_fit, R_fit = euler_seir(
    timepoints_fit, N, S0, E0, I0, R0,
    best_beta, best_sigma, best_gamma
)

plt.figure(figsize=(10, 6))
plt.plot(x_data_actual, y_data_actual, 'o', label='Actual data')
plt.plot(timepoints_fit, I_fit, label='Fitted SEIR model')
plt.xlabel('Day')
plt.ylabel('Active cases')
plt.title('SEIR Fit to VT Epidemic Data')
plt.legend()
plt.show()

print("Fitted model peak over days 1-120:", np.max(I_fit))

# ============================================================
# BASELINE THROUGH DAY 120
# ============================================================
timepoints_full = np.arange(1, 121)

S_base, E_base, I_base, R_base = euler_seir(
    timepoints_full, N, S0, E0, I0, R0,
    best_beta, best_sigma, best_gamma
)

# -----------------------------
# EXTRACT DAY 70 STATE
# -----------------------------
day70_index = np.where(timepoints_full == 70)[0][0]

S70 = S_base[day70_index]
E70 = E_base[day70_index]
I70 = I_base[day70_index]
R70 = R_base[day70_index]

# -----------------------------
# DEFINE POST-70 TIMEPOINTS
# -----------------------------
timepoints_post70 = np.arange(70, 121)

# -----------------------------
# BASELINE FROM DAY 70-120
# -----------------------------
S_post_base, E_post_base, I_post_base, R_post_base = euler_seir(
    timepoints_post70, N, S70, E70, I70, R70,
    best_beta, best_sigma, best_gamma
)

# ============================================================
# INTERVENTION 1: MASKING MANDATE
# Reduces transmission by 40%, so beta becomes 60% of original
# ============================================================
beta_mask = 0.6 * best_beta # 40% reduction in transmission

S_mask, E_mask, I_mask, R_mask = euler_seir(
    timepoints_post70, N, S70, E70, I70, R70,
    beta_mask, best_sigma, best_gamma
)

# ============================================================
# INTERVENTION 2: VACCINE CAMPAIGN
# Day 70 only: vaccinate 2000 students with 90% efficacy
# Effective protected students = 1800 moved from S to R
# ============================================================
effective_vax_campaign = 2000 * 0.90 # 90% efficacy means 1800 students effectively protected

S70_vax_campaign = max(S70 - effective_vax_campaign, 0)
R70_vax_campaign = R70 + min(effective_vax_campaign, S70)

S_vax_campaign, E_vax_campaign, I_vax_campaign, R_vax_campaign = euler_seir(
    timepoints_post70, N, S70_vax_campaign, E70, I70, R70_vax_campaign,
    best_beta, best_sigma, best_gamma
)

# ============================================================
# INTERVENTION 3: VACCINE ROLLOUT
# Vaccinate 1000 students on days 70, 80, 90 with 90% efficacy
# Effective protected students = 900 each time
# ============================================================
def euler_seir_vaccine_rollout(timepoints, N, S0, E0, I0, R0, 
                               beta, sigma, gamma,
                               vax_days, vax_amount, efficacy):
    dt = timepoints[1] - timepoints[0]

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    effective_vax = vax_amount * efficacy # number of students effectively protected each vaccination day

    for i in range(len(timepoints) - 1):
        current_day = timepoints[i]

        if current_day in vax_days:
            moved = min(effective_vax, S[i])
            S[i] -= moved
            R[i] += moved

        dSdt = -beta * S[i] * I[i] / N
        dEdt = beta * S[i] * I[i] / N - sigma * E[i]
        dIdt = sigma * E[i] - gamma * I[i]
        dRdt = gamma * I[i]

        S[i + 1] = S[i] + dSdt * dt
        E[i + 1] = E[i] + dEdt * dt
        I[i + 1] = I[i] + dIdt * dt
        R[i + 1] = R[i] + dRdt * dt

    return S, E, I, R

S_vax_rollout, E_vax_rollout, I_vax_rollout, R_vax_rollout = euler_seir_vaccine_rollout(
    timepoints_post70, N, S70, E70, I70, R70,
    best_beta, best_sigma, best_gamma,
    vax_days=[70, 80, 90],
    vax_amount=1000,
    efficacy=0.90
)

# ============================================================
# INTERVENTION 4: TESTING + QUARANTINE
# Reduces infectious period by 2 days
# infectious period = 1/gamma
# ============================================================
infectious_period = 1 / best_gamma # original infectious period in days
new_infectious_period = infectious_period - 2 # reduce by 2 days

if new_infectious_period <= 0:
    raise ValueError("New infectious period is not valid. Check gamma.")

gamma_test = 1 / new_infectious_period

S_test, E_test, I_test, R_test = euler_seir(
    timepoints_post70, N, S70, E70, I70, R70,
    best_beta, best_sigma, gamma_test
)

# ============================================================
# INTERVENTION 5: TWO-WEEK SCHOOL CLOSURE
# Day 70-84: only 20% of normal contacts
# After day 84: contact rate returns to normal
# ============================================================
def euler_seir_school_closure(timepoints, N, S0, E0, I0, R0,
                              beta_normal, sigma, gamma,
                              closure_start, closure_end, closure_contact_fraction):
    dt = timepoints[1] - timepoints[0]

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    for i in range(len(timepoints) - 1):
        current_day = timepoints[i]

        if closure_start <= current_day < closure_end:
            beta_current = beta_normal * closure_contact_fraction
        else:
            beta_current = beta_normal

        dSdt = -beta_current * S[i] * I[i] / N
        dEdt = beta_current * S[i] * I[i] / N - sigma * E[i]
        dIdt = sigma * E[i] - gamma * I[i]
        dRdt = gamma * I[i]

        S[i + 1] = S[i] + dSdt * dt
        E[i + 1] = E[i] + dEdt * dt
        I[i + 1] = I[i] + dIdt * dt
        R[i + 1] = R[i] + dRdt * dt

    return S, E, I, R

S_close, E_close, I_close, R_close = euler_seir_school_closure(
    timepoints_post70, N, S70, E70, I70, R70,
    best_beta, best_sigma, best_gamma,
    closure_start=70,
    closure_end=84,
    closure_contact_fraction=0.20
)

# ============================================================
# METRICS FUNCTION
# Peak infections = max I(t) from day 70-120
# Cases from day 70-120 = S(day70) - S(day120)
# ============================================================
def intervention_metrics(name, S_int, I_int, S_base_window, I_base_window):
    peak_base = np.max(I_base_window)
    peak_int = np.max(I_int)

    total_cases_base = S_base_window[0] - S_base_window[-1]
    total_cases_int = S_int[0] - S_int[-1]

    peak_reduction = peak_base - peak_int
    cases_prevented = total_cases_base - total_cases_int

    print(f'--- {name} ---')
    print(f'Peak infections (baseline): {peak_base:.2f}')
    print(f'Peak infections (intervention): {peak_int:.2f}')
    print(f'Peak reduction: {peak_reduction:.2f}')
    print(f'Total cases day 70-120 (baseline): {total_cases_base:.2f}')
    print(f'Total cases day 70-120 (intervention): {total_cases_int:.2f}')
    print(f'Cases prevented: {cases_prevented:.2f}')
    print()

# ============================================================
# PRINT METRICS
# ============================================================
intervention_metrics('Masking mandate', S_mask, I_mask, S_post_base, I_post_base)
intervention_metrics('Vaccine campaign', S_vax_campaign, I_vax_campaign, S_post_base, I_post_base)
intervention_metrics('Vaccine rollout', S_vax_rollout, I_vax_rollout, S_post_base, I_post_base)
intervention_metrics('Testing + quarantine', S_test, I_test, S_post_base, I_post_base)
intervention_metrics('2-week school closure', S_close, I_close, S_post_base, I_post_base)

# ============================================================
# PLOT ALL INTERVENTIONS VS BASELINE
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(timepoints_post70, I_post_base, label='Baseline')
plt.plot(timepoints_post70, I_mask, label='Masking mandate')
plt.plot(timepoints_post70, I_vax_campaign, label='Vaccine campaign')
plt.plot(timepoints_post70, I_vax_rollout, label='Vaccine rollout')
plt.plot(timepoints_post70, I_test, label='Testing + quarantine')
plt.plot(timepoints_post70, I_close, label='2-week school closure')

plt.xlabel('Day')
plt.ylabel('Active infections')
plt.title('VT Interventions Compared to Baseline (Days 70-120)')
plt.legend()
plt.show()

