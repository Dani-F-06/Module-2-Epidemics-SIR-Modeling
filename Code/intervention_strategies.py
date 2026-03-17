 # Importing necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Loading data from data release 3
data_3 = pd.read_csv(r'Data/mystery_virus_daily_active_counts_RELEASE#3.csv', parse_dates=['date'], header=0, index_col=None)
x_data_actual = data_3['day'].values.astype(float)
y_data_actual = data_3['active reported daily cases'].values.astype(float)

# Best values for beta, sigma, and gamma from Euler calculation
best_beta = 0.25172413793103443
best_sigma = 0.4854166666666666
best_gamma = 0.07291666666666667

# VT Infection Day 0-70
def euler_seir(timepoints, N, S0, E0, I0, R0, beta, sigma, gamma):
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


S0 = 38857 # VT student population
E0 = 1
I0 = 1
R0 = 0
timepoints = range(1,71)
N = int(S0) + int(E0) + int(I0) + int(R0)

S_fit, E_fit, I_fit, R_fit = euler_seir(
    timepoints, N, S0, E0, I0, R0,
    best_beta, best_sigma, best_gamma
)

plt.figure(figsize=(8, 5))
plt.plot(timepoints, I_fit, label='Best-fit SEIR I(t)')
plt.xlabel('Day')
plt.ylabel('Population')
plt.title('SEIR Fit to Epidemic Data')
plt.legend()
plt.show()


# ===== ENSURE DAY 70 STATE + TIMEPOINTS EXIST =====

# full baseline
timepoints_full = np.arange(1, 121)
S_base, E_base, I_base, R_base = euler_seir(
    timepoints_full, N, S0, E0, I0, R0,
    best_beta, best_sigma, best_gamma
)

# day 70 state
day70_index = np.where(timepoints_full == 70)[0][0]
S70 = S_base[day70_index]
E70 = E_base[day70_index]
I70 = I_base[day70_index]
R70 = R_base[day70_index]

# post-70 timepoints
timepoints_post70 = np.arange(70, 121)

# post-70 baseline
S_post_base, E_post_base, I_post_base, R_post_base = euler_seir(
    timepoints_post70, N, S70, E70, I70, R70,
    best_beta, best_sigma, best_gamma
)

# interventions
S_mask, E_mask, I_mask, R_mask = euler_seir(
    timepoints_post70, N, S70, E70, I70, R70,
    best_beta, best_sigma, best_gamma
)


# Intervention 1: Masking mandate
# reduces transmission by 40%

beta_mask = 0.6 * best_beta

S_mask, E_mask, I_mask, R_mask = euler_seir(
    timepoints_post70, N, S70, E70, I70, R70,
    beta_mask, best_sigma, best_gamma
)

# -----------------------------
# Intervention 2: Vaccine campaign
# single event on day 70
# vaccinate 2000 students with 90% efficacy
# move 1800 from S to R
# -----------------------------
effective_vax_campaign = 2000 * 0.90

S70_vax_campaign = max(S70 - effective_vax_campaign, 0)
R70_vax_campaign = R70 + min(effective_vax_campaign, S70)

S_vax_campaign, E_vax_campaign, I_vax_campaign, R_vax_campaign = euler_seir(
    timepoints_post70, N, S70_vax_campaign, E70, I70, R70_vax_campaign,
    best_beta, best_sigma, best_gamma
)

# -----------------------------
# Intervention 3: Vaccine rollout
# vaccinate 1000 students on day 70, 80, 90 with 90% efficacy
# move 900 from S to R at each event
# -----------------------------
def euler_seir_vaccine_rollout(timepoints, N, S0, E0, I0, R0, beta, sigma, gamma, vax_days, vax_amount, efficacy):
    dt = timepoints[1] - timepoints[0]

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    effective_vax = vax_amount * efficacy

    for i in range(len(timepoints) - 1):
        current_day = timepoints[i]

        # apply vaccination at the start of specified days
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

# -----------------------------
# Intervention 4: Testing + quarantine
# reduces infectious period by 2 days
# infectious period = 1/gamma
# new gamma = 1/(old infectious period - 2)
# -----------------------------
infectious_period = 1 / best_gamma
new_infectious_period = infectious_period - 2

if new_infectious_period <= 0:
    raise ValueError("New infectious period is not valid. Check gamma.")

gamma_test = 1 / new_infectious_period

S_test, E_test, I_test, R_test = euler_seir(
    timepoints_post70, N, S70, E70, I70, R70,
    best_beta, best_sigma, gamma_test
)

# -----------------------------
# Intervention 5: 2-week school closure
# day 70-84: only 20% of normal contacts
# after closure: return to normal
# -----------------------------
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

# -----------------------------
# Metrics function
# Peak infections and total cases prevented from day 70-120
# cases = S(day70) - S(day120)
# -----------------------------
def intervention_metrics(name, S_int, I_int, S_base, I_base):
    peak_base = np.max(I_base)
    peak_int = np.max(I_int)

    total_cases_base = S_base[0] - S_base[-1]
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

intervention_metrics('Masking mandate', S_mask, I_mask, S_post_base, I_post_base)
intervention_metrics('Vaccine campaign', S_vax_campaign, I_vax_campaign, S_post_base, I_post_base)
intervention_metrics('Vaccine rollout', S_vax_rollout, I_vax_rollout, S_post_base, I_post_base)
intervention_metrics('Testing + quarantine', S_test, I_test, S_post_base, I_post_base)
intervention_metrics('2-week school closure', S_close, I_close, S_post_base, I_post_base)

# -----------------------------
# Plot all interventions vs baseline
# -----------------------------
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

