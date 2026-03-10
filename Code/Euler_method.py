#%% SEIR model fitting with Euler's method
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# Helper: safely find data columns
# -------------------------------

data = pd.read_csv(r"Data/mystery_virus_daily_active_counts_RELEASE#2.csv", parse_dates=['date'], header=0, index_col=None)



possible_day_cols = ['day', 'day_number', 'Day', 'Day Number']
possible_case_cols = ['active_cases', 'active case', 'active', 'cases', 'Active Cases']

day_col = None
case_col = None

for col in data.columns:
    if col in possible_day_cols:
        day_col = col
    if col in possible_case_cols:
        case_col = col

# fallback: try lowercase matching
if day_col is None:
    for col in data.columns:
        if 'day' in col.lower():
            day_col = col
            break

if case_col is None:
    for col in data.columns:
        c = col.lower()
        if 'active' in c or 'case' in c:
            case_col = col
            break

if day_col is None or case_col is None:
    raise ValueError(f"Could not find required columns. Columns found: {list(data.columns)}")

timepoints = data[day_col].to_numpy()
infected_data = data[case_col].to_numpy()

# -------------------------------
# Initial conditions
# -------------------------------
N = 10000          
I0 = infected_data[0]
E0 = I0             # reasonable first guess: exposed starts similar to infected
R0_init = 0         # initial recovered
S0 = N - E0 - I0 - R0_init

# -------------------------------
# Euler method for SEIR
# -------------------------------
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

# -------------------------------
# SSE calculation
# -------------------------------
def calculate_sse(model_I, data_I):
    return np.sum((model_I - data_I) ** 2)

# -------------------------------
# Parameter search
# -------------------------------
def fit_seir_parameters(timepoints, N, S0, E0, I0, R0, infected_data):
    # These are reasonable starter ranges for a daily-timescale outbreak
    beta_values = np.linspace(0.1, 1.2, 30)
    sigma_values = np.linspace(0.05, 0.6, 25)
    gamma_values = np.linspace(0.05, 0.6, 25)

    best_sse = np.inf
    best_beta = None
    best_sigma = None
    best_gamma = None

    sse_results = []

    for b in beta_values:
        for s in sigma_values:
            for g in gamma_values:
                S, E, I, R = euler_seir(timepoints, N, S0, E0, I0, R0, b, s, g)
                sse = calculate_sse(I, infected_data)
                sse_results.append((b, s, g, sse))

                if sse < best_sse:
                    best_sse = sse
                    best_beta = b
                    best_sigma = s
                    best_gamma = g

    return best_beta, best_sigma, best_gamma, best_sse, sse_results

# -------------------------------
# Fit the model
# -------------------------------
best_beta, best_sigma, best_gamma, best_sse, sse_results = fit_seir_parameters(
    timepoints, N, S0, E0, I0, R0_init, infected_data
)

print("Best beta =", best_beta)
print("Best sigma =", best_sigma)
print("Best gamma =", best_gamma)
print("Best SSE =", best_sse)

# -------------------------------
# Run best-fit model on observed time window
# -------------------------------
S_fit, E_fit, I_fit, R_fit = euler_seir(
    timepoints, N, S0, E0, I0, R0_init,
    best_beta, best_sigma, best_gamma
)

# -------------------------------
# Plot model fit vs data
# -------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(timepoints, infected_data, label='Observed active cases')
plt.plot(timepoints, I_fit, label='Best-fit SEIR I(t)')
plt.xlabel('Day')
plt.ylabel('Population')
plt.title('SEIR Fit to Epidemic Data')
plt.legend()
plt.show()

# -------------------------------
# Predict the future peak
# -------------------------------
future_days = 120
future_timepoints = np.arange(timepoints[0], timepoints[0] + future_days + 1, 1)

S_future, E_future, I_future, R_future = euler_seir(
    future_timepoints, N, S0, E0, I0, R0_init,
    best_beta, best_sigma, best_gamma
)

peak_index = np.argmax(I_future)
peak_day = future_timepoints[peak_index]
peak_value = I_future[peak_index]

print("Predicted peak infected population =", peak_value)
print("Predicted peak day =", peak_day)

# -------------------------------
# Plot future prediction
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(future_timepoints, I_future, label='Predicted I(t)')
plt.scatter(peak_day, peak_value, label=f'Peak: day {peak_day}, I = {peak_value:.2f}')
plt.xlabel('Day')
plt.ylabel('Population')
plt.title('SEIR Future Prediction')
plt.legend()
plt.show()