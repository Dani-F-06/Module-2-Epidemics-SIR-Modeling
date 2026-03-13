# Importing necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Euler_method import best_beta, best_sigma, best_gamma

# Loading data from data release 3
data_3 = pd.read_csv(r'/Users/saraelster/Desktop/UVA/Computational BME/Module 2/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#3.csv', parse_dates=['date'], header=0, index_col=None)

x_data_actual = data_3['day'].values.astype(float)
y_data_actual = data_3['active reported daily cases'].values.astype(float)


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


S0 = 38,857 # VT student population
E0 = 1
I0 = 1
R0 = 0
timepoints = range(1,71)
N = S0 + E0 + I0 + R0

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

# Intervention strategy 1: Masking Mandates (starting at day 70) - reduces transmission by 40%



# Intervention strategy 2: Testing/Quarantine (starting at day 70) - reduces infectious period by 2 days



# Intervention strategy 3: 2 week school closure (starting at day 70) - day 70-84 - 20% normal contacts; day 84-120 - normal contacts
