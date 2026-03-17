# Importing necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Loading data from data release 3
data_3 = pd.read_csv(r'Data/mystery_virus_daily_active_counts_RELEASE#3.csv', parse_dates=['date'], header=0, index_col=None)

x_data_actual = data_3['day'].values.astype(float)
y_data_actual = data_3['active reported daily cases'].values.astype(float)

<<<<<<< Updated upstream
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

# Intervention strategy 1: Masking Mandates (starting at day 70) - reduces transmission by 40%
new_beta = 0.4 * best_beta
timepoints_2 = range(70,121)

S_fit2, E_fit2, I_fit2, R_fit2 = euler_seir(
    timepoints_2, N, S0, E0, I0, R0,
    new_beta, best_sigma, best_gamma
)

plt.figure(figsize=(8, 5))
plt.plot(timepoints, I_fit, label='No Masking Mandate')
plt.plot(timepoints_2, I_fit2, label='Masking Mandate')
plt.xlabel('Day')
plt.ylabel('Population')
plt.title('Modeling Effects of Masking Mandate on Infection Spread at VT')
plt.legend()
plt.show()

# Intervention strategy 2: Testing/Quarantine (starting at day 70) - reduces infectious period by 2 days



# Intervention strategy 3: 2 week school closure (starting at day 70) - day 70-84 - 20% normal contacts; day 84-120 - normal contacts
=======
# Plotting full dataset against SEIR model
plt.figure(figsize=(8, 5))
plt.scatter(x_data_actual, y_data_actual, label = "Actual Data")
plt.xlabel('Day')
plt.ylabel('Population')
plt.title('SEIR Future Prediction')
plt.legend()
plt.show()
>>>>>>> Stashed changes
