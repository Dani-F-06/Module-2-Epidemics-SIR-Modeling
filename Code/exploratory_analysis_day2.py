#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
#%%
# Load the data
data = pd.read_csv(r'/Users/saraelster/Desktop/UVA/Computational BME/Module 2/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)
#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

# Fit the exponential growth model to the data. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.
x_data = data['day'].values.astype(float)
y_data = data['active reported daily cases'].values.astype(float)
popt, pcov = curve_fit(exponential_growth, x_data, y_data)

r_fit = popt[0]

# Approximate R0 using this fit
D = 6 # It seems from the data that there is a 6 day infectious period

r0 = np.exp(r_fit * D)

print("Estimated growth rate R0: ", r0)

# Add the fit as a line on top of your scatterplot.
# Generate fitted curve
y_fitted = exponential_growth(x_data, r_fit)

# Plot fitted curve and actual data
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label = "Actual Data")
plt.plot(x_data, y_fitted, color = "red", label = "Estimated Curve")
plt.xlabel('Day')
plt.ylabel('Active Cases')
plt.title('Exponential Growth Model Fit to Virus Data')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show() 

# %%
