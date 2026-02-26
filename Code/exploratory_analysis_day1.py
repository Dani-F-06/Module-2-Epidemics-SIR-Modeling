#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv(r'Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%
# Make a plot of the active cases over time
plt.figure(figsize=(10, 6))
plt.plot(data['day'], data['active reported daily cases'], marker='o', linestyle='-')
plt.xlabel('Day')
plt.ylabel('Active Cases')
plt.title('Active Cases Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show() 

# What do you notice about initial infections?
#   The initial infections seem to be very low, with only a few cases reported in the early days. This could indicate that the virus was not spreading widely at the beginning, or that there was limited testing and reporting.
# How could we measure how quickly its spreading?
#   We could measure how quickly the virus is spreading by calculating the growth rate of active cases over time. This can be done by taking the difference in active cases between consecutive days and dividing it by the number of active cases on the previous day to get a percentage growth rate. 
# What information about the virus would be helpful in determining the shape of the outbreak curve?
#   Information about the virus's transmission rate, incubation period, and recovery time would be helpful in determining the shape of the outbreak curve.