# Importing necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Loading data from data release 3
data_3 = pd.read_csv(r'/Users/saraelster/Desktop/UVA/Computational BME/Module 2/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#3.csv', parse_dates=['date'], header=0, index_col=None)

x_data_actual = data_3['day'].values.astype(float)
y_data_actual = data_3['active reported daily cases'].values.astype(float)

# Intervention strategy 1: Masking Mandates (starting at day 70) - reduces transmission by 40%



# Intervention strategy 2: Testing/Quarantine (starting at day 70) - reduces infectious period by 2 days



# Intervention strategy 3: 2 week school closure (starting at day 70) - day 70-84 - 20% normal contacts; day 84-120 - normal contacts
