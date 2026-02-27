# Generative AI was used as assistance for code development relating to fitting the data to an exponential growth curve and estimating R0.


#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
#%%
# Load the data
data = pd.read_csv('C:\\Users\\kiran\\Desktop\\UVA\\Classes\\Semester 4\\Computational BME\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)
#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.

t = data['day'].values
I = data['active reported daily cases'].values

# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

# Fit the exponential growth model to the data. 
# Initial infected population I0 = 1

initial_guess = [0.1]  # initial guess for r
params, covariance = curve_fit(exponential_growth, t, I, p0=initial_guess)

r_est = params[0]

# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.

# Approximate R0 using this fit

gamma = 1 / 9  # recovery rate based on 9-day infectious period
R0_est = 1 + r_est / gamma

# Add the fit as a line on top of your scatterplot.

t_fit = np.linspace(min(t), max(t), 200)
I_fit = exponential_growth(t_fit, r_est)

plt.figure()
plt.scatter(t, I, label='Reported data')
plt.plot(t_fit, I_fit, label='Exponential fit (I0 = 1)')
plt.xlabel('Day')
plt.ylabel('Active Infections')
plt.title('Exponential Growth Fit to Mystery Virus Data')
plt.legend()
plt.grid(True)
plt.show()

print(f"Estimated growth rate r = {r_est:.4f} per day")
print(f"Estimated R0 = {R0_est:.2f}")