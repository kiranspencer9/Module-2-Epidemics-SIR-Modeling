# Generative AI was used as assistance for code development relating to fitting the data to an exponential growth curve and estimating R0.


#%%
import pandas as pd # import pandas for data manipulation
import matplotlib.pyplot as plt # import matplotlib for plotting
import numpy as np # import numpy for numerical operations
from scipy.optimize import curve_fit # import curve_fit for fitting the exponential growth model to the data
#%%
# Load the data
data = pd.read_csv('C:\\Users\\kiran\\Desktop\\UVA\\Classes\\Semester 4\\Computational BME\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)
#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.

t = data['day'].values # time variable (day number)
I = data['active reported daily cases'].values # active infections variable (number of active cases)

# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

# Fit the exponential growth model to the data. 
# Initial infected population I0 = 1

initial_guess = [0.1]  # initial guess for r
params, covariance = curve_fit(exponential_growth, t, I, p0=initial_guess) # Fit the model to the data

r_est = params[0] # Extract the estimated growth rate r from the fitted parameters

# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.

# Approximate R0 using this fit

gamma = 1 / 9  # recovery rate based on 9-day infectious period
R0_est = 1 + r_est / gamma # R0 can be approximated as 1 + (growth rate / recovery rate) for early exponential growth phase

# Add the fit as a line on top of your scatterplot.

t_fit = np.linspace(min(t), max(t), 200) # Generate a range of time values for plotting the fitted curve
I_fit = exponential_growth(t_fit, r_est) # Calculate the fitted values using the estimated growth rate r_est

plt.figure() 
plt.scatter(t, I, label='Reported data') # Plot the original data points
plt.plot(t_fit, I_fit, label='Exponential fit (I0 = 1)') # Plot the fitted exponential growth curve
plt.xlabel('Day') # Label for x-axis
plt.ylabel('Active Infections') # Label for y-axis
plt.title('Exponential Growth Fit to Mystery Virus Data') # Title for the plot
plt.legend() # Add a legend to differentiate between the data points and the fitted curve
plt.grid(True) # Add a grid for better visibility
plt.show()

print(f"Estimated growth rate r = {r_est:.4f} per day")
print(f"Estimated R0 = {R0_est:.2f}")