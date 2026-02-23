#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv('C:\\Users\\kiran\\Desktop\\UVA\\Classes\\Semester 4\\Computational BME\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%
# Plot
plt.figure() # Create a new figure
plt.plot(data["day"], data["active reported daily cases"]) # Plot active infections vs day
plt.xlabel("Day") # Label for x-axis
plt.ylabel("Active Infections") # Label for y-axis
plt.title("DATA RELEASE #1: Active Infections vs Day") # Title for the plot
plt.grid(True) # Add a grid for better visibility
plt.show()