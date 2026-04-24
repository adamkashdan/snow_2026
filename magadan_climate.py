import datetime
from meteostat import daily
import pandas as pd
import matplotlib.pyplot as plt
import pyhomogeneity as hg

# 1. Define location for Magadan (Russia)
# Station ID for Magadan is 25913
magadan = '25913'

# Set time period (e.g., last 30 years)
start = datetime.datetime(1993, 1, 1)
end = datetime.datetime(2023, 12, 31)

# 2. Fetch daily data using Meteostat
print("Fetching daily weather data for Magadan...")
data = daily(magadan, start, end)
data = data.fetch()

# Convert the index to a proper datetime if it isn't already
# and ensure we have a continuous date range (introducing NaNs for missing days)
full_index = pd.date_range(start=start, end=end, freq='D')
data = data.reindex(full_index)

# 3. Gap Filling (Infilling missing data)
# `climatol` in R uses surrounding stations or statistical methods to infill data.
# In Python, we can use Pandas interpolation (linear, spline, polynomial) or specialized libraries.
print("\n--- Missing Data Information before infilling ---")
print(data[['temp', 'prcp']].isnull().sum())

# We fill missing temperatures via linear interpolation and missing precipitation with 0
data['temp_filled'] = data['temp'].interpolate(method='linear')
data['prcp_filled'] = data['prcp'].fillna(0)

print("\n--- Missing Data Information after infilling ---")
print(data[['temp_filled', 'prcp_filled']].isnull().sum())


# 4. Homogenization Testing
# `climatol` checks for breaks/inhomogeneities in climate data.
# In Python, `pyhomogeneity` offers tests like Pettitt, SNHT, Buishand.

# Let's perform a Pettitt's test for homogeneity on the Annual Mean Temperature
# Resample to annual mean temperature
annual_temp = data['temp_filled'].resample('YE').mean()

# Perform Pettitt's test
print("\n--- Running Pettitt's Homogeneity Test ---")
h, cp, p, U, mu = hg.pettitt_test(annual_temp, alpha=0.05)
print(f"Homogeneous (True) or Non-Homogeneous (False): {h}")
cp_date = pd.to_datetime(cp)
print(f"Change point: {cp} (which corresponds to {cp_date.year})")
print(f"P-value: {p:.4f}")


# 5. Visualizing the Data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot Daily Average Temperature
ax1.plot(data.index, data['temp_filled'], color='orange', alpha=0.6, label='Daily Temp (°C)')
ax1.plot(annual_temp.index, annual_temp, color='red', linewidth=2, label='Annual Mean Temp')

# Highlight the change point if the data is not homogeneous
if not h: # If False, it means there's a breakpoint
    ax1.axvline(x=cp_date, color='blue', linestyle='--', label='Breakpoint (Pettitt Test)')
    
ax1.set_title('Average Daily Temperature in Magadan (1993 - 2023)')
ax1.set_ylabel('Temperature (°C)')
ax1.legend()
ax1.grid(True)

# Plot Daily Precipitation
# Group by year for a clearer bar chart of annual precipitation
annual_prcp = data['prcp_filled'].resample('YE').sum()
ax2.bar(annual_prcp.index.year, annual_prcp, color='teal', label='Annual Precipitation (mm)')
ax2.set_title('Total Annual Precipitation in Magadan (1993 - 2023)')
ax2.set_ylabel('Precipitation (mm)')
ax2.set_xlabel('Year')
ax2.legend()
ax2.grid(True, axis='y')

plt.tight_layout()
plt.savefig('magadan_climate_analysis.png')
print("\nPlot saved as 'magadan_climate_analysis.png'.")
