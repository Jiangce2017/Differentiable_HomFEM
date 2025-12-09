import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Example data (in Celsius)
T_celsius = np.array([13, 20, 30, 40, 50, 60, 70, 80])  # °C
R_t = np.array([735000, 390405, 248158, 238378, 155455, 105968, 68333, 54767])  # Ohms

# Convert to Kelvin
T = T_celsius + 273.15

# Calculate ln(R_t) and 1/T
ln_Rt = np.log(R_t)
inv_T = 1 / T

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(inv_T, ln_Rt)

# Calculate β (negative slope)
beta = -slope

print(f"β (material constant) = {beta:.2f} K")
print(f"R² = {r_value**2:.4f}")

# Plot
plt.scatter(inv_T, ln_Rt, color="blue", label="Data points")
plt.plot(inv_T, slope * inv_T + intercept, color="red", label=f"Fit (β = {beta:.0f} K)")
plt.xlabel("1 / T")
plt.ylabel("ln(Rt)")
plt.title("Thermistor β Constant Determination")
plt.legend()
plt.grid(True)
plt.show()
