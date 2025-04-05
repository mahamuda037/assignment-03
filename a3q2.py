import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp
def lotka_volterra(t,z):
    x,y=z
    dx_dt=-0.1*x+0.02*x*y
    dy_dt=0.2*y-0.025*x*y
    return[dx_dt,dy_dt]
x0=6
y0=6
z0=[x0,y0]
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 2000)
sol = solve_ivp(lotka_volterra, t_span, z0, t_eval=t_eval)
t = sol.t
x = sol.y[0]
y = sol.y[1]

# Display x(t) and y(t) in a table
df = pd.DataFrame({'t': t, 'x(t)': x, 'y(t)': y})
pd.set_option('display.float_format', '{:.4f}'.format)
print("Values of x(t) and y(t):")
print(df)

# Find the first time when x(t) ≈ y(t), ignoring t = 0
diff = np.abs(x - y)
nonzero_diff_indices = np.where(t > 0)[0]
equal_index = nonzero_diff_indices[np.argmin(diff[nonzero_diff_indices])]
equal_time = t[equal_index]
equal_value = x[equal_index]
print(f"\nThe populations are approximately equal at first t ≈ {equal_time:.2f}(t>0), with x(t) ≈ y(t) ≈ {equal_value:.2f}")

plt.figure()
plt.plot(t,x,label='predators(x)',color='red')
plt.plot(t,y,label='prey(y)',color='green')
plt.axvline(equal_time, color='blue', linestyle='--', label=f'x ≈ y ≈ {equal_value:.2f} at t ≈ {equal_time:.2f}')
plt.xlabel("Time")
plt.ylabel("Populations (in thousands)")
plt.title('Lotka-Volterra Predator-Prey Model')
plt.legend()
plt.grid(True)
plt.show()