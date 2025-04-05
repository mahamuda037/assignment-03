import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
def competition_model(z, t):
    x, y = z
    dxdt = x * (2 - 0.4 * x - 0.3 * y)
    dydt = y * (1 - 0.1 * y - 0.3 * x)
    return [dxdt, dydt]
t = np.linspace(0, 50, 500)
plt.figure(figsize=(12,10))
initial_conditions = [(1.5, 3.5),(1, 1),(2, 7),(4.5, 0.5)]
for i, (x0, y0) in enumerate(initial_conditions):
    z0 = [x0, y0]
    z = odeint(competition_model, z0, t)
    x, y = z.T
    plt.subplot(2, 2, i + 1)
    plt.plot(t, x, label='x(t)')
    plt.plot(t, y, label='y(t)')
    plt.xlabel('Time (years)')
    plt.ylabel('Population (thousands)')
    plt.title(f'Initial conditions: x(0)={x0}, y(0)={y0}')
    plt.legend()
plt.show()