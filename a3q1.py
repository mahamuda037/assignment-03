import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

# ----------------------------
# Problem 1: y' = t*e^{3t} - 2y, y(0) = 0
# ----------------------------

def f1(y, t):
    return t * np.exp(3*t) - 2*y

def exact1(t):
    return (1/5)*t*np.exp(3*t) - (1/25)*np.exp(3*t) + (1/25)*np.exp(-2*t)

t1 = np.linspace(0, 1, 100)
y1_odeint = odeint(f1, 0, t1).flatten()
sol1 = solve_ivp(lambda t, y: f1(y, t), [0, 1], [0], t_eval=t1)
y1_solve_ivp = sol1.y[0]
y1_exact = exact1(t1)

error1_odeint = np.abs(y1_odeint - y1_exact)
error1_solve_ivp = np.abs(y1_solve_ivp - y1_exact)

# Iteration Table for Problem 1
print("Problem 1: Iteration Table")
print(" t_n     y_n (odeint)    y_n (solve_ivp)   Exact y_n    Error (odeint)    Error (solve_ivp)")
for i in range(len(t1)):
    print(f"{t1[i]:.2f}    {y1_odeint[i]:.5f}        {y1_solve_ivp[i]:.5f}        {y1_exact[i]:.5f}        {error1_odeint[i]:.5f}        {error1_solve_ivp[i]:.5f}")

print("\nApproximated Result (Problem 1 at t=1):")
print(f"odeint      y(1) ≈ {y1_odeint[-1]:.5f}")
print(f"solve_ivp   y(1) ≈ {y1_solve_ivp[-1]:.5f}")

# ----------------------------
# Problem 2: y' = 1 + (t - y)^2, y(2) = 1
# ----------------------------

def f2(t, y):
    return 1 + (t - y)**2

def exact2(t):
    return t + 1/(1 - t)

t2 = np.linspace(2, 3, 100)
y2_odeint = odeint(lambda y, t: f2(t, y), 1, t2).flatten()
sol2 = solve_ivp(f2, [2, 3], [1], t_eval=t2)
y2_solve_ivp = sol2.y[0]
y2_exact = exact2(t2)

error2_odeint = np.abs(y2_odeint - y2_exact)
error2_solve_ivp = np.abs(y2_solve_ivp - y2_exact)

# Iteration Table for Problem 2
print("\nProblem 2: Iteration Table")
print(" t_n     y_n (odeint)    y_n (solve_ivp)   Exact y_n    Error (odeint)    Error (solve_ivp)")
for i in range(len(t2)):
    print(f"{t2[i]:.2f}    {y2_odeint[i]:.5f}        {y2_solve_ivp[i]:.5f}        {y2_exact[i]:.5f}        {error2_odeint[i]:.5f}        {error2_solve_ivp[i]:.5f}")

print("\nApproximated Result (Problem 2 at t=3):")
print(f"odeint      y(3) ≈ {y2_odeint[-1]:.5f}")
print(f"solve_ivp   y(3) ≈ {y2_solve_ivp[-1]:.5f}")

# ----------------------------
# Plotting
# ----------------------------
plt.figure(figsize=(14, 10))

# Problem 1 - Solutions
plt.subplot(2, 2, 1)
plt.plot(t1, y1_exact, 'k-', label='Exact')
plt.plot(t1, y1_odeint, 'r--', label='odeint')
plt.plot(t1, y1_solve_ivp, 'b:', label='solve_ivp')
plt.title("Problem 1: Solutions")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()

# Problem 1 - Errors
plt.subplot(2, 2, 2)
plt.plot(t1, error1_odeint, 'r--', label='Error (odeint)')
plt.plot(t1, error1_solve_ivp, 'b:', label='Error (solve_ivp)')
plt.title("Problem 1: Errors")
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid()

# Problem 2 - Solutions
plt.subplot(2, 2, 3)
plt.plot(t2, y2_exact, 'k-', label='Exact')
plt.plot(t2, y2_odeint, 'r--', label='odeint')
plt.plot(t2, y2_solve_ivp, 'b:', label='solve_ivp')
plt.title("Problem 2: Solutions")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()

# Problem 2 - Errors
plt.subplot(2, 2, 4)
plt.plot(t2, error2_odeint, 'r--', label='Error (odeint)')
plt.plot(t2, error2_solve_ivp, 'b:', label='Error (solve_ivp)')
plt.title("Problem 2: Errors")
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()