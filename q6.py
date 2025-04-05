import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp
def shooting_system(x, y):
    y1, y2 = y
    return [y2, 100*y1]
def shooting_method(h, y0, y1, x_end):
    x = np.arange(0, x_end + h, h)
    sol1 = solve_ivp(shooting_system, [0, x_end], [0, 1], t_eval=x, method='RK45')
    sol2 = solve_ivp(shooting_system, [0, x_end], [1, 0], t_eval=x, method='RK45')
    v1, v1p = sol1.y
    v2, v2p = sol2.y
    s = (y1 - v2[-1]) / v1[-1]
    y_shoot = v2 + s * v1  # y = v2 + s*v1
    return x, y_shoot
x1, y1 = shooting_method(0.1, 1, np.exp(-10), 1)
x2, y2 = shooting_method(0.05, 1, np.exp(-10), 1)
print("\nShooting Method (h=0.1):")
for i in range(len(x1)):
    print(f"x = {x1[i]:.2f}, y(x) = {y1[i]:.6f}")
print("\nShooting Method (h=0.05):")
for i in range(len(x2)):
    print(f"x = {x2[i]:.2f}, y(x) = {y2[i]:.6f}")
def bvp_system(x, y):
    y1, y2 = y
    return np.vstack([y2, 100*y1])
def bc(Ya, Yb):
    return np.array([Ya[0] - 1, Yb[0] - np.exp(-10)])
x_mesh = np.linspace(0, 1, 50)
Y_guess = np.zeros((2, x_mesh.size))  
bvp_solution = solve_bvp(bvp_system, bc, x_mesh, Y_guess)
y_bvp = bvp_solution.sol(x_mesh)[0]
print("\nsolve_bvp Solution:")
for i in range(len(bvp_solution.x)):
    print(f"x = {bvp_solution.x[i]:.2f}, y(x) = {bvp_solution.y[0, i]:.6f}")
x_exact = np.linspace(0, 1, 100)
y_exact = np.exp(-10 * x_exact)
print("\nExact Solution:")
for i in range(len(x_exact)):
    print(f"x = {x_exact[i]:.2f}, y(x) = {y_exact[i]:.6f}")
plt.figure(figsize=(10, 6))
plt.plot(x_exact, y_exact, 'k-', label="Analytical Solution")
plt.plot(x1, y1, 'ro-', label="Shooting Method (h=0.1)")
plt.plot(x2, y2, 'bs-', label="Shooting Method (h=0.05)")
plt.plot(x_mesh, y_bvp, 'g*-', label="solve_bvp Solution")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Numerical and Analytical Solutions")
plt.legend()
plt.grid()
plt.show()
