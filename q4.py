import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
g = 32.17  
L = 2     
def pendulum_eq(theta, t):
    dtheta_dt = theta[1]
    d2theta_dt2 = -(g / L) * np.sin(theta[0])
    return [dtheta_dt, d2theta_dt2]
theta0 = [np.pi/ 6,0]
t = np.arange(0,2.1,0.1) 
theta = odeint(pendulum_eq, theta0,t)
plt.plot(t, theta[:, 0], label='Angle (radians)')
plt.xlabel('Time (s)')
plt.ylabel('Theta (radians)')
plt.title('Motion of a Swinging Pendulum')
plt.legend()
plt.grid(True)
plt.show()
print("Theta:",theta)