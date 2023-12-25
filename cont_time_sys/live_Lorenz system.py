import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Lorenz system equations
def lorenz(t, xyz):
    x, y, z = xyz
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Initial conditions
initial_conditions = [1.0, 0.0, 20.0]

# Time span for simulation
t_span = (0, 25)
t_eval = np.linspace(*t_span, 1000)

# Function to update the plot in each animation frame
def update(frame):
    ax.cla()
    ax.set_title(f"Lorenz System - Time: {t_eval[frame]:.2f}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    # Plot the trajectory up to the current frame
    sol = solve_ivp(lorenz, t_span, initial_conditions, t_eval=t_eval[:frame+1])
    x, y, z = sol.y

    # Plot the trajectory line
    ax.plot(x, y, z, color='b', alpha=0.7)

    # Plot the current point
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color='r', s=50, label='Current Point')

    ax.legend()


# Create the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set up the initial plot
ax.set_title("Lorenz System")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

# Create the animation
#animation = FuncAnimation(fig, update, frames=len(t_eval), interval=50, repeat=False)
# Create the animation with a faster frame speed
animation = FuncAnimation(fig, update, frames=len(t_eval), interval=10, repeat=False)


# Show the plot
plt.show()
