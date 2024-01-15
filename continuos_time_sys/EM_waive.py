import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a grid of points
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)

# Define parameters of the wave
amplitude = 1.0
frequency = 1.0
wavelength = 2 * np.pi / frequency

# Calculate the electromagnetic wave equation
z = amplitude * np.sin(2 * np.pi * frequency * (np.sqrt(x**2 + y**2) - wavelength))

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the electromagnetic wave
ax.plot_surface(x, y, z, cmap='viridis')

# Set labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Amplitude')

# Set title
ax.set_title('3D Electromagnetic Wave')

# Show the plot
plt.show()
