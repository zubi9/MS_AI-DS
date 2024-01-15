import numpy as np
import matplotlib.pyplot as plt

# Parameters
Tstop = 10   # simulation time [sec]
dt = 0.001   # sampling [sec]
a1 = 1
a2 = 0.5
a3 = 2
f1 = 30
f2 = 76
f3 = 150
A_noise = 2
T1 = 5  # anomaly appears since t > T1
a4 = 1
f4 = 50
m = 200  # number of columns of DEM (slided windows)

# Generate the normal and anomaly signals
t = np.arange(0, Tstop, dt)  # time vector
N = len(t)
y = (
    a1 * np.sin(2 * np.pi * f1 * t)
    + a2 * np.sin(2 * np.pi * f2 * t)
    + a3 * np.sin(2 * np.pi * f3 * t)
    + np.random.randn(N) * A_noise
)  # normal signal with noise

y[t > T1] = y[t > T1] + a4 * np.sin(2 * np.pi * f4 * t[t > T1])

# Prepare the data for the neural network
DEM = np.zeros((N - m, m))
for i in range(m):
    DEM[:, i] = y[i : N - m + i]
X = DEM
X = (X - np.mean(X)) / np.std(X)  # z-score

# Define the neural network architecture
input_size = m  # Number of columns
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 9  # Number of principal components

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size)

# Training parameters

learn_rate = 0.01
epochs = 1000

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1)  # ReLU activation
    z2 = np.dot(a1, W2) + b2

    # Loss calculation
    loss = np.mean(np.square(z2 - X))

    # Backpropagation
    d_z2 = 2 * (z2 - X) / N
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * (z1 > 0)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # Update weights and biases
    W2 -= learn_rate * d_W2
    b2 -= learn_rate * d_b2
    W1 -= learn_rate * d_W1
    b1 -= learn_rate * d_b1

# Get the compressed signal from the neural network
z1 = np.dot(X, W1) + b1
a1 = np.maximum(0, z1)
C = np.dot(a1, W2) + b2

# Decompress the signal
Xdecompressed = C

# Plot the results
plt.figure()
plt.subplot(411)
plt.plot(t, y)
plt.subplot(412)
plt.plot(C)
plt.subplot(413)
plt.plot(Xdecompressed[:, 3])
plt.show()
