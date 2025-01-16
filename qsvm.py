

import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib as plt
from tqdm import tqdm
from time import time

import matplotlib.pyplot as plt

#matplotlib.use('TkAgg')  # Use TkAgg backend

#from qiskit.qasm2 import dumps
from qiskit.circuit.library import ZZFeatureMap, pauli_feature_map
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_machine_learning.algorithms import QSVC
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.visualization import plot_histogram
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
#from qiskit.utils import algorithm_globals

import qiskit
print(qiskit.__version__)

from qiskit.qasm3 import dumps


# Classical Machine Learning imports
from sklearn.svm import SVC

# Utilities for timing
from time import time


from qiskit.providers.basic_provider import BasicProvider
 
backend = BasicProvider().get_backend('basic_simulator')



feature_dim = 2
seed = 10240

"""
Variables:
X_train -> X train data
y_train -> y train data
X_test -> X test data
y_test -> y test data
class_labels -> labels
"""

# X_train, y_train, X_test, y_test, class_labels = ad_hoc_data(
#     training_size=100,
#     test_size=40,
#     n=feature_dim,
#     gap=0,
#     plot_data=True,
#     one_hot=False,
#     include_sample_total=True,
# )

from sklearn.datasets import make_circles

# Create a synthetic dataset (not linearly separable)
X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)


from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer

# Example: Load the Iris dataset
# data = load_wine()
# X = data.data
# y = data.target

# from sklearn.datasets import make_moons
# X, y = make_moons(n_samples=200, noise=0.1)


# from datetime import datetime


# points_red = []
# points_blue = []
# points_green = []

# # Current color to use
# current_color = "red"

# # Event handler for mouse clicks
# def on_click(event):
#     global current_color
#     if event.xdata is not None and event.ydata is not None:  # Check if click is within the plot
#         x, y = event.xdata, event.ydata
#         if current_color == "red":
#             points_red.append((x, y))
#         elif current_color == "blue":
#             points_blue.append((x, y))
#         elif current_color == "green":
#             points_green.append((x, y))
#         # Plot the point
#         plt.scatter(x, y, c=current_color, s=50)
#         plt.draw()

# # Event handler for key presses
# def on_key(event):
#     global current_color
#     if event.key == "r":  # Change color to red
#         current_color = "red"
#         print("Current color: Red")
#     elif event.key == "b":  # Change color to blue
#         current_color = "blue"
#         print("Current color: Blue")
#     elif event.key == "g":  # Change color to green
#         current_color = "green"
#         print("Current color: Green")
#     elif event.key == "q":  # Quit and print the points
#         print("Red points:", points_red)
#         print("Blue points:", points_blue)
#         print("Green points:", points_green)
#         plt.close()

# # Setup the canvas
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.set_title("Click to add points. Press 'r', 'b', 'g' to change color. Press 'q' to quit.")
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
# ax.grid(True)

# # Connect the event handlers
# fig.canvas.mpl_connect("button_press_event", on_click)
# fig.canvas.mpl_connect("key_press_event", on_key)

# # Show the canvas
# plt.show()


# # Getting the current date and time
# dt = datetime.now()

# np.savetxt("red_points_"+str(dt)+".csv", points_red, delimiter=",")
# np.savetxt("blue_points_"+str(dt)+".csv", points_blue, delimiter=",")

# np.savetxt("red_points.csv", points_red, delimiter=",")
# np.savetxt("blue_points.csv", points_blue, delimiter=",")


# Load red points from the CSV file
points_red = np.loadtxt("red_points.csv", delimiter=",")

# Load blue points from the CSV file
points_blue = np.loadtxt("blue_points.csv", delimiter=",")


print(points_blue)
print("stufff")
print(points_red)


#X = np.stack([points_red, points_blue],axis=1)
X = np.vstack((points_red, points_blue))

print("asdfasdfas")
print(X)
y = np.array([0] * len(points_red) + [1] * len(points_blue))  # Class labels

# Separate x and y coordinates
xp = X[:, 0]
yp = X[:, 1]

# Normalize x and y to the interval [0, 2π]
x_norm = 2 * np.pi * (xp - np.min(xp)) / (np.max(xp) - np.min(xp))
y_norm = 2 * np.pi * (yp - np.min(yp)) / (np.max(yp) - np.min(yp))

# Combine normalized coordinates
points_normalized = np.column_stack((x_norm, y_norm))

X = points_normalized





# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scatter plot for the two classes
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0', alpha=0.7, edgecolor='k')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1', alpha=0.7, edgecolor='k')

# Add labels, legend, and grid
plt.title("2D Dataset - Two Classes", fontsize=14)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Set a seed for reproducibility
#algorithm_globals.random_seed = 42


reps = 2
#feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=reps, entanglement="linear")

feature_map = pauli_feature_map(feature_dimension=feature_dim, reps=reps, entanglement="linear", paulis=["zz","zz"])



sampler = Sampler()

fidelity = ComputeUncompute(sampler=sampler)

kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)


quantum_circuit = feature_map.decompose()
print("Unbound parameters:", quantum_circuit.parameters)

# # Bind parameters
# param_values = {param: 1.0 for param in quantum_circuit.parameters}  # Replace 1.0 with actual values if needed
# bound_circuit = quantum_circuit.bind_parameters(param_values)

qasm_file = "feature_map.qasm"
with open(qasm_file, "w") as f:
    f.write(dumps(quantum_circuit))

print(f"Feature map circuit saved to {qasm_file}")

print(quantum_circuit)


parameters = quantum_circuit.parameters
print("Unbound parameters:", parameters)

x = ParameterVector('x', 2)

array1 = [1.23, 4.56]  # Values for x[0] and x[1]

# Map the parameters to the array values
param_bindings = {param: value for param, value in zip(x, array1)}

# bc = quantum_circuit.assign_parameters(param_bindings)
# bc.measure_all()
# bc.draw('mpl')

# simulator = AerSimulator()
# compiled_circuit = transpile(bc, simulator)
# sim_result = simulator.run(compiled_circuit).result()
# counts = sim_result.get_counts()
# plot_histogram(counts)
# plt.show()


# QSVC (Quantum Support Vector Classifier)
qsvc = QSVC(quantum_kernel=kernel)
t0 = time()
qsvc.fit(X_train, y_train)

# with tqdm(total=1, desc="QSVC Fitting", leave=True) as pbar:
#     qsvc.fit(X_train, y_train)
#     pbar.update(1)  # Update when done


qsvc_score = qsvc.score(X_test, y_test)
t1 = time()

print(f"QSVC classification test score: {qsvc_score}")
print(f"Time taken: {t1-t0}")

Z = qsvc.predict(X_test)

print("Predictions:", Z)

# Classical SVC with Quantum Kernel
svc = SVC(kernel=kernel.evaluate)
t0 = time()
svc.fit(X_train, y_train)
t1 = time()
svc_score = svc.score(X_test, y_test)
print(f"Classical SVC w/ Quantum Kernel classification test score: {svc_score}")
print(f"Time taken: {t1-t0}")

Z = svc.predict(X_test)

print("Predictions:", Z)

# Classical SVC with RBF Kernel
svc2 = SVC(kernel='rbf')
t0 = time()
svc2.fit(X_train, y_train)
svc2_score = svc2.score(X_test, y_test)
t1 = time()


# Create a grid for visualization
xx, yy = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

# Predict the class for each point in the grid
Z = svc2.predict(X_test)

print("Predictions:", Z)
# Z = Z.reshape(xx.shape)

# # Plot the decision boundary
# plt.figure(figsize=(8, 6))
# plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', label='Train')
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')
# plt.title("SVM Decision Boundary")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.legend()
# plt.grid(True)
# plt.show()


print(f"Classical Kernel SVC classification test score: {svc2_score}")
print(f"Time taken: {t1-t0}")


