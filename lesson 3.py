import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
    return x * x

def derivative(x):
    return 2*x

def gradient_descent(starting_point, learning_rate, epochs):
    x = starting_point
    path = [x]
    for i in range(epochs):
        grad = derivative(x)
        x -= learning_rate * grad
        path.append(x)
    return path

starting_point = 10
learning_rate = 0.2
epochs = 20
path = gradient_descent(starting_point, learning_rate, epochs)
x_values = np.linspace(-10,10,400)
y_values = cost_function(x_values)
plt.figure(figsize = (10, 6))
plt.plot(x_values, y_values, label = "cost function")
plt.scatter(path, cost_function(np.array(path)), color = "red", s = 100)
plt.xlabel("x")
plt.ylabel("cost")
plt.legend()
plt.grid()
plt.show()