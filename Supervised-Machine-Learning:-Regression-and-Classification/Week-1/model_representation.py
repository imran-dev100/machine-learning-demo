import numpy as np
import matplotlib.pyplot as plt
# importing dependencies
plt.style.use('/Users/imran/jupyter-workspace/machine-learning-demo/dependencies/deeplearning.mplstyle')

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
# creating array of numpy for training examples
x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])


print(f"x_train = {x_train}")
print(f"y_train = {y_train}")


# identifying length of the training example array
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

# iterate each training example individually
for i in range(m):
    x_i = x_train[i]
    y_i = y_train[i]

    print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# plotting the data points of training set in graph
plt.scatter(x_train, y_train, marker = 'x', c = 'r')

# setting the title of graph
plt.title("Housing prices")

# setting the label of x-axis
plt.xlabel("Size (in 1000 sq. ft.)")

# setting the label of y-axis
plt.ylabel("Price (in 1000s of USD)")

# showing the plot in new tab
plt.show()






