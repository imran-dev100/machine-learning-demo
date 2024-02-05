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

print(f"Iterating each training example")
for i in range(m):
    x_i = x_train[i]
    y_i = y_train[i]
    print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


def plot_graph(x_train, y_train, label):
    # plotting the data points of training set in graph
    plt.scatter(x_train, y_train, marker = 'x', c = 'r', label = label)

    # setting the title of graph
    plt.title("Housing prices")

    # setting the label of x-axis
    plt.xlabel("Size (in 1000 sq. ft.)")

    # setting the label of y-axis
    plt.ylabel("Price (in 1000s of USD)")

    # showing the plot in new tab
    plt.show()

plot_graph(x_train, y_train, '')


## MODEL FUNCTION ##

print(f"Defining the model function with parameters w and b with formula f = wx +b")

w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")

# Calculating or computing model function for given training model

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    # Length or size of the training model
    m = x.shape[0]

    # initialing an array with zeros on the same size to compute each input's model function
    f_wb = np.zeros(m)
    for i in range(m):
        # f = wx + b
        f_wb[i] = w * x[i] + b

    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)

# plotting a prediction graph
plt.plot(x_train, tmp_f_wb, c='b', label='Our prediction')
    
plot_graph(x_train, y_train, 'Actual Values')

print(f"Predicting the price of a house with 1200 sqft. Since the units of  𝑥 are in 1000's of sqft, 𝑥 is 1.2.")
w = 200
b = 100
x_i = 1.2
print(f"w: {w}")
print(f"b: {b}")

cost_1200sqft = w * x_i + b
print(f"${cost_1200sqft:.0f} thousand dollars")

