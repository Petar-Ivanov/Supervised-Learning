# for mathematics
import numpy as np
# for plotting the data graphically
import matplotlib.pyplot as plt 


def main():
    '''# choosing an ML style for the plots
    plt.style.use('./deeplearning.mplstyle')'''

    # Input variables
    x_train = np.array([1.0, 2.0, 1.5])
    # Output variables
    y_train = np.array([300.0, 500.0, 400.0])

    # Training the weight and the bias
    w, b = gradient_descent(x_train, y_train, 0, 0, 0.01, 1500, compute_gradient)

    # Calculate predictions (a list of predictions for each x)
    tmp_f_wb = compute_model_output(x_train, w, b,)

    # Prints results
    print("Predictions: ")
    for i in range(len(x_train)):
        print(f"x{i}: {x_train[i]} / ŷ{i}: {tmp_f_wb[i]} / y{i}: {y_train[i]}")

    # Calculates a single example
    print(f"x: 1.6 / ŷ: {predict(1.6, w, b)}")
    
    # Prints the weight and the bias
    print(f"w: {round(w, 2)} / b: {round(b, 2)}")
    # Prints the total cost
    print(f"Total cost: {compute_total_cost(tmp_f_wb, y_train)}")
    
    # Plot our model prediction
    plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
    # Plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
    # Set the title
    plt.title("Housing Prices")
    # Set the y-axis label
    plt.ylabel('Price (in 1000s of dollars)')
    # Set the x-axis label
    plt.xlabel('Size (1000 sqft)')
    plt.legend()
    plt.show()

# calculates a particular example
def predict(x, w, b):
    return round(w * x + b, 2)

# calculates all the y's 
def compute_model_output(x, w, b):
    # the number of training examples
    m = len(x)
    # an empty array
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        f_wb[i] = round(f_wb[i], 2)
    return f_wb

def compute_total_cost(y_estimate, y_actual):
    m = len(y_estimate)
    cost_sum = 0
    for i in range(m):
        cost_sum += (y_estimate[i]-y_actual[i])**2
    return round(cost_sum/(2*m), 2)

# computes the gradient part of the gradient descent formula
def compute_gradient(x, y, w, b):

    m = len(x)
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb-y[i]) * x[i]
        dj_db += (f_wb-y[i])
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, iterations, gradient):
    w = w_in
    b = b_in
    for _ in range(iterations):
        dj_dw, dj_db = gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    
    return w, b

if __name__ == "__main__":
    main()