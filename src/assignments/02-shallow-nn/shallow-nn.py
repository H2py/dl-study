import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from utils.grad_check import grad_check, flatten_params, unflatten_params

sys.path.insert(0, "../01-logistic-regression")
import logistic_regression as logistic
import utils.grad_check

def flatten_nn_params(parameters):
    """
    Args:
        parameters (dict): W1, b1, W2, b2

    Returns:
        theta (np.ndarray): All parameters flattened (p,)
    """
    return np.concatenate([
        parameters["W1"].ravel(),
        parameters["b1"].ravel(),
        parameters["W2"].ravel(),
        parameters["b2"].ravel(),
    ])

def unflatten_nn_params(theta, shapes):
    """
    Args:
        theta (np.ndarray): Flattened parameters (p,)
        shapes (dict): Original shapes of W1, b1, W2, b2

    Returns:
        parameters (dict): W1, b1, W2, b2 with original shapes
    """
    parameters = {}
    start = 0

    for key in ["W1", "b1", "W2", "b2"]:
        size = int(np.prod(shapes[key]))
        parameters[key] = theta[start:start + size].reshape(shapes[key])
        start += size

    return parameters

def check_nn_gradients(X, Y, n_h):
    """
    Args:
        X (np.ndarray): input data (n_x, m)
        Y (np.ndarray): labels (1, m)
        n_h (int): The number of hidden neurons

    Returns:
        difference (float): Relative error
    """
    n_x, _, n_y = layer_sizes(X, Y, n_h)
    parameters = initialize_parameters(n_x, n_h, n_y)

    shapes = {
        key: value.shape
        for key, value in parameters.items()
    }

    theta = flatten_nn_params(parameters)

    def cost_fn(theta):
        check_parameters = unflatten_nn_params(theta, shapes)
        _, cache = forward_propagation(X, check_parameters)
        return compute_cost(cache["Z2"], Y)

    _, cache = forward_propagation(X, parameters)
    grads = backward_propagation(parameters, cache, X, Y)

    analytic_grad = np.concatenate([
        grads["dW1"].ravel(),
        grads["db1"].ravel(),
        grads["dW2"].ravel(),
        grads["db2"].ravel(),
    ])

    difference = grad_check(cost_fn, theta, analytic_grad)

    print(f"기울기 검증 상대 오차: {difference:.10e}")

    return difference

def sigmoid(z):
    """sigmoid function
    Args:
        z (np.ndarray): (1, m)

    Returns:
        np.ndarray : (1, m)
    """
    z = np.asarray(z)
    result = np.zeros_like(z, dtype=np.float64)
    
    mask = z > 0
    result[mask] = 1 / (1 + np.exp(-z[mask]))
    result[~mask] = np.exp(z[~mask]) / (1 + np.exp(z[~mask]))
    
    return result

def make_data(n, noise, seed):
    X, y = make_moons(
        n_samples=n,
        noise=noise,
        random_state=seed,
    )
    return X.T, y.reshape(1, -1)

def plot_decision_boundary(predict_fn, X, Y, title=""):
    x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
    y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01),
    )

    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z = predict_fn(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.4)
    plt.scatter(X[0], X[1], c=Y.ravel(), s=12, cmap=plt.cm.Spectral)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)

def plot_cost(iterations, train_cost_list, test_cost_list, title):
    plt.figure(figsize=(10, 4))

    plt.plot(iterations, train_cost_list, label="Train cost")
    plt.plot(iterations, test_cost_list, label="Test cost")

    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./images/compare_two_cost_{title}.png")
    plt.close()

def make_initial_plot(X_train, X_test, Y_train, Y_test):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, X, Y, title in [
        (axes[0], X_train, Y_train, "Train data"),
        (axes[1], X_test, Y_test, "Test data"),
    ]:
        ax.scatter(X[0], X[1], c=Y.ravel(), s=12)
        ax.set(xlabel="x1", ylabel="x2", title=title)

    fig.tight_layout()
    fig.savefig("./images/train_test_data.png")

    # 기존 로지스틱 회귀로 학습
    result = logistic.model(
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_iterations=2000,
        learning_rate=0.1,
    )

    w = result["params"]["w"]
    b = result["params"]["b"]

    print(f"Training accuracy: {result['train_accuracy']:.2f}%")
    print(f"Test accuracy: {result['test_accuracy']:.2f}%")

    # 시험 데이터 위에 결정 경계 표시
    plt.figure(figsize=(7, 5))
    plot_decision_boundary(
        lambda X: logistic.predict(w, b, X),
        X_test,
        Y_test,
        title="Logistic regression",
    )
    plt.tight_layout()
    plt.savefig("./images/logistic_result.png")

def layer_sizes(X,Y,n_h):
    """
    Args:
        X (np.ndarray): (2, n)
        Y (np.ndarray): (1, n)
        n_h (int): The number of hidden neurons (hyperparameter)
    Returns:
        tuple: 
            n_x (int): Size of the input layer
            n_h (int): Size of the hidden layer
            n_y (int): Size of the output layer
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]

    return n_x, n_h, n_y

def initialize_parameters(n_x, n_h, n_y):
    """
    Args:
        n_x (int): Size of the input layer
        n_h (int): Size of the hidden layer
        n_y (int): Size of the output layer

    Returns:
        parameters : dict
            W1 : weight matrix (n_h, n_x)
            b1 : bias (n_h, 1)
            W2 : weight matrix (n_y, n_h)
            b2 : bias (n_y, 1)
    """
    W1 = np.random.randn(n_h, n_x) * 0.5
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.5
    b2 = np.zeros((n_y,1))

    parameters = {"W1":W1, "b1": b1, "W2":W2, "b2":b2}

    return parameters

def forward_propagation(X, parameters):
    """
    Args:
        X (np.ndarray): (2, n)
        parameters (dict): 
            W1 : weight matrix (n_h, n_x)
            b1 : bias (n_h, 1)
            W2 : weight matrix (n_y, n_h)
            b2 : bias (n_y, 1)

    Returns:
        A2: 0~1
        cache (dict):
            Z1 : Weighted sum (n_h, m)
            A1 : Activation value from tanh (n_h, m)
            Z2 : Weighted sum (n_y, m)
            A2 : Activation value from sigmoid (n_y, m)
    """
    W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    # --------------------실험 C------------------------#
    # A1 = np.tanh(Z1)
    A1 = Z1
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2}

    return A2, cache

def compute_cost(Z2, Y):
    """
    Args:
        Z2 : Weighted sum (n_y, 1)
        Y (np.ndarray): (1, n)

    Returns:
        scalar : cost
    """
    m = Y.shape[1]
    cost = np.sum(np.logaddexp(0, (1-2*Y)*Z2)) / m

    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Args:
        parameters (dictionary): W1, b1, W2, b2
        cache (dictionary): Z1, A1, Z2, A2
        X (np.ndarray): (2, n)
        Y (np.ndarray): (1, n)

    Returns:
        grads: dictionary (dW1, db1, dW2, db2)
    """
    m = X.shape[1]
    W2 = parameters["W2"]

    A1, A2 = cache["A1"], cache["A2"]

    dZ2 = A2 - Y # (1, m)
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    # --------------------실험 C------------------------#
    # dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    
    dZ1 = np.dot(W2.T, dZ2)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def update_parameters(parameters, grads, learning_rate = 0.05):
    """
    Arguments:
    parameters -- dictionary(W1, b1, W2, b2)
    grads -- dictionary(dW1, db1, dW2, db2)
    
    Returns:
    parameters(updated) -- dictionary(W1, b1, W2, b2)
    """
    W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
    
    dW1, db1, dW2, db2 = grads["dW1"], grads["db1"], grads["dW2"], grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2    
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

def nn_model (X_train, Y_train, n_h, X_test, Y_test, num_iterations = 100, learning_rate = 0.5005):
    """
    Args:
        X_train (np.ndarray): Training data (n_x, m)
        Y_train (np.ndarray): Training labels (1, m)
        n_h (int): The number of hidden neurons (hyperparameter)
        X_test (np.ndarray): Test data (n_x, m_test)
        Y_test (np.ndarray): Test labels (1, m_test)
        num_iterations (int): The number of training iterations
        learning_rate (float): Learning rate for updating parameters

    Returns:
        parameters : dict
            W1 : weight matrix (n_h, n_x)
            b1 : bias (n_h, 1)
            W2 : weight matrix (n_y, n_h)
            b2 : bias (n_y, 1)
        history : dict
            iteration : the number of iterations(list)
            train_cost : list
            test_cost : list
            train_accuracy : the number of train accuracy(list) 
            test_accuracy : the number of test accuracy(list) 
    """
    n_x, n_y = layer_sizes(X_train, Y_train, n_h)[0], layer_sizes(X_train, Y_train, n_h)[2]
    history = { "iteration": [], "train_cost": [], "test_cost": [], "train_accuracy": [], "test_accuracy": [] }
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2_train, train_cache = forward_propagation(X_train, parameters)

        train_cost = compute_cost(train_cache["Z2"], Y_train)
        grads = backward_propagation(parameters, train_cache, X_train, Y_train)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            A2_test, test_cache = forward_propagation(X_test, parameters)
            test_cost = compute_cost(test_cache["Z2"], Y_test)

            train_predictions = (A2_train > 0.5).astype(int)
            test_predictions = (A2_test > 0.5).astype(int)

            train_accuracy = np.mean(train_predictions == Y_train) * 100
            test_accuracy = np.mean(test_predictions == Y_test) * 100

            history["iteration"].append(i)
            history["train_cost"].append(train_cost)
            history["test_cost"].append(test_cost)
            history["train_accuracy"].append(train_accuracy)
            history["test_accuracy"].append(test_accuracy)

    return parameters, history

def predict(parameters, X):
    """
    Args:
        parameters (dict): W1, b1, W2, b2
        X (np.ndarray): (n_x, m)

    Returns:
        predictions (np.ndarray): Predicted values (1, m)
    """
    
    A2, _ =forward_propagation(X, parameters)
    predictions = (A2 > 0.5).astype(int)
        
    return predictions

def main():
    # 데이터 생성
    X_train, Y_train = make_data(400, 0.2, 0)
    X_test, Y_test = make_data(1000, 0.2, 99)

    X_small, Y_small = make_data(40, 0.3, 7)
    X_test3, Y_test3 = make_data(1000, 0.3, 99)

    experiments = [
        # ("A-1", 1,  X_train, Y_train, X_test,  Y_test),
        # ("A-2", 4,  X_train, Y_train, X_test,  Y_test),
        # ("A-3", 50, X_train, Y_train, X_test,  Y_test),
        # ("A-4", 50, X_small, Y_small, X_test3, Y_test3),
        # ("B-1", 50, X_train, Y_train, X_test, Y_test)
        ("C-1", 4, X_train, Y_train, X_test, Y_test),
        ("C-2", 50, X_train, Y_train, X_test, Y_test)
    ]

    # difference = check_nn_gradients(X_train[:, :10], Y_train[:, :10], n_h=4)
    # assert difference < 1e-7

    # make_initial_plot(X_train, X_test, Y_train, Y_test)

    for name, n_h, X_tr, Y_tr, X_te, Y_te in experiments:
        np.random.seed(42)    
        
        parameters, history = nn_model(
            X_train=X_tr,
            Y_train=Y_tr,
            n_h=n_h,
            X_test=X_te,
            Y_test=Y_te,
            num_iterations=20000,
            learning_rate=0.05,
        )
        train_predictions = predict(parameters, X_tr)
        test_predictions = predict(parameters, X_te)
        
        train_accuracy = np.mean(train_predictions == Y_tr) * 100
        test_accuracy = np.mean(test_predictions == Y_te) * 100

        W1 = parameters["W1"]
        
        # min_index = np.argmin(history["test_cost"])
        # min_iteration = history["iteration"][min_index]

        print(f"{name}")
        print(f"Training accuracy: {train_accuracy:.2f}%")
        print(f"Test accuracy: {test_accuracy:.2f}%")        
        
        # --------------------실험 A------------------------#
        # print(f"Iteration at minimum test cost: {min_iteration}")
        
        # plot_cost(history["iteration"], history["train_cost"], history["test_cost"], title=name)
        plt.figure(figsize=(7, 5))
        
        plot_decision_boundary(
            lambda X: predict(parameters, X),
            X_te,
            Y_te,
            title=f"{name} accuracy : {test_accuracy}",
        )
        plt.tight_layout()
        plt.savefig(f"./images/{name}_result.png")
        plt.close()

        # --------------------실험 B------------------------#
        # print(np.unique(np.round(W1, 8), axis=0).shape[0], "개의 서로 다른 은닉 유닛")
        # for i in range(W1.shape[0]):
        #     print(f"{i}번째 행 : {W1[i,:]}")
    

if __name__ == "__main__":
    main()

