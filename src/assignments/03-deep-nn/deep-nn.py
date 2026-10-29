import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.grad_check import grad_check, flatten_deep_params, unflatten_deep_params

def reduction_validate(X_train, Y_train, X_test, Y_test, layer_dims = [2,1]):
    parameters, _ = L_layer_model(X_train, Y_train, layer_dims, X_test, Y_test, learning_rate=0.5, iterations=30000)

    print("W:", parameters["W1"])
    print("b:", parameters["b1"].item())

def check_gradients(X, Y):
    """
    Args:
        X (n_input, m), Y (1, m): data

    Returns:
        error (float): relative error
    """
    rng = np.random.default_rng(42)
    indices = rng.permutation(X.shape[1])[:20]
    X_check = X[:, indices]
    Y_check = Y[:, indices]

    parameters = initialize_parameters_deep([2, 4, 3, 1])

    # 검증할 때만 편향을 작은 난수로 변경
    for l in range(1, len(parameters) // 2 + 1):
        shape = parameters[f"b{l}"].shape
        parameters[f"b{l}"] = rng.normal(size=shape) * 0.1

    _, ZL, caches = L_model_forward(X_check, parameters)
    grads = L_model_backward(ZL, Y_check, caches)

    parameter_grads = {}
    for key in parameters:
        parameter_grads[key] = grads["d" + key]

    theta = flatten_deep_params(parameters)
    analytic_grad = flatten_deep_params(parameter_grads)

    def cost_fn(theta):
        restored = unflatten_deep_params(theta, parameters)
        _, ZL, _ = L_model_forward(X_check, restored)
        return compute_cost(ZL, Y_check)

    error = grad_check(cost_fn, theta, analytic_grad)
    print(f"Gradient check error: {error:.3e}")

    return error

def make_spiral(n_per_class=200, noise=0.12, seed=0, turns=1.6):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in (0, 1):
        r = np.linspace(0.08, 1.0, n_per_class)
        t = np.linspace(0, turns*2*np.pi, n_per_class) + k*np.pi
        xx = r*np.sin(t) + rng.normal(scale=noise, size=n_per_class)*r
        yy = r*np.cos(t) + rng.normal(scale=noise, size=n_per_class)*r
        X.append(np.stack([xx, yy], 1)); y.append(np.full(n_per_class, k))
    return np.concatenate(X).T, np.concatenate(y).reshape(1, -1)

def plot_data(X, Y):
    """
    Args:
        X (2, m), Y (1, m): data

    Returns:
        None
    """
    plt.figure()
    plt.scatter(X[0], X[1], c=Y.squeeze(), cmap="coolwarm")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    plt.close()

def plot_decision_boundary(X, Y, parameters, name):
    """
    Args:
        X (2, m), Y (1, m): data
        parameters (dict): trained W, b
        name (str): image name

    Returns:
        None
    """
    xx, yy = np.meshgrid(
        np.linspace(X[0].min(), X[0].max(), 100),
        np.linspace(X[1].min(), X[1].max(), 100)
    )

    grid = np.array([xx.ravel(), yy.ravel()])
    predictions = predict(grid, parameters).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, predictions, levels=[-0.5, 0.5, 1.5], cmap="coolwarm", alpha=0.3)
    plt.scatter(X[0], X[1], c=Y.squeeze(),cmap="coolwarm")
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(name)
    plt.savefig(f"images/{name}.png")
    plt.close()

def check_initial_data(X_train, Y_train):
    plt.figure()
    plt.scatter(X_train[0, :], X_train[1, :], c=Y_train.squeeze(), cmap='coolwarm')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def initialize_parameters_deep(layer_dims, seed=42):
    """
    Args:
        layer_dims (list): list representing # of neurons in each layer

    Returns:
        parameters(dict): W1...Wl b1...bl dictionary
    """
    np.random.seed(seed)
    L = len(layer_dims)
    parameters = {}


    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
        # parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01 # For experiment B
        # parameters["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1])) # For reduction-validate
        

    return parameters

def linear_forward(A_prev, W, b):
    """
    n_cur = # of neurons in layer l 
    n_prev = # of neurons in layer l-1
    Args:
        A_prev (n_prev, m): activation from previous layer
        W (n_cur, n_prev): weight matrix
        b (n_cur, 1): bias

    Returns:
        Z, cache: Z = X @ A_prev + b, cache = tuple for calculating backpropagation
    """
    Z = np.dot(W, A_prev) + b

    cache = (A_prev, W, b)

    return Z, cache

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

def relu(Z):
    return np.maximum(0, Z)

def linear_activation_forward(A_prev, W, b, activation):
    """
    Args:
        A_prev (n_prev, m): activation from previous layer
        W (n_cur, n_prev): weight matrix
        b (n_cur, 1): bias
        activation (string): 'relu' or 'sigmoid'

    Returns:
        A(n_cur, m), cache: A = activation_function(Z), cache = (linear_cache, activation_cache)
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A = relu(Z)
    elif activation == 'sigmoid':
        A = sigmoid(Z)
    else:
        raise ValueError("Activation must be 'relu' or 'sigmoid'")

    cache = (linear_cache, Z)
    return A, cache

def L_model_forward(X, parameters):
    """
    Args:
        X (input size, m(# of training data))
        parameters (dict): W1...Wl and b1...bl dictionary

    Returns:
        A_last(1,m), Z_last(1,m), caches: Last activation value & Last logit value, cache(linear_cache, activation_cache) list
    """
    caches = []
    L = len(parameters) // 2
    A = X
    for l in range(1, L):
        A, cache = linear_activation_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    ZL = caches[-1][1]

    return AL, ZL, caches

def compute_cost(ZL, Y):
    """
    Args:
        ZL (n_cur, m): logit from last layer
        Y (1, m) : label list

    Returns:
        cost : cross-entropy loss
    """
    m = Y.shape[1]

    cost = 1/m * np.sum(np.logaddexp(0, (1-2*Y) * ZL))

    return cost

def linear_backward(dZ, cache):
    """
    Args:
        dZ (n_cur, m): derivative of Z
        cache (tuple): (A_prev, W, b)

    Returns:
        dA_prev, dW, db: derivatives of each A_prev, W, b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Args:
        dA (n_cur, m): activation gradient
        cache (tuple): (linear_cache, Z)
        activation (string): 'relu' or 'sigmoid'

    Returns:
        dA_prev, dW, db: gradients of A_prev, W, b
    """
    linear_cache, Z = cache
    
    if activation == 'relu':
        dZ = dA * (Z > 0)
    elif activation == 'sigmoid':        
        A = sigmoid(Z)
        dZ = dA * A * (1-A)
    else :
        raise ValueError("Activation must be 'relu' or 'sigmoid'")

    return linear_backward(dZ, linear_cache)

def L_model_backward(ZL, Y, caches):
    """
    Args:
        ZL (n_cur, m): logit from last layer
        Y (1, m): label list
        caches (tuple list): (linear_cache, Z)

    Returns:
        grads(dict): dA, dW, db
    """
    grads = {}
    L = len(caches)

    AL = sigmoid(ZL) # A_last
    dZ = (AL - Y)
    linear_cache, _ = caches[-1]
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    grads["dA" + str(L-1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    for l in range(L-1, 0, -1):
        current_cache = caches[l-1]
        dA = grads["dA" + str(l)]

        dA_prev, dW, db = linear_activation_backward(dA, current_cache, 'relu')

        grads["dA" + str(l-1)] = dA_prev
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Args:
        parameters (dict): W, b
        grads (dict): dW, db
        learning_rate (float): step size

    Returns:
        parameters (dict): updated in place
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters

def predict(X, parameters):
    """
    Args:
        X (n_input, m): input data
        parameters (dict): W, b

    Returns:
        predictions (1, m): predicted labels
    """
    AL, _, _ = L_model_forward(X, parameters)
    return (AL >= 0.5).astype(int)

def L_layer_model(X_train, Y_train, layer_dims, X_test, Y_test, learning_rate=0.5, iterations=30000, seed=42):
    """
    Args:
        X_train (n_input, m_train), Y (1, m_train): train data
        X_test (n_input, m_test), Y_test (1, m_test): test data
        layer_dims (list): layer sizes
        learning_rate (float): step size
        iterations (int): number of updates

    Returns:
        parameters (dict), history (dict)
    """
    history = {
        "iterations": [],
        "train_cost": [],
        "test_cost": [],
    }
    parameters = initialize_parameters_deep(layer_dims, seed)

    for i in range(iterations+1):
        _, ZL_train, caches = L_model_forward(X_train, parameters)

        if i % 100 == 0 or i == iterations:
            _, ZL_test, _ = L_model_forward(X_test, parameters)

            train_cost = compute_cost(ZL_train, Y_train)
            test_cost = compute_cost(ZL_test, Y_test)

            history["iterations"].append(i)
            history["train_cost"].append(train_cost)
            history["test_cost"].append(test_cost)

            if i == iterations : 
                break

        grads = L_model_backward(ZL_train, Y_train, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
    
    train_accuracy = np.mean(predict(X_train, parameters) == Y_train)
    test_accuracy = np.mean(predict(X_test, parameters) == Y_test)

    print(f"Experiment : {layer_dims}")
    print(f"Prediction of train datasets : {train_accuracy:.2%}")
    print(f"Prediction of test datasets : {test_accuracy:.2%}")
    
    return parameters, history

def count_params(dims):
    return sum(dims[l]*dims[l-1] + dims[l] for l in range(1, len(dims)))

def experiment_b(mode="small"):
    """mode (str): 'small' or 'he'"""
    X_train, Y_train = make_spiral(200, 0.12, 0)
    X_test, Y_test = make_spiral(500, 0.12, 99)

    experiments = {
        "B1": [2, 16, 1],
        "B2": [2, 16, 16, 1],
        "B3": [2, 32, 16, 8, 1],
    }
    results = {}

    for name, layer_dims in experiments.items():
        print(f"\n실험: {name}, 초기화: {mode}")

        parameters, history = L_layer_model(
            X_train, Y_train, layer_dims, X_test, Y_test,
            learning_rate=0.5,
            iterations=100000,
            seed=1
        )

        accuracy = np.mean(predict(X_test, parameters) == Y_test)

        if mode == "small":
            np.savez(
                f"images/{name}_small_seed42.npz",
                **history,
                accuracy=accuracy
            )
            continue

        with np.load(f"images/{name}_small_seed42.npz") as saved:
            small = dict(saved)

        results[name] = (accuracy, small["accuracy"].item())

        if name in ("B2", "B3"):
            plt.figure()

            for data, label, color in (
                (history, "He", "blue"),
                (small, "0.01", "orange"),
            ):
                plt.plot(
                    data["iterations"], data["train_cost"],
                    label=f"{label} Train", color=color
                )
                plt.plot(
                    data["iterations"], data["test_cost"],
                    label=f"{label} Test", color=color,
                    linestyle="--"
                )

            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.title(name)
            plt.legend(loc="upper right")
            plt.savefig(f"images/{name}_comparison_seed42.png")
            plt.close()

    if mode == "he":
        print("\n| 실험 | He 초기화 | * 0.01 |")
        print("| --- | --- | --- |")

        for name, (he_accuracy, small_accuracy) in results.items():
            print(f"| {name} | {he_accuracy:.2%} | {small_accuracy:.2%} |")

def main():
    X_train, Y_train = make_spiral(200, 0.12, 0)    # (2, 400)
    X_test,  Y_test  = make_spiral(500, 0.12, 99)   # (2, 1000)
    
    # check_initial_data(X_train, Y_train)
    error = check_gradients(X_train, Y_train)
    if not error < 1e-6:
        print("fail to verify gradient")
        return

    # reduction_validate(X_train, Y_train, X_test, Y_test)

    name = "A5"
    layer_dims = [2, 256, 1]

    parameters, history = L_layer_model(X_train, Y_train, layer_dims, X_test, Y_test, iterations=30000)

    plot_decision_boundary(X_test, Y_test, parameters, name)
    
    idx = np.argmin(history["test_cost"])
    best_iteration = history["iterations"][idx]

    print(f"Best iteration and idx = {best_iteration} {idx}")
    print("파라미터 수:", count_params(layer_dims))

    plt.figure()
    plt.plot(history["iterations"], history["train_cost"], label="Train")
    plt.plot(history["iterations"], history["test_cost"], label="Test")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    # plt.savefig(f"images/{name}_learning_curve.png")
    plt.show()

if __name__ == "__main__":
    # main()
    experiment_b('he')