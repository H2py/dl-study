import numpy as np, sklearn.datasets
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.grad_check import grad_check, flatten_deep_params, unflatten_deep_params

def experiment_1(layer_dims):
    np.random.seed(1)
    X_train, Y_train = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    X_test,  Y_test  = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    X_train, X_test = X_train.T, X_test.T
    Y_train, Y_test = Y_train.reshape(1, -1), Y_test.reshape(1, -1)

    for method in ['zero', 'random', 'he']:
        print(f"초기화 방법 : {method}\n")
        parameters, history = L_layer_model(X_train, Y_train, layer_dims, X_test, Y_test, learning_rate=0.01, iterations=15000, seed=3, initialize=method)
    
        first_cost = history["train_cost"][0]
        final_cost = history["train_cost"][-1]

        print(f"0회 시점 비용 : {first_cost:.4f}")
        print(f"최종 학습 비용 : {final_cost:.4f}")

        plot_decision_boundary(X_train, Y_train, parameters, method)

def make_field(n, seed, flip=0.10, amp=0.25, freq=4.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-0.8, 0.8, size=(n, 2))
    y = (X[:, 1] > amp * np.sin(freq * X[:, 0]) + 0.05).astype(int)
    y = np.where(rng.random(n) < flip, 1 - y, y)
    return X.T, y.reshape(1, -1)

def experiment_2(layer_dims):
    train_X, train_Y = make_field(150, 0)
    dev_X, dev_Y = make_field(1000, 1)

    l2_models, dropout_models = {}, {}
    histories = {}

    print("\nlambda | train | dev | W 크기")
    best_dev = -1

    for lambd in [0, 0.1, 0.7, 3, 30]:
        parameters, history = L_layer_model(train_X, train_Y, layer_dims, dev_X, dev_Y,learning_rate=0.3, iterations=30000, seed=3,normalize="L2", lambd=lambd)
        l2_models[lambd] = parameters
        histories[f"L2_{lambd}"] = history

        train_acc = np.mean(predict(train_X, parameters) == train_Y)
        dev_acc = np.mean(predict(dev_X, parameters) == dev_Y)
        norm = np.sqrt(sum(
            np.sum(parameters[f"W{l}"] ** 2)
            for l in range(1, len(layer_dims))
        ))
        print(f"{lambd} | {train_acc:.2%} | {dev_acc:.2%} | {norm:.4f}")

        if dev_acc > best_dev:
            best_dev, best_lambda = dev_acc, lambd

    print("keep_prob | train | dev \n")
    best_dev = -1

    for keep_prob in [1.0, 0.86, 0.5]:
        if keep_prob == 1.0:
            parameters, history = l2_models[0], histories["L2_0"]
        else:
            parameters, history = L_layer_model(train_X, train_Y, layer_dims, dev_X, dev_Y,learning_rate=0.3, iterations=30000, seed=3,normalize="dropout", keep_prob=keep_prob)
        dropout_models[keep_prob] = parameters
        histories[f"dropout_{keep_prob}"] = history

        train_acc = np.mean(predict(train_X, parameters) == train_Y)
        dev_acc = np.mean(predict(dev_X, parameters) == dev_Y)
        print(f"{keep_prob} | {train_acc:.2%} | {dev_acc:.2%}")

        if dev_acc > best_dev:
            best_dev, best_keep_prob = dev_acc, keep_prob

    models = [("None", l2_models[0]), (f"L2: {best_lambda}", l2_models[best_lambda]), (f"Dropout: {best_keep_prob}", dropout_models[best_keep_prob])]

    test_X, test_Y = make_field(1000, 2)

    print("모델 | train | dev | test \n")
    for name, parameters in models:
        train_acc = np.mean(predict(train_X, parameters) == train_Y)
        dev_acc = np.mean(predict(dev_X, parameters) == dev_Y)
        test_acc = np.mean(predict(test_X, parameters) == test_Y)
        print(f"{name} | {train_acc:.2%} | {dev_acc:.2%} | {test_acc:.2%}")

    x = np.linspace(-0.8, 0.8, 200)
    xx, yy = np.meshgrid(x, x)
    grid = np.array([xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, parameters) in zip(axes, models):
        predictions = predict(grid, parameters).reshape(xx.shape)
        ax.contourf(xx, yy, predictions,levels=[-0.5, 0.5, 1.5], cmap="coolwarm", alpha=0.3)
        ax.scatter(train_X[0], train_X[1],c=train_Y.ravel(), cmap="coolwarm", s=15)
        ax.set(title=name, xlabel="Feature 1", ylabel="Feature 2")

    fig.tight_layout()
    fig.savefig("images/4-2_boundaries.png")
    plt.close(fig)

    return models, histories

def experiment_3(layer_dims):
    train_X, train_Y = make_field(150, 0)

    A_train = train_X.copy()
    A_train[0] *= 100

    mu = np.mean(A_train, axis=1, keepdims=True)
    sigma = np.std(A_train, axis=1, keepdims=True)
    normalized_X = (A_train - mu) / sigma

    settings = [
        ("Raw, lr=0.3", A_train, 0.3),
        ("Raw, lr=0.003", A_train, 0.003),
        ("Normalized, lr=0.3", normalized_X, 0.3),
    ]

    histories = {}
    plt.figure()
    print("설정 | 0회 비용 | 500회 비용 | 최종 비용 | 학습 정확도\n")

    for name, X, learning_rate in settings:
        parameters, history = L_layer_model(
            X, train_Y, layer_dims, X, train_Y,
            learning_rate=learning_rate,
            iterations=10000, seed=3,
            initialize="he", normalize=None,
        )
        histories[name] = history

        first_cost = history["train_cost"][0]
        index_500 = history["iterations"].index(500)
        cost_500 = history["train_cost"][index_500]
        final_cost = history["train_cost"][-1]
        accuracy = np.mean(predict(X, parameters) == train_Y)

        print(
            f"{name} | {first_cost:.4f} | {cost_500:.4f} | "
            f"{final_cost:.4f} | {accuracy:.2%}"
        )

        plt.plot(history["iterations"], history["train_cost"],label=name)

    plt.xlabel("Iterations")
    plt.ylabel("Training cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/4-3_cost.png")
    plt.close()

    return histories

def experiment_4():
    # Used for (가) experiment
    x = 2.0
    theta_1d = np.array([3.0])

    def cost_1d(t):
        return t[0] * x
    epsilon = 1e-7
    numerical = (cost_1d(theta_1d + epsilon) - cost_1d(theta_1d - epsilon)) / (2 * epsilon)
    analytic = np.array([x])
    error = grad_check(cost_1d, theta_1d, analytic, epsilon)

    print(f"수치 미분: {numerical:.10f}")
    print(f"해석 미분: {analytic[0]:.10f}")
    print(f"상대 오차: {error:.3e}")

    np.random.seed(1)
    X = np.random.randn(4, 20)
    Y = np.random.randint(0, 2, size=(1, 20))

    layer_dims = [4, 5, 3, 1]
    parameters = initialize_parameters_deep(layer_dims, seed=3, initialize="he")

    for l in range(1, len(layer_dims)):
        parameters["b" + str(l)] = np.random.randn(layer_dims[l], 1) * 0.1

    theta = flatten_deep_params(parameters)

    def check(cost_fn, grads):
        param_grads = {}
        for name in parameters:
            param_grads[name] = grads["d" + name]

        analytic_grad = flatten_deep_params(param_grads)
        return grad_check(cost_fn, theta, analytic_grad, epsilon)

    def cost_fn(t):
        p = unflatten_deep_params(t, parameters)
        _, ZL, _ = L_model_forward(X, p)
        return compute_cost(ZL, Y)

    # 4-4 (나-a)
    _, ZL, caches = L_model_forward(X, parameters)
    grads = L_model_backward(ZL, Y, caches)
    normal_error = check(cost_fn, grads)

    # 4-4 (나-b)
    grads["dW2"] *= 2
    bug_error = check(cost_fn, grads)

    # 4-4 (나-c)
    keep_prob = 0.86    
    def dropout_forward(p):
        np.random.seed(7)
        return L_model_forward_with_dropout(X, p, keep_prob)

    def dropout_cost_fn(t):
        p = unflatten_deep_params(t, parameters)
        _, ZL, _, _ = dropout_forward(p)
        return compute_cost(ZL, Y)

    _, ZL, caches, masks = dropout_forward(parameters)
    grads = L_model_backward_with_dropout(ZL, Y, caches, keep_prob, masks)
    dropout_error = check(dropout_cost_fn, grads)

    print("n조건 | 상대 오차 \n")
    print(f"정상 | {normal_error:.3e}")
    print(f"dW2 × 2 | {bug_error:.3e}")
    print(f"dropout | {dropout_error:.3e}")

def plot_decision_boundary(X,Y,parameters, name):
    x = np.linspace(X[0].min(), X[0].max(), 100)
    y = np.linspace(X[1].min(), X[1].max(), 100)
    xx, yy = np.meshgrid(x, y)

    grid = np.array([xx.ravel(), yy.ravel()])
    predictions = predict(grid, parameters).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, predictions, levels=[-0.5, 0.5, 1.5], cmap="coolwarm", alpha=0.3)
    plt.scatter(X[0], X[1], c=Y.squeeze(), cmap="coolwarm")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(name)
    plt.savefig(f"images/{name}.png")
    plt.close()

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

def check_initial_data(X_train, Y_train):
    plt.figure()
    plt.scatter(X_train[0, :], X_train[1, :], c=Y_train.squeeze(), cmap='coolwarm')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def initialize_parameters_deep(layer_dims, seed=42, initialize='he'):
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
        if initialize == 'he':
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
        elif initialize == 'random':
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 10
        elif initialize == 'zero':
            parameters["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        else:
            raise ValueError("잘못된 초기화 설정값")
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        

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

def L_model_forward_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Args:
        X (input size, m(# of training data))
        parameters (dict): W1...Wl and b1...bl dictionary
        keep_prob (scalar) : probability to keep neurons

    Returns:
        A_last(1,m) : Last activation value
        Z_last(1,m) : Last logit value
        caches:  cache(linear_cache, activation_cache) list
        dropout_mask_list(list) : list of D_n(1,m) n = 1... m
    """
    caches = []
    dropout_mask_list = []

    L = len(parameters) // 2
    A = X
    for l in range(1, L):
        A, cache = linear_activation_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        D = np.random.rand(*A.shape) < keep_prob
        A = np.multiply(A, D) / keep_prob
        caches.append(cache)
        dropout_mask_list.append(D)

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    ZL = caches[-1][1]

    return AL, ZL, caches, dropout_mask_list

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

def compute_regularized_cost(ZL, Y, parameters, lambd):
    """
    Args:
        ZL (n_cur, m): logit from last layer
        Y (1, m) : label list
        parameters(dict) : W1...WL * B1...BL
        lambd (scalar): hyperparameter 

    Returns:
        cost : cross-entropy loss + regularized_cost
    """
    m = Y.shape[1]
    L = len(parameters) // 2
    cross_entropy_cost = compute_cost(ZL, Y)
    weighted_square_sum = 0.0

    for l in range(1, L+1):
        weighted_square_sum += np.sum(np.square(parameters["W" + str(l)]))

    l2_regularization_term = (lambd / (2*m)) * weighted_square_sum
    cost = cross_entropy_cost + l2_regularization_term
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

def L_model_backward_with_dropout(
    ZL, Y, caches, keep_prob, dropout_mask_list
):
    grads = {}
    L = len(caches)

    dZ = sigmoid(ZL) - Y
    linear_cache, _ = caches[-1]
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    grads[f"dA{L-1}"] = dA_prev
    grads[f"dW{L}"] = dW
    grads[f"db{L}"] = db

    for l in range(L-1, 0, -1):
        D = dropout_mask_list[l-1]
        dA = grads["dA" + str(l)] * D / keep_prob

        dA_prev, dW, db = linear_activation_backward(dA, caches[l-1], "relu")

        grads[f"dA{l-1}"] = dA_prev
        grads[f"dW{l}"] = dW
        grads[f"db{l}"] = db

    return grads

def L_model_regularized_backward(ZL, Y, caches, lambd):
    """
    Args:
        ZL (n_cur, m): logit from last layer
        Y (1, m): label list
        caches (tuple list): (linear_cache, Z)
        lambd (scalar): hyperparameter 

    Returns:
        grads(dict): dA, dW, db
    """
    grads = {}
    L = len(caches)
    m = Y.shape[1]

    AL = sigmoid(ZL) # A_last
    dZ = (AL - Y)
    linear_cache, _ = caches[-1]
    _, W, _ = linear_cache
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    grads["dA" + str(L-1)] = dA_prev
    grads["dW" + str(L)] = dW + (lambd / m) * W
    grads["db" + str(L)] = db

    for l in range(L-1, 0, -1):
        current_cache = caches[l-1]
        linear_cache, _ = current_cache
        _, W, _ = linear_cache
        dA = grads["dA" + str(l)]

        dA_prev, dW, db = linear_activation_backward(dA, current_cache, 'relu')

        grads["dA" + str(l-1)] = dA_prev
        grads["dW" + str(l)] = dW + (lambd / m) * W
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

def L_layer_model(
    X_train, Y_train, layer_dims, X_test, Y_test,
    learning_rate=0.01, iterations=15000, seed=3,
    initialize="he", normalize="L2", lambd=0, keep_prob=0.5
):
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
        "train_objective": [],
    }

    parameters = initialize_parameters_deep(layer_dims, seed, initialize)

    for i in range(iterations + 1):
        if normalize == "dropout":
            _, ZL_train, caches, masks = L_model_forward_with_dropout(X_train, parameters, keep_prob)
        else:
            _, ZL_train, caches = L_model_forward(X_train, parameters)

        if i % 100 == 0 or i == iterations:
            if normalize == "dropout":
                _, ZL_train_eval, _ = L_model_forward(X_train, parameters)
            else:
                ZL_train_eval = ZL_train

            _, ZL_eval, _ = L_model_forward(X_test, parameters)

            train_cost = compute_cost(ZL_train_eval, Y_train)
            eval_cost = compute_cost(ZL_eval, Y_test)

            if normalize == "L2":
                objective = compute_regularized_cost(ZL_train, Y_train, parameters, lambd)
            else:
                objective = compute_cost(ZL_train, Y_train)

            history["iterations"].append(i)
            history["train_cost"].append(train_cost)
            history["test_cost"].append(eval_cost)
            history["train_objective"].append(objective)

        if i == iterations:
            break

        if normalize == "L2":
            grads = L_model_regularized_backward(ZL_train, Y_train, caches, lambd)
        elif normalize == "dropout":
            grads = L_model_backward_with_dropout(ZL_train, Y_train, caches, keep_prob=keep_prob,dropout_mask_list=masks)
        else:
            grads = L_model_backward(ZL_train, Y_train, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

    return parameters, history

def main():
    # experiment_1(layer_dims)
    # models, histories = experiment_2(layer_dims)
    # histories = experiment_3([2, 20, 3, 1])
    experiment_4()

    # plt.figure()
    # plt.plot(history["iterations"], history["train_cost"], label="Train")
    # plt.plot(history["iterations"], history["test_cost"], label="Test")
    # plt.xlabel("Iterations")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.savefig(f"images/{name}_learning_curve.png")
    plt.show()

if __name__ == "__main__":
    main()