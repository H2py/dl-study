import numpy as np, sklearn.datasets
import matplotlib.pyplot as plt
import math, time

def plot_decision_boundary(X, Y, parameters, name, ax=None):
    """
    Args:
        X (np.ndarray): inputs, shape (2, m)
        Y (np.ndarray): labels, shape (1, m)
        parameters (dict): trained W, b
        name (str): plot title
        ax (Axes or None): subplot to draw on

    Returns:
        None
    """
    x = np.linspace(X[0].min(), X[0].max(), 100)
    y = np.linspace(X[1].min(), X[1].max(), 100)
    xx, yy = np.meshgrid(x, y)

    grid = np.array([xx.ravel(), yy.ravel()])
    predictions = predict(grid, parameters).reshape(xx.shape)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots()

    ax.contourf(
        xx, yy, predictions,
        levels=[-0.5, 0.5, 1.5], cmap="coolwarm", alpha=0.3
    )
    ax.scatter(X[0], X[1], c=Y.squeeze(), cmap="coolwarm")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(name)

    if standalone:
        fig.tight_layout()
        fig.savefig(f"images/{name}.png")
        plt.close(fig)

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

def update_parameters_with_gd(parameters, grads, learning_rate):
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

def random_mini_batches(X, Y, mini_batch_size, seed):
    """
    Args:
        X (np.ndarray): input data, shape (n_x, m)
        Y (np.ndarray): labels, shape (1, m)
        mini_batch_size (int): number of samples per mini-batch
        seed (int): random seed for shuffling

    Returns:
        mini_batches (list): shuffled (mini_batch_X, mini_batch_Y) tuples
    """

    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    permutation = np.random.permutation(m)

    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation]

    floor_num_of_size = math.floor(m / mini_batch_size)
    for i in range(0, floor_num_of_size):
        mini_batch_X = shuffled_X[:, i * mini_batch_size : (i+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, i * mini_batch_size : (i+1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0 :
        mini_batch_X = shuffled_X[:, floor_num_of_size * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, floor_num_of_size * mini_batch_size : m]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def experiment_1(train_X, train_Y, dev_X, dev_Y):
    layer_dims = [2,5,2,1]
    
    histories = {}
    for mbs in [300, 64, 1]:
        tic = time.time()

        parameters, history = L_layer_model(train_X, train_Y, layer_dims, dev_X, dev_Y,learning_rate=0.01,iterations=1000, seed=3, mini_batch_size=mbs)

        toc = time.time()
        train_accuracy = np.mean(predict(train_X, parameters) == train_Y)
        dev_accuracy = np.mean(predict(dev_X, parameters) == dev_Y)

        final_cost = history["epoch_costs"][-1]
        cost_100 = history["epoch_costs"][99]

        histories[mbs] = history
        print(
            f"크기: {mbs} \n "
            f"업데이트/epoch: {(train_X.shape[1] + mbs - 1) // mbs} \n "
            f"100 epoch 비용: {cost_100:.6f} \n "
            f"최종 비용: {final_cost:.6f} \n "
            f"학습 정확도: {train_accuracy:.2%} \n "
            f"dev 정확도: {dev_accuracy:.2%} \n "
            f"시간: {toc-tic:.2f}초"
        )
    
    plt.figure()
    for mbs in [300,64]:
        costs = histories[mbs]["update_costs"]
        updates = np.arange(1, len(costs) + 1)

        plt.plot(updates, costs, label=f"Batch size {mbs}")

    plt.xlabel("Update")
    plt.ylabel("Mini-batch costs")
    plt.title("First 200 costs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/mini_batch_costs.png")

def experiment_2():
    np.random.seed(0)
    days = np.arange(1, 366)
    true = 12 - 8 * np.cos(2 * np.pi * days / 365)
    theta = true + 3 * np.random.randn(365)

    results = {
        "theta": theta,
        "true": true
    }

    plt.figure()
    plt.scatter(days, theta, s=8, alpha=0.3, label="theta")

    for beta in [0.5, 0.9, 0.98]:
        v = 0.0
        averages = []

        for day in range(1, len(days) + 1):
            v = beta * v + (1 - beta) * theta[day - 1]
            averages.append(v)

        averages = np.array(averages)
        results[f"beta:{beta}"] = averages
        plt.plot(days, averages, label=f"beta:{beta}")

        if beta == 0.98:
            corrected_averages = averages / (1 - beta ** days)
            results[f"beta:{beta}_corrected"] = corrected_averages
            plt.plot(
                days, corrected_averages,
                label=f"beta:{beta}_corrected"
            )
    plt.plot(days, true, label="true")

    plt.xlabel("Day")
    plt.ylabel("Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/ewa.png")
    plt.close()

    indices = np.array([1, 10, 50, 200]) - 1

    print("| 방법 | 1일 | 10일 | 50일 | 200일 |")
    print("|---|---|---|---|---|")
    for name, values in results.items():
        numbers = " | ".join(f"{value:.4f}" for value in values[indices])
        print(f"| {name} | {numbers} |")

def make_field(n, seed, flip=0.10, amp=0.25, freq=4.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-0.8, 0.8, size=(n, 2))
    y = (X[:, 1] > amp * np.sin(freq * X[:, 0]) + 0.05).astype(int)
    y = np.where(rng.random(n) < flip, 1 - y, y)
    return X.T, y.reshape(1, -1)

def initialize_state(parameters):
    """
    Args:
        parameters (dict): W, b

    Returns:
        state (dict): zero arrays matching each parameter's shape
    """
    L = len(parameters) // 2
    v = {}

    for l in range(1, L+1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)

    return v

def initialize_adam(parameters):
    L = len(parameters) // 2 
    v = {}
    s = {}
    
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)    
    
    return v, s

def update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)] 
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)] 

        parameters["W" + str(l)] -= learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * v["db" + str(l)]

    return parameters, v

def update_parameters_with_rmsprop(parameters, grads, s, beta=0.999, learning_rate=0.01, epsilon=1e-8):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        s["dW" + str(l)] = beta * s["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]**2 
        s["db" + str(l)] = beta * s["db" + str(l)] + (1 - beta) * grads["db" + str(l)]**2

        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)] /  (np.sqrt(s["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)] /  (np.sqrt(s["db" + str(l)]) + epsilon)

    return parameters, s

def update_parameters_with_adam(parameters, grads, v, s, t, beta1 = 0.9, beta2 = 0.999, learning_rate=0.01, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L+1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)] 
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)] 

        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1**t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1**t)
        
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * grads["dW" + str(l)]**2 
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * grads["db" + str(l)]**2

        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2**t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2**t)

        parameters["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] /  (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] /  (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return parameters, v, s

def experiment_3():
    # Used for (가) experiment
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)

    np.random.seed(4)
    dev_X, dev_Y = sklearn.datasets.make_moons(n_samples=1000, noise=.2)

    train_X, dev_X = train_X.T, dev_X.T
    train_Y = train_Y.reshape(1, -1)
    dev_Y = dev_Y.reshape(1, -1)

    layer_dims = [2, 5, 2, 1]
    results_a = {}

    settings = [
        ("gd", 0.0007),
        ("momentum", 0.0007),
        ("rmsprop", 0.0007),
        ("adam", 0.0007),
        ("gd", 0.1),
    ]

    for num, (method, lr) in enumerate(settings, start=1):
        parameters, history = L_layer_model(
            train_X, train_Y, layer_dims, dev_X, dev_Y,
            optimizer_method=method,
            learning_rate=lr,
            iterations=5000,
            seed=3,
            mini_batch_size=64,
            decay=None
        )

        results_a[num] = {
            "method": method,
            "learning_rate": lr,
            "parameters": parameters,
            "history": history,
        }

    print("| # | 방법 | 학습률 | 1,000 epoch 비용 | 최종 비용 | 학습 정확도 | dev 정확도 |")
    print("|---|---|---|---|---|---|---|")

    plt.figure()
    for num, result in results_a.items():
        parameters = result["parameters"]
        costs = result["history"]["epoch_costs"]
        epochs = np.arange(1, len(costs) + 1)

        train_accuracy = np.mean(predict(train_X, parameters) == train_Y)
        dev_accuracy = np.mean(predict(dev_X, parameters) == dev_Y)

        print(
            f"| {num} | {result['method']} | {result['learning_rate']} | "
            f"{costs[999]:.6f} | {costs[-1]:.6f} | "
            f"{train_accuracy:.2%} | {dev_accuracy:.2%} |"
        )

        plt.plot(epochs, costs, label=f"{result['method']}, lr = {result['learning_rate']} ")

    plt.xlabel("Epoch")
    plt.ylabel("Costs")
    plt.legend()
    plt.title("Optimizer Comparison")
    plt.tight_layout()
    plt.savefig("images/optimizer_moons_costs.png")
    plt.close()

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4), sharex=True, sharey=True
    )

    for ax, num in zip(axes, [1, 4, 5]):
        result = results_a[num]
        name = f"{num}: {result['method']}, lr={result['learning_rate']}"

        plot_decision_boundary(train_X, train_Y, result["parameters"], name, ax=ax)

    fig.tight_layout()
    fig.savefig("images/optimizer_moons_boundaries.png")
    plt.close(fig)    


    # Used for (나) experiment
    field_X, field_Y = make_field(150, 0)

    A_train = field_X.copy()
    A_train[0] *= 100

    mu = np.mean(A_train, axis=1, keepdims=True)
    sigma = np.std(A_train, axis=1, keepdims=True)
    A_train_normalized = (A_train - mu) / sigma

    layer_dims = [2, 20, 3, 1]
    results_b = {}

    settings = [
        ("정규화하지 않음", A_train, "gd"),
        ("정규화하지 않음", A_train, "adam"),
        ("정규화함", A_train_normalized, "gd"),
        ("정규화함", A_train_normalized, "adam"),
    ]

    for num, (input_name, X, method) in enumerate(settings, start=1):
        parameters, history = L_layer_model(
            X, field_Y, layer_dims, None, None,
            optimizer_method=method,
            learning_rate=0.003,
            iterations=10000,
            seed=3,
            mini_batch_size=150,
            decay=None
        )

        results_b[num] = {
            "input_name": input_name,
            "method": method,
            "X": X,
            "Y": field_Y,
            "parameters": parameters,
            "history": history,
        }

    print(
        "| # | 입력 | 방법 | 500 epoch 비용 | 최종 비용 | "
        "마지막 1,000 epoch 비용의 최솟값 ~ 최댓값 | 학습 정확도 |"
    )
    print("|---|---|---|---|---|---|---|")

    plt.figure()

    for num, result in results_b.items():
        parameters = result["parameters"]
        costs = np.array(result["history"]["epoch_costs"])
        epochs = np.arange(1, len(costs) + 1)

        cost_500 = costs[499]
        final_cost = costs[-1]
        min_cost = costs[-1000:].min()
        max_cost = costs[-1000:].max()

        train_accuracy = np.mean(
            predict(result["X"], parameters) == result["Y"]
        )

        print(
            f"| {num} | {result['input_name']} | {result['method']} | "
            f"{cost_500:.6f} | {final_cost:.6f} | "
            f"{min_cost:.6f} ~ {max_cost:.6f} | "
            f"{train_accuracy:.2%} |"
        )

        input_label = (
            "Normalized"
            if result["input_name"] == "정규화함"
            else "Unnormalized"
        )

        plt.plot(
            epochs, costs,
            label=f"{num}: {input_label}, {result['method']}"
        )

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Epoch cost (log scale)")
    plt.title("Input normalization and optimizer comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/optimizer_field_costs.png")
    plt.close()

    return results_a, results_b

def experiment_4(train_X, train_Y, dev_X, dev_Y):
    layer_dims = [2, 5, 2, 1]
    histories = {}

    colors = {
        None: "tab:blue",
        "A": "tab:orange",
        "B": "tab:green",
    }

    for method in [None, "A", "B"]:
        parameters, history = L_layer_model(
            train_X, train_Y, layer_dims, dev_X, dev_Y,
            learning_rate=0.1,
            iterations=5000,
            seed=3,
            mini_batch_size=64,
            decay=method
        )
        histories[method] = history

        costs = np.array(history["epoch_costs"])
        final_cost = costs[-1]
        cost_range = costs[-1000:].max() - costs[-1000:].min()

        train_accuracy = np.mean(predict(train_X, parameters) == train_Y)
        dev_accuracy = np.mean(predict(dev_X, parameters) == dev_Y)

        print(
            f"감쇠 방법: {method}\n"
            f"최종 비용: {final_cost:.6f}\n"
            f"마지막 1,000 epoch 비용 범위 폭: {cost_range:.6f}\n"
            f"학습 정확도: {train_accuracy:.2%}\n"
            f"dev 정확도: {dev_accuracy:.2%}\n"
        )

    plt.figure()
    for method in [None, "A", "B"]:
        costs = histories[method]["epoch_costs"]
        epochs = np.arange(1, len(costs) + 1)
        plt.plot(epochs, costs, label=f"Decay: {method}", color=colors[method])

    plt.xlabel("Epoch")
    plt.ylabel("Epoch cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/lr_decay.png")
    plt.close()

    plt.figure()
    for method in [None, "B"]:
        costs = histories[method]["epoch_costs"][-1000:]
        epochs = np.arange(4001, 5001)
        plt.plot(epochs, costs, label=f"Decay: {method}", color=colors[method])

    plt.xlabel("Epoch")
    plt.ylabel("Epoch cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/lr_decay_last_1000.png")
    plt.close()

def L_layer_model(
    X_train, Y_train, layer_dims, X_test, Y_test, optimizer_method = "gd",
    learning_rate=0.01, iterations=1000, seed=3, mini_batch_size=64, decay=None, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8
):
    """
    Args:
        X_train (np.ndarray): train inputs, shape (n_input, m_train)
        Y_train (np.ndarray): train labels, shape (1, m_train)
        layer_dims (list): layer sizes
        X_test (np.ndarray): dev inputs, shape (n_input, m_test)
        Y_test (np.ndarray): dev labels, shape (1, m_test)
        learning_rate (float): step size
        iterations (int): number of epochs
        seed (int): random seed for initialization
        mini_batch_size (int): number of samples per mini-batch

    Returns:
        parameters (dict): trained W, b
        history (dict): costs and final accuracies
    """
    m = X_train.shape[1]
    history = {
        "iterations": [],
        "train_cost": [],
        "test_cost": [],
        "epoch_costs": [],
        "update_costs": [],
    }
    parameters = initialize_parameters_deep(layer_dims, seed)
    lr0 = learning_rate
    t = 0

    if optimizer_method == "gd":
        pass
    elif optimizer_method == "momentum" or optimizer_method == "rmsprop":
        v = initialize_state(parameters)
    elif optimizer_method == "adam":
        v,s = initialize_adam(parameters)

    for i in range(1, iterations + 1):
        mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size, seed=10 + i)
        epoch_loss_sum = 0.0

        for mini_batch_X, mini_batch_Y in mini_batches:
            _, ZL_train, caches = L_model_forward(mini_batch_X, parameters)
            cost = compute_cost(ZL_train, mini_batch_Y)
            epoch_loss_sum += cost * mini_batch_X.shape[1]

            if len(history["update_costs"]) < 200:
                history["update_costs"].append(cost)

            grads = L_model_backward(ZL_train, mini_batch_Y, caches)
            
            if optimizer_method == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer_method == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta1, learning_rate)
            elif optimizer_method == "rmsprop":
                parameters, s = update_parameters_with_rmsprop(parameters, grads, v, beta2, learning_rate, epsilon)
            elif optimizer_method == "adam":
                t = t + 1 
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, beta1, beta2, learning_rate, epsilon)

        epoch_cost = epoch_loss_sum / m
        history["epoch_costs"].append(epoch_cost)

        if decay == "A":
            learning_rate = lr0 / (1 + i)
        elif decay == "B":
            learning_rate = lr0 / (1 + i // 1000)

        # if i == 100:
        #     print(f"Learning rate (100 epoch) : {learning_rate}")
        # elif i == 5000:
        #     print(f"Learning rate (5000 epoch): {learning_rate}")

        if i % 100 == 0 or i == iterations:
            history["iterations"].append(i)
            history["train_cost"].append(epoch_cost)

            if X_test is not None and Y_test is not None:
                _, ZL_test, _ = L_model_forward(X_test, parameters)
                test_cost = compute_cost(ZL_test, Y_test)
                history["test_cost"].append(test_cost)

    return parameters, history

def main():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)
    np.random.seed(4)
    dev_X,   dev_Y   = sklearn.datasets.make_moons(n_samples=1000, noise=.2)
    train_X, dev_X = train_X.T, dev_X.T
    train_Y, dev_Y = train_Y.reshape(1, -1), dev_Y.reshape(1, -1)

    # mini_batches = random_mini_batches(train_X, train_Y, mini_batch_size=64, seed=11)
    # print([mini_batch_X.shape[1] for mini_batch_X, _ in mini_batches])

    # ============= Experiment 1 ===================
    # experiment_1(train_X, train_Y, dev_X, dev_Y)

    # ============= Experiment 2 ===================
    # experiment_2()

    # ============= Experiment 3 ===================
    experiment_3()
    # ============= Experiment 4 ===================
    #experiment_4(train_X,train_Y, dev_X, dev_Y)
if __name__ == "__main__":
    main()