import math, sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-logistic-regression"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.grad_check import grad_check, flatten_deep_params, unflatten_deep_params
import logistic_regression as logistic


def initialize_parameters_deep(layer_dims, seed=3, initialize="he", use_bn=False):
    """
    Args:
        layer_dims (list): list representing # of neurons in each layer
        seed (int): random seed
        initialize (string): 'he', 'random', 'small' or 'zero'
        use_bn (bool): whether to use BN in hidden layers

    Returns:
        parameters (dict): Wl (n_cur, n_prev), bl/gammal/betal (n_cur, 1)
    """
    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims) - 1

    for l in range(1, L + 1):
        shape = (layer_dims[l], layer_dims[l - 1])

        if initialize == "he":
            parameters[f"W{l}"] = np.random.randn(*shape) * np.sqrt(
                2.0 / layer_dims[l - 1]
            )
        elif initialize == "random":
            parameters[f"W{l}"] = np.random.randn(*shape) * 10
        elif initialize == "small":
            parameters[f"W{l}"] = np.random.randn(*shape) * 0.01
        elif initialize == "zero":
            parameters[f"W{l}"] = np.zeros(shape)
        else:
            raise ValueError("initialize must be 'he', 'random', 'small' or 'zero'")

        if use_bn and l < L:
            parameters[f"gamma{l}"] = np.ones((layer_dims[l], 1))
            parameters[f"beta{l}"] = np.zeros((layer_dims[l], 1))
        else:
            parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    return parameters


def initialize_running_stats(parameters):
    """
    Args:
        parameters (dict): Wl (n_cur, n_prev), bl/gammal/betal (n_cur, 1)

    Returns:
        running (dict): meanl/varl (n_cur, 1), updates (scalar)
    """
    running = {"updates": 0}

    for name in parameters:
        if name.startswith("gamma"):
            l = name[5:]
            running[f"mean{l}"] = np.zeros_like(parameters[name])
            running[f"var{l}"] = np.ones_like(parameters[name])

    return running


def linear_forward(A_prev, W, b):
    """
    n_cur = # of neurons in layer l
    n_prev = # of neurons in layer l-1
    Args:
        A_prev (n_prev, m): activation from previous layer
        W (n_cur, n_prev): weight matrix
        b (n_cur, 1) or scalar: bias

    Returns:
        Z (n_cur, m), cache: Z = W @ A_prev + b, cache = (A_prev, W, b)
    """
    Z = W @ A_prev + b
    return Z, (A_prev, W, b)


def sigmoid(z):
    """sigmoid function
    Args:
        z (np.ndarray): (1, m)

    Returns:
        np.ndarray: (1, m)
    """
    z = np.asarray(z)
    result = np.zeros_like(z, dtype=np.float64)
    mask = z > 0
    result[mask] = 1 / (1 + np.exp(-z[mask]))
    result[~mask] = np.exp(z[~mask]) / (1 + np.exp(z[~mask]))
    return result


def relu(Z):
    """
    Args:
        Z (n_cur, m): logits

    Returns:
        A (n_cur, m): activations
    """
    return np.maximum(0, Z)


def softmax(Z):
    """
    Args:
        Z (C, m): logits

    Returns:
        A (C, m): class probabilities
    """
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    return A


def one_hot(y, C):
    """
    Args:
        y (1, m): correct class indices
        C (int): class number

    Returns:
        Y (C, m): one-hot labels
    """
    m = y.shape[1]
    Y = np.zeros((C, m))

    for i in range(m):
        Y[y[0, i], i] = 1

    return Y


def batch_norm_forward(Z, gamma, beta, mean, var):
    """
    Args:
        Z (n_cur, m): logits
        gamma, beta (n_cur, 1): BN parameters
        mean, var (n_cur, 1): mean and variance

    Returns:
        Z_tilde (n_cur, m), cache: (Z_norm, inv_std, gamma)
    """
    inv_std = 1 / np.sqrt(var + 1e-5)
    Z_norm = (Z - mean) * inv_std
    Z_tilde = gamma * Z_norm + beta

    return Z_tilde, (Z_norm, inv_std, gamma)


def batch_norm_backward(dZ_tilde, cache):
    """
    Args:
        dZ_tilde (n_cur, m): gradient before ReLU
        cache (tuple): Z_norm (n_cur, m), inv_std/gamma (n_cur, 1)

    Returns:
        dZ (n_cur, m), dgamma (n_cur, 1), dbeta (n_cur, 1)
    """
    Z_norm, inv_std, gamma = cache
    m = dZ_tilde.shape[1]

    dgamma = np.sum(dZ_tilde * Z_norm, axis=1, keepdims=True) / m
    dbeta = np.sum(dZ_tilde, axis=1, keepdims=True) / m

    dZ_norm = dZ_tilde * gamma

    dZ = inv_std * (
        dZ_norm
        - np.mean(dZ_norm, axis=1, keepdims=True)
        - Z_norm * np.mean(dZ_norm * Z_norm, axis=1, keepdims=True)
    )

    return dZ, dgamma, dbeta


def linear_activation_forward(A_prev, W, b, activation):
    """
    Args:
        A_prev (n_prev, m): activation from previous layer
        W (n_cur, n_prev): weight matrix
        b (n_cur, 1): bias
        activation (string): 'relu' or 'softmax'

    Returns:
        A (n_cur, m), cache: (linear_cache, Z)
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "relu":
        A = relu(Z)
    elif activation == "softmax":
        A = softmax(Z)
    else:
        raise ValueError("activation must be 'relu' or 'softmax'")

    return A, (linear_cache, Z)


def L_model_forward(
    X, parameters, running=None, training=True, update_running=False
):
    """
    Args:
        X (n_input, m): input data
        parameters (dict): Wl (n_cur, n_prev), bl/gammal/betal (n_cur, 1)
        running (dict): meanl/varl (n_cur, 1), updates (scalar)
        training (bool): whether to use batch statistics
        update_running (bool): whether to update running statistics

    Returns:
        AL (C, m), ZL (C, m), caches: (linear_cache, activation_cache, bn_cache) list
    """
    L = sum(name.startswith("W") for name in parameters)
    A = X
    caches = []

    for l in range(1, L):
        Z, linear_cache = linear_forward(
            A, parameters[f"W{l}"], parameters.get(f"b{l}", 0)
        )
        bn_cache = None

        if f"gamma{l}" in parameters:
            if training:
                mean = np.mean(Z, axis=1, keepdims=True)
                var = np.var(Z, axis=1, keepdims=True)

                if update_running:
                    running[f"mean{l}"] = 0.9 * running[f"mean{l}"] + 0.1 * mean
                    running[f"var{l}"] = 0.9 * running[f"var{l}"] + 0.1 * var
            else:
                mean = running[f"mean{l}"]
                var = running[f"var{l}"]

            Z, bn_cache = batch_norm_forward(
                Z, parameters[f"gamma{l}"], parameters[f"beta{l}"], mean, var
            )

        A = relu(Z)
        caches.append((linear_cache, Z, bn_cache))

    AL, cache = linear_activation_forward(
        A, parameters[f"W{L}"], parameters[f"b{L}"], "softmax"
    )
    caches.append((cache[0], cache[1], None))

    if training and update_running and "gamma1" in parameters:
        running["updates"] += 1

    ZL = caches[-1][1]
    return AL, ZL, caches


def compute_cost(ZL, Y):
    """
    Args:
        ZL (C, m): logit from last layer
        Y (C, m): one-hot labels

    Returns:
        cost (scalar): cross-entropy loss
    """
    Z_shifted = ZL - np.max(ZL, axis=0, keepdims=True)

    log_sum_exp = np.log(np.sum(np.exp(Z_shifted), axis=0, keepdims=True))
    correct_logits = np.sum(Y * Z_shifted, axis=0, keepdims=True)

    loss = log_sum_exp - correct_logits

    return np.mean(loss)


def linear_backward(dZ, cache):
    """
    Args:
        dZ (n_cur, m): derivative of Z
        cache (tuple): A_prev (n_prev, m), W (n_cur, n_prev), b (n_cur, 1) or scalar

    Returns:
        dA_prev (n_prev, m), dW (n_cur, n_prev), db (n_cur, 1)
    """
    A_prev, W, _ = cache
    m = A_prev.shape[1]

    dW = (dZ @ A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = W.T @ dZ

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation="relu"):
    """
    Args:
        dA (n_cur, m): activation gradient
        cache (tuple): (linear_cache, Z)
        activation (string): 'relu'

    Returns:
        dA_prev (n_prev, m), dW (n_cur, n_prev), db (n_cur, 1)
    """
    if activation != "relu":
        raise ValueError("은닉층 역전파는 ReLU만 사용합니다.")

    linear_cache, Z = cache
    dZ = dA * (Z > 0)

    return linear_backward(dZ, linear_cache)


def L_model_backward(ZL, Y, caches):
    """
    Args:
        ZL (C, m): logit from last layer
        Y (C, m): one-hot labels
        caches (tuple list): (linear_cache, activation_cache, bn_cache)

    Returns:
        grads (dict): dAl (n_cur, m), dWl (n_cur, n_prev),
                      dbl/dgammal/dbetal (n_cur, 1)
    """
    grads = {}
    L = len(caches)

    AL = softmax(ZL)
    dZ = AL - Y

    linear_cache, _, _ = caches[-1]
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    grads[f"dA{L - 1}"] = dA_prev
    grads[f"dW{L}"] = dW
    grads[f"db{L}"] = db

    for l in range(L - 1, 0, -1):
        linear_cache, Z, bn_cache = caches[l - 1]
        dA = grads[f"dA{l}"]

        if bn_cache is None:
            dA_prev, dW, db = linear_activation_backward(
                dA, (linear_cache, Z), "relu"
            )
        else:
            dZ_tilde = dA * (Z > 0)
            dZ, dgamma, dbeta = batch_norm_backward(dZ_tilde, bn_cache)

            grads[f"dgamma{l}"] = dgamma
            grads[f"dbeta{l}"] = dbeta

            dA_prev, dW, db = linear_backward(dZ, linear_cache)

        grads[f"dA{l - 1}"] = dA_prev
        grads[f"dW{l}"] = dW
        grads[f"db{l}"] = db

    return grads


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Args:
        parameters (dict): Wl (n_cur, n_prev), bl/gammal/betal (n_cur, 1)
        grads (dict): parameter gradients with corresponding shapes
        learning_rate (float): step size

    Returns:
        parameters (dict): updated in place, same shapes
    """
    for name in parameters:
        parameters[name] -= learning_rate * grads["d" + name]

    return parameters


def initialize_adam(parameters):
    """
    Args:
        parameters (dict): Wl (n_cur, n_prev), bl/gammal/betal (n_cur, 1)

    Returns:
        v, s (dict): moment arrays with corresponding parameter shapes
    """
    v, s = {}, {}

    for name in parameters:
        v["d" + name] = np.zeros_like(parameters[name])
        s["d" + name] = np.zeros_like(parameters[name])

    return v, s


def update_parameters_with_adam(
    parameters, grads, v, s, t,
    beta1=0.9, beta2=0.999, learning_rate=0.01, epsilon=1e-8,
):
    """
    Args:
        parameters (dict): Wl (n_cur, n_prev), bl/gammal/betal (n_cur, 1)
        grads, v, s (dict): arrays with corresponding parameter shapes
        t (int): update count
        beta1, beta2, learning_rate, epsilon (float): Adam settings

    Returns:
        parameters, v, s (dict): updated in place, same shapes
    """
    for name in parameters:
        grad_name = "d" + name

        v[grad_name] = beta1 * v[grad_name] + (1 - beta1) * grads[grad_name]
        s[grad_name] = beta2 * s[grad_name] + (1 - beta2) * grads[grad_name] ** 2

        v_corrected = v[grad_name] / (1 - beta1 ** t)
        s_corrected = s[grad_name] / (1 - beta2 ** t)

        parameters[name] -= learning_rate * v_corrected / (
            np.sqrt(s_corrected) + epsilon
        )

    return parameters, v, s


def predict(X, parameters, running=None, training=False):
    """
    Args:
        X (n_input, m): input data
        parameters (dict): Wl (n_cur, n_prev), bl/gammal/betal (n_cur, 1)
        running (dict): meanl/varl (n_cur, 1), updates (scalar)
        training (bool): whether to use batch statistics

    Returns:
        predictions (1, m): predicted labels
    """
    _, ZL, _ = L_model_forward(
        X, parameters, running, training=training
    )
    return np.argmax(ZL, axis=0, keepdims=True)


def random_mini_batches(X, Y, mini_batch_size, seed):
    """
    Args:
        X (np.ndarray): input data, shape (n_x, m)
        Y (np.ndarray): labels, shape (C, m)
        mini_batch_size (int): number of samples per mini-batch
        seed (int): random seed for shuffling

    Returns:
        mini_batches (list): X (n_x, batch_m), Y (C, batch_m) tuples
    """
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    floor_num_of_size = math.floor(m / mini_batch_size)

    for i in range(0, floor_num_of_size):
        mini_batch_X = shuffled_X[:, i * mini_batch_size : (i + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, i * mini_batch_size : (i + 1) * mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, floor_num_of_size * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, floor_num_of_size * mini_batch_size : m]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches


def L_layer_model(
    X_train, Y_train, layer_dims, X_dev=None, Y_dev=None,
    optimizer_method="gd", learning_rate=0.01, iterations=1000,
    seed=3, mini_batch_size=64, initialize="he",
    beta1=0.9, beta2=0.999, epsilon=1e-8, use_bn=False,
):
    """
    Args:
        X_train (np.ndarray): train inputs, shape (n_input, m_train)
        Y_train (np.ndarray): train labels, shape (C, m_train)
        layer_dims (list): layer sizes
        X_dev (np.ndarray): dev inputs, shape (n_input, m_dev)
        Y_dev (np.ndarray): dev labels, shape (C, m_dev)
        optimizer_method (string): 'gd' or 'adam'
        learning_rate (float): step size
        iterations (int): number of epochs
        seed (int): random seed for initialization
        mini_batch_size (int): number of samples per mini-batch
        initialize (string): initialization method
        beta1, beta2, epsilon (float): Adam settings
        use_bn (bool): whether to use BN in hidden layers

    Returns:
        parameters (dict): trained Wl (n_cur, n_prev), bl/gammal/betal (n_cur, 1)
        history (dict): initial_cost (scalar), epoch_costs/dev_costs (iterations,),
                        running statistics
    """
    if optimizer_method not in ("gd", "adam"):
        raise ValueError("optimizer_method must be 'gd' or 'adam'")

    m = X_train.shape[1]
    parameters = initialize_parameters_deep(layer_dims, seed, initialize, use_bn)
    running = initialize_running_stats(parameters)

    _, ZL, _ = L_model_forward(X_train, parameters)

    history = {
        "initial_cost": compute_cost(ZL, Y_train),
        "epoch_costs": [],
        "dev_costs": [],
        "running": running,
    }

    t = 0
    if optimizer_method == "adam":
        v, s = initialize_adam(parameters)

    for epoch in range(1, iterations + 1):
        mini_batches = random_mini_batches(
            X_train, Y_train, mini_batch_size, seed=10 + epoch
        )
        epoch_loss_sum = 0.0

        for batch_X, batch_Y in mini_batches:
            _, ZL, caches = L_model_forward(
                batch_X, parameters, running, update_running=use_bn
            )

            cost = compute_cost(ZL, batch_Y)
            epoch_loss_sum += cost * batch_X.shape[1]
            grads = L_model_backward(ZL, batch_Y, caches)

            if optimizer_method == "gd":
                parameters = update_parameters_with_gd(
                    parameters, grads, learning_rate
                )
            else:
                t += 1
                parameters, v, s = update_parameters_with_adam(
                    parameters, grads, v, s, t,
                    beta1=beta1, beta2=beta2,
                    learning_rate=learning_rate, epsilon=epsilon
                )

        history["epoch_costs"].append(epoch_loss_sum / m)

        if X_dev is not None and Y_dev is not None:
            _, ZL_dev, _ = L_model_forward(
                X_dev, parameters, running, training=False
            )
            history["dev_costs"].append(compute_cost(ZL_dev, Y_dev))

    return parameters, history


def check_gradients(X, Y, layers_dims, use_bn=False):
    """
    Args:
        X (n_input, m): input data
        Y (C, m): one-hot labels
        layers_dims (list): [n_input, n_1, ..., C]
        use_bn (bool): whether to check BN parameters

    Returns:
        error (scalar): relative gradient error
    """
    rng = np.random.default_rng(0)
    indices = rng.permutation(X.shape[1])[:20]

    X_check = X[:, indices]
    Y_check = Y[:, indices]

    parameters = initialize_parameters_deep(layers_dims, use_bn=use_bn)

    for name in parameters:
        if name.startswith("W"):
            continue

        parameters[name] = rng.normal(size=parameters[name].shape) * 0.1

        if name.startswith("gamma"):
            parameters[name] += 1

    _, ZL, caches = L_model_forward(X_check, parameters)
    grads = L_model_backward(ZL, Y_check, caches)

    parameter_grads = {}
    for name in parameters:
        parameter_grads[name] = grads["d" + name]

    theta = flatten_deep_params(parameters)
    analytic_grad = flatten_deep_params(parameter_grads)

    def cost_fn(theta):
        """
        Args:
            theta (P,): flattened parameters

        Returns:
            cost (scalar): cross-entropy loss
        """
        restored = unflatten_deep_params(theta, parameters)
        _, ZL, _ = L_model_forward(X_check, restored)

        return compute_cost(ZL, Y_check)

    return grad_check(cost_fn, theta, analytic_grad)


def check_reduction():
    """
    Args:
        None

    Returns:
        None: prints parameter and probability differences
    """
    digits = load_digits()
    mask = (digits.target == 0) | (digits.target == 1)
    X, y = digits.data[mask], digits.target[mask]

    X_tr, X_te, y_tr, _ = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    X_train, X_test = X_tr.T / 16.0, X_te.T / 16.0
    y_train = y_tr.reshape(1, -1)
    Y_train = one_hot(y_train, 2)

    w, b = logistic.initialize(X_train.shape[0])

    params, _, _ = logistic.optimize(
        w, b, X_train, y_train,
        num_iterations=2000, learning_rate=0.5
    )

    logistic_prob = logistic.sigmoid(
        params["w"].T @ X_test + params["b"]
    )

    print("\n| softmax 학습률 | W 최대 차이 | b 차이 | 확률 최대 차이 |")
    print("|---|---|---|---|")

    for learning_rate in (0.5, 0.25):
        parameters, _ = L_layer_model(
            X_train, Y_train, [64, 2],
            learning_rate=learning_rate,
            iterations=2000,
            mini_batch_size=X_train.shape[1],
            initialize="zero"
        )

        W, b = parameters["W1"], parameters["b1"]
        A, _, _ = L_model_forward(X_test, parameters)

        w_diff = np.max(np.abs(W[1] - W[0] - params["w"].ravel()))
        b_diff = abs((b[1] - b[0]).item() - params["b"])
        prob_diff = np.max(np.abs(A[1:2] - logistic_prob))

        print(
            f"| {learning_rate} | {w_diff:.10e} "
            f"| {b_diff:.10e} | {prob_diff:.10e} |"
        )


def check_bn(X, Y):
    """
    Args:
        X (64, m): input data
        Y (10, m): one-hot labels

    Returns:
        passed (bool): whether the gradient check passed
    """
    error = check_gradients(X, Y, [64, 8, 6, 10], use_bn=True)

    print(f"\nBN 기울기 검증 상대 오차: {error:.10e}")
    print("통과:", error < 1e-7)

    if not error < 1e-7:
        return False

    # 검증 2: BN 층에 b를 남겨 둔 경우
    rng = np.random.default_rng(0)
    indices = rng.permutation(X.shape[1])[:20]

    parameters = initialize_parameters_deep([64, 8, 6, 10], use_bn=True)

    for l in (1, 2):
        shape = parameters[f"beta{l}"].shape
        parameters[f"b{l}"] = rng.normal(size=shape) * 0.1

    _, ZL, caches = L_model_forward(X[:, indices], parameters)
    grads = L_model_backward(ZL, Y[:, indices], caches)

    for l in (1, 2):
        print(f"\ndb{l}:")
        print(grads[f"db{l}"])

    # 검증 3: 한 층의 W만 10배로 변경
    parameters = initialize_parameters_deep([64, 32, 32, 10], use_bn=True)

    _, ZL, caches = L_model_forward(X, parameters)
    cost_before = compute_cost(ZL, Y)
    grads_before = L_model_backward(ZL, Y, caches)

    print("\n| 층 | 비용 전 | 비용 후 | dW 크기 전 | dW 크기 후 |")
    print("|---|---|---|---|---|")

    for l in (1, 2, 3):
        changed = {name: value.copy() for name, value in parameters.items()}
        changed[f"W{l}"] *= 10

        _, ZL, caches = L_model_forward(X, changed)
        cost_after = compute_cost(ZL, Y)
        grads_after = L_model_backward(ZL, Y, caches)

        norm_before = np.linalg.norm(grads_before[f"dW{l}"])
        norm_after = np.linalg.norm(grads_after[f"dW{l}"])

        print(
            f"| W{l} | {cost_before:.6f} | {cost_after:.6f} "
            f"| {norm_before:.6e} | {norm_after:.6e} |"
        )

    return True


def load_data():
    """
    Args:
        None

    Returns:
        train_X (64, 1077), dev_X (64, 360), test_X (64, 360)
        train_y (1, 1077), dev_y (1, 360), test_y (1, 360)
    """
    digits = load_digits()
    X, y = digits.data, digits.target

    X_rest, X_te, y_rest, y_te = train_test_split(
        X, y, test_size=360, random_state=0, stratify=y
    )
    X_tr, X_dv, y_tr, y_dv = train_test_split(
        X_rest, y_rest, test_size=360, random_state=0, stratify=y_rest
    )

    train_X, dev_X, test_X = X_tr.T / 16.0, X_dv.T / 16.0, X_te.T / 16.0
    train_y = y_tr.reshape(1, -1)
    dev_y = y_dv.reshape(1, -1)
    test_y = y_te.reshape(1, -1)

    return train_X, dev_X, test_X, train_y, dev_y, test_y


def experiment_1():
    train_X, dev_X, test_X, train_y, dev_y, test_y = load_data()

    train_Y = one_hot(train_y, 10)
    dev_Y = one_hot(dev_y, 10)
    test_Y = one_hot(test_y, 10)

    print(one_hot(np.array([[0, 2, 1]]), 3))

    # 아래부터는 6-1 실험 검증 과정
    Z = np.array([[1000., 1, -1000], [999, 2, -1001], [0, 3, -999]])
    A = softmax(Z)

    print("\nsoftmax:")
    print(A)
    print("열별 합:", np.sum(A, axis=0))

    exp_Z = np.exp(Z)
    A_naive = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    print("\n최댓값을 빼지 않은 softmax:")
    print(A_naive)

    parameters = initialize_parameters_deep([64, 10], initialize="zero")
    _, ZL, _ = L_model_forward(train_X, parameters)

    print("\n0 초기화 비용:", compute_cost(ZL, train_Y))
    print(f"ln(10): {np.log(10):.4f}")

    error = check_gradients(train_X, train_Y, [64, 8, 10])

    print(f"\n기울기 검증 상대 오차: {error:.10e}")
    print("통과:", error < 1e-7)

    if not error < 1e-7:
        return

    experiments = [
        ([64, 10], "zero"),
        ([64, 32, 10], "he"),
    ]

    print("\n| 모델 | 초기화 | 초기 비용 | 최종 비용 | 학습 정확도 | dev 정확도 |")
    print("|---|---|---|---|---|---|")

    epochs = np.arange(1, 201)
    plt.figure()

    for layer_dims, initialization in experiments:
        parameters, history = L_layer_model(
            train_X, train_Y, layer_dims, dev_X, dev_Y,
            learning_rate=0.1, iterations=200, initialize=initialization
        )

        initial_cost = history["initial_cost"]
        final_cost = history["epoch_costs"][-1]
        train_accuracy = np.mean(predict(train_X, parameters) == train_y)
        dev_accuracy = np.mean(predict(dev_X, parameters) == dev_y)

        print(
            f"| {layer_dims} | {initialization} "
            f"| {initial_cost:.6f} | {final_cost:.6f} "
            f"| {train_accuracy:.4%} | {dev_accuracy:.4%} |"
        )

        plt.plot(epochs, history["epoch_costs"], label=f"{layer_dims} Train")
        plt.plot(epochs, history["dev_costs"], "--", label=f"{layer_dims} Dev")

    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.savefig("images/softmax_costs.png")
    plt.close()

    check_reduction()


def experiment_2():
    train_X, dev_X, _, train_y, dev_y, _ = load_data()

    train_Y = one_hot(train_y, 10)
    dev_Y = one_hot(dev_y, 10)

    if not check_bn(train_X, train_Y):
        return

    layer_dims = [64, 32, 32, 32, 32, 10]
    epochs = np.arange(1, 101)
    results = {}

    initializations = [
        ("small", False),
        ("small", True),
        ("he", False),
        ("he", True),
        ("random", False),
        ("random", True),
    ]

    experiments = [
        (initialization, use_bn, 0.1)
        for initialization, use_bn in initializations
    ]
    experiments += [
        ("he", use_bn, learning_rate)
        for use_bn in (False, True)
        for learning_rate in (0.01, 1.0)
    ]

    for initialization, use_bn, learning_rate in experiments:
        print(
            f"\n학습: initialize={initialization}, "
            f"BN={use_bn}, lr={learning_rate}"
        )

        parameters, history = L_layer_model(
            train_X, train_Y, layer_dims, dev_X, dev_Y,
            learning_rate=learning_rate, iterations=100,
            initialize=initialization, use_bn=use_bn
        )

        running = history["running"]
        train_accuracy = np.mean(
            predict(train_X, parameters, running) == train_y
        )
        dev_accuracy = np.mean(
            predict(dev_X, parameters, running) == dev_y
        )

        results[(initialization, use_bn, learning_rate)] = {
            "parameters": parameters,
            "history": history,
            "train_accuracy": train_accuracy,
            "dev_accuracy": dev_accuracy,
        }

    for use_bn in (False, True):
        parameters = results[("he", use_bn, 0.1)]["parameters"]
        count = sum(value.size for value in parameters.values())
        print(f"BN={use_bn}, 파라미터 수: {count}")

    print("\n| # | 초기화 | BN | 초기 비용 | 10 epoch 비용 | 최종 비용 | 학습 정확도 | dev 정확도 |")
    print("|---|---|---|---|---|---|---|---|")

    plt.figure()

    for i, (initialization, use_bn) in enumerate(initializations, 1):
        result = results[(initialization, use_bn, 0.1)]
        history = result["history"]

        print(
            f"| {i} | {initialization} | {use_bn} "
            f"| {history['initial_cost']:.6f} "
            f"| {history['epoch_costs'][9]:.6f} "
            f"| {history['epoch_costs'][-1]:.6f} "
            f"| {result['train_accuracy']:.4%} "
            f"| {result['dev_accuracy']:.4%} |"
        )

        if i != 5:
            plt.plot(
                epochs, history["epoch_costs"],
                label=f"{initialization}, BN={use_bn}"
            )

    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.yscale("log")
    plt.legend()
    plt.savefig("images/bn_initialization.png")
    plt.close()

    print("\n| BN | 학습률 | 10 epoch 비용 | 최종 비용 | 학습 정확도 | dev 정확도 |")
    print("|---|---|---|---|---|---|")

    for use_bn in (False, True):
        plt.figure()

        for learning_rate in (0.01, 0.1, 1.0):
            result = results[("he", use_bn, learning_rate)]
            history = result["history"]

            print(
                f"| {use_bn} | {learning_rate} "
                f"| {history['epoch_costs'][9]:.6f} "
                f"| {history['epoch_costs'][-1]:.6f} "
                f"| {result['train_accuracy']:.4%} "
                f"| {result['dev_accuracy']:.4%} |"
            )

            plt.plot(
                epochs, history["epoch_costs"], label=f"lr={learning_rate}"
            )

        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.legend()
        plt.savefig(f"images/bn_learning_rate_{use_bn}.png")
        plt.close()

    result = results[("he", True, 0.1)]
    parameters = result["parameters"]
    running = result["history"]["running"]

    print("\n| 평가 방법 | dev 정확도 |")
    print("|---|---|")

    predictions = predict(dev_X, parameters, running)
    print(f"| 이동 평균 | {np.mean(predictions == dev_y):.4%} |")

    for batch_size in (360, 4, 1):
        predictions = []

        for start in range(0, dev_X.shape[1], batch_size):
            batch_X = dev_X[:, start:start + batch_size]
            predictions.append(
                predict(batch_X, parameters, running, training=True)
            )

        predictions = np.concatenate(predictions, axis=1)

        print(
            f"| dev {batch_size}개씩 계산한 통계 "
            f"| {np.mean(predictions == dev_y):.4%} |"
        )

        if batch_size == 1:
            single_predictions = predictions

    print(
        "\n1개씩 평가한 예측 분포:",
        np.unique(single_predictions, return_counts=True)
    )
    print("층당 이동 평균 갱신 횟수:", running["updates"])
    print(f"이동 평균 처음값의 가중치: {0.9 ** running['updates']:.3e}")

    for key in (("small", False, 0.1), ("he", False, 1.0)):
        parameters = results[key]["parameters"]
        _, ZL, caches = L_model_forward(dev_X, parameters)

        A_last_hidden = caches[-1][0][0]
        predictions = np.argmax(ZL, axis=0, keepdims=True)

        print(f"\n설정: {key}")
        print("마지막 은닉층 최솟값:", np.min(A_last_hidden))
        print("마지막 은닉층 최댓값:", np.max(A_last_hidden))
        print(
            "유닛별 샘플 간 표준편차의 최댓값:",
            np.max(np.std(A_last_hidden, axis=1))
        )
        print("예측 분포:", np.unique(predictions, return_counts=True))

    return results

def train_candidates(
    train_X, train_Y, dev_X, dev_Y, dev_y, candidates, method
):
    """
    Args:
        train_X (64, m_train): train inputs
        train_Y (10, m_train): train one-hot labels
        dev_X (64, m_dev): dev inputs
        dev_Y (10, m_dev): dev one-hot labels
        dev_y (1, m_dev): dev labels
        candidates (list): (learning_rate, epsilon) pairs
        method (string): 'grid', 'random' or 'fine'

    Returns:
        results (list): candidate results, best first
            learning_rate, epsilon, dev_accuracy, dev_cost (scalar)
            parameters (dict): Wl (n_cur, n_prev), bl (n_cur, 1)
            history (dict): epoch_costs/dev_costs (50,)
    """
    results = []

    print(f"\n{method} 후보별 결과")
    print("| # | α | ε | dev 정확도 | dev 비용 |")
    print("|---|---|---|---|---|")

    for i, (learning_rate, epsilon) in enumerate(candidates, 1):
        parameters, history = L_layer_model(
            train_X, train_Y, [64, 32, 10], dev_X, dev_Y,
            optimizer_method="adam",
            learning_rate=learning_rate,
            iterations=50,
            seed=3,
            mini_batch_size=64,
            initialize="he",
            beta1=0.9,
            beta2=0.999,
            epsilon=epsilon,
            use_bn=False,
        )

        dev_accuracy = np.mean(predict(dev_X, parameters) == dev_y)
        dev_cost = history["dev_costs"][-1]

        results.append({
            "learning_rate": learning_rate,
            "epsilon": epsilon,
            "parameters": parameters,
            "history": history,
            "dev_accuracy": dev_accuracy,
            "dev_cost": dev_cost,
        })

        print(
            f"| {i} | {learning_rate:.8e} | {epsilon:.8e} "
            f"| {dev_accuracy:.4%} | {dev_cost:.6f} |"
        )

    return sorted(
        results,
        key=lambda result: (-result["dev_accuracy"], result["dev_cost"])
    )

def experiment_3():
    train_X, dev_X, test_X, train_y, dev_y, test_y = load_data()

    train_Y = one_hot(train_y, 10)
    dev_Y = one_hot(dev_y, 10)

    rng = np.random.default_rng(0)
    uniform_lr = rng.uniform(1e-4, 1, 1000)
    log_lr = 10 ** rng.uniform(-4, 0, 1000)

    intervals = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    print("\n| 구간 | 균등 | 로그 척도 |")
    print("|---|---|---|")

    for lo, hi in zip(intervals[:-1], intervals[1:]):
        uniform_count = np.sum((lo <= uniform_lr) & (uniform_lr < hi))
        log_count = np.sum((lo <= log_lr) & (log_lr < hi))

        print(
            f"| [{lo:.0e}, {hi:.0e}) "
            f"| {uniform_count} | {log_count} |"
        )

    grid = [
        (10.0 ** a, 10.0 ** e)
        for a in (-4, -3, -2, -1, 0)
        for e in (-8, -7, -6, -5, -4)
    ]

    rng = np.random.default_rng(0)
    r_lr = rng.uniform(-4, 0, 25)
    r_eps = rng.uniform(-8, -4, 25)
    rand = list(zip(10 ** r_lr, 10 ** r_eps))

    results = {}

    for method, candidates in (("grid", grid), ("random", rand)):
        results[method] = train_candidates(
            train_X, train_Y, dev_X, dev_Y, dev_y, candidates, method
        )

    print(
        "\n| 방법 | 학습 횟수 | 서로 다른 α의 개수 | 최고 dev 정확도 "
        "| 그때의 α, ε | dev 정확도 97% 이상인 후보 수 |"
    )
    print("|---|---|---|---|---|---|")

    for method in ("grid", "random"):
        group = results[method]
        best = group[0]

        num_lr = len({result["learning_rate"] for result in group})
        num_good = sum(result["dev_accuracy"] >= 0.97 for result in group)

        print(
            f"| {method} | {len(group)} | {num_lr} "
            f"| {best['dev_accuracy']:.4%} "
            f"| {best['learning_rate']:.8e}, {best['epsilon']:.8e} "
            f"| {num_good} |"
        )

    print("\n| α | ε 5개에서 dev 정확도의 최솟값 ~ 최댓값 |")
    print("|---|---|")

    grid_lr = sorted({
        result["learning_rate"] for result in results["grid"]
    })

    for learning_rate in grid_lr:
        accuracies = [
            result["dev_accuracy"]
            for result in results["grid"]
            if result["learning_rate"] == learning_rate
        ]

        print(
            f"| {learning_rate:.0e} "
            f"| {min(accuracies):.4%} ~ {max(accuracies):.4%} |"
        )

    top_five = results["random"][:5]
    lo = min(result["learning_rate"] for result in top_five)
    hi = max(result["learning_rate"] for result in top_five)

    print(f"\nlo: {lo:.8e}")
    print(f"hi: {hi:.8e}")

    rng = np.random.default_rng(1)
    r_lr = rng.uniform(np.log10(lo), np.log10(hi), 10)
    r_eps = rng.uniform(-8, -4, 10)
    fine = list(zip(10 ** r_lr, 10 ** r_eps))

    results["fine"] = train_candidates(
        train_X, train_Y, dev_X, dev_Y, dev_y, fine, "fine"
    )

    random_best = results["random"][0]
    fine_best = results["fine"][0]

    print(
        "\n범위를 좁힌 후 최고 dev 정확도: "
        f"{fine_best['dev_accuracy']:.4%}"
    )
    print(
        "random 최고 대비 변화: "
        f"{100 * (fine_best['dev_accuracy'] - random_best['dev_accuracy']):.4f}%p"
    )

    plt.figure()

    for method, marker in (("grid", "o"), ("random", "x"), ("fine", "^")):
        group = results[method]

        plt.scatter(
            [result["learning_rate"] for result in group],
            [100 * result["dev_accuracy"] for result in group],
            marker=marker,
            label=method,
        )

    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Dev accuracy (%)")
    plt.legend()
    plt.savefig("images/hyperparameter_search.png")
    plt.close()

    all_results = results["grid"] + results["random"] + results["fine"]

    best = min(
        all_results,
        key=lambda result: (-result["dev_accuracy"], result["dev_cost"])
    )

    parameters = best["parameters"]
    best["train_accuracy"] = np.mean(
        predict(train_X, parameters) == train_y
    )

    best["test_accuracy"] = np.mean(
        predict(test_X, parameters) == test_y
    )

    print("\n| α | ε | 학습 정확도 | dev 정확도 | 시험 정확도 |")
    print("|---|---|---|---|---|")
    print(
        f"| {best['learning_rate']:.8e} | {best['epsilon']:.8e} "
        f"| {best['train_accuracy']:.4%} "
        f"| {best['dev_accuracy']:.4%} "
        f"| {best['test_accuracy']:.4%} |"
    )

    results["best"] = best
    return results

def main():
    # experiment_1()
    # experiment_2()
    experiment_3()


if __name__ == "__main__":
    main()