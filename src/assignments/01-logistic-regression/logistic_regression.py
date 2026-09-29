import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from utils.grad_check import grad_check, flatten_params, unflatten_params

def save_cost_plot(costs, iter):
    iterations = np.arange(0, iter, 100)
    plt.figure(figsize=(10,4))
    plt.plot(iterations, costs)
    plt.xlabel("Iterations")
    plt.ylabel("Costs")
    plt.grid(True)
    plt.savefig("./images/cost_plot_6.png") 

def sigmoid(z):
    """sigmoid 함수

    Args:
        z (np.array): (1, m)

    Returns:
        np.array : (1, m)
    """
    z = np.asarray(z)
    result = np.zeros_like(z, dtype=np.float64)
    
    mask = z > 0
    
    result[mask] = 1 / (1 + np.exp(-z[mask]))
    
    result[~mask] = np.exp(z[~mask]) / (1 + np.exp(z[~mask]))
    
    return result

def initialize(n):
    """w, b 초기화함수

    Args:
        n (integer): 이미지 한 장의 픽셀 개수

    Returns:
        (np.array, scalar) : ((n, 1), scalar)
    """
    w = np.zeros((n, 1))
    b = 0.0

    return w, b

# def propagate(w, b, X, Y):
#     """순전파 및 cost, dw, db를 계산하는 함수

#     Args:
#         w (np.array) : (n, 1)
#         b (float): bias
#         X (np.array): (n, m)
#         Y (np.array): (1, m)

#     Returns:
#         (dict, float): grads{dw, db}, cost
#     """

#     m = X.shape[1]
#     a = sigmoid(np.dot(w.T, X) + b)
#     cost_equation = Y * np.log(a) + (1 - Y) * np.log(1 - a)
#     cost = -1/m * np.sum(cost_equation)

#     dw = 1/m * np.dot(X, (a-Y).T)
#     db = 1/m * np.sum(a-Y)

#     grads = {"dw" : dw, "db" : db}

#     return grads, cost

def propagate(w, b, X, Y):
    # 입력: w (n,1), b 스칼라, X (n,m), Y (1,m)
    # 출력: grads {dw (n,1), db 스칼라}, cost 스칼라
    m = X.shape[1]

    z = np.dot(w.T, X) + b
    a = sigmoid(z)

    cost = np.sum(np.logaddexp(0, (1 - 2 * Y) * z)) / m

    dw = np.dot(X, (a - Y).T) / m
    db = np.sum(a - Y) / m

    grads = {"dw": dw, "db": db}
    return grads, cost



def optimize(w, b, X, Y, num_iterations, learning_rate):
    """propagate에서 구한 dw, db를 이용하여 파라미터 최적화를 진행하는 함수 num_iterations 동안 cost가 줄어들도록 만듦

    Args:
        w (np.array) : (n, 1)
        b (float): bias
        X (np.array): (n, m)
        Y (np.array): (1, m)
        num_iterations (integer) : 학습 횟수
        learning_rate (float): 학습률

    Returns:
        (dict, dict, list): params{w, b}, grads{dw, db}, costs
    """

    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w -= learning_rate * dw
        b -= learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

    params = {"w": w, "b": b}

    return params, grads, costs

def predict(w, b, X):
    """ (1,m) np.array 반환, sigmoid 값 threshold 0.5를 기준으로 1 or 0 값을 내놓음

    Args:
        w (n, 1)
        b (scalar)
        X (n, m)
    """

    m = X.shape[1]
    predicted = np.zeros((1, m))

    a = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(m):
        if a[0,i] > 0.5:
            predicted[0,i] = 1
        else :
            predicted[0,i] = 0
    return predicted

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    """초기화, 학습, 예측을 진행하는 함수

    Args:
        X_train (np.array): (n, m_train)
        Y_train (np.array): (1, m_train)
        X_test (np.array): (n, m_test)
        Y_test (np.array): (1, m_test)
        num_iterations (int): 학습 횟수
        learning_rate (float): 학습률

        이때, n은 특징 수, m_train과 m_test는 각각 학습 or 테스트 샘플 수

    Returns:
        dict:
            params: w (n, 1), b (scalar)
            grads: dw (n, 1), db (scalar)
            costs: list
            test_accuracy: 테스트 정확도 (scalar)
            train_accuracy: 학습 정확도 (scalar)
    """

    w, b = initialize(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    w = params["w"]
    b = params["b"]

    test_predicted = predict(w, b, X_test)
    train_predicted = predict(w, b, X_train)

    # 1 or 0 값이 들어 있음

    test_accuracy = (np.sum(test_predicted == Y_test) / Y_test.shape[1]) * 100
    train_accuracy = (np.sum(train_predicted == Y_train) / Y_train.shape[1]) * 100

    result = {"params" : params, "grads": grads, "costs": costs, "test_accuracy" : test_accuracy, "train_accuracy" : train_accuracy}

    return result



def main():
    digits = load_digits()
    mask = (digits.target == 5) | (digits.target == 3)
    X, y = digits.data[mask], digits.target[mask]        # (360, 64), (360,)

    y = (y == 5).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

    X_train = X_tr.T / 16.0        # (64, 288)
    X_test  = X_te.T / 16.0        # (64, 72)
    Y_train = y_tr.reshape(1, -1)  # (1, 288)
    Y_test  = y_te.reshape(1, -1)  # (1, 72)

    # 기울기 검증
    X_small = X_train[:, :10]
    Y_small = Y_train[:, :10]

    w, b = initialize(X_small.shape[0])
    w_shape = w.shape

    def cost_fn(theta):
        check_w, check_b = unflatten_params(theta, w_shape)
        _, cost = propagate(check_w, check_b, X_small, Y_small)
        return cost

    theta = flatten_params(w, b)

    grads, _ = propagate(w, b, X_small, Y_small)
    analytic_grad = flatten_params(grads["dw"], grads["db"])

    difference = grad_check(cost_fn, theta, analytic_grad)
    print(f"기울기 검증 상대 오차: {difference:.10e}")
    print(f"기울기 검증 통과: {difference < 1e-7}")

    # 이후 기존 sigmoid 검사와 model 학습 코드

    # print(f"X_train shape = {X_train.shape}")
    # print(f"X_train [0,:] = \n {X_train[:,0]}") # X_train[:,i] = i번째 이미지에 대한 모든 행, 열 pixel value들 나열
    # print(f"X_train = \n {X_train[0,:]}") # 의미없는 값, 각 이미지의 0번째 행 pixel value를 가져옴
    #최댓값 최솟값은 각각 1, 0으로 고정 (pixel value)

    test_1 = sigmoid([-1000, 0, 1000])
    test_2 = sigmoid([-100, 0, 100])
    print(f"Sigmoid test done : {test_1, test_2}")

    iter = 2000
    lr = 500000000000
    result = model(X_train, Y_train, X_test, Y_test, num_iterations=iter, learning_rate=lr)
    
    costs = result["costs"]
    
    print(f"Train 정확도: {result['train_accuracy']} Iteration : {iter} Learning rate : {lr}")
    print(f"Test 정확도: {result['test_accuracy']}")
    
    # save_cost_plot(costs, iter)


if __name__ == "__main__":
    main()