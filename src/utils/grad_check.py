# 미분의 정의로 기울기를 근사하고, 직접 유도한 기울기와 비교한다.
# ∂J/∂θ ≈ ( J(θ + ε) − J(θ − ε) ) / (2ε)

# 절차는 다음과 같다.

# 1. `theta` 의 각 원소마다 `epsilon` 을 더한 경우와 뺀 경우의 cost를 계산해 수치 기울기를 만든다.
# 2. 상대 오차를 계산한다.
#    `np.linalg.norm(수치 - 해석) / (np.linalg.norm(수치) + np.linalg.norm(해석))`
# 3. **결과가 1e-7 보다 작으면 통과**한다. 1e-5 보다 크면 역전파에 버그가 있다.

import numpy as np

def flatten_params(w, b):
    return np.append(w.flatten(), b)

def unflatten_params(flat_params, w_shape):
    w = flat_params[:-1].reshape(w_shape)
    b = flat_params[-1]

    return w,b


def grad_check(cost_fn, theta, analytic_grad, epsilon=1e-7):
    """
    cost_fn       : theta(1차원 벡터)를 받아 스칼라 cost를 반환하는 함수
    theta         : 모든 파라미터를 1차원으로 펼친 벡터, shape (n,)
    analytic_grad : 역전파로 구한 기울기, theta와 같은 shape
    반환          : 상대 오차 (스칼라)
    """
    n = theta.shape[0]
    numerical_grad = np.zeros_like(theta, dtype=float)

    for i in range(n):
        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon

        numerical_grad[i] = (cost_fn(theta_plus) - cost_fn(theta_minus)) / (2 * epsilon)

    result = np.linalg.norm(numerical_grad - analytic_grad) / (np.linalg.norm(numerical_grad) + np.linalg.norm(analytic_grad))
    return result
