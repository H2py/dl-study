import time
import numpy as np
import matplotlib.pyplot as plt

MILLION = 1000000

def compare_for_and_dot(arr1, arr2):
    result_for, result_dot = 0, 0
    
    tic = time.time()
    for i in range(arr1.shape[0]):
        result_for += arr1[i] * arr2[i]
    toc = time.time()

    print(f"Time for For loop : {1000 * (toc - tic)} results : {result_for}")

    tic = time.time()
    result_dot = np.dot(arr1, arr2)
    toc = time.time()

    print(f"Time for dot product : {1000*(toc - tic)} results : {result_dot}")

def relu(x):
    return np.maximum(x, 0)

def main():
    # ------ Problem 1 ------
    # arr1 = np.random.randn(MILLION)
    # arr2 = np.random.randn(MILLION)

    # compare_for_and_dot(arr1, arr2)


    # ------ Problem 2 ------
    # arr = np.random.randint(1, 10, size = (3, 4))
    # print(f"Original array : \n {arr}")
    # arr_sum = np.sum(arr, axis=0)
    # print(f"Sum of arr : {arr_sum}")
    # arr = arr / arr_sum
    # print(f"Array after division : \n {arr}")

    # ------ Problem 3 ------
    # a = np.random.randn(5)
    # b = np.random.randn(5, 1)

    # print(f"a shape : {a.shape}\nb shape : {b.shape}\n")
    # print(f"a.T shape : {a.T.shape}\nb.T shape : {b.T.shape}\n")
    # print(f"a dot product : {np.dot(a, a.T)}\nb dot product : {np.dot(b, b.T)}")

    # ------ Problem 4 ------
    x = np.linspace(-5, 5, 200)
    relu_one = relu(x)

    plt.figure(figsize=(8, 4))

    # 4-1
    # plt.plot(x, relu_one, label="relu(x)") 

    # 4-2
    # plt.plot(x, relu(x + 2), label="relu(x + 2)")
    # plt.plot(x, relu(x - 1), label="relu(x - 1)")
    # plt.plot(x, relu(-2*x + 3), label="relu(-2x + 3)")

    np.random.seed(42)

    w = np.random.randn(50)
    b = np.random.randn(50)
    c = np.random.randn(50)

    for name, n in [("relu_3", 3), ("relu_50", 50), ("linear_50", 50)]:
        y = np.zeros_like(x)

        for i in range(n):
            z = w[i] * x + b[i]

            if name.startswith("relu"):
                y += c[i] * relu(z)
            else:
                y += c[i] * z
        plt.figure()
        plt.plot(x, y)
        plt.title(name)
        plt.grid(True)
        plt.savefig(f"{name}.png", bbox_inches="tight")
        plt.show()
        plt.close()



if __name__ == "__main__":
    main()