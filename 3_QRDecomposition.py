import numpy as np

SIZE = 5
A = np.random.uniform(0, 10, (SIZE, SIZE))

# QR-разложение с помощью модифицрованного алгоритма ортогонализации Грама Шмидта:
def get_QR(matrix):
    Q = np.zeros((SIZE, SIZE))
    R = np.zeros((SIZE, SIZE))
    for j in range(SIZE):
        Q[:, j] = np.copy(matrix[:, j])
        for i in range(SIZE)[0:j]:
            if j == 0:
                break
            R[i, j] = np.transpose(Q[:, i]).dot(Q[:, j])
            Q[:, j] -= R[i, j] * Q[:, i]
        R[j, j] = np.sqrt(np.transpose(Q[:, j]).dot(Q[:, j]))
        if R[j, j] == 0:
            print("a" + str(j + 1) + " линейно зависим ai, i = 1.." + str(j))
            exit()
        Q[:, j] /= R[j, j]
    return Q, R


print("A")
print(A)
print("A=QR")
Q, R = get_QR(A)
print("Q")
print(Q)
print("R")
print(R)
print("transpQ*Q")
print(np.transpose(Q).dot(Q))
print("Q*R")
print(Q.dot(R))

# A*x = b => Q*R*x = b => x = invR*transpQ*b
b = np.random.uniform(0, 10, SIZE)
print("x = invR*transpQ*b")
print(np.linalg.inv(R).dot(np.transpose(Q)).dot(b))
print("x = invA*b")
print(np.linalg.inv(A).dot(b))

