from copy import deepcopy
import numpy as np

# Тут немного криво, т.к. сначала начинал на простых вложенных списках, а потом перешел на numpy
# поэтому A_arr - матрица из вложенных списков, A - та же матрица, но в numpy.array
# SIZE = 3
# A_arr = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]
# A = np.array(A_arr)
#

# A = np.random.uniform(0, 10, (SIZE, SIZE))
# A_arr = list(A)

SIZE = 5
A_arr = np.random.uniform(-20, 21, (SIZE, SIZE))
A = np.array(A_arr)
A[:, 1] = 2 * A[:, 0]
A[:, 3] = 3 * A[:, 0] - A[:, 2]
A[:, 4] = -2 * A[:, 0] + A[:, 2]
A_arr = np.array(A)
print(A)


def get_PAQ_LU(matrix):
    n = 0
    C = deepcopy(matrix)
    P = list(np.eye(SIZE))
    Q = list(np.eye(SIZE))

    for i in range(SIZE):
        pivotValue = 0
        pivot1 = -1
        pivot2 = -1
        for row in range(SIZE)[i:SIZE]:
            for column in range(SIZE)[i:SIZE]:
                if np.fabs(C[row][column]) > pivotValue:
                    pivotValue = np.fabs(C[row][column])
                    pivot1 = row
                    pivot2 = column
        if pivotValue != 0:
            if pivot1 != i:
                n += 1
                swap_rows(P, pivot1, i)
                swap_rows(C, pivot1, i)
            if pivot2 != i:
                n += 1
                swap_columns(Q, pivot2, i)
                swap_columns(C, pivot2, i)
            for j in range(SIZE)[i + 1:SIZE]:
                C[j][i] /= C[i][i]
                for s in range(SIZE)[i + 1:SIZE]:
                    C[j][s] -= C[j][i] * C[i][s]

    return C, P, Q, n


def swap_columns(your_list, pos1, pos2):
    for item in your_list:
        item[pos1], item[pos2] = item[pos2], item[pos1]


def swap_rows(input_array, row1, row2):
    temp = np.copy(input_array[row1][:])
    input_array[row1][:] = input_array[row2][:]
    input_array[row2][:] = temp


print("PAQ=LU")
C, P, Q, n = get_PAQ_LU(A_arr)

print("A")
print(np.array(A_arr))

# C = L + U - E
print("C")
print(np.array(C))

print("P")
print(np.array(P))

print("Q")
print(np.array(Q))

print("L")
L = np.tril(np.array(C), -1) + np.identity(SIZE)
print(L)

print("U")
U = np.triu(np.array(C), 0)
print(U)

print("L*U")
print(L.dot(U))

print("P*A*Q")
PAQ = np.array(P).dot(np.array(A_arr).dot(np.array(Q)))
print(PAQ)


def trapez_rank(matrix):
    rank = SIZE
    for row in range(SIZE):
        if row_zero(matrix[SIZE - 1 - row]):
            rank -= 1
        else:
            break
    return rank


def row_zero(row):
    for i in range(SIZE):
        if abs(row[i]) - 0.0000001 > 0:
            return False
    return True


print("Rank")
rank = trapez_rank(U)
print(rank)


# Решение совместной системы

# A*x = b => P*A*Q*invQ*x = L*U*invQ*x = P*b
def eq_LUx_b(L, U, P, Q, b):
    b = list(np.array(P).dot(np.array(b).transpose()))
    y = []
    for i in range(SIZE):
        k = b[i]
        for j in range(SIZE)[0:i]:
            k -= L[i][j] * y[j]
        y.append(k)
    x = []
    for i in range(SIZE):
        k = y[SIZE - i - 1]
        for j in range(SIZE)[0:i]:
            k -= U[SIZE - i - 1][SIZE - j - 1] * x[j]
        x.append(k / U[SIZE - i - 1][SIZE - i - 1])
    x.reverse()
    x = np.array(Q).dot(np.array(x))
    return x


# Поиск частного решения СЛАУ с вырожденной матрицой
def solve_degen(L, U, P, Q, b, rank):
    cU = np.copy(U)
    g = np.linalg.inv(L).dot(P).dot(b)
    y = list(np.zeros(SIZE))
    for i in range(rank)[::-1]:
        g[i] = g[i] / cU[i, i]
        cU[i, :] /= cU[i, i]
        for j in range(i):
            g[j] -= g[i] * cU[j, i]
            cU[j, :] -= cU[i, :] * cU[j, i]
        y[i] = g[i]
    x = np.array(Q).dot(np.array(y))
    return x


print("b")
b = np.random.uniform(0, 10, SIZE)
# b = A[:, 0]
print(b)
Ab = np.zeros((SIZE, SIZE + 1))
for i in range(SIZE):
    Ab[:, i] = A[:, i]
Ab[:, SIZE] = b
print("Расширенная матрица А")
print(Ab)
print(np.linalg.matrix_rank(Ab))
if np.linalg.matrix_rank(Ab) == rank:
    if rank == SIZE:
        print("A*x = b")
        print(eq_LUx_b(L, U, P, Q, b))
    else:
        print("x")
        x = solve_degen(L, U, P, Q, b, rank)
        print(x)
        print("check(b)")
        print(A.dot(x))

else:
    print("Система не совместна")
