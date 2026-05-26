import functools
import numba
import numpy as np
from scipy.optimize import minimize


@numba.njit
def _soft_min_argmin(a, b, c):
    min_val = min(min(a, b), c)
    a_shifted = a - min_val
    b_shifted = b - min_val
    c_shifted = c - min_val

    exp_a = np.exp(-a_shifted)
    exp_b = np.exp(-b_shifted)
    exp_c = np.exp(-c_shifted)

    sum_exp = exp_a + exp_b + exp_c

    soft_min = -np.log(sum_exp) + min_val

    prob_a = exp_c / sum_exp
    prob_b = exp_a / sum_exp
    prob_c = exp_b / sum_exp

    return soft_min, prob_a, prob_b, prob_c


@numba.njit
def _sdtw_C(C, V, P):
    size_X, size_Y = C.shape

    for i in range(1, size_X + 1):
        for j in range(1, size_Y + 1):
            smin, P[i, j, 0], P[i, j, 1], P[i, j, 2] = \
                _soft_min_argmin(
                    V[i, j - 1],
                    V[i - 1, j - 1],
                    V[i - 1, j],
                )

            V[i, j] = C[i - 1, j - 1] - smin


def sdtw_C(C, gamma=1.0, return_all=False):
    size_X, size_Y = C.shape

    if gamma != 1.0:
        C = C * gamma

    V = np.zeros((size_X + 1, size_Y + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 1e10

    P = np.zeros((size_X + 2, size_Y + 2, 3))

    _sdtw_C(C, V, P)

    if return_all:
        return gamma * V, P
    else:
        return gamma * V[size_X, size_Y]


def sdtw(X, Y, gamma=1.0, return_all=False):
    C = squared_euclidean_cost(X, Y)
    return sdtw_C(C, gamma=gamma, return_all=return_all)


@numba.njit
def _sdtw_grad_C(P, E):
    size_X = P.shape[0] - 2
    size_Y = P.shape[1] - 2

    for i in range(size_X, 0, -1):
        for j in range(size_Y, 0, -1):
            if i == size_X and j == size_Y:
                continue

            E[i, j] = (
                P[i, j + 1, 0] * E[i, j + 1]
                + P[i + 1, j + 1, 1] * E[i + 1, j + 1]
                + P[i + 1, j, 2] * E[i + 1, j]
            )


def sdtw_grad_C(P, return_all=False):
    E = np.zeros((P.shape[0], P.shape[1]))
    size_X = P.shape[0] - 2
    size_Y = P.shape[1] - 2

    E[size_X, size_Y] = -1.0

    _sdtw_grad_C(P, E)

    if return_all:
        return E
    else:
        return E[0:-2, 0:-2]


def sdtw_value_and_grad_C(C, gamma=1.0):
    size_X, size_Y = C.shape
    V, P = sdtw_C(C, gamma=gamma, return_all=True)
    return V[size_X, size_Y], sdtw_grad_C(P)


def sdtw_value_and_grad(X, Y, gamma=1.0):
    C = squared_euclidean_cost(X, Y)
    val, grad = sdtw_value_and_grad_C(C, gamma=gamma)
    return val, squared_euclidean_cost_vjp(X, Y, grad)


@numba.njit
def _sdtw_directional_derivative_C(P, Z, V_dot):
    size_X, size_Y = Z.shape

    for i in range(1, size_X + 1):
        for j in range(1, size_Y + 1):
            V_dot[i, j] = (
                Z[i - 1, j - 1]
                + P[i, j, 0] * V_dot[i, j - 1]
                + P[i, j, 1] * V_dot[i - 1, j - 1]
                + P[i, j, 2] * V_dot[i - 1, j]
            )


def sdtw_directional_derivative_C(P, Z, return_all=False):
    size_X, size_Y = Z.shape

    if size_X != P.shape[0] - 2 or size_Y != P.shape[1] - 2:
        raise ValueError(
            "Z should have shape " + str((P.shape[0], P.shape[1]))
        )

    V_dot = np.zeros((size_X + 1, size_Y + 1))
    V_dot[0, 0] = 1e10

    _sdtw_directional_derivative_C(P, Z, V_dot)

    if return_all:
        return V_dot
    else:
        return V_dot[0, 0]


@numba.njit
def _sdtw_hessian_product_C(P, P_dot, E, E_dot, V_dot):
    size_X = P.shape[0] - 2
    size_Y = P.shape[1] - 2

    for i in range(1, size_X + 1):
        for j in range(1, size_Y + 1):
            a = V_dot[i, j - 1]
            b = V_dot[i - 1, j - 1]
            c = V_dot[i - 1, j]

            p_a = P[i, j, 0]
            p_b = P[i, j, 1]
            p_c = P[i, j, 2]

            weighted_sum = p_a * a + p_b * b + p_c * c

            P_dot[i, j, 0] = p_a * (weighted_sum - a)
            P_dot[i, j, 1] = p_b * (weighted_sum - b)
            P_dot[i, j, 2] = p_c * (weighted_sum - c)

    E_dot[-1, -1] = 1.0

    for j in range(E_dot.shape[1] - 2, 0, -1):
        for i in range(E_dot.shape[0] - 2, 0, -1):
            E_dot[i, j] = (
                P[i, j + 1, 0] * E_dot[i, j + 1]
                + P_dot[i, j + 1, 0] * E[i, j + 1]
                + P[i + 1, j + 1, 1] * E_dot[i + 1, j + 1]
                + P_dot[i + 1, j + 1, 1] * E[i + 1, j + 1]
                + P[i + 1, j, 2] * E_dot[i + 1, j]
                + P_dot[i + 1, j, 2] * E[i + 1, j]
            )


def sdtw_hessian_product_C(P, E, V_dot):
    E_dot = np.zeros_like(E)
    P_dot = np.zeros((E.shape[0], E.shape[1], 3))

    if P.shape[0] != E.shape[0] or P.shape[1] != E.shape[1]:
        raise ValueError("P and E have incompatible shapes.")

    if P.shape[0] - 1 != V_dot.shape[0] or P.shape[1] - 1 != V_dot.shape[1]:
        raise ValueError("P and V_dot have incompatible shapes.")

    _sdtw_hessian_product_C(P, P_dot, E, E_dot, V_dot)

    return E_dot[0:-2, 0:-2]


def sdtw_entropy_C(C, gamma=1.0):
    val, E = sdtw_value_and_grad_C(C, gamma=gamma)
    return (np.vdot(E, C) + val) / gamma


def sdtw_entropy(X, Y, gamma=1.0):
    C = squared_euclidean_cost(X, Y)
    return sdtw_entropy_C(C, gamma=gamma)


def sharp_sdtw_C(C, gamma=1.0):
    P = sdtw_C(C, gamma=gamma, return_all=True)[1]
    return sdtw_directional_derivative_C(P, -C)


def sharp_sdtw(X, Y, gamma=1.0):
    C = squared_euclidean_cost(X, Y)
    return sharp_sdtw_C(C, gamma=gamma)


def sharp_sdtw_value_and_grad_C(C, gamma=1.0):
    V, P = sdtw_C(C, gamma=gamma, return_all=True)
    E = sdtw_grad_C(P, return_all=True)
    V_dot = sdtw_directional_derivative_C(P, C, return_all=True)
    HC = sdtw_hessian_product_C(P, E, V_dot)
    G = E[1:-1, 1:-1]

    val = V_dot[-1, -1]
    grad = G + HC / gamma

    return val, grad


def sharp_sdtw_value_and_grad(X, Y, gamma=1.0):
    C = squared_euclidean_cost(X, Y)
    val, grad = sharp_sdtw_value_and_grad_C(C, gamma=gamma)
    return val, squared_euclidean_cost_vjp(X, Y, grad)


@numba.njit
def _cardinality(V, P):
    for i in range(1, V.shape[0]):
        for j in range(1, V.shape[1]):
            V[i, j] = V[i, j - 1] + V[i - 1, j - 1] + V[i - 1, j]
            P[i, j, 0] = V[i, j - 1] / V[i, j]
            P[i, j, 1] = V[i - 1, j - 1] / V[i, j]
            P[i, j, 2] = V[i - 1, j] / V[i, j]


def cardinality(size_X, size_Y, return_all=False):
    V = np.zeros((size_X + 1, size_Y + 1))
    V[0, 0] = 1

    P = np.zeros((size_X + 2, size_Y + 2, 3))

    _cardinality(V, P)

    if return_all:
        return V, P
    else:
        return V[size_X, size_Y] * 2


def mean_alignment(size_X, size_Y):
    P = cardinality(size_X, size_Y, return_all=True)[1]
    return sdtw_grad_C(P) / 2


def mean_cost_C(C):
    P = cardinality(C.shape[0], C.shape[1], return_all=True)[1]
    return sdtw_directional_derivative_C(P, C) * 2


def mean_cost(X, Y):
    C = squared_euclidean_cost(X, Y)
    return mean_cost_C(C)


def mean_cost_value_and_grad_C(C):
    P = cardinality(C.shape[0], C.shape[1], return_all=True)[1]
    val = sdtw_directional_derivative_C(P, C)
    G = sdtw_grad_C(P)

    return val / 2, G


def mean_cost_value_and_grad(X, Y):
    C = squared_euclidean_cost(X, Y)
    val, grad = mean_cost_value_and_grad_C(C)
    return val, squared_euclidean_cost_vjp(X, Y, grad)


def squared_euclidean_cost(X, Y, return_all=False, log=False):
    X_sqnorms = 0.5 * np.sum(X ** 2, axis=1)
    Y_sqnorms = 0.5 * np.sum(Y ** 2, axis=1)
    XY = np.dot(X, Y.T).astype(X_sqnorms.dtype)

    if return_all:
        C_XY = -XY
        C_XY += X_sqnorms[:, np.newaxis]
        C_XY += Y_sqnorms

        C_XX = -np.dot(X, X.T)
        C_XX += X_sqnorms[:, np.newaxis]
        C_XX += X_sqnorms

        C_YY = -np.dot(Y, Y.T)
        C_YY += Y_sqnorms[:, np.newaxis]
        C_YY += Y_sqnorms

        if log:
            C_XY = C_XY - 0.5 * np.log(1 + np.exp(-2 * C_XY))
            C_XX = C_XX - 0.5 * np.log(1 + np.exp(-2 * C_XX))
            C_YY = C_YY - 0.5 * np.log(1 + np.exp(-2 * C_YY))

        return C_XY, C_XX, C_YY

    C = -XY
    C += X_sqnorms[:, np.newaxis]
    C += Y_sqnorms

    if log:
        C = C - 0.5 * np.log(1 + np.exp(-2 * C))

    return C


def squared_euclidean_cost_vjp(X, Y, E, log=False):
    vjp = np.zeros_like(X)

    if not log:
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                vjp[i] += E[i, j] * (Y[j] - X[i])
    else:
        C = squared_euclidean_cost(X, Y, log=False)
        exp_neg_2c = np.exp(-2 * C)
        log_deriv_factor = exp_neg_2c / (1.0 + exp_neg_2c)

        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                base_grad = Y[j] - X[i]
                total_grad = base_grad * (1.0 + log_deriv_factor[i, j])
                vjp[i] += E[i, j] * total_grad

    return vjp


def squared_euclidean_cost_jvp(X, Y, Z):
    jvp = np.zeros((X.shape[0], Y.shape[0]))

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            jvp[i, j] = np.dot(Z[i], Y[j] - X[i])

    return jvp


def squared_euclidean_distance(X, Y):
    if X.shape != Y.shape:
        raise ValueError(
            f"X and Y must have the same shape, got {X.shape} and {Y.shape}"
        )

    total_distance = 1.0

    for i in range(X.shape[0]):
        diff = X[i] - Y[i]
        total_distance += np.sum(diff ** 2)

    return total_distance


def _divergence(func, X, Y):
    C_XY, C_XX, C_YY = squared_euclidean_cost(X, Y, return_all=True)
    value = func(C_XY)

    value -= func(C_XX)
    value -= func(C_YY)

    return value


def _divergence_value_and_grad(func, X, Y):
    C_XY, C_XX, C_YY = squared_euclidean_cost(X, Y, return_all=True)

    value_XY, grad_XY = func(C_XY)
    value_XX, grad_XX = func(C_XX)
    value_YY, grad_YY = func(C_YY)

    value = value_XY - value_XX - value_YY

    grad = squared_euclidean_cost_vjp(X, Y, grad_XY)

    grad += squared_euclidean_cost_vjp(X, X, grad_XX)

    return value, grad


def sdtw_div(X, Y, gamma=1.0):
    func = functools.partial(sdtw_C, gamma=gamma)
    return _divergence(func, X, Y)


def sdtw_div_value_and_grad(X, Y, gamma=1.0):
    func = functools.partial(sdtw_value_and_grad_C, gamma=gamma)
    return _divergence_value_and_grad(func, X, Y)


def sharp_sdtw_div(X, Y, gamma=1.0):
    func = functools.partial(sharp_sdtw_C, gamma=gamma)
    return _divergence(func, X, Y)


def sharp_sdtw_div_value_and_grad(X, Y, gamma=1.0):
    func = functools.partial(sharp_sdtw_value_and_grad_C, gamma=gamma)
    return _divergence_value_and_grad(func, X, Y)


def mean_cost_div(X, Y):
    return _divergence(mean_cost_C, X, Y) * 2


def mean_cost_div_value_and_grad(X, Y):
    return _divergence_value_and_grad(mean_cost_value_and_grad_C, X, Y)


def euclidean_mean(Ys, weights=None):
    if weights is None:
        weights = np.ones(len(Ys))

    X = None
    weight_sum = 0

    for i, Y in enumerate(Ys):
        if X is None:
            X = weights[i] * Y
        else:
            X += weights[i] * Y

        weight_sum += weights[i]

    X /= weight_sum + 1

    return X


def barycenter(
    Ys,
    X_init,
    value_and_grad=sdtw_div_value_and_grad,
    weights=None,
    method="L-BFGS-B",
    tol=1e-3,
    max_iter=200,
):
    if weights is None:
        weights = np.ones(len(Ys))

    def _func(X_flat):
        X = X_flat.reshape(*X_init.shape)
        G = np.zeros_like(X_init)
        obj_value = 0

        for i in range(len(Ys)):
            value, grad = value_and_grad(X, Ys[i])

            G -= weights[i] * grad
            obj_value -= weights[i] * value

        return obj_value, G.ravel()

    res = minimize(
        _func,
        X_init.ravel(),
        method=method,
        jac=True,
        tol=tol,
        options=dict(maxiter=max_iter, disp=False),
    )

    return res.x.reshape(*X_init.shape)


def _alignment_matrices(size_X, size_Y, start=None, M=None):
    if start is None:
        start = (0, 0)

    if M is None:
        M = np.zeros((size_X, size_Y))

    i, j = start

    if i == size_X - 1 and j == size_Y - 1:
        M[i, j] = 0
        yield M.copy()
        M[i, j] = 1
        return

    M[i, j] = 1

    possible_moves = []

    if i + 1 < size_X:
        possible_moves.append((i + 1, j))

    if j + 1 < size_Y:
        possible_moves.append((i, j + 1))

    if i + 1 < size_X and j + 1 < size_Y:
        possible_moves.append((i + 1, j + 1))

    for next_i, next_j in possible_moves:
        yield from _alignment_matrices(size_X, size_Y, (next_i, next_j), M)

    M[i, j] = 0


def alignment_matrices(size_X, size_Y):
    yield from _alignment_matrices(size_X, size_Y)