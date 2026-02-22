import numpy as np
from scipy.linalg import blas, lapack


# TODO : JAX
def backward_pass(policy, problem, mode="nominal"):
    np.set_printoptions(precision=12, suppress=True)
    H = len(problem.states)

    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_action

    gx = problem.objective.gradient_state
    gu = problem.objective.gradient_action

    gxx = problem.objective.hessian_state_state
    guu = problem.objective.hessian_action_action
    gux = problem.objective.hessian_action_state

    if mode == "nominal":
        K = policy.K
        k = policy.k
    else:
        K = policy.K_candidate
        k = policy.k_candidate

    # Value function approximation
    P = policy.value.hessian
    p = policy.value.gradient

    # Action-value function approximation
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    Qxx = policy.action_value.hessian_state_state
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state

    P[-1] = gxx[-1]
    p[-1] = gx[-1]

    for t in reversed(range(H - 1)):
        fxF = np.asfortranarray(fx[t])
        pF = np.asfortranarray(p[t + 1])
        gxF = np.asfortranarray(gx[t])
        tmp_x = blas.dgemv(alpha=1.0, a=fxF.T, x=p[t + 1], trans=0)
        Qx[t] = gx[t] + tmp_x

        fuF = np.asfortranarray(fu[t])
        guF = np.asfortranarray(gu[t])
        tmp_u = blas.dgemv(alpha=1.0, a=fuF.T, x=p[t + 1], trans=0)
        Qu[t] = gu[t] + tmp_u

        PF = np.asfortranarray(P[t + 1])
        gxxF = np.asfortranarray(gxx[t])
        tmp_xx = blas.dgemm(alpha=1.0, a=fxF.T, b=PF, trans_a=False, trans_b=False)
        Qxx[t] = gxxF + blas.dgemm(alpha=1.0, a=tmp_xx, b=fxF)

        guuF = np.asfortranarray(guu[t])
        tmp_uu = blas.dgemm(alpha=1.0, a=fuF.T, b=PF, trans_a=False, trans_b=False)
        Quu[t] = guuF + blas.dgemm(alpha=1.0, a=tmp_uu, b=fuF)

        guxF = np.asfortranarray(gux[t])
        tmp_ux = blas.dgemm(alpha=1.0, a=fuF.T, b=PF, trans_a=False, trans_b=False)
        Qux[t] = guxF + blas.dgemm(alpha=1.0, a=tmp_ux, b=fxF)

        # Cholesky Decompose
        QuuF = np.asfortranarray(Quu[t])
        U, info = lapack.dpotrf(QuuF, lower=0)  # 'U' → upper triangular
        if info != 0:
            raise RuntimeError(f"Cholesky failed at t={t}, info={info}")

        QuxF = np.asfortranarray(Qux[t])
        K[t], info = lapack.dpotrs(U, QuxF, lower=0)
        if info != 0:
            raise RuntimeError(f"potrs failed for K at t={t}, info={info}")
        K[t] *= -1.0

        QuF = np.asfortranarray(Qu[t])
        k[t], info = lapack.dpotrs(U, QuF, lower=0)
        if info != 0:
            raise RuntimeError(f"potrs failed for k at t={t}, info={info}")
        k[t] *= -1.0

        P[t] = Qxx[t] + K[t].T @ Quu[t] @ K[t] + K[t].T @ Qux[t] + Qux[t].T @ K[t]

        p[t] = Qx[t] + K[t].T @ Quu[t] @ k[t] + K[t].T @ Qu[t] + Qux[t].T @ k[t]

    return policy, problem
