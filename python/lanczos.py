import scipy.linalg as la
import numpy as np
import scipy as sp

from lanczos_util import estimate_orthonormality, zgemm, orthogonalize, hermitian_csr_matmat

class Hermitian_operator(sp.sparse.linalg.LinearOperator):
    def __init__(self, H):
        self.shape = H.shape
        self.U = sp.sparse.tril(H, k = 0, format = 'csr')
        self.dtype = H.dtype

    def _matmat(self, m):
        return hermitian_csr_matmat(self.shape[0], self.U.data, self.U.indices, self.U.indptr, m)

    def _adjoint(self):
        return self


class LanczosGen:
    import numpy as np

    def __init__(self, A, v, krylov_size, beta=None, v_old=None, start=0):
        self.block = len(v.shape) == 2
        if not self.block:
            v = v.reshape((len(v), 1))
        n = v.shape[1]
        if beta is None:
            beta = np.zeros((v.shape[1], v.shape[1]), dtype=complex)
        elif not self.block:
            beta = beta * np.eye((n, n))
        if v_old is None:
            v_old = np.zeros_like(v)
        elif not self.block:
            v_old = v_old.reshape((len(v_old), 1))
        self.A = A
        self.v = v
        self.v_old = v_old
        self.beta = beta
        self.i = start * n
        self.krylov_size = krylov_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.i + 1 * self.v.shape[1] > self.krylov_size:
            raise StopIteration
        self.i += self.v.shape[1]

        v_new = self.A @ self.v
        alpha = np.conj(self.v.T) @ v_new
        v_new -= self.v @ alpha + self.v_old @ np.conj(self.beta.T)

        self.v_old = self.v
        q, r = sp.linalg.qr(v_new, mode="full", overwrite_a = True, check_finite = False)
        self.v, self.beta = q[:, :self.v.shape[1]], r[:self.beta.shape[0], :]

        return alpha, self.beta, self.v


def estimate_orthonormality_old(
    W, alphas, betas, eps=np.finfo(float).eps, N=1, rng=np.random.default_rng()
):
    """Estimate the overlap between obtained Lanczos vectors at a ceratin iteration.
    W, alphas and betas contain all the required information for estimating the overlap.
    The stats dictionary contains the following keys:
    * w_bar  - The absolute values of the estimated overlaps for the second row
               of W.
    Parameters:
    W      - Array containing the two latest etimates of overlap. Dimensions (2, i+1, n, n)
    alphas - Array containing the (block) diagonal elements obtained from the
             (block) Lanczos method. Dimensions (i+1, n, n)
    betas  - Array containing the (block) off diagonal elements obtained from the
             (block) Lanczos method. Dimensions (i+1, n, n)
    eps    - Precision of orthogonality. Default: machine precision
    A_norm - Estimate of the norm of the matrix A. Default: 1
    Returns:
    W_out  - Estimated overlaps of the last two vectors obtained from the (block)
             Lanczos method. Dimensions (2, i+1, n, n)
    """
    # i is the index of the latest calculated vector
    i = alphas.shape[0] - 1
    n = alphas.shape[1]
    W_out = np.zeros((2, i + 2, n, n), dtype=complex)
    w_bar = np.zeros((i + 2, n, n), dtype=complex)
    w_bar[i + 1, :, :] = np.identity(n)
    w_bar[i, :, :] = (
        eps
        * N
        * sp.linalg.solve_triangular(betas[i], betas[0], lower = False, trans = 'C', check_finite = False)
        # * sp.linalg.solve_triangular(np.conj(betas[i].T), betas[0], lower = True)
        # * 0.6
        # * rng.standard_normal(size=(n, n))
    )
    if i == 0:
        W_out[0, : i + 1] = W[1]
        W_out[1, : i + 2] = w_bar
        return W_out

    if n > 1:
        w_bar[0] = (
            W[1, 1] @ betas[0]
            + W[1, 0] @ alphas[0]
            - alphas[i] @ W[1, 0]
            - betas[i - 1] @ W[0, 0]
        )
        w_bar[0] = sp.linalg.solve_triangular(betas[i], w_bar[0], lower = False, trans = 'C', check_finite = False)
        # w_bar[0] = sp.linalg.solve_triangular(np.conj(betas[i].T), w_bar[0], lower = True)
        w_bar[1:i] = (
            W[1, 2 : i + 1] @ betas[1:i]
            + W[1, 1:i] @ alphas[1:i]
            - alphas[i][np.newaxis, :, :] @ W[1, 1:i]
            + W[1, 0 : i - 1] @ np.conj(np.transpose(betas[0 : i - 1], axes=[0, 2, 1]))
            - betas[i - 1][np.newaxis, :, :] @ W[0, 1:i]
        )
        for j in range(1, i):
            w_bar[j] = sp.linalg.solve_triangular(betas[i], w_bar[j], lower = False, trans = 'C', check_finite = False)
            # w_bar[j] = sp.linalg.solve_triangular(np.conj(betas[i].T), w_bar[j], lower = True)
        # w_bar[1:i] = np.linalg.solve(np.conj(betas[i].T)[np.newaxis, :, :], w_bar[1:i])
    elif n == 1:
        # For standard Lanczos, broadcasting is faster than looping
        w_bar[:i] = (
            W[1, 1 : i + 1] * betas[:i]
            + (alphas[:i] - alphas[i]) * W[1, :i]
            + np.append(
                np.zeros((1, 1, 1), dtype=complex),
                W[1, 0 : i - 1] * betas[0 : i - 1],
                axis=0,
            )
            - betas[i - 1] * W[0, :i]
        )
        w_bar[:i] = w_bar[:i] / betas[i]

    w_bar[:i] += (
        eps * (betas[i] + betas[:i]) # * 0.3 * rng.standard_normal(size=(i, n, n))
    )
    W_out[0, : i + 1] = W[1]
    W_out[1, : i + 2] = w_bar

    return W_out

def ric(
    W,
    alphas,
    betas,
    Q,
    q,
    force_reort=False,
    eps=np.finfo("float").eps,
    rng=np.random.default_rng(),
):
    """Reorthogonalization with improved convergence-check (RIC)
    Parameters:
    alphas    - Diagonal (blocks) of T_i. Dimensions (i, n, n)
    betas     - Off-diagonal (blocks) of T_i. Dimensions (i, n, n)
    Q         - Lanczos vectors. Dimensions (N, i*n + 2)
    q         - Vector (block) to reorthogonalize. Dimensions (N, n)
    eps       - Machine precision. Max overlap allowed is sqrt(eps)
    Returns:
    q_out     - The orthogonalized vector (block). Dimensions (N, n)
    """
    n = alphas.shape[1]
    l, s = eigsh(alphas, betas, eigvals_only=False)
    if alphas.shape[0] == 1:
        mask = np.array([True] * n)
    else:
        lp = eigsh(alphas[:-1], betas[:-1], eigvals_only=True)
        m1 = np.append(np.abs(lp - l[:-n]) < np.sqrt(eps), [[False] * n])
        m2 = np.append([False] * n, [np.abs(lp - l[n:]) < np.sqrt(eps)])
        mask = np.logical_or(m1, m2)
    y = Q @ s

    reort_mask = np.array(
        [mask[i * n : (i + 1) * n] for i in range(mask.shape[0] // n)]
    )
    block_mask = np.any(np.abs(W[1, :-1]) > np.sqrt(eps), axis=1)
    reset_mask = np.logical_or(reort_mask, block_mask)
    W[1, :-1][reset_mask] = (
        eps * 1.5 * rng.standard_normal(size=W[1, :-1][reset_mask].shape)
    )
    orthogonalize(q, y[:, mask].copy())
    return W, q, np.any(reort_mask)


def partial_reorthonormalization(
    W,
    Q,
    q,
    N=1,
    eps=np.finfo("float").eps,
    force_reort=None,
    rng=np.random.default_rng(),
):
    """Perform partial reorthonormalization as part of the (block) Lanczos method.
    Parameters:
    W      - E.T).T)stimated overlaps for the last two vector blocks obtained from the
             Lanczos method. Dimensions (2, i+1, n, n)
    Q      - Vector blocks previously obtained from the Lanczos method. Dimensions
             (N, i*n)
    betas  - Obtained off diagonal (block) elements obtained from the Lanczos
             method (only the two latest ones are used). Dimensions (i+1, n, n)
    A_norm - Estimate of the norm of the matrix A. Default: 1
    eps    - Precision of orthogonality. Default machine precision
    """
    n = q.shape[1]
    i = Q.shape[1] // n - 2
    reort = np.any(np.abs(W[1, :-1]) > np.sqrt(eps))
    if reort:
        # mask = np.any(np.abs(W[1, :-1]) > eps ** (3 / 4), axis=1)
        mask = np.array([[True] * n] * (i + 2))
    else:
        mask = np.array([[False] * n] * (i + 2))

    combined_mask = (
        np.logical_or(mask, force_reort) if force_reort is not None else mask
    )
    # Update orthonormality estimates
    W[1, :-1][combined_mask] = (
        eps # * 1.5 * rng.standard_normal(size=W[1, :-1][combined_mask].shape)
    )

    # orthogonalize(q=q, Q=Q[:, combined_mask.flatten()].copy()),
    return (
        W,
        q - Q[:, combined_mask.flatten()] @ (np.conj(Q[:, combined_mask.flatten()].T) @ q),
        mask if reort else None,
    )


def eigsh(alphas, betas, eigvals_only=False, select="a", select_range=None):
    if alphas.shape[1] == 1:
        return la.eigh_tridiagonal(
            np.real(alphas.flatten()),
            np.real(betas[:-1].flatten()),
            eigvals_only=eigvals_only,
            select=select,
            select_range=select_range,
        )
    else:
        p = alphas.shape[1]
        bands = np.empty((p + 1, p * alphas.shape[0]), dtype=complex)
        for i in range(p + 1):
            if i == 0:
                diag = np.diagonal(alphas, offset=0, axis1=1, axis2=2).flatten()
            else:
                diag = []
                for j in range(alphas.shape[0]):
                    diag = np.append(
                        diag,
                        np.append(
                            np.diagonal(alphas[j], offset=-i),
                            [np.diagonal(betas[j], offset=p - i)],
                        ),
                    )
                diag = diag.flatten()

            bands[i, : diag.shape[0]] = diag
        return la.eig_banded(
            bands,
            lower=True,
            eigvals_only=eigvals_only,
            select=select,
            select_range=select_range,
            overwrite_a_band=True,
        )


def errors_scalar(alphas, betas, k):
    t, s = la.eigh_tridiagonal(alphas, betas[:-1])
    return np.abs(betas[-1] * s[-1, :k])


def errors_block(alphas, betas, k):
    n = alphas.shape[1]
    bands = np.empty((n + 1, n * alphas.shape[0]), dtype=complex)
    for i in range(n + 1):
        if i == 0:
            diag = np.diagonal(alphas, offset=0, axis1=1, axis2=2).flatten()
        else:
            diag = []
            for j in range(alphas.shape[0]):
                diag = np.append(
                    diag,
                    np.append(
                        np.diagonal(alphas[j], offset=-i),
                        [np.diagonal(betas[j], offset=n - i)],
                    ),
                )
            diag = diag.flatten()

        bands[i, : diag.shape[0]] = diag

    t, s = la.eig_banded(bands, lower=True)
    norms = betas[-1] @ s[-n:, :k]
    # return la.norm(norms, axis = 0)
    return la.norm(norms, ord=2)


def converged(alphas, betas, k, tol):
    n = 0
    block_size = 1
    if len(alphas.shape) == 1:
        n = alphas.shape[0]
    elif len(alphas.shape) == 3:
        n, block_size, _ = alphas.shape
    else:
        print(f"ERROR! alphas and betas must be vectors of scalars or matrices!")
        print(f"alphas.shape = {alphas.shape}, betas.shape = {betas.shape}")

    if n * block_size < k:
        return False

    if len(alphas.shape) == 1:
        errors = errors_scalar(alphas, betas, k)
    else:
        errors = errors_block(alphas, betas, k)
    return np.all(errors < tol)


def bench_lanczos(
    A, v0, k=10, krylov_size=1200, partial_reorth=True, improved_conv=True
):
    from time import perf_counter
    import scipy.sparse.linalg as spl

    eps = np.finfo(float).eps

    N = A.shape[0]
    n = v0.shape[1]

    rng = np.random.default_rng()

    eigvals = np.zeros((0, k))
    estimated_orth = []
    exact_orth = []
    delta_gs = []
    alphas = np.zeros((0, n, n), dtype=complex)
    betas = np.zeros((0, n, n), dtype=complex)
    Q = np.empty((N, n), dtype=complex)
    Q[:, :] = v0
    W = np.zeros((2, 1, n, n), dtype=complex)
    W[1] = np.identity(n)
    t_start = perf_counter()
    lanczos_it = LanczosGen(A, v0, krylov_size)
    force_reort = None
    force_ric = False
    for alpha, beta, q in lanczos_it:
        alphas = np.append(alphas, [alpha], axis=0)
        betas = np.append(betas, [beta], axis=0)

        W = estimate_orthonormality(W, alphas, betas, eps=eps, N=A.shape[0], rng=rng)
        estimated_orth.append(np.max(np.abs(W[1, :-1])))
        # Reorthogonalization.
        if (partial_reorth or improved_conv) and (
            force_ric
            or force_reort is not None
            or np.any(np.abs(W[1, :-1]) > np.sqrt(eps))
        ):
            if partial_reorth or improved_conv:
                q = q @ beta

            if partial_reorth:
                if force_reort is not None:
                    # We cannot have orthogonalized against the (current) last vector last time
                    force_reort = np.append(force_reort, [[False] * n], axis=0)
                W, q, force_reort = partial_reorthonormalization(
                    W, Q, q, N=A.shape[0], force_reort=force_reort, eps=eps, rng=rng
                )
            if improved_conv:
                W, q, _ = ric(
                    W, alphas, betas, Q, q, force_reort=None, eps=eps, rng=rng
                )

            if partial_reorth or improved_conv:
                v, r = sp.linalg.qr(q, mode = 'full', overwrite_a = True, check_finite = False)
                q, betas[-1, :, :] = v[:, :n], r[:n, :]
                # q, betas[-1, :, :] = np.linalg.qr(q)
                lanczos_it.beta = betas[-1]
                lanczos_it.v = q
        Q = np.append(Q, q, axis=1)
        delta_gs.append(test_G(alphas, betas))

        # if converged(alphas, betas, k, tol=eps):
        #     break

    t_stop = perf_counter()

    for i in range(1, lanczos_it.i // n):
        exact_orth.append(
            np.max(np.abs(np.conj(Q[:, : i * n].T) @ Q[:, i * n : (i + 1) * n]))
        )
        if i == 1:
            es = np.linalg.eigvalsh(alphas[0])
        else:
            es = eigsh(
                alphas[:i],
                betas[:i],
                eigvals_only=True,
                select="i",
                select_range=(0, min(k, i * n) - 1),
            )
        es = np.sort(es)
        if i * n < k:
            es = np.append(es[: i * n], [np.nan] * (k - i * n))
        eigvals = np.append(eigvals, [es[:k]], axis=0)

    results = {
        "eigenvalues": eigvals,
        "delta G": delta_gs,
        "estimated orth": estimated_orth,
        "exact orth": exact_orth,
        "t": t_stop - t_start,
    }

    return results


def test_G(alphas, betas):
    delta = 0.010j
    omegas = np.linspace(-1, 0.5, 100)

    n = alphas.shape[1]
    wIs = (omegas + delta)[:, np.newaxis, np.newaxis] * np.identity(n, dtype=complex)[
        np.newaxis, :, :
    ]
    gs_new = wIs - alphas[-1]
    if alphas.shape[0] == 1:
        return np.max(np.abs(gs_new))
    gs_new = (
        wIs
        - alphas[-2]
        - np.conj(betas[-2].T)[np.newaxis, :, :]
        @ np.linalg.solve(gs_new, betas[-2][np.newaxis, :, :])
    )
    gs_prev = wIs - alphas[-2]
    for alpha, beta in zip(alphas[-3::-1], betas[-3::-1]):
        gs_new = (
            wIs
            - alpha
            - np.conj(beta.T)[np.newaxis, :, :]
            @ np.linalg.solve(gs_new, beta[np.newaxis, :, :])
        )
        gs_prev = (
            wIs
            - alpha
            - np.conj(beta.T)[np.newaxis, :, :]
            @ np.linalg.solve(gs_prev, beta[np.newaxis, :, :])
        )
    return np.max(np.abs(gs_new - gs_prev))


def reorthogonalize_plot(N=5000, max_degeneracy=1, n=1, k=10, krylov_size=1200):
    from scipy import stats
    import scipy.sparse.linalg as spl
    from numpy.random import default_rng

    eps = np.finfo(float).eps

    # eigvals = [np.arange(0, N//max_degeneracy)]*max_degeneracy
    # eigvals = np.append(eigvals, [N//max_degeneracy + np.arange((N%max_degeneracy)*max_degeneracy)])
    # eigvals = np.sort(eigvals)
    # print (f"eigvals.shape = {eigvals.shape}")
    # A = sp.sparse.diags(eigvals)
    # rng = default_rng()
    # rvs = stats.norm(loc = 0, scale = 100).rvs
    # A = (
    #              sp.sparse.random(N, N, density=0.001,  dtype=complex, random_state = rng, data_rvs = rvs)
    #         + 1j*sp.sparse.random(N, N, density=0.001,  dtype=complex, random_state = rng, data_rvs = rvs)
    #         ).tocsr()
    # A @= A.H
    eigvals = np.repeat(np.arange(np.ceil(N/max_degeneracy)), max_degeneracy)[:N]
    A = sp.sparse.diags(eigvals, format = 'csr', dtype = complex)
    hamiltonian = Hermitian_operator(A)
    init_block = np.random.rand(N, n)  + 1j*np.random.rand(N, n)
    init_block[:, 0] = 1
    v0, _ = sp.linalg.qr(init_block, mode="full", overwrite_a = True, check_finite = False)
    v0 = v0[:, :n]

    eigvals = np.sort(eigvals)

    results = {}
    print(f"No reort")
    results["no reorth"] = bench_lanczos(
        # A=A,
        A=hamiltonian,
        v0=v0,
        k=k,
        krylov_size=krylov_size,
        partial_reorth=False,
        improved_conv=False,
    )
    print(f"Partial reort")
    results["partial reorth"] = bench_lanczos(
        # A=A,
        A=hamiltonian,
        v0=v0,
        k=k,
        krylov_size=krylov_size,
        partial_reorth=True,
        improved_conv=False,
    )
    # print(f"Improved convergence reort")
    # results["reorth improved conv"] = bench_lanczos(
    #     A=A,
    #     v0=v0,
    #     k=k,
    #     krylov_size=krylov_size,
    #     partial_reorth=False,
    #     improved_conv=True,
    # )
    # print(f"PRO + RIC")
    # results["partial reorth improved conv"] = bench_lanczos(
    #     A=A,
    #     v0=v0,
    #     k=k,
    #     krylov_size=krylov_size,
    #     partial_reorth=True,
    #     improved_conv=True,
    # )
    # print(f"eigvals:\n{eigvals}")

    its = results["no reorth"]["eigenvalues"].shape[0]
    krylov_space_size = range(0, its*n, n)

    import matplotlib.pyplot as plt

    axes = {
        "labelsize": 16,
        "titlesize": 16,
        "titleweight": "bold",
    }
    plt.rc("axes", **axes)
    figure = {"titlesize": 20, "titleweight": "bold"}
    plt.rc("figure", **figure)
    from matplotlib.markers import MarkerStyle

    sm = MarkerStyle("o", fillstyle="full")
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex="all", sharey="row")
    # fig, axes = plt.subplots(nrows=2, ncols=4, sharex="all", sharey="row")
    # plt.suptitle(
    #     f"Benchmarking of (block) Lanczos method\n(Matrix dimension, N = {N}; Block size, n = {n}; Krylov size = {krylov_size})"
    # )
    axes[0, 0].set_title(
        f"Without reorthogonalization\n(took {results['no reorth']['t']:.6f} seconds)"
    )
    axes[0, 0].plot(krylov_space_size, results["no reorth"]["eigenvalues"], "--", marker=sm, alpha=0.3)
    axes[0, 0].plot(krylov_space_size, [eigvals[:k]] * its, "--", color="black", alpha=0.3)
    axes[1, 0].plot(
            krylov_space_size,
            np.log10(results["no reorth"]["estimated orth"])[:-1],
            "--",
            color="tab:blue",
            label=r"$|\omega_{i + 1}|$",
            )
    axes[1, 0].plot(
            krylov_space_size,
            np.log10(results["no reorth"]["exact orth"]),
            "--",
            color="tab:red",
            label=r"$Q^{\dag}q_{i}$",
            )
    axes[1, 0].plot(krylov_space_size, [np.log10(np.sqrt(eps))] * its, "--", color="black", alpha=0.3)
    axes[1, 0].plot(krylov_space_size, [np.log10(1.0)] * its, "--", color="black", alpha=0.1)
    axes[2, 0].plot(
            krylov_space_size,
            np.log10(results["no reorth"]["delta G"])[:-1],
            "--",
            color="tab:blue",
            label=r"$\Delta G$",
            )

    axes[0, 1].set_title(
            f"Partial reorthogonalization (PRO)\n(took {results['partial reorth']['t']:.6f} seconds)"
            )
    axes[0, 1].plot(
            krylov_space_size,
            results["partial reorth"]["eigenvalues"], "--", marker=sm, alpha=0.3
            )
    axes[0, 1].plot(krylov_space_size, [eigvals[:k]] * its, "--", color="black", alpha=0.3)

    axes[1, 1].plot(
            krylov_space_size,
            np.log10(results["partial reorth"]["estimated orth"])[:-1],
            "--",
            color="tab:blue",
            label=r"$|\omega_{i + 1}|$",
            )

    axes[1, 1].plot(
            krylov_space_size,
            np.log10(results["partial reorth"]["exact orth"]),
            "--",
            color="tab:red",
            label=r"$Q^{\dag}q_{i}$",
            )
    axes[1, 1].plot(krylov_space_size, [np.log10(np.sqrt(eps))] * its, "--", color="black", alpha=0.3)
    axes[1, 1].plot(krylov_space_size, [np.log10(1.0)] * its, "--", color="black", alpha=0.1)
    axes[2, 1].plot(
            krylov_space_size,
            np.log10(results["partial reorth"]["delta G"])[:-1],
            "--",
            color="tab:blue",
            label=r"$\Delta G$",
            )
    # axes[0, 2].set_title(
    #     f"Improved convergence (RIC)\n(took {results['reorth improved conv']['t']:.6f} seconds)"
    # )
    # axes[0, 2].plot(
    #     results["reorth improved conv"]["eigenvalues"], "--", marker=sm, alpha=0.3
    # )
    # axes[0, 2].plot(range(its), [eigvals[:k]] * its, "--", color="black", alpha=0.3)
    # axes[1, 2].plot(
    #     np.log10(results["reorth improved conv"]["estimated orth"]),
    #     "--",
    #     color="tab:blue",
    #     label=r"$|\omega_{i + 1}|$",
    # )
    # axes[1, 2].plot(
    #     np.log10(results["reorth improved conv"]["exact orth"]),
    #     "--",
    #     color="tab:red",
    #     label=r"$Q^{\dag}q_{i}$",
    # )
    # axes[1, 2].plot([np.log10(np.sqrt(eps))] * its, "--", color="black", alpha=0.3)
    # axes[1, 2].plot([np.log10(1.0)] * its, "--", color="black", alpha=0.1)

    # axes[0, 3].set_title(
    #     f"PRO and RIC\n(took {results['partial reorth improved conv']['t']:.6f} seconds)"
    # )
    # axes[0, 3].plot(
    #     results["partial reorth improved conv"]["eigenvalues"],
    #     "--",
    #     marker=sm,
    #     alpha=0.3,
    # )
    # axes[0, 3].plot(range(its), [eigvals[:k]] * its, "--", color="black", alpha=0.3)
    # axes[1, 3].plot(
    #     np.log10(results["partial reorth improved conv"]["estimated orth"]),
    #     "--",
    #     color="tab:blue",
    #     label=r"$|\omega_{i + 1}|$",
    # )
    # axes[1, 3].plot(
    #     np.log10(results["partial reorth improved conv"]["exact orth"]),
    #     "--",
    #     color="tab:red",
    #     label=r"$Q^{\dag}q_{i}$",
    # )
    # axes[1, 3].plot([np.log10(np.sqrt(eps))] * its, "--", color="black", alpha=0.3)
    # axes[1, 3].plot([np.log10(1.0)] * its, "--", color="black", alpha=0.1)

    for ax in axes[-1, :]:
        ax.set_xlabel("Krylov space size")
    for ax in axes[0, :]:
        ax.set_ylabel("Eigenvalues")
        ax.set_ylim(bottom=np.min(eigvals[:k]), top=np.max(eigvals[:k]))
    for ax in axes[1, :]:
        ax.set_ylabel(
            r"$log_{10}\left(max\left(\left|\langle q_i | q_j\rangle\right|\right)\right)$"
        )
        ax.legend()
        ax.set_ylim(top=0)
    for ax in axes[2, :]:
        ax.set_ylabel(r"$log_{10}(\Delta G)$")
        ax.legend()
    plt.tight_layout()
    plt.show()
