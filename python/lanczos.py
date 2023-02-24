import scipy.linalg as la
import numpy as np
import scipy as sp


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
        self.v, self.beta = np.linalg.qr(v_new, mode="reduced")

        return alpha, self.beta, self.v


def estimate_orthonormality(
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
        * np.linalg.solve(np.conj(betas[i].T), betas[0])
        * 0.6
        * rng.standard_normal(size=(n, n))
    )
    if i == 0:
        W_out[0, : i + 1] = W[1]
        W_out[1, : i + 2] = w_bar
        return W_out

    if n > 1:
        # For block Lanczos, looping over the blocks is faster than einsum, or broadcasting
        w_bar[0] = (
            W[1, 1] @ betas[0]
            + W[1, 0] @ alphas[0]
            - alphas[i] @ W[1, 0]
            - betas[i - 1] @ W[0, 0]
        )
        w_bar[0] = np.linalg.solve(np.conj(betas[i].T), w_bar[0])
        for j in range(1, i):
            w_bar[j] = (
                W[1, j + 1] @ betas[j]
                + W[1, j] @ alphas[j]
                - alphas[i] @ W[1, j]
                + W[1, j - 1] @ np.conj(betas[j - 1].T)
                - betas[i - 1] @ W[0, j]
            )
            w_bar[j] = np.linalg.solve(np.conj(betas[i].T), w_bar[j])
        # w_bar[0] = W[1, 1] * np.linalg.norm(betas[0], ord = 2) + W[1, 0] * np.linalg.norm(alphas[0], ord = 2) +\
        #            np.linalg.norm(alphas[i], ord = 2) * W[1, 0] -\
        #            np.linalg.norm(betas[i-1], ord = 2) * W[0, 0]
        # w_bar[0] = np.linalg.norm(np.linalg.inv(betas[i]), ord = 2) * w_bar[0]
        # for j in range(1,i):
        #     w_bar[j] = W[1, j+1] * np.linalg.norm(betas[j], ord =2) + W[1, j] * np.linalg.norm(alphas[j], ord = 2) +\
        #                np.linalg.norm(alphas[i], ord = 2) * W[1, j] + W[1, j-1] * np.linalg.norm(betas[j-1], ord = 2) +\
        #                np.linalg.norm(betas[i-1], ord = 2) * W[0, j]
        #     w_bar[j] = np.linalg.norm(np.linalg.inv(betas[i]), ord = 2) * w_bar[j]
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
        eps * (betas[i] + betas[:i]) * 0.3 * rng.standard_normal(size=(i, n, n))
    )
    W_out[0, : i + 1] = W[1]
    W_out[1, : i + 2] = w_bar

    return W_out


def orthogonalize(q, Q):
    """Reorthonormalise the vector blocks in q against all vectors in Q.
    Parameters:
    q        - Array containing the vectors to reorthonormalise. Dimensions (M, N, n)
    Q        - Matrix containing the column vectors to orthogonalize against. Dimensions (N, n)
    Returns:
    q_ort    - Orthonormal vector blocks, orthogonal to all vectors in Q. Dimensions (M, N, n)
    """
    return q - Q @ (np.conj(Q.T) @ q)


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

    return W, orthogonalize(q, y[:, mask]), np.any(reort_mask)


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
        mask = np.any(np.abs(W[1, :-1]) > eps ** (3 / 4), axis=1)
    else:
        mask = np.array([[False] * n] * (i + 2))

    combined_mask = (
        np.logical_or(mask, force_reort) if force_reort is not None else mask
    )
    # Update orthonormality estimates
    W[1, :-1][combined_mask] = (
        eps * 1.5 * rng.standard_normal(size=W[1, :-1][combined_mask].shape)
    )

    return (
        W,
        orthogonalize(q=q, Q=Q[:, combined_mask.flatten()]),
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

    try:
        if len(alphas.shape) == 1:
            errors = errors_scalar(alphas, betas, k)
        else:
            errors = errors_block(alphas, betas, k)
    except np.linalg.LinAlgError as er:
        return False
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

            q, betas[-1] = np.linalg.qr(q)

            lanczos_it.beta = betas[-1]
            lanczos_it.v = q
        Q = np.append(Q, q, axis=1)

        if converged(alphas, betas, k, tol=eps):
            break

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
        "estimated orth": estimated_orth,
        "exact orth": exact_orth,
        "t": t_stop - t_start,
    }

    return results


def reorthogonalize_plot(N=5000, max_degeneracy=1, n=1, k=10, krylov_size=1200):
    import scipy.sparse.linalg as spl

    eps = np.finfo(float).eps

    eigvals = np.sort(
        [
            4
            * (
                np.sin(np.pi * i / (2 * (max_degeneracy + 1))) ** 2
                + np.sin(np.pi * j / (2 * (N // max_degeneracy + 1))) ** 2
            )
            for i in range(1, max_degeneracy + 1)
            for j in range(1, N // max_degeneracy + 1)
        ]
    )
    A = sp.sparse.diags(eigvals)
    v0 = np.random.rand(N, n) + 1j * np.random.rand(N, n)
    v0, _ = np.linalg.qr(v0, mode="reduced")

    results = {}
    print(f"No reort")
    results["no reorth"] = bench_lanczos(
        A=A,
        v0=v0,
        k=k,
        krylov_size=krylov_size,
        partial_reorth=False,
        improved_conv=False,
    )
    print(f"Partial reort")
    results["partial reorth"] = bench_lanczos(
        A=A,
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
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="row")
    #fig, axes = plt.subplots(nrows=2, ncols=4, sharex="all", sharey="row")
    # plt.suptitle(
    #     f"Benchmarking of (block) Lanczos method\n(Matrix dimension, N = {N}; Block size, n = {n}; Krylov size = {krylov_size})"
    # )
    axes[0, 0].set_title(
        f"Without reorthogonalization\n(took {results['no reorth']['t']:.6f} seconds)"
    )
    axes[0, 0].plot(results["no reorth"]["eigenvalues"], "--", marker=sm, alpha=0.3)
    axes[0, 0].plot(range(its), [eigvals[:k]] * its, "--", color="black", alpha=0.3)
    axes[1, 0].plot(
        np.log10(results["no reorth"]["estimated orth"]),
        "--",
        color="tab:blue",
        label=r"$|\omega_{i + 1}|$",
    )
    axes[1, 0].plot(
        np.log10(results["no reorth"]["exact orth"]),
        "--",
        color="tab:red",
        label=r"$Q^{\dag}q_{i}$",
    )
    axes[1, 0].plot([np.log10(np.sqrt(eps))] * its, "--", color="black", alpha=0.3)
    axes[1, 0].plot([np.log10(1.0)] * its, "--", color="black", alpha=0.1)

    axes[0, 1].set_title(
        f"Partial reorthogonalization (PRO)\n(took {results['partial reorth']['t']:.6f} seconds)"
    )
    axes[0, 1].plot(
        results["partial reorth"]["eigenvalues"], "--", marker=sm, alpha=0.3
    )
    axes[0, 1].plot(range(its), [eigvals[:k]] * its, "--", color="black", alpha=0.3)
    axes[1, 1].plot(
        np.log10(results["partial reorth"]["estimated orth"]),
        "--",
        color="tab:blue",
        label=r"$|\omega_{i + 1}|$",
    )
    axes[1, 1].plot(
        np.log10(results["partial reorth"]["exact orth"]),
        "--",
        color="tab:red",
        label=r"$Q^{\dag}q_{i}$",
    )
    axes[1, 1].plot([np.log10(np.sqrt(eps))] * its, "--", color="black", alpha=0.3)
    axes[1, 1].plot([np.log10(1.0)] * its, "--", color="black", alpha=0.1)

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

    for row in axes:
        for ax in row:
            ax.set_xlabel("Iteration")
    for ax in axes[0, :]:
        ax.set_ylabel("Eigenvalues")
        ax.set_ylim(bottom=np.min(eigvals[:k]), top=np.max(eigvals[:k]))
    for ax in axes[1, :]:
        ax.set_ylabel(r"$log_{10}|max\left(\left|\langle q_i | q_j\rangle\right|\right)|$")
        ax.legend()
    plt.tight_layout()
    plt.show()
