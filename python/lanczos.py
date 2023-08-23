import scipy.linalg as la
import numpy as np
import scipy as sp

from lanczos_util import (
    estimate_orthonormality,
    hermitian_csr_matvec,
    hermitian_csr_matmat,
)


class Hermitian_operator(sp.sparse.linalg.LinearOperator):
    def __init__(self, H):
        self.shape = H.shape
        self.U = sp.sparse.tril(H, k=-1, format="csr")
        self.diag = np.real(H.diagonal())
        self.dtype = H.dtype

    def _matmat(self, m):
        return hermitian_csr_matmat(
            self.shape[0], self.diag, self.U.data, self.U.indices, self.U.indptr, m
        )

    def _matvec(self, v):
        return hermitian_csr_matvec(
            self.shape[0],
            self.diag,
            self.U.data,
            self.U.indices,
            self.U.indptr,
            v[:, 0],
        )

    def _adjoint(self):
        return self


class LanczosBasis:
    def __init__(self, v0):
        p = v0.shape[1] if len(v0.shape) > 1 else 1
        self._vectors = np.empty((2 * p, v0.shape[0]), dtype=v0.dtype)
        self.size = p
        self.capacity = self._vectors.shape[0]
        self._vectors[:p] = v0.T

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self._vectors[i].T

    def __iter__(self):
        return self._vectors[: self.size].__iter__()

    def add(self, v):
        p = v.shape[1] if len(v.shape) == 2 else 1
        if self.size + p > self.capacity:
            self._vectors = np.append(
                self._vectors, np.empty_like(self._vectors), axis=0
            )
            self.capacity = self._vectors.shape[0]
        self._vectors[self.size : self.size + p] = v.T
        self.size += p

    def get_projecttion(self, v):
        return self._vectors[: self.size].T @ (np.conj(self._vectors[: self.size]) @ v)


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
        if self.i + 1 * self.v.shape[1] >= self.krylov_size:
            raise StopIteration
        self.i += self.v.shape[1]

        v_new = self.A @ self.v
        alpha = np.conj(self.v.T) @ v_new
        v_new -= self.v @ alpha + self.v_old @ np.conj(self.beta.T)

        self.v_old = self.v
        self.v, self.beta = sp.linalg.qr(
            v_new, mode="economic", overwrite_a=True, check_finite=False
        )

        return alpha, self.beta, self.v


class LanczosBasisGen:
    import numpy as np

    def __init__(self, A, v0, alphas, betas, start=0):
        self.block = len(v0.shape) == 2
        if not self.block:
            v0 = v0.reshape((len(v0), 1))
        self.A = A
        self.v_old = np.zeros_like(v0)
        self.v = v0
        self.alphas = alphas
        self.betas = betas
        self.i = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.i + 1 > self.alphas.size[0]:
            raise StopIteration
        self.i += 1

        v_new = (
            self.A @ self.v
            - self.v @ self.alphas[self.i]
            - self.v_old @ np.conj(self.self.betas[self.i - 1].T)
        )
        self.v_old = self.v
        self.v, _ = sp.linalg.qr(
            v_new, mode="economic", overwrite_a=True, check_finite=False
        )

        return self.v


def implicit_restart(alphas, betas, Q, k):
    """
    Reduce the number of Krylov vectors by k, removing unwanted eigenvalues from the search space.
    k has to be a multiple of the block size.
    """
    n = alphas.shape[1]
    assert k % n == 0

    N = alphas.shape[0]
    eigvals = eigsh(alphas, betas, eigvals_only=True)
    Tm = sp.sparse.dia_matrix((n * N, n * N), dtype=alphas.dtype)
    em = np.zeros((N * n, n), dtype=alphas.dtype)
    em[-n:, :] = np.identity(n)
    v = np.zeros((N * n, n), dtype=alphas.dtype)
    v[-n:, :] = np.identity(n)
    if n == 1:
        Tm.setdiag(alphas.flatten(), k=0)
        Tm.setdiag(betas[:-1].flatten(), k=1)
        Tm.setdiag(betas[:-1].flatten(), k=-1)
    else:
        bands = build_banded_matrix(alphas, betas)
        for i in range(n + 1):
            Tm.setdiag(np.conj(bands[i]), i)
            Tm.setdiag(bands[i], -i)
    Tm = Tm.todense()
    np.savetxt("Tm_pre_reort.txt", Tm)
    indices = np.argsort(eigvals)
    shifts = eigvals[indices[-k:]]
    V = np.identity(Tm.shape[0])
    for sigma in shifts:
        Vi, R = sp.linalg.qr(
            Tm - sigma * np.identity(n * N), mode="economic", check_finite=False
        )
        Tm = np.conj(Vi.T) @ Tm @ Vi
        Q[:, :-n] = Q[:, :-n] @ Vi
        v = np.conj((np.conj(v.T) @ Vi).T)
        V = V @ Vi
    # assert np.allclose(V[-n:, -(K+n):-k], v[-n:])
    print(f"{k=} {n=}")
    print(f"{Q.shape=} {Tm.shape=}")
    rp = Q[:, -k : -k + n] @ betas[-1] + Q[:, -n:] @ V[-n:, -k : -k + n]
    # rp = Q[:, -k: -k + n] @ Tm[-k:-k+n, -(k+n):-k] + Q[:, -n:] @ v[-n:]
    # (N, n) x (n, n) + (N, n) x (n, n)
    # print (f"{v.shape=} {V.shape=}")
    np.savetxt("Tm_post_reort.txt", Tm)
    Tm = Tm[:-k, :-k]
    Q = Q[:, :-k]
    alphas = np.zeros((Tm.shape[0] // n, n, n), dtype=alphas.dtype)
    betas = np.zeros((Tm.shape[0] // n, n, n), dtype=betas.dtype)
    for i in range(Tm.shape[0] // n):
        alphas[i] = Tm[i * n : (i + 1) * n, i * n : (i + 1) * n]
        if i > 0:
            betas[i - 1] = Tm[i * n : (i + 1) * n, (i - 1) * n : i * n]
    qp, _ = sp.linalg.qr(rp, mode="economic", check_finite=False, overwrite_a=True)
    print(f"{np.max(np.abs(np.conj(Q.T) @ Q - np.identity(Q.shape[1])))=}")
    return alphas, betas, Q, qp


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
    W_out = np.empty((2, i + 2, n, n), dtype=complex)
    w_bar = np.empty((i + 2, n, n), dtype=complex)
    w_bar[i + 1, :, :] = np.identity(n)
    w_bar[i, :, :] = (
        eps
        * N
        * sp.linalg.solve_triangular(
            betas[i], betas[0], lower=False, trans="C", check_finite=False
        )
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
        w_bar[0] = sp.linalg.solve_triangular(
            betas[i], w_bar[0], lower=False, trans="C", check_finite=False
        )
        # w_bar[0] = sp.linalg.solve_triangular(np.conj(betas[i].T), w_bar[0], lower = True)
        w_bar[1:i] = (
            W[1, 2 : i + 1] @ betas[1:i]
            + W[1, 1:i] @ alphas[1:i]
            - alphas[i][np.newaxis, :, :] @ W[1, 1:i]
            + W[1, 0 : i - 1] @ np.conj(np.transpose(betas[0 : i - 1], axes=[0, 2, 1]))
            - betas[i - 1][np.newaxis, :, :] @ W[0, 1:i]
        )
        for j in range(1, i):
            w_bar[j] = sp.linalg.solve_triangular(
                betas[i], w_bar[j], lower=False, trans="C", check_finite=False
            )
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

    w_bar[:i] += eps * (
        betas[i] + betas[:i]
    )  # * 0.3 * rng.standard_normal(size=(i, n, n))
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
    l, y = eigsh(alphas, betas, Q, eigvals_only=False)
    if alphas.shape[0] == 1:
        mask = np.array([True] * n)
    else:
        lp = eigsh(alphas[:-1], betas[:-1], eigvals_only=True)
        m1 = np.append(np.abs(lp - l[:-n]) < np.sqrt(eps), [[False] * n])
        m2 = np.append([False] * n, [np.abs(lp - l[n:]) < np.sqrt(eps)])
        mask = np.logical_or(m1, m2)

    reort_mask = np.array(
        [mask[i * n : (i + 1) * n] for i in range(mask.shape[0] // n)]
    )
    block_mask = np.any(np.abs(W[1, :-1]) > np.sqrt(eps), axis=1)
    reset_mask = np.logical_or(reort_mask, block_mask)
    W[1, :-1][reset_mask] = (
        eps * 1.5 * rng.standard_normal(size=W[1, :-1][reset_mask].shape)
    )
    # orthogonalize(q, y[:, mask])
    q -= y[:, mask] @ (np.conj(y[:, mask].T) @ q)
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
    W[1, :-1][
        combined_mask
    ] = eps  # * 1.5 * rng.standard_normal(size=W[1, :-1][combined_mask].shape)

    # orthogonalize(q=q, Q=Q[:, combined_mask.flatten()].copy()),
    return (
        W,
        q
        - Q[:, combined_mask.flatten()]
        @ (np.conj(Q[:, combined_mask.flatten()].T) @ q),
        mask if reort else None,
    )


def build_banded_matrix(alphas, betas):
    p = alphas.shape[1]
    bands = np.empty((p + 1, p * alphas.shape[0]), dtype=complex)
    bands[0, :] = np.diagonal(alphas, offset=0, axis1=1, axis2=2).flatten()
    for i in range(1, p + 1):
        for j in range(alphas.shape[0]):
            bands[i, j * p : (j + 1) * p] = np.append(
                np.diagonal(alphas[j], offset=-i),
                [np.diagonal(betas[j], offset=p - i)],
            ).flatten()
    return bands


def eigsh(alphas, betas, Q=None, eigvals_only=False, select="a", select_range=None):
    if not eigvals_only:
        assert Q is not None

    if alphas.shape[1] == 1:
        if eigvals_only:
            return la.eigh_tridiagonal(
                np.real(alphas.flatten()),
                np.real(betas[:-1].flatten()),
                eigvals_only=eigvals_only,
                select=select,
                select_range=select_range,
            )

        eigvals, eigvecs = la.eigh_tridiagonal(
            np.real(alphas.flatten()),
            np.real(betas[:-1].flatten()),
            eigvals_only=eigvals_only,
            select=select,
            select_range=select_range,
        )
        return eigvals, Q @ eigvecs
    else:
        bands = build_banded_matrix(alphas, betas)
        if eigvals_only:
            return la.eig_banded(
                bands,
                lower=True,
                eigvals_only=eigvals_only,
                select=select,
                select_range=select_range,
                overwrite_a_band=True,
            )

        eigvals, eigvecs = la.eig_banded(
            bands,
            lower=True,
            eigvals_only=eigvals_only,
            select=select,
            select_range=select_range,
            overwrite_a_band=True,
        )

        return eigvals, Q @ eigvecs


def errors_scalar(alphas, betas, k):
    t, s = la.eigh_tridiagonal(alphas, betas[:-1])
    return np.abs(betas[-1] * s[-1, :k])


def errors_block(alphas, betas, k):
    n = alphas.shape[1]
    try:
        t, s = eigsh(alphas, betas, np.identity(n * alphas.shape[0]))
        norms = betas[-1] @ s[-n:, :k]
    except:
        norms = np.array([np.finfo(float).max])
    return la.norm(norms, ord=2)


def converged(alphas, betas, k, tol):
    n = 0
    block_size = 1
    if len(alphas.shape) == 1:
        n = alphas.shape[0]
    elif len(alphas.shape) == 3:
        n, block_size, _ = alphas.shape
    else:
        print("ERROR! alphas and betas must be vectors of scalars or matrices!")
        print(f"alphas.shape = {alphas.shape}, betas.shape = {betas.shape}")

    if n * block_size < k:
        return False

    if len(alphas.shape) == 1:
        errors = errors_scalar(alphas, betas, k)
    else:
        errors = errors_block(alphas, betas, k)
    return np.all(errors < tol)


def implicit_restart_lanczos(A, v0, k=10, max_basis_size=None, tol=np.finfo(float).eps):
    tol = max(np.finfo(float).eps, tol)
    N = A.shape[0]
    n = v0.shape[1]
    alphas = np.zeros((0, n, n), dtype=complex)
    betas = np.zeros((0, n, n), dtype=complex)
    Q = np.empty((N, n), dtype=complex)
    Q[:, :] = v0

    if max_basis_size is None:
        max_basis_size = (np.ceil(0.75 * A.shape[0] // n)) * n

    lanczos_it = LanczosGen(A, v0, A.shape[0])
    for alpha, beta, q in lanczos_it:
        alphas = np.append(alphas, [alpha], axis=0)
        betas = np.append(betas, [beta], axis=0)
        Q = np.append(Q, q, axis=1)

        if converged(alphas, betas, k, tol=tol):
            break
        if max_basis_size is not None and Q.shape[1] > max_basis_size:
            alphas, betas, Q, q = implicit_restart(alphas, betas, Q, Q.shape[1] - n)
            lanczos_it.v_old[:] = np.zeros((N, n), dtype=complex)
            lanczos_it.v[:] = q[:, :n]
            lanczos_it.beta[:] = np.zeros((n, n), dtype=complex)  # betas[-1]
            lanczos_it.i = A.shape[0]

    return eigsh(alphas, betas, Q=Q[:, :-n], select="i", select_range=(0, k))


def test_implicit_restart(
    N=5000, max_degeneracy=1, n=1, k=10, max_basis_size=None, tol=np.finfo(float).eps
):
    from scipy import stats
    import scipy.sparse.linalg as spl
    from numpy.random import default_rng

    eps = np.finfo(float).eps

    eigvals = [np.arange(0, N // max_degeneracy)] * max_degeneracy
    eigvals = np.append(
        eigvals,
        [N // max_degeneracy + np.arange((N % max_degeneracy) * max_degeneracy)],
    )
    eigvals = np.sort(eigvals)
    print(f"eigvals.shape = {eigvals.shape}")
    A = sp.sparse.diags(eigvals)
    rng = default_rng()
    rvs = stats.norm(loc=0, scale=100).rvs
    A = (
        sp.sparse.random(
            N, N, density=0.001, dtype=complex, random_state=rng, data_rvs=rvs
        )
        + 1j
        * sp.sparse.random(
            N, N, density=0.001, dtype=complex, random_state=rng, data_rvs=rvs
        )
    ).tocsr()
    A @= A.H
    eigvals = np.repeat(np.arange(np.ceil(N / max_degeneracy)), max_degeneracy)[:N]
    # eigvals = np.arange(1, N + 1)
    # eigvals = 1/(1+np.arange(1, N + 1))
    A = sp.sparse.diags(eigvals, format="csr", dtype=complex)
    hamiltonian = Hermitian_operator(A)
    init_block = np.random.rand(N, n) + 1j * np.random.rand(N, n)
    init_block[:, 0] = 1
    v0, _ = sp.linalg.qr(init_block, mode="full", overwrite_a=True, check_finite=False)
    v0 = v0[:, :n]

    eigvals = np.sort(eigvals)
    eigvals_ir, eigvecs_ir = implicit_restart_lanczos(A, v0, k, max_basis_size, tol)
    print(f"{np.max(np.abs(eigvals[:k] - eigvals_ir))}")


def bench_lanczos(
    A,
    v0,
    k=10,
    krylov_size=1200,
    partial_reorth=True,
    improved_conv=True,
    max_basis_size=None,
    tol=np.finfo(float).eps,
):
    tol = max(np.finfo(float).eps, tol)
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
    it = 0
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
                q[:] = q @ beta

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
                q, betas[-1] = sp.linalg.qr(
                    q, mode="economic", overwrite_a=True, check_finite=False
                )
                mask = np.abs(np.diagonal(betas[-1])) < eps
                while np.any(mask):
                    q[:] = q @ betas[-1]
                    q[:, mask] = np.random.rand(
                        q.shape[0], sum(mask)
                    ) + 1j * np.random.rand(q.shape[0], sum(mask))
                    q[:] -= Q @ (np.conj(Q.T) @ q)
                    q[:], betas[-1] = sp.linalg.qr(
                        q, mode="economic", overwrite_a=True, check_finite=False
                    )
                    mask = np.abs(np.diagonal(betas[-1])) < eps
                lanczos_it.beta[:] = betas[-1]
                lanczos_it.v[:] = q
        Q = np.append(Q, q, axis=1)
        delta_gs.append(test_G(alphas, betas))
        it += 1

        if converged(alphas, betas, k, tol=tol):
            break
        # if max_basis_size is not None and Q.shape[1] > min(max(k + 10*n, int(np.ceil(2*k/n)*n)), A.shape[0]):
        if max_basis_size is not None and Q.shape[1] > min(
            A.shape[0] / 2, np.ceil(100 / n) * n
        ):
            _, q = eigsh(alphas, betas, Q[:, :-n], eigvals_only=False, select="a")
            Q = np.empty((N, n), dtype=complex)
            alphas = np.zeros((0, n, n), dtype=complex)
            betas = np.zeros((0, n, n), dtype=complex)
            Q[:, :] = q[:, :n]
            # alphas, betas, Q, q = implicit_restart(alphas, betas, Q, Q.shape[1] - k - n)
            lanczos_it.v_old[:] = np.zeros((N, n), dtype=complex)
            lanczos_it.v[:] = q[:, :n]
            lanczos_it.beta[:] = np.zeros((n, n), dtype=complex)
            lanczos_it.i = 0
            max_basis_size = None

    t_stop = perf_counter()

    for i in range(1, Q.shape[1] // n - 1):
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

    Ap = np.zeros((Q.shape[1], Q.shape[1]), dtype=complex)
    for i in range(alphas.shape[0]):
        Ap[i * n : (i + 1) * n, i * n : (i + 1) * n] = alphas[i, :, :]
        if i > 0:
            Ap[(i - 1) * n : i * n, i * n : (i + 1) * n] = np.conj(betas[i - 1].T)
            Ap[i * n : (i + 1) * n, (i - 1) * n : i * n] = betas[i - 1]

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
    # return np.max(np.abs(np.linalg.inv(gs_new) - np.linalg.inv(gs_prev)))
    return np.max(np.abs(gs_new - gs_prev))


def reorthogonalize_plot(
    N=5000, max_degeneracy=1, n=1, k=10, krylov_size=1200, tol=np.finfo(float).eps
):
    from scipy import stats
    import scipy.sparse.linalg as spl
    from numpy.random import default_rng

    eps = np.finfo(float).eps

    eigvals = [np.arange(0, N // max_degeneracy)] * max_degeneracy
    eigvals = np.append(
        eigvals,
        [N // max_degeneracy + np.arange((N % max_degeneracy) * max_degeneracy)],
    )
    eigvals = np.sort(eigvals)
    print(f"eigvals.shape = {eigvals.shape}")
    A = sp.sparse.diags(eigvals)
    rng = default_rng()
    rvs = stats.norm(loc=0, scale=100).rvs
    A = (
        sp.sparse.random(
            N, N, density=0.001, dtype=complex, random_state=rng, data_rvs=rvs
        )
        + 1j
        * sp.sparse.random(
            N, N, density=0.001, dtype=complex, random_state=rng, data_rvs=rvs
        )
    ).tocsr()
    A @= A.H
    eigvals = np.repeat(np.arange(np.ceil(N / max_degeneracy)), max_degeneracy)[:N]
    # eigvals = np.arange(1, N + 1)
    # eigvals = 1/(1+np.arange(1, N + 1))
    A = sp.sparse.diags(eigvals, format="csr", dtype=complex)
    hamiltonian = Hermitian_operator(A)
    init_block = np.random.rand(N, n) + 1j * np.random.rand(N, n)
    init_block[:, 0] = 1
    v0, _ = sp.linalg.qr(init_block, mode="full", overwrite_a=True, check_finite=False)
    v0 = v0[:, :n]

    eigvals = np.sort(eigvals)

    results = {}
    print(f"No reort")
    results["no reorth"] = bench_lanczos(
        A=A,
        # A=hamiltonian,
        v0=v0,
        k=k,
        krylov_size=krylov_size,
        partial_reorth=False,
        improved_conv=False,
        tol=tol,
        max_basis_size=10 * n * (k // n),
    )
    print(np.max(results["no reorth"]["eigenvalues"][-1][:k] - eigvals[:k]))
    print(f"Partial reort")
    results["partial reorth"] = bench_lanczos(
        A=A,
        # A=hamiltonian,
        v0=v0,
        k=k,
        krylov_size=krylov_size,
        partial_reorth=True,
        improved_conv=False,
        tol=tol,
    )
    print(np.max(results["partial reorth"]["eigenvalues"][-1][:k] - eigvals[:k]))
    print(f"Improved convergence reort")
    results["reorth improved conv"] = bench_lanczos(
        A=A,
        v0=v0,
        k=k,
        krylov_size=krylov_size,
        partial_reorth=False,
        improved_conv=True,
        tol=tol,
    )
    print(np.max(results["reorth improved conv"]["eigenvalues"][-1][:k] - eigvals[:k]))
    print(f"PRO + RIC")
    results["partial reorth improved conv"] = bench_lanczos(
        A=A,
        v0=v0,
        k=k,
        krylov_size=krylov_size,
        partial_reorth=True,
        improved_conv=True,
        tol=tol,
    )
    print(
        np.max(
            results["partial reorth improved conv"]["eigenvalues"][-1][:k] - eigvals[:k]
        )
    )

    its = results["no reorth"]["eigenvalues"].shape[0]
    krylov_space_size = range(0, its * n, n)
    partial_krylov = range(0, len(results["partial reorth"]["eigenvalues"]) * n, n)
    no_krylov = range(0, len(results["no reorth"]["eigenvalues"]) * n, n)
    partial_ric_krylov = range(
        0, len(results["partial reorth improved conv"]["eigenvalues"]) * n, n
    )
    ric_krylov = range(0, len(results["reorth improved conv"]["eigenvalues"]) * n, n)

    import matplotlib.pyplot as plt

    # plt.spy(A)
    # plt.show()

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
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex="all", sharey="row")
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex="none", sharey="row")
    # plt.suptitle(
    #     f"Benchmarking of (block) Lanczos method\n(Matrix dimension, N = {N}; Block size, n = {n}; Krylov size = {krylov_size})"
    # )
    axes[0, 0].set_title(
        f"Without reorthogonalization\n(took {results['no reorth']['t']:.6f} seconds)"
    )
    axes[0, 0].plot(
        krylov_space_size,
        results["no reorth"]["eigenvalues"],
        "--",
        marker=sm,
        alpha=0.3,
    )
    axes[0, 0].plot(
        krylov_space_size, [eigvals[:k]] * its, "--", color="black", alpha=1.0 / k
    )
    # axes[1, 0].plot(
    #         # krylov_space_size[:n_no],
    #         no_krylov,
    #         np.log10(results["no reorth"]["estimated orth"])[:-1],
    #         "--",
    #         color="tab:blue",
    #         label=r"$|\omega_{i + 1}|$",
    #         )
    axes[1, 0].plot(
        # krylov_space_size[:n_no],
        no_krylov,
        np.log10(results["no reorth"]["exact orth"]),
        "--",
        color="tab:red",
        label=r"$Q^{\dag}q_{i}$",
    )
    axes[1, 0].plot(
        krylov_space_size,
        [np.log10(np.sqrt(eps))] * its,
        "--",
        color="black",
        alpha=0.3,
    )
    axes[1, 0].plot(
        krylov_space_size, [np.log10(1.0)] * its, "--", color="black", alpha=0.1
    )
    # axes[2, 0].plot(
    #         # krylov_space_size[:n_no],
    #         no_krylov,
    #         np.log10(results["no reorth"]["delta G"])[:-1],
    #         "--",
    #         color="tab:blue",
    #         label=r"$\Delta G$",
    #         )

    axes[0, 1].set_title(
        f"Partial reorthogonalization (PRO)\n(took {results['partial reorth']['t']:.6f} seconds)"
    )
    axes[0, 1].plot(
        # krylov_space_size[:n_partial],
        partial_krylov,
        results["partial reorth"]["eigenvalues"],
        "--",
        marker=sm,
        alpha=0.3,
    )
    axes[0, 1].plot(
        partial_krylov,
        [eigvals[:k]] * len(partial_krylov),
        "--",
        color="black",
        alpha=1.0 / k,
    )

    axes[1, 1].plot(
        partial_krylov,
        np.log10(results["partial reorth"]["estimated orth"])[:-1],
        "--",
        color="tab:blue",
        label=r"$|\omega_{i + 1}|$",
    )

    axes[1, 1].plot(
        partial_krylov,
        np.log10(results["partial reorth"]["exact orth"]),
        "--",
        color="tab:red",
        label=r"$Q^{\dag}q_{i}$",
    )
    axes[1, 1].plot(
        krylov_space_size,
        [np.log10(np.sqrt(eps))] * len(krylov_space_size),
        "--",
        color="black",
        alpha=0.3,
    )
    axes[1, 1].plot(
        krylov_space_size, [np.log10(1.0)] * its, "--", color="black", alpha=0.1
    )
    # axes[2, 1].plot(
    #        partial_krylov,
    #        np.log10(results["partial reorth"]["delta G"]),
    #        "--",
    #        color="tab:blue",
    #        label=r"$\Delta G$",
    #        )
    axes[0, 2].set_title(
        f"Improved convergence (RIC)\n(took {results['reorth improved conv']['t']:.6f} seconds)"
    )
    axes[0, 2].plot(
        krylov_space_size, [eigvals[:k]] * its, "--", color="black", alpha=1.0 / k
    )
    axes[0, 2].plot(
        # krylov_space_size[:n_ric],
        ric_krylov,
        results["reorth improved conv"]["eigenvalues"],
        "--",
        marker=sm,
        alpha=0.3,
    )
    axes[1, 2].plot(
        krylov_space_size,
        [np.log10(np.sqrt(eps))] * len(krylov_space_size),
        "--",
        color="black",
        alpha=0.3,
    )
    axes[1, 2].plot(
        krylov_space_size,
        [np.log10(1.0)] * len(krylov_space_size),
        "--",
        color="black",
        alpha=0.1,
    )
    # axes[1, 2].plot(
    #         # krylov_space_size[:n_ric],
    #         ric_krylov,
    #         np.log10(results["reorth improved conv"]["estimated orth"]),
    #         "--",
    #         color="tab:blue",
    #         label=r"$|\omega_{i + 1}|$",
    # )
    axes[1, 2].plot(
        # krylov_space_size[:n_ric],
        ric_krylov,
        np.log10(results["reorth improved conv"]["exact orth"]),
        "--",
        color="tab:red",
        label=r"$Q^{\dag}q_{i}$",
    )
    # axes[2, 2].plot(
    #         # krylov_space_size[:n_ric],
    #         ric_krylov,
    #         np.log10(results["reorth improved conv"]["delta G"]),
    #         "--",
    #         color="tab:blue",
    #         label=r"$\Delta G$",
    #         )

    axes[0, 3].set_title(
        f"PRO and RIC\n(took {results['partial reorth improved conv']['t']:.6f} seconds)"
    )
    axes[0, 3].plot(
        krylov_space_size,
        [eigvals[:k]] * len(krylov_space_size),
        "--",
        color="black",
        alpha=1.0 / k,
    )
    axes[0, 3].plot(
        # krylov_space_size[:n_partial_ric],
        partial_ric_krylov,
        results["partial reorth improved conv"]["eigenvalues"],
        "--",
        marker=sm,
        alpha=0.3,
    )
    axes[1, 3].plot(
        krylov_space_size,
        [np.log10(np.sqrt(eps))] * len(krylov_space_size),
        "--",
        color="black",
        alpha=0.3,
    )
    axes[1, 3].plot(
        krylov_space_size,
        [np.log10(1.0)] * len(krylov_space_size),
        "--",
        color="black",
        alpha=0.1,
    )
    # axes[1, 3].plot(
    #         # krylov_space_size[:n_partial_ric],
    #         partial_ric_krylov,
    #         np.log10(results["partial reorth improved conv"]["estimated orth"]),
    #         "--",
    #         color="tab:blue",
    #         label=r"$|\omega_{i + 1}|$",
    # )
    axes[1, 3].plot(
        # krylov_space_size[:n_partial_ric],
        partial_ric_krylov,
        np.log10(results["partial reorth improved conv"]["exact orth"]),
        "--",
        color="tab:red",
        label=r"$Q^{\dag}q_{i}$",
    )
    axes[1, 3].plot(
        [np.log10(np.sqrt(eps))] * len(krylov_space_size),
        "--",
        color="black",
        alpha=0.3,
    )
    axes[1, 3].plot(
        [np.log10(1.0)] * len(krylov_space_size), "--", color="black", alpha=0.1
    )
    # axes[2, 3].plot(
    #         # krylov_space_size[:n_partial_ric],
    #         partial_ric_krylov,
    #         np.log10(results["partial reorth improved conv"]["delta G"]),
    #         "--",
    #         color="tab:blue",
    #         label=r"$\Delta G$",
    #         )

    for ax in axes[-1, :]:
        ax.set_xlabel("Krylov space size")
    for ax in axes[0, :]:
        ax.set_ylabel("Eigenvalues")
        eig_min = np.min(eigvals[:k])
        eig_max = np.max(eigvals[:k])
        ax.set_ylim(
            bottom=eig_min - 0.1 * np.abs(eig_min), top=eig_max + 0.1 * np.abs(eig_max)
        )
        # ax.set_xlim(left = 0, right = n_no)
    for ax in axes[1, :]:
        ax.set_ylabel(
            r"$log_{10}\left(max\left(\left|\langle q_i | q_j\rangle\right|\right)\right)$"
        )
        ax.legend()
        ax.set_ylim(top=0)
        # ax.set_xlim(left = 0, right = n_no)
    for ax in axes[2, :]:
        ax.set_ylabel(r"$log_{10}(\Delta G)$")
        ax.legend()
        # ax.set_xlim(left = 0, right = n_no)
    plt.tight_layout()
    plt.show()
