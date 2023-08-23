import numpy as np
cimport cython
# from libc.complex cimport conj
cdef extern from "<complex.h>" nogil:
    double complex conj(double complex z)

cimport scipy.linalg.cython_blas as blas
cimport  scipy.linalg.cython_lapack as lapack

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void csr_dense_matvec(const complex[:] csr_data, const int [:] csr_indices, const int[:] csr_index_ptr, const complex[:] dense_v, complex[:] res) noexcept:
    cdef Py_ssize_t csr_rows = res.shape[0]
    cdef Py_ssize_t csr_cols = dense_v.shape[0]
    cdef Py_ssize_t row, k
    cdef Py_ssize_t  col, index, indices_start, indices_end
    cdef complex csr_val
    for row in range(csr_rows):
        indices_start, indices_end = csr_index_ptr[row], csr_index_ptr[row + 1]
        for index in range(indices_start, indices_end):
            # csr_val = A[row, k]
            k, csr_val = csr_indices[index] , csr_data[index]
            # csr_val_dag = A[k, row] = A[row, k]*
            res[row] += csr_val*dense_v[k]
            # A[k, k] = A[k, k]* = A[k, k], don't add it twice
            if k != row:
                res[k] += conj(csr_val)*dense_v[row]

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void csr_dense_no_diagonal_matvec(const complex[:] csr_data, const int [:] csr_indices, const int[:] csr_index_ptr, const complex[:] dense_v, complex[:] res) noexcept:
    cdef Py_ssize_t csr_rows = res.shape[0]
    cdef Py_ssize_t csr_cols = dense_v.shape[0]
    cdef Py_ssize_t row, k
    cdef Py_ssize_t  col, index, indices_start, indices_end
    cdef complex csr_val
    for row in range(csr_rows):
        indices_start, indices_end = csr_index_ptr[row], csr_index_ptr[row + 1]
        for index in range(indices_start, indices_end):
            # csr_val = A[row, k]
            k, csr_val = csr_indices[index] , csr_data[index]
            # csr_val_dag = A[k, row] = A[row, k]*
            res[row] += csr_val*dense_v[k]
            res[k] += conj(csr_val)*dense_v[row]

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void csr_dense_diagonal_matvec(const double[:] diagonal, const complex[:] dense_v, complex[:] res) noexcept:
    # [a  0  0 ...   ]  [ v0]
    # [0  b  0 ...   ]  [ v1]
    # [.  .  . ...   ]  [ . ]
    # [0  0  0 ...  n]  [ vn]
    cdef Py_ssize_t rows = diagonal.shape[0]
    cdef Py_ssize_t row
    for row in range(rows):
            res[row] = diagonal[row]*dense_v[row]

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void csr_dense_matmat(const complex[:] csr_data, const int[:] csr_indices, const int[:] csr_index_ptr, const complex[:, :] dense_m, complex[:, :] res) noexcept:
    cdef Py_ssize_t csr_rows = res.shape[0]
    cdef Py_ssize_t dense_cols = dense_m.shape[1]
    cdef Py_ssize_t row, k
    cdef Py_ssize_t col, index, indices_start, indices_end
    cdef complex csr_val
    for row in range(csr_rows):
        indices_start, indices_end = csr_index_ptr[row], csr_index_ptr[row + 1]
        for index in range(indices_start, indices_end):
            # csr_val = A[row, k]
            k, csr_val = csr_indices[index] , csr_data[index]
            # csr_val_dag = A[k, row]*
            csr_val_dag = conj(csr_val)
            for col in range(dense_cols):
                res[row, col] += csr_val*dense_m[k, col]
                # A[k, k] = A[k, k]* = A[k, k], don't add it twice
                if k != row:
                    res[k, col] += conj(csr_val)*dense_m[row, col]

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void csr_dense_no_diagonal_matmat(const complex[:] csr_data, const int[:] csr_indices, const int[:] csr_index_ptr, const complex[:, :] dense_m, complex[:, :] res) noexcept:
    cdef Py_ssize_t csr_rows = res.shape[0]
    cdef Py_ssize_t dense_cols = dense_m.shape[1]
    cdef Py_ssize_t row, k
    cdef Py_ssize_t col, index, indices_start, indices_end
    cdef complex csr_val
    for row in range(csr_rows):
        indices_start, indices_end = csr_index_ptr[row], csr_index_ptr[row + 1]
        for index in range(indices_start, indices_end):
            # csr_val = A[row, k]
            k, csr_val = csr_indices[index] , csr_data[index]
            # csr_val_dag = A[k, row]*
            for col in range(dense_cols):
                res[row, col] += csr_val*dense_m[k, col]
                res[k, col] += conj(csr_val)*dense_m[row, col]

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void csr_dense_diagonal_matmat(const double[:] diagonal, const complex[:, :] dense_m, complex[:, :] res) noexcept:
    # [a  0  0 ...   ]  [ v00  v01 ... v0m]
    # [0  b  0 ...   ]  [ v10  v11 ... v1m]
    # [.  .  . ...   ]  [ .    .   ... .  ]
    # [0  0  0 ...  n]  [ vn0  vn1 ... vnm]
    cdef Py_ssize_t rows = diagonal.shape[0]
    cdef Py_ssize_t cols = dense_m.shape[1]
    cdef Py_ssize_t row, col
    for row in range(rows):
        for col in range(cols):
            res[row, col] = diagonal[row]*dense_m[row, col]

def hermitian_csr_matmat(int csr_nrows, const double[:] diagonal, const complex[:] csr_data, const int[:] csr_indices, const int[:] csr_index_ptr, const complex[:, :] dense_m):
    res = np.empty((csr_nrows, dense_m.shape[1]), dtype = complex)
    cdef complex[:, :] res_view = res
    csr_dense_diagonal_matmat(diagonal, dense_m, res_view)
    csr_dense_no_diagonal_matmat(csr_data, csr_indices, csr_index_ptr, dense_m, res_view)
    return res

def hermitian_csr_matvec(int csr_nrows, const double[:] diagonal, const complex[:] csr_data, const int[:] csr_indices, const int[:] csr_index_ptr, const complex[:] dense_v):
    res = np.empty((csr_nrows), dtype = complex)
    cdef complex[:] res_view = res
    csr_dense_diagonal_matvec(diagonal, dense_v, res_view)
    csr_dense_no_diagonal_matvec(csr_data, csr_indices, csr_index_ptr, dense_v, res_view)
    return res

@cython.initializedcheck(False)
cpdef void zgemm(
        char trans_a,
        char trans_b,
        complex alpha,
        complex[:, :] a,
        complex[:, :] b,
        complex beta,
        complex[:, :] c,
        bint column_major
        ):
    cdef int lda = a.strides[0]/a.itemsize
    cdef int ldb = b.strides[0]/b.itemsize
    cdef int ldc = c.strides[0]/c.itemsize
    cdef int M = c.shape[0]
    cdef int N = c.shape[1]
    cdef int K = a.shape[1] if (trans_a == ord('N') or trans_a == ord('n')) else a.shape[0]
    blas.zgemm(
            &trans_a if column_major else &trans_b,
            &trans_b if column_major else &trans_a,
            &M if column_major else &N,
            &N if column_major else &M,
            &K,
            &alpha,
            &a[0,0] if column_major else &b[0,0],
            &lda if column_major else &ldb,
            &b[0,0] if column_major else &a[0,0],
            &ldb if column_major else &lda,
            &beta,
            &c[0, 0],
            &ldc
            )

# @cython.initializedcheck(False)
# def orthogonalize(complex[:, :] q, complex[:, :] Q):
def orthogonalize(q: np.ndarray, Q: np.ndarray):
    if not q.flags.forc:
        q = q.copy(order = 'C')
    if not Q.flags.forc:
        Q = Q.copy(order = 'C')
    cdef complex[:, :] tmp = np.empty((Q.shape[1], q.shape[1]), dtype = complex)
    zgemm(ord('C'), ord('N'), 1, Q, q, 0, tmp, Q.flags.f_contiguous)
    zgemm(ord('N'), ord('N'), -1, Q, tmp, 1.0, q, Q.flags.f_contiguous)

def dot(a: np.ndarray, b: np.ndarray):
    if not a.flags.forc:
        a = a.copy(order = 'C')
    if not b.flags.forc:
        b = b.copy(order = 'C')
    res = np.empty((a.shape[0], b.shape[1]), dtype = a.dtype)
    zgemm(ord('N'), ord('N'), 1, a, b, 0, res, a.flags.f_contiguous)
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void assign_2_index(complex[:, :] a, complex[:, :] b):
    cdef Py_ssize_t i_max = a.shape[0]
    cdef Py_ssize_t j_max = a.shape[1]

    cdef Py_ssize_t i, j
    for i in range(i_max):
        for j in range(j_max):
            a[i, j] = b[i, j]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void assign_3_index_1(
        complex[:, :, :, :] a,
        const complex[:, :, :, :] b,
        Py_ssize_t a0, Py_ssize_t i_max,
        Py_ssize_t b0) nogil :
    cdef Py_ssize_t j_max = a.shape[2]
    cdef Py_ssize_t k_max = a.shape[3]

    cdef Py_ssize_t i, j, k
    for i in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                a[a0, i, j, k] = b[b0, i, j, k]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void assign_3_index_2(
        complex[:, :, :, :] a,
        const complex[:, :, :] b,
        Py_ssize_t a0,
        Py_ssize_t i_max
        ) nogil :
    cdef Py_ssize_t j_max = a.shape[2]
    cdef Py_ssize_t k_max = a.shape[3]

    cdef Py_ssize_t i, j, k
    for i in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                a[a0, i, j, k] = b[i, j, k]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void f1(
        complex[:, :, :] wbar0,
        const complex[:, :] W1betas0,
        const complex[:, :] W1alphas0,
        const complex[:, :] alphasiW1,
        const complex[:, :] betasim1W0
        ) nogil :
    cdef Py_ssize_t i_max = wbar0.shape[1]
    cdef Py_ssize_t j_max = wbar0.shape[2]

    cdef Py_ssize_t i, j
    for i in range(i_max):
        for j in range(j_max):
            wbar0[0, i, j] = W1betas0[i, j] + W1alphas0[i, j] - alphasiW1[i, j] -betasim1W0[i, j]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void f2(
        complex[:, :, :] wbar,
        const complex[:, :, :] Wbeta,
        const complex[:, :, :] Walpha,
        const complex[:, :, :] alphaW,
        const complex[:, :, :] Wbetadagg,
        const complex[:, :, :] betaW,
        Py_ssize_t i_max
        ) nogil :
    cdef Py_ssize_t j_max = wbar.shape[1]
    cdef Py_ssize_t k_max = wbar.shape[2]

    cdef Py_ssize_t i, j, k
    for i in range(i_max - 1):
        for j in range(j_max):
            for k in range(k_max):
                wbar[1 + i, j, k] = Wbeta[i, j, k] + Walpha[i, j, k] - alphaW[i, j, k] + Wbetadagg[i, j, k] - betaW[i, j, k]

@cython.boundscheck(False)
@cython.wraparound(False)
def estimate_orthonormality(
        W,
        alphas,
        betas,
        eps = np.finfo(float).eps,
        N = 1,
        rng = np.random.default_rng()
        ):
    i = alphas.shape[0] - 1
    n = alphas.shape[1]
    W_out= np.zeros((2, i + 2, n, n), dtype=complex)
    cdef complex[:, :, :, :] W_out_view = W_out
    w_bar = np.zeros((i + 2, n, n), dtype=complex)
    w_bar[i + 1, :, :] = np.identity(n)
    cdef complex[:, :, :] w_bar_view = w_bar
    w_bar[i, :, :] = (
            eps
            * N
            * np.linalg.solve(np.conj(betas[i].T), betas[0])
            * 0.6
            * rng.standard_normal(size=(n, n))
            )
    if i == 0:
        assign_3_index_1(W_out_view, W, 0, i+1, 1)
        # W_out[0, : i + 1] = W[1]
        assign_3_index_2(W_out_view, w_bar, 1, i + 2)
        # W_out[1, : i + 2] = w_bar
        return W_out

    cdef Py_ssize_t m
    if n > 1:
        # w_bar[0] = (
        #     W[1, 1] @ betas[0]
        #     + W[1, 0] @ alphas[0]
        #     - alphas[i] @ W[1, 0]
        #     - betas[i - 1] @ W[0, 0]
        # )
        zgemm(ord('N'), ord('N'), -1.0, betas[i - 1], W[0, 0], 0., w_bar[0], False)
        zgemm(ord('N'), ord('N'), -1.0, alphas[i], W[1, 0], 1.0, w_bar[0], False)
        zgemm(ord('N'), ord('N'), 1.0, W[1, 0], alphas[0], 1.0, w_bar[0], False)
        zgemm(ord('N'), ord('N'), 1.0, W[1, 1], betas[0], 1.0, w_bar[0], False)


        # f1(w_bar, W[1, 1] @ betas[0], W[1, 0] @ alphas[0], alphas[i] @ W[1, 0], betas[i - 1] @ W[0, 0])
        w_bar[0] = np.linalg.solve(np.conj(betas[i].T), w_bar[0])
        # w_bar[1:i] = (
        #     W[1, 2 : i + 1] @ betas[1:i]
        #     + W[1, 1:i] @ alphas[1:i]
        #     - alphas[i][np.newaxis, :, :] @ W[1, 1:i]
        #     + W[1, 0 : i - 1] @ np.conj(np.transpose(betas[0 : i - 1], axes=[0, 2, 1]))
        #     - betas[i - 1][np.newaxis, :, :] @ W[0, 1:i]
        # )
        for m in range(1, i):
            zgemm(ord('N'), ord('N'), -1.0, betas[i - 1], W[0, m], 0., w_bar[m], False)
            # zgemm(ord('N'), ord('N'), 1.0, W[1, m - 1], np.conj(betas[m].T), 1.0, w_bar[m], False)
            zgemm(ord('N'), ord('C'), 1.0, W[1, m - 1], betas[m], 1.0, w_bar[m], False)
            zgemm(ord('N'), ord('N'), -1.0, alphas[i], W[1, m], 1.0, w_bar[m], False)
            zgemm(ord('N'), ord('N'), 1.0, W[1, m], alphas[m], 1.0, w_bar[m], False)
            zgemm(ord('N'), ord('N'), 1.0, W[1, m + 1], betas[m], 1.0, w_bar[m], False)
        # f2(w_bar, W[1, 2 : i + 1] @ betas[1:i], W[1, 1:i] @ alphas[1:i], alphas[i][np.newaxis, :, :] @ W[1, 1:i], W[1, 0 : i - 1] @ np.conj(np.transpose(betas[0 : i - 1], axes=[0, 2, 1])), betas[i - 1][np.newaxis, :, :] @ W[0, 1:i], i)
        w_bar[1:i] = np.linalg.solve(np.conj(betas[i].T)[np.newaxis, :, :], w_bar[1:i])
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
    assign_3_index_1(W_out_view, W, 0, i+1, 1)
    # W_out[0, : i + 1] = W[1]
    assign_3_index_2(W_out_view, w_bar, 1, i+2)
    # W_out[1, : i + 2] = w_bar
    return W_out
