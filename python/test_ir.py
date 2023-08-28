import numpy as np
import lanczos

lanczos.test_implicit_restart(
    N=30, max_degeneracy=1, n=1, k=3, max_basis_size=10, tol=np.finfo(float).eps
    )
