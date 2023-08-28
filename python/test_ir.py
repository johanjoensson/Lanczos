import numpy as np
import lanczos

lanczos.test_implicit_restart(
    N=30, max_degeneracy=2, n=1, k=4, max_basis_size=14, tol=1e-8#np.finfo(float).eps
    )
