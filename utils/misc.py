import numpy as np

def get_rand_normalized_herm_matrix(d: int, dtype: np.dtype = np.float32):
    if dtype == np.float32:
        rand_herm_matrix = np.random.randn(d, d)
        rand_herm_matrix = rand_herm_matrix + rand_herm_matrix.T
        rand_herm_matrix = rand_herm_matrix / np.linalg.norm(rand_herm_matrix)
    else:
        rand_herm_matrix = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        rand_herm_matrix = rand_herm_matrix + rand_herm_matrix.T.conj()
        rand_herm_matrix = rand_herm_matrix / np.linalg.norm(rand_herm_matrix)

    return rand_herm_matrix