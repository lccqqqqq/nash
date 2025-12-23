import torch

def group_legs(a, axes):
    """ Given list of lists like axes = [ [l1, l2], [l3], [l4 . . . ]]

        does a transposition of "a" according to l1 l2 l3... followed by a reshape according to parantheses.

        Return the reformed tensor along with a "pipe" which can be used to undo the move
    """

    nums = [len(k) for k in axes]

    flat = []
    for ax in axes:
        flat.extend(ax)

    a = a.permute(*flat)
    perm = torch.argsort(torch.tensor(flat)).tolist()

    oldshape = a.shape

    shape = []
    oldshape_list = []
    m = 0
    for n in nums:
        shape.append(int(torch.prod(torch.tensor(a.shape[m:m+n])).item()))
        oldshape_list.append(a.shape[m:m+n])
        m += n

    a = a.reshape(shape)

    pipe = (oldshape_list, perm)

    return a, pipe

def ungroup_legs(a, pipe):
    """
        Given the output of group_legs, recovers the original tensor (inverse operation)

        For any singleton grouping [l], allows the dimension to have changed (the new dim is inferred from 'a').
    """
    if a.ndim != len(pipe[0]):
        raise ValueError
    shape = []
    for j in range(a.ndim):
        if len(pipe[0][j]) == 1:
            shape.append(a.shape[j])
        else:
            shape.extend(pipe[0][j])

    a = a.reshape(shape)
    a = a.permute(*pipe[1])
    return a

def transpose_mpo(Psi):
    """Transpose row / column of an MPO"""
    return [b.permute(1, 0, 2, 3) for b in Psi]

def mps_group_legs(Psi, axes='all'):
    """ Given an 'MPS' with a higher number of physical legs (say, 2 or 3), with B tensors

            physical leg_1 x physical leg_2 x . . . x virtual_left x virtual_right

        groups the physical legs according to axes = [ [l1, l2], [l3], . .. ] etc,

        Example:


            Psi-rank 2,    axes = [[0, 1]]  will take MPO--> MPS
            Psi-rank 2, axes = [[1], [0]] will transpose MPO
            Psi-rank 3, axes = [[0], [1, 2]] will take to MPO

        If axes = 'all', groups all of them together.

        Returns:
            Psi
            pipes: list which will undo operation
    """

    if axes == 'all':
        axes = [list(range(Psi[0].ndim - 2))]

    psi = []
    pipes = []
    for j in range(len(Psi)):
        ndim = Psi[j].ndim
        b, pipe = group_legs(Psi[j], axes + [[ndim-2], [ndim-1]])

        psi.append(b)
        pipes.append(pipe)

    return psi, pipes

def mps_ungroup_legs(Psi, pipes):
    """Inverts mps_group_legs given its output"""
    psi = []
    for j in range(len(Psi)):
        psi.append(ungroup_legs(Psi[j], pipes[j]))

    return psi

def mps_invert(Psi):
    np = Psi[0].ndim - 2
    return [b.permute(*list[int](range(np)) + [-1, -2]) for b in Psi[::-1]]

def mps_2form(Psi, form='A'):
    """Puts an mps with an arbitrary # of legs into A or B-canonical form

        hahaha so clever!!!
    """
    Psi, pipes = mps_group_legs(Psi, axes='all')

    if form == 'B':
        Psi = [b.permute(0, 2, 1) for b in Psi[::-1]]

    L = len(Psi)
    T = Psi[0]
    for j in range(L-1):
        T, pipe = group_legs(T, [[0, 1], [2]])  # view as matrix
        A, s = torch.linalg.qr(T)  # T = A s can be given from QR
        Psi[j] = ungroup_legs(A, pipe)
        T = torch.tensordot(s, Psi[j+1], dims=([1], [1])).permute(1, 0, 2)  # Absorb s into next tensor

    Psi[L-1] = T

    if form == 'B':
        Psi = [b.permute(0, 2, 1) for b in Psi[::-1]]

    Psi = mps_ungroup_legs(Psi, pipes)

    return Psi

def mps_entanglement_spectrum(Psi):

    Psi, pipes = mps_group_legs(Psi, axes='all')

    # First bring to A-form
    L = len(Psi)
    T = Psi[0]
    for j in range(L-1):
        T, pipe = group_legs(T, [[0, 1], [2]])  # view as matrix
        A, s = torch.linalg.qr(T)  # T = A s can be given from QR
        Psi[j] = ungroup_legs(A, pipe)
        T = torch.tensordot(s, Psi[j+1], dims=([1], [1])).permute(1, 0, 2)  # Absorb s into next tensor

    Psi[L-1] = T

    # Flip the MPS around
    Psi = [b.permute(0, 2, 1) for b in Psi[::-1]]

    T = Psi[0]
    Ss = []
    for j in range(L-1):
        T, pipe = group_legs(T, [[0, 1], [2]])  # view as matrix
        U, s, Vh = torch.linalg.svd(T)  # T = U s Vh
        Ss.append(s)
        Psi[j] = ungroup_legs(U, pipe)
        s = (Vh.T * s).T
        T = torch.tensordot(s, Psi[j+1], dims=([1], [1])).permute(1, 0, 2)  # Absorb sV into next tensor

    return Ss

def mpo_on_mpo(X, Y, form=None):
    """ Multiplies two two-sided MPS, XY = X*Y and optionally puts in a canonical form
    """
    if X[0].ndim != 4 or Y[0].ndim != 4:
        raise ValueError

    XY = [group_legs(torch.tensordot(x, y, dims=([1], [0])), [[0], [3], [1, 4], [2, 5]])[0] for x, y in zip(X, Y)]

    if form is not None:
        XY = mps_2form(XY, form)

    return XY

def mpo_on_mps(w_list, B_list):
    " Apply the MPO to an MPS."
    d = B_list[0].shape[0]
    D = w_list[0].shape[1]
    L = len(B_list)

    chi1 = B_list[0].shape[1]
    chi2 = B_list[0].shape[2]

    B = torch.tensordot(B_list[0], w_list[0][0, :, :, :], dims=(0, 1))
    B = B.permute(3, 0, 1, 2).reshape(d, chi1, chi2*D)
    B_new = [B]

    for i_site in range(1, L-1):
        chi1 = B_list[i_site].shape[1]
        chi2 = B_list[i_site].shape[2]
        B = torch.tensordot(B_list[i_site], w_list[i_site][:, :, :, :], dims=(0, 2))
        D = w_list[i_site].shape[1]
        B = B.permute(4, 0, 2, 1, 3).reshape(d, chi1*D, chi2*D)
        B_new.append(B)

    chi1 = B_list[L-1].shape[1]
    chi2 = B_list[L-1].shape[2]
    B = torch.tensordot(B_list[L-1], w_list[L-1][:, 0, :, :], dims=(0, 1))
    B = B.permute(3, 0, 2, 1).reshape(d, D*chi1, chi2)
    B_new.append(B)

    return B_new

def svd_theta(theta, truncation_par, return_XYZ=None, normalize=True):
    """ SVD and truncate a matrix based on truncation_par """

    U, s, Vh = torch.linalg.svd(theta, full_matrices=False)
    s[torch.abs(s) < 1e-14] = 0.
    nrm = torch.linalg.norm(s)
    eta = min(
        torch.count_nonzero((1 - torch.cumsum(s**2, dim=0) / nrm**2) > truncation_par['p_trunc']).item() + 1,
        truncation_par['chi_max']
    )
    nrm_t = torch.linalg.norm(s[:eta])

    if return_XYZ:
        Y = s[:eta] / nrm_t
        X = U[:, :eta]
        Z = Vh[:eta, :]

        info = {'p_trunc': 1 - (nrm_t / nrm)**2, 's': s[:eta], 'nrm': nrm}
        return X, Y, Z, info

    A = U[:, :eta]

    if normalize:
        SB = (Vh[:eta, :].T * s[:eta] / nrm_t).T
    else:
        SB = (Vh[:eta, :].T * s[:eta] / nrm_t * nrm).T

    info = {'p_trunc': 1 - (nrm_t / nrm)**2, 's': s[:eta], 'nrm': nrm}
    return A, SB, info

def compress(Psi, chi, normalize=True):
    L = len(Psi)

    def sweep(Psi, truncation_par, normalize=True):
        """ Left to right sweep """
        psi = Psi[0]
        num_p = psi.ndim - 2  # number of physical legs
        p_trunc = 0.
        nrm = 1.
        expectation_O = 0.
        d = psi.shape[0]
        for j in range(L-1):
            psi, pipe1L = group_legs(psi, [[0], list(range(1, num_p+1)), [num_p + 1]])   # bring to 3-leg form
            B, pipe1R = group_legs(Psi[j+1], [[0], [num_p], list(range(1, num_p)) + [num_p + 1]])  # bring to 3-leg form

            theta = torch.tensordot(psi, B, dims=[[-1], [-2]])  # Theta = s B
            theta = torch.tensordot(torch.eye(d**2, device=psi.device, dtype=psi.dtype).reshape(d, d, d, d), theta, dims=[[2, 3], [0, 2]])  # Theta = U Theta
            theta, pipeT = group_legs(theta, [[0, 2], [1, 3]])  # Turn into a matrix

            A, SB, info = svd_theta(theta, truncation_par, normalize=normalize)  # Theta = A s

            # Back to 3-leg form
            A = A.reshape(*pipeT[0][0], -1)
            SB = SB.reshape(-1, *pipeT[0][1]).permute(1, 0, 2)

            A = ungroup_legs(A, pipe1L)
            SB = ungroup_legs(SB, pipe1R)

            p_trunc += info['p_trunc'].item() if isinstance(info['p_trunc'], torch.Tensor) else info['p_trunc']
            nrm *= info['nrm'].item() if isinstance(info['nrm'], torch.Tensor) else info['nrm']

            Psi[j] = A
            psi = SB

        Psi[L-1] = psi

        return p_trunc, nrm

    chi_max = max([max(A.shape) for A in Psi])

    p_trunc, nrm = sweep(Psi, {'p_trunc': 1e-14, 'chi_max': chi_max}, normalize=normalize)

    Psi = mps_invert(Psi)
    p_trunc, nrm = sweep(Psi, {'p_trunc': 1e-14, 'chi_max': chi}, normalize=normalize)
    Psi = mps_invert(Psi)

    info = {'p_trunc': p_trunc, 'nrm': nrm}
    return Psi, info

def get_transfer_MPO(M):
    """
    Get the transfer MPO
    """
    T = []
    for t in M:
        _, a, b, c, d = t.shape
        T.append(torch.tensordot(t, torch.conj(t), dims=(0, 0)))
        T[-1] = T[-1].permute(0, 4, 1, 5, 2, 6, 3, 7)
        T[-1] = T[-1].reshape(a*a, b*b, c*c, d*d)
    return T

def mpo_to_full(M):
    D = M[0].shape[0]

    L = len(M)
    d = M[0].shape[2]

    M_full = M[0].squeeze(dim=0)  # v1 h1 h2
    for i in range(0, L-1):
        M_full = torch.tensordot(M_full, M[i+1], dims=(2*i, 0))  # h1 h2 v1 h3 h4 ...

    M_full = M_full.squeeze()  # h1 h2 h3 h4 ...
    nd = M_full.ndim
    M_full = M_full.permute(*list(range(0, nd, 2)), *list(range(1, nd, 2)))
    M_full = M_full.reshape(d**L, d**L)

    return M_full

def mps_overlap(psi1, psi2):
    N = torch.ones(1, 1, device=psi1[0].device, dtype=psi1[0].dtype)  # a ap
    L = len(psi1)
    for i in range(L):
        N = torch.tensordot(N, torch.conj(psi1[i]), dims=([1], [1]))  # a ap
        N = torch.tensordot(N, psi2[i], dims=([0, 1], [1, 0]))  # ap a
        N = N.permute(1, 0)
    N = torch.trace(N)
    return N
