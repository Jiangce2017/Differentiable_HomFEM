import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def elasticity(xPhys):
    xPhys = xPhys.astype(np.float64)  # ensure input is double precision

    penal = 1.0
    nely, nelx = np.shape(xPhys)

    ## MATERIAL PROPERTIES
    EO = np.float64(1.0)
    Emin = np.float64(1e-9)
    nu = np.float64(0.3)

    ## PREPARE FINITE ELEMENT ANALYSIS
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]], dtype=np.float64)
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]], dtype=np.float64)
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]], dtype=np.float64)
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]], dtype=np.float64)

    KE = (1 / (1 - nu**2) / 24) * (
        np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12], [B12.T, B11]])
    )

    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1, dtype=np.int32).reshape((1 + nely, 1 + nelx), order="F")
    edofVec = np.reshape(2 * nodenrs[:-1, :-1] + 1, (nelx * nely, 1), order="F")
    edofMat = np.tile(edofVec, (1, 8)) + np.tile(
        np.concatenate(([0, 1], 2 * nely + np.array([2, 3, 0, 1]), [-2, -1])), (nelx * nely, 1)
    )

    iK = np.reshape(np.kron(edofMat, np.ones((8, 1), dtype=np.int32)).T, (64 * nelx * nely, 1), order='F')
    jK = np.reshape(np.kron(edofMat, np.ones((1, 8), dtype=np.int32)).T, (64 * nelx * nely, 1), order='F')

    ## PERIODIC BOUNDARY CONDITIONS
    e0 = np.eye(3, dtype=np.float64)
    ufixed = np.zeros((8, 3), dtype=np.float64)
    U = np.zeros((2 * (nely + 1) * (nelx + 1), 3), dtype=np.float64)

    alldofs = np.arange(1, 2 * (nely + 1) * (nelx + 1) + 1, dtype=np.int32)
    n1 = np.concatenate((nodenrs[-1, [0, -1]], nodenrs[0, [-1, 0]]))
    d1 = np.reshape(([[(2 * n1 - 1)], [2 * n1]]), (1, 8), order='F')
    n3 = np.concatenate((nodenrs[1:-1, 0].T, nodenrs[-1, 1:-1]))
    d3 = np.reshape(([[(2 * n3 - 1)], [2 * n3]]), (1, 2 * (nelx + nely - 2)), order='F')
    n4 = np.concatenate((nodenrs[1:-1, -1].flatten(), nodenrs[0, 1:-1].flatten()))
    d4 = np.reshape(([[(2 * n4 - 1)], [2 * n4]]), (1, 2 * (nelx + nely - 2)), order='F')
    d2 = np.setdiff1d(alldofs, np.hstack([d1, d3, d4]))

    for j in range(3):
        ufixed[2:4, j] = np.array([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]], dtype=np.float64) @ np.array([nelx, 0], dtype=np.float64)
        ufixed[6:8, j] = np.array([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]], dtype=np.float64) @ np.array([0, nely], dtype=np.float64)
        ufixed[4:6, j] = ufixed[2:4, j] + ufixed[6:8, j]

    wfixed = np.concatenate((
        np.tile(ufixed[2:4, :], (nely - 1, 1)),
        np.tile(ufixed[6:8, :], (nelx - 1, 1))
    ))

    ## INITIALIZE ITERATION
    qe = np.empty((3, 3), dtype=object)
    Q = np.zeros((3, 3), dtype=np.float64)

    ## FE-ANALYSIS
    sK = (KE.flatten(order='F')[:, np.newaxis] *
         (Emin + xPhys.flatten(order='F').T ** penal * (EO - Emin)))[np.newaxis, :].reshape(-1, 1, order='F')

    K = sp.coo_matrix((sK.flatten(order='F'), (iK.flatten(order='F') - 1, jK.flatten(order='F') - 1)),
                      shape=(alldofs.shape[0], alldofs.shape[0]))
    K = (K + K.transpose()) / 2
    K = sp.csr_matrix(np.nan_to_num(K.toarray(), nan=0.0))
    K = K.tocsc()

    k1 = K[d2 - 1][:, d2 - 1]
    k2 = K[d2[:, None] - 1, d3 - 1] + K[d2[:, None] - 1, d4 - 1]
    k3 = K[d3 - 1, d2[:, None] - 1] + K[d4 - 1, d2[:, None] - 1]
    k4 = K[d3 - 1, d3.T - 1] + K[d4 - 1, d4.T - 1] + K[d3 - 1, d4.T - 1] + K[d4 - 1, d3.T - 1]

    Kr_top = sp.hstack((k1, k2))
    Kr_bottom = sp.hstack((k3.T, k4))
    Kr = sp.vstack((Kr_top, Kr_bottom)).tocsc()

    rhs = (
        -sp.vstack((K[d2[:, None] - 1, d1 - 1],
                    (K[d3 - 1, d1.T - 1] + K[d4 - 1, d1.T - 1]).T)) @ ufixed
        - sp.vstack((K[d2 - 1, d4.T - 1].T,
                     (K[d3 - 1, d4.T - 1] + K[d4 - 1, d4.T - 1]).T)) @ wfixed
    )

    U[d1 - 1, :] = ufixed
    U[np.concatenate((d2 - 1, d3.ravel() - 1)), :] = spsolve(Kr, rhs)
    U[d4 - 1, :] = U[d3 - 1, :] + wfixed

    for i in range(3):
        for j in range(3):
            U1 = U[:, i]
            U2 = U[:, j]
            qe[i, j] = np.reshape(np.sum((U1[edofMat - 1] @ KE) * U2[edofMat - 1], axis=1), (nely, nelx), order='F') / (nelx * nely)
            Q[i, j] = np.sum((Emin + xPhys ** penal * (EO - Emin)) * qe[i, j])

    Q[np.abs(Q) < 1e-6] = 0
    return Q
