import numpy as np
from numpy.linalg import inv
        
def match_multi(keys, relation, known, dim=6):
    V = np.mat(np.zeros((dim*len(keys), 1)))
    Dk = np.mat(np.diag(np.ones(dim*len(keys))*1e6))

    for n, pts1, pts2 in known:
        for p1, p2 in zip(pts1, pts2):
            L = np.mat(p2[:2]).T
            Dl = np.mat(np.diag(np.ones(2)))
            T = np.mat(np.zeros(len(keys)*2*dim)).reshape((2,-1))
            T[0,keys.index(n)*dim:keys.index(n)*dim+3] = p1
            T[1,keys.index(n)*dim+3:keys.index(n)*dim+6] = p1
            if dim==8:
                T[0,keys.index(n)*dim+6:keys.index(n)*dim+8] = \
                    [-p1[0]*p2[0], -p1[1]*p2[0]]
                T[1,keys.index(n)*dim+6:keys.index(n)*dim+8] = \
                    [-p1[0]*p2[1], -p1[1]*p2[1]]
            CX = (Dk.I + T.T * Dl.I * T).I
            CL = CX * T.T* Dl.I
            CV = np.mat(np.diag(np.ones(dim*len(keys)))) - CX * T.T * Dl.I * T
            V = CL * L + CV * V
            Dk = CV * Dk * CV.T + CL * Dl * CL.T

    for (n1,n2), (pts1, pts2) in relation:
        for p1, p2 in zip(pts1, pts2):
            L = np.mat([0,0]).T
            Dl = np.mat(np.diag(np.ones(2)))
            T = np.mat(np.zeros(len(keys)*2*dim)).reshape((2,-1))
            T[0,keys.index(n1)*dim:keys.index(n1)*dim+3] = p1
            T[1,keys.index(n1)*dim+3:keys.index(n1)*dim+6] = p1
            T[0,keys.index(n2)*dim:keys.index(n2)*dim+3] = -p2
            T[1,keys.index(n2)*dim+3:keys.index(n2)*dim+6] = -p2
            if dim==8:
                T[0,keys.index(n1)*dim+6:keys.index(n1)*dim+8] = \
                    [-p1[0]*p2[0], -p1[1]*p2[0]]
                T[1,keys.index(n1)*dim+6:keys.index(n1)*dim+8] = \
                    [-p1[0]*p2[1], -p1[1]*p2[1]]
                
                T[0,keys.index(n2)*dim+6:keys.index(n2)*dim+8] = \
                    [p1[0]*p2[0], p1[1]*p2[0]]
                T[1,keys.index(n2)*dim+6:keys.index(n2)*dim+8] = \
                    [p1[0]*p2[1], p1[1]*p2[1]]
                
            CX = (Dk.I + T.T * Dl.I * T).I
            CL = CX * T.T* Dl.I
            CV = np.mat(np.diag(np.ones(dim*len(keys)))) - CX * T.T * Dl.I * T
            V = CL * L + CV * V
            Dk = CV * Dk * CV.T + CL * Dl * CL.T
    return V.reshape((-1, dim))

if __name__ == '__main__':
    pts1 = np.array([(0,0),(1,0),(1,1),(0,1)]).T
    pts1 = np.vstack((pts1, np.ones(pts1.shape[1]))).astype(np.float32)

    trans = np.array([(1,0.5, 1),(0.2,1,1),(0.1,0.1,1)])
    pts2 = np.dot(trans, pts1).astype(np.float32)
    pts2 /= pts2[2]
    print(pts2)
    rst = match_multi([1,2], [((1,2),(pts1.T, pts2.T))], [(2, pts2.T, pts2.T)], 8)
    print(rst)
