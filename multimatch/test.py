import cv2, numpy as np
from skimage.data import camera
import matplotlib.pyplot as plt

def wiener():
    B = np.zeros((pts1.shape[1]*2, 6))
    B[::2,:3] = pts1.T
    B[1::2,3:] = pts1.T
    Y = pts2.T[:,:2].ravel()
    X = np.dot(inv(np.dot(B.T, B)), np.dot(B.T, Y))
    return X.reshape((2,3))

def kalman(pts1, pts2):
    V = np.mat(np.zeros((6, 1)))
    Dk = np.mat(np.diag(np.ones(6)*1e8))
    for p1,p2 in zip(pts1.T, pts2.T):
        L = np.mat(p2[:2]).T
        Dl = np.mat(np.diag(np.ones(2)))
        T = np.mat([p1,[0,0,0],[0,0,0],p1]).reshape((2,-1))
        CX = (Dk.I + T.T * Dl.I * T).I
        CL = CX * T.T* Dl.I
        CV = np.mat(np.diag(np.ones(6))) - CX * T.T * Dl.I * T
        V = CL * L + CV * V
        Dk = CV * Dk * CV.T + CL * Dl * CL.T
    return V.reshape((2,3))
