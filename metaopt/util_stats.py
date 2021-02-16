import numpy as np


def get_correlation(X, Y):

    Nx,Dx = X.shape
    Ny,Dy = Y.shape

    corr_matrix = np.zeros((Nx,Ny))
    for x_i in range(Nx):
        for y_j in range(Ny):

            x = X[x_i,:]
            y = Y[y_j,:]
            norm = np.sqrt((x**2).sum())*np.sqrt((y**2).sum())
            corr_ij = np.dot(x,y) / norm
            corr_matrix[x_i,y_j] = corr_ij

    return  corr_matrix                  

