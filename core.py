
import numpy as np
from ball_tree import BallTree  ## TODO: force python to only look locally
from faster_sandwich_filling import multiply_XeeX, CutoffError

def cross_section(y, X, lat_long, cutoff, kernel = 'uniform'):
    N = y.shape[0]
    assert lat_long.shape == (N, 2)
    assert X.shape[0] == N
    assert y.shape == (N,) or y.shape[1] == 1
    #kernel_fn = get_kernel_fn(kernel)
    assert cutoff > 0
    k = X.shape[1]
    bread = np.linalg.inv(X.T @ X)
    # I have no idea if this leaf_size is reasonable.  If running out of memory, divide N by a larger number.
    # 40 is the default.
    leaf_size = max(40, N // 1000)
    betahat =  bread @ X.T @ y  # '@' is matrix multiplication, equivalent to np.dot or __mul__ on matrices
    #betahat = sm.OLS(Y,X).fit().params  # probably better
    residuals = (y - X @ betahat)[:, 0]  # reshape from (N, 1) to (N,)
    balltree = BallTree(lat_long, metric = 'greatcircle', leaf_size = leaf_size)
    if kernel == 'uniform':
        neighbors = balltree.query_radius(lat_long, r = cutoff)
        filling = multiply_XeeX(neighbors, residuals, X, kernel)
    else:
        neighbors, neighbor_distances = balltree.query_radius(lat_long, r = cutoff, return_distance = True)
        filling = multiply_XeeX(neighbors, residuals, X, kernel, distances = neighbor_distances, cutoff = cutoff)
    del balltree
    sandwich = N * (bread.T @ filling @ bread)
    se = np.sqrt(np.diag(sandwich)).reshape(-1, 1)
    #results = np.hstack((betahat, se))
    return se


def panel():
    pass
