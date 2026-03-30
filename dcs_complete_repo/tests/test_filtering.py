import numpy as np

from dcs.filtering import StableIncPCAKNNFilter


def test_projection_dim_stable():
    filt = StableIncPCAKNNFilter(proj_dim=5, n_neighbors=3)

    # first round: fewer than proj_dim samples
    X1 = np.random.randn(3, 100).astype(np.float32)
    Z1 = filt.project(X1)
    assert Z1.shape == (3, 5)

    # second round: more samples enables PCA
    X2 = np.random.randn(10, 100).astype(np.float32)
    Z2 = filt.project(X2)
    assert Z2.shape == (10, 5)

    # reference update and detect
    filt.update_reference(Z2)
    anom, scores = filt.detect(Z1)
    assert anom.shape == (3,)
    assert scores.shape == (3,)
