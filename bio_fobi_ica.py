import numpy as np

class bio_fobi_ica:
    """
 Parameters:
 ====================
 y_dim         -- Dimension of output
 x_dim         -- Dimensions of inputs
 dataset       -- Input dataset to use the optimal learning rates that were found using a grid search
 M0            -- Initialization for the lateral weight matrix M, must be of size z_dim by z_dim
 Wx0           -- Initialization for the feedforward weight matrix Wx, must be of size y_dim by x_dim
 Lambda_Matrix -- Setting of the diagonal matrix containng the kurtosis of the sources
 eta           -- Learning rate
 tau           -- Ratio of Wx learning rate and M learning rate

 Methods:
 ========
 fit_next()
 """

    def __init__(self, x_dim, y_dim, dataset_name=None, M0=None, Wx0=None, Wy0=None, eta=None, tau=0.1,
                 Lambda_Matrix=None):

        # synaptic weight initializations

        if M0 is not None:
            assert M0.shape == (y_dim, y_dim)
            M = M0
        else:
            M = np.eye(z_dim)

        if Wx0 is not None:
            assert Wx0.shape == (y_dim, x_dim)
            Wx = Wx0
        else:
            Wx = np.random.randn(y_dim, x_dim) / np.sqrt(x_dim)

        # optimal hyperparameters for test datasets

        if dataset_name is not None:
            if dataset_name == 'synthetic':
                def eta(t):
                    return 1e-3 / (1 + 1e-4 * t)

                tau = 0.1
            elif dataset_name == 'speech':
                def eta(t):
                    return 1e-2 / (1 + 1e-4 * t)

                tau = 0.1
            elif dataset_name == 'images':
                def eta(t):
                    return 1e-2 / (1 + 1e-4 * t)

                tau = 0.1
            else:
                print('The optimal learning rates for this dataset are not stored')

        # default learning rate:

        if eta is None:
            def eta(t): return 1e-3 / (1 + 1e-4 * t)

        self.Lambda_Matrix = Lambda_Matrix
        self.t = 0
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.M = M
        self.Minv = np.linalg.inv(M)
        self.Wx = Wx
        self.eta = eta
        self.tau = tau

    def fit_next(self, x):

        t, tau, Wx, M, Minv = self.t, self.tau, self.Wx, self.M, self.Minv

        # project inputs

        c = Wx @ x

        # neural dynamics

        y = Minv @ (c)

        # synaptic updates

        eta = self.eta

        nonlin_alpha = (sum(y * y) ** 1)
        scaled_c = nonlin_alpha * (np.diag(self.Lambda_Matrix) @ c)

        Wx += 2 * eta * np.outer(y-scaled_c, x)

        M += (eta / tau) * (np.outer(y, y) - np.eye(self.Lambda_Matrix.shape[0]))
        Minv = np.linalg.inv(M)

        self.Wx = Wx
        self.M = M
        self.Minv = Minv

        self.t += 1

        return y