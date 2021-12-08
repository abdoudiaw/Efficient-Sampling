class Hartmann(object):

    def __init__(self, ndim=6):
        r"""Hartmann test function.

    Most commonly used is the six-dimensional version (typically evaluated on
    `[0, 1]^6`):

        H(x) = - sum_{i=1}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij)**2 )

    H has a 6 local minima and a global minimum at

        z = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)

    with `H(z) = -3.32237`.
        """
        if ndim not in (3, 4, 6):
            raise ValueError(f"Hartmann with dim {ndim} not defined")
        self.dim = ndim
        # optimizers and optimal values
        optvals = {3: -3.86278, 6: -3.32237}
        optimizers = {
            3: [(0.114614, 0.555649, 0.852547)],
            6: [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)],
        }
        self._optimal_value = optvals.get(self.dim)
        self._optimizers = optimizers.get(self.dim)
        self.ALPHA = [1.0, 1.2, 3.0, 3.2]
        if ndim == 3:
            A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            P = [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        elif ndim == 4:
            A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            P = [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        else:#if ndim == 6:
            A = [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
            P = [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        self.A = A
        self.P = P

    def function(self, x):
        import numpy as np
        x = np.expand_dims(np.array(x), axis=-2)
        inner_sum = np.sum(self.A * (x - 0.0001 * np.array(self.P))**2, axis=-1)
        H = -(np.sum(self.ALPHA * np.exp(-inner_sum), axis=-1))
        if self.dim == 4:
            H = (1.1 + H) / 0.839
        return H

    def __call__(self, x):
        return self.function(x)


# instances
hartmann = Hartmann(ndim=6).function
