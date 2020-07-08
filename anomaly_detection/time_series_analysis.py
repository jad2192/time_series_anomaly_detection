import numpy as np


class HMM(object):
    def __init__(self, brand, counts, num_states=2, num_obs=3):
        self.mem = 20
        self.brand = brand
        self.counts = np.asanyarray(counts, dtype='i4')
        self.N = num_states  # number of hidden states
        self.K = num_obs  # number of possible observations
        # Initialize the transition probability matrix A.
        if self.N == 2:
            self.A = np.asanyarray([[0.9, 0.1], [0.1, 0.9]])
        else:
            A = np.random.uniform(size=(self.N, self.N))
            d = 1 / A.sum(axis=1)
            self.A = np.einsum('ij,i->ij', A, d)
        # Initialize hidden state probabilities, assume last position
        # is safety issue.
        if self.N == 2:
            self.pi = np.asanyarray([0.95, 0.05])
        else:
            pi = np.random.uniform(size=self.N)
            self.pi = pi / pi.sum()
        # Initialize the emission probability matrix B.
        if self.N == 2 and self.K == 3:
            self.B = np.asanyarray([[0.70, 0.2, 0.1], [0.01, 0.05, 0.94]])
        else:
            B = np.random.uniform(size=(self.N, self.K))
            d = 1 / B.sum(axis=1)
            self.B = np.einsum('ij,i->ij', B, d)

    # The next three functions are to carry out the Baum-Welch Algorithm as
    # described here:
    # https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
    # I have tried to use the same variable names as the artcile when possible
    def forward(self, Y):  # Y is the sequence of observations
        T = len(Y)
        alpha = np.zeros((self.N, T))
        alpha[:, 0] = self.pi * self.B[:, Y[0]]
        for t in range(1, T):
            alpha[:, t] = self.B[:, Y[t]] * np.einsum('n,nm->m',
                                                      alpha[:, t - 1], self.A)
        return alpha

    def backward(self, Y):
        T = len(Y)
        beta = np.ones((self.N, T))
        for t in range(2, T + 1):
            beta[:, -t] = np.einsum('m,nm->n',
                                    beta[:, 1 - t] * self.B[:, Y[1 - t]], self.A)
        return beta

    def fit_params(self, Y):
        T = len(Y)
        alpha = self.forward(Y)
        beta = self.backward(Y)
        # compute gamma
        gamma = alpha * beta
        d = 1 / gamma.sum(axis=0)
        gamma = np.einsum('m,nm->nm', d, gamma)
        # update pi
        self.pi = gamma[:, 0]
        # compute xi
        xi_1 = np.einsum('nm,nt->tnm', self.A, alpha[:, :-1])
        beta_y = np.zeros((self.N, T - 1))
        for t in range(T - 1):
            beta_y[:, t] = beta[:, t + 1] * self.B[:, Y[t + 1]]
        xi = np.einsum('tnm,mt->tnm', xi_1, beta_y)
        xi = np.einsum('t,tnm->tnm', xi.sum(axis=(1, 2)), xi)
        # update A
        self.A = np.einsum('n,nm->nm', 1 / gamma[:, :-1].sum(axis=1),
                           xi.sum(axis=0))
        d = 1 / self.A.sum(axis=1)
        self.A = np.clip(np.einsum('ij,i->ij', self.A, d), 1e-6, 1 - 1e-6)
        # update B
        chi = np.zeros((T, self.K))
        for t in range(T):
            chi[t, Y[t]] = 1
        b = np.dot(gamma, chi)
        self.B = np.einsum('n,nm->nm', 1 / gamma.sum(axis=1), b)
        d = 1 / self.B.sum(axis=1)
        self.B = np.clip(np.einsum('ij,i->ij', self.B, d), 1e-6, 1 - 1e-6)

    def fit(self, int_len=7):
        Y_f = self.counts
        T = len(Y_f)
        self.mem = int_len
        for k in range(-1, -T, -int_len):
            if k == -1 and Y_f[k - int_len + 1:].sum() > int_len * 0.1:
                self.fit_params(Y_f[k - int_len + 1:])
            elif all([len(Y_f[k - int_len + 1:k + 1]) == 7,
                      Y_f[k - int_len + 1:k + 1].sum() > int_len * 0.1]):
                self.fit_params(Y_f[k - int_len + 1:k + 1])

    def anomaly_prob(self, t_i):
        if t_i < self.mem:
            Y = self.counts[:t_i + 1]
        else:
            Y = self.counts[t_i - self.mem + 1: t_i + 1]
        alpha = self.forward(Y)
        beta = self.backward(Y)
        # print(alpha)
        # print(beta)
        gamma = alpha * beta
        d = 1 / gamma.sum(axis=0)
        # print(d)
        gamma = np.einsum('m,nm->nm', d, gamma)
        return gamma[1, -1]


class EARS(object):
    def __init__(self, brand, counts, c=3):
        self.brand = brand
        self.counts = np.asanyarray(counts)
        self.c = c

    def get_c_stat(self, t, window_size=7):
        T = window_size + 2
        mean = self.counts[t - T:t - 2].mean()
        std = np.sqrt((1 / (window_size - 1)) *
                      ((self.counts[t - T:t - 2] - mean) ** 2).sum())
        if std != 0:
            return (self.counts[-1] - mean) / std
        else:
            return self.counts[-1] - mean

    def predict_anomaly(self, window_size=7):
        if self.c == 3:
            c3 = (max(0, self.get_c_stat(-1, window_size) - 1) +
                  max(0, self.get_c_stat(-2, window_size) - 1) +
                  max(0, self.get_c_stat(-3, window_size) - 1))
            c3_n = (max(0, -self.get_c_stat(-1, window_size) - 1) +
                    max(0, -self.get_c_stat(-2, window_size) - 1) +
                    max(0, -self.get_c_stat(-3, window_size) - 1))
            if c3 > 4:
                return True, 'inc'
            elif c3_n > 5:
                return True, 'dec'
            else:
                return False, ''
        elif self.c == 2:
            return self.get_c_stat(-1, window_size) > 4
        else:
            return self.get_c_stat(1, window_size) > 4
