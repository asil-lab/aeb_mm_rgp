from typing import List, Tuple, Sequence
import numpy as np
import scipy as sp
from filterpy.kalman import KalmanFilter, IMMEstimator


## IMM-CV algorithm
class IMMCV(IMMEstimator):
    def __init__(self, q_levels, r, dt, dim: int = 2):
        num_modes = len(q_levels)
        no_tran_prob = 0.9
        tran_probs = (1 - no_tran_prob) / (num_modes - 1)
        M = (
            np.eye(num_modes) * (no_tran_prob - tran_probs)
            + np.ones(num_modes) * tran_probs
        )
        filters: List[KalmanFilter] = [self.cv_kf(dt, q, r, dim) for q in q_levels]
        mu = np.full(num_modes, 1.0 / num_modes)
        super().__init__(filters, mu, M)

    # constant vel KF template
    def cv_kf(self, dt: float, q: float, r: float, dim: int) -> KalmanFilter:
        kf = KalmanFilter(dim_x=2 * dim, dim_z=dim)
        kf.F = np.kron(
            np.eye(dim),
            np.array(
                [
                    [1, dt],
                    [0, 1],
                ]
            ),
        )
        kf.H = np.kron(
            np.eye(dim),
            np.array(
                [
                    1,
                    0,
                ]
            ),
        )
        kf.Q = q * np.eye(2 * dim)
        kf.R = r * np.eye(dim)
        kf.P *= 100.0  # large initial uncertainty
        return kf


## RGP*MT algorithm in 1D
class RGPMT:
    def __init__(
        self,
        window: int,
        t_init: np.ndarray,
        f_init: np.ndarray,
        kernel,
        *,
        theta_init: np.ndarray = None,
        q_f: float = 1e-1,
        q_th: float = 1e-5,
        jitter: float = 1e-6,
    ):
        self.kernel = kernel
        self.W = window
        self.n = window + 3  # state dimension W + 3 hyper‑parameters
        self.jitter = jitter

        self.x = np.zeros(self.n)
        self.x[:window] = np.asarray(np.nan_to_num(f_init), dtype=float)
        if theta_init is None:
            theta_init = np.log([350.0, 10.0, 5.0])
        self.x[-3:] = np.log(theta_init.astype(float))

        # initial cov
        ell, sigma_f, sigma_n = np.exp(self.x[-3:])
        K_ff = self.kernel(t_init, t_init, ell, sigma_f)
        K_ff += (sigma_n**2 + jitter) * np.eye(window)

        # cross cov
        cross = np.full((window, 3), 0)
        P_theta = np.diag([1e-6, 1e-6, 1e-6])
        self.P = np.block([[K_ff, cross], [cross.T, P_theta]])

        # process noise
        self.Q = np.diag(np.concatenate([np.full(window, q_f**2), np.full(3, q_th**2)]))

        # ukf params
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self.compute_ut_weights()

        # time window
        self.t_win = np.asarray(t_init, dtype=float)

        # est mean and var
        self.est_mean = []
        self.est_var = []

    def compute_ut_weights(self) -> None:
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n
        c = n + lam
        self.gamma = np.sqrt(c)

        self.Wm = np.full(2 * n + 1, 1.0 / (2 * c))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / c
        self.Wc[0] = lam / c + (1 - self.alpha**2 + self.beta)

    def sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        A = self.gamma * np.linalg.cholesky(cov + self.jitter * np.eye(self.n))
        return np.column_stack([mean, mean[:, None] + A, mean[:, None] - A])

    def propagate_sigma_point(
        self, s: np.ndarray, t_prev: np.ndarray, t_k: float
    ) -> Tuple[np.ndarray, float]:
        W = self.W
        f_prev = s[:W]
        log_theta = s[W:]
        ell, sigma_f, _ = np.exp(log_theta)

        K = self.kernel(t_prev, t_prev, ell, sigma_f) + self.jitter * np.eye(W)
        k_star = self.kernel(t_prev, np.array([t_k]), ell, sigma_f).flatten()

        mu_pred = k_star @ np.linalg.solve(K, f_prev)
        f_new = np.concatenate([f_prev[1:], [mu_pred]])
        y = np.concatenate([f_new, log_theta])
        return y, mu_pred  # propagated state, predicted measurement

    # update
    def step(self, z_k: float, t_k: float) -> Tuple[float, float, np.ndarray]:
        t_prev = self.t_win.copy()

        # ut
        chi = self.sigma_points(self.x, self.P)
        n_sigma = chi.shape[1]
        Y_sigma = np.zeros_like(chi)
        z_sigma = np.zeros(n_sigma)

        for i in range(n_sigma):
            Y_sigma[:, i], z_sigma[i] = self.propagate_sigma_point(
                chi[:, i], t_prev, t_k
            )

        # pred mean cov
        x_pred = Y_sigma @ self.Wm
        dY = Y_sigma - x_pred[:, None]
        P_pred = dY @ np.diag(self.Wc) @ dY.T + self.Q
        P_pred = 0.5 * (P_pred + P_pred.T)  # numerical symmetry

        # pred meas
        z_pred = np.dot(self.Wm, z_sigma)
        sigma_n2_pred = np.exp(2 * x_pred[-1])
        P_zz = np.sum(self.Wc * (z_sigma - z_pred) ** 2) + sigma_n2_pred
        P_xz = (Y_sigma - x_pred[:, None]) @ np.diag(self.Wc) @ (z_sigma - z_pred)

        # update
        if not np.isnan(z_k):
            K = P_xz / P_zz  # kalman gain
            self.x = x_pred + K * (z_k - z_pred)
            self.P = P_pred - np.outer(K, K) * P_zz
        else:
            self.x, self.P = x_pred, P_pred
        self.t_win = np.concatenate([t_prev[1:], [t_k]])

        # output
        mu_k = self.x[self.W - 1]
        var_k = self.P[self.W - 1, self.W - 1]
        self.est_mean.append(mu_k)
        self.est_var.append(var_k)
        return mu_k, var_k, np.exp(self.x[-3:])

    # hyperparams
    @property
    def theta(self) -> np.ndarray:
        return np.exp(self.x[-3:])


# N dimesnional generalization
class RGPMT_ND:
    def __init__(
        self,
        window: int,
        t_init: np.ndarray,
        f_init: Sequence[np.ndarray],
        kernel,
        *,
        theta_init: np.ndarray = None,
        q_f: float = 1e-1,
        q_th: float = 1e-5,
        jitter: float = 1e-6,
        dim: int = 2,
    ) -> None:
        self.dim = dim
        self.filters: List[RGPMT] = []
        for i in range(dim):
            self.filters.append(
                RGPMT(
                    window,
                    t_init,
                    f_init[i],
                    kernel,
                    theta_init=theta_init,
                    q_f=q_f,
                    q_th=q_th,
                    jitter=jitter,
                )
            )

    def step(self, z_k: np.ndarray, t_k: float) -> np.ndarray:
        mu_k = np.zeros(self.dim)
        for i in range(self.dim):
            mu, var, theta = self.filters[i].step(z_k[i], t_k)
            mu_k[i] = mu
        return mu_k

    def predicted_mean(self):
        preds = []
        for f in self.filters:
            preds.append(f.est_mean[:-1])
        return preds


## Proposed
class MMRGP(IMMEstimator):
    def __init__(
        self,
        kernel,
        end_time: float,
        length_scales: Sequence[float],
        r: float,
        num_gp_states: int,
        q: float,
        sigma_prior: float,
        dim: int = 2,
    ):
        self.kernel = kernel
        self.length_scales = length_scales
        self.num_gp_states = num_gp_states
        self.q = q
        self.r = r
        self.sigma_prior = sigma_prior
        self.dim = dim

        self.num_models = len(length_scales)
        no_tran_prob = 0.9
        tran_probs = (1 - no_tran_prob) / (self.num_models - 1)
        M = (
            np.eye(self.num_models) * (no_tran_prob - tran_probs)
            + np.ones(self.num_models) * tran_probs
        )
        mu = np.full(self.num_models, 1.0 / self.num_models)

        self.gp_inputs = np.linspace(0, end_time, num_gp_states, endpoint=True)
        gp_cov = [
            kernel(self.gp_inputs, self.gp_inputs, ls, sigma_prior)
            for ls in length_scales
        ]
        self.inv_prior_cov = [np.linalg.pinv(c) for c in gp_cov]
        filters = [
            self.rgp_kf(num_gp_states, gp_cov[n], r, q, dim)
            for n in range(self.num_models)
        ]
        super().__init__(filters, mu, M)

    # TODO check
    def time_update(self, u=None):
        for f in self.filters:
            f.Q = self.q * f.P_post
        return super().predict(u)

    def meas_update(self, z, meas_time):
        for i in range(self.num_models):
            H = (
                self.kernel(
                    meas_time, self.gp_inputs, self.length_scales[i], self.sigma_prior
                )
                @ self.inv_prior_cov[i]
            )
            R = (
                self.kernel(
                    meas_time, meas_time, self.length_scales[i], self.sigma_prior
                )
                - self.kernel(
                    meas_time, self.gp_inputs, self.length_scales[i], self.sigma_prior
                )
                @ self.inv_prior_cov[i]
                @ self.kernel(
                    meas_time, self.gp_inputs, self.length_scales[i], self.sigma_prior
                ).T
                + self.r
            )

            self.filters[i].H = sp.linalg.block_diag(*[H] * self.dim)
            self.filters[i].R = sp.linalg.block_diag(*[R] * self.dim)
        return super().update(z)

    def predicted_mean(self, pred_time):
        pred_mat = np.sum(
            [
                self.mu[i]
                * self.kernel(
                    pred_time, self.gp_inputs, self.length_scales[i], self.sigma_prior
                )
                @ self.inv_prior_cov[i]
                for i in range(self.num_models)
            ],
            axis=0,
        )

        return sp.linalg.block_diag(*[pred_mat] * self.dim) @ self.x_post

    def rgp_kf(
        self,
        num_gp_states: int,
        init_cov: float,
        r: float,
        q: float = 1e-3,
        dim: int = 2,
    ) -> KalmanFilter:
        kf = KalmanFilter(dim_x=dim * num_gp_states, dim_z=dim)
        kf.R = r * np.eye(dim)
        kf.Q = q * sp.linalg.block_diag(*[init_cov] * dim)
        kf.P = sp.linalg.block_diag(*[init_cov] * dim)
        return kf
