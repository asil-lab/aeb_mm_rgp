import numpy as np
import matplotlib.pyplot as plt


def generate_simulated_data(
    plotting=False,
    t_total=1500,
    dt=0.01,
    sigma_meas=0.1,
    meas_drop_rate=0.9,
    mc_runs=250,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
    time = np.arange(t_total) * dt

    # Ground truth
    true_x = time * np.cos(time + np.log(time + 1.05) / 5) + 0.011
    true_y = time * np.log(np.sin(time) ** 2 + 2) + 0.01

    meas_x_list = [None] * mc_runs
    meas_y_list = [None] * mc_runs
    mask_list = [None] * mc_runs

    for n in range(mc_runs):
        # random mask to drop measurements
        mask = np.random.rand(t_total) > meas_drop_rate  # True = measurement available
        mask[0] = True
        mask[-1] = True

        meas_x = true_x + sigma_meas * np.random.randn(t_total)
        meas_y = true_y + sigma_meas * np.random.randn(t_total)

        # insert nans where missing.
        meas_x[~mask] = np.nan
        meas_y[~mask] = np.nan

        meas_x_list[n] = meas_x
        meas_y_list[n] = meas_y
        mask_list[n] = mask

    # save
    np.savez(
        "./data/simulated_data/data.npz",
        time=time,
        true_x=true_x,
        true_y=true_y,
        meas_x=meas_x_list,
        meas_y=meas_y_list,
        mask=mask_list,
    )

    # plotting
    if plotting:
        plt.figure(figsize=(10, 6))

        # trajectory
        plt.plot(true_x, true_y, label="Ground Truth", color="b", linewidth=2)

        # measurements
        plt.scatter(
            meas_x_list[0],
            meas_y_list[0],
            c="r",
            label="Noisy Measurements",
            s=10,
            alpha=0.7,
        )

        plt.title("2D Trajectory with Noisy Measurements")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()
