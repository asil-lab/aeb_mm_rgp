import urllib.request
import zipfile
import tarfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def extract_real_data(
    plotting=False,
    meas_drop_rate=0.9,
    url="https://download.ifi.uzh.ch/rpg/AGZ_data/AGZ_subset.zip",
    download_dir="./data/uzh_sample_data",
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    download_path = download_dir / filename

    # Download the file if it doesn't exist
    if not download_path.exists():
        print(f"Downloading data from {url}...")
        urllib.request.urlretrieve(url, download_path)

        print(f"Extracting {filename}...")
        if filename.endswith(".zip"):
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(download_dir)
        elif filename.endswith((".tar.gz", ".tgz")):
            with tarfile.open(download_path, "r:gz") as tar_ref:
                tar_ref.extractall(download_dir)
        elif filename.endswith(".tar"):
            with tarfile.open(download_path, "r:") as tar_ref:
                tar_ref.extractall(download_dir)
        else:
            print(f"Unknown archive format: {filename}, skipping extraction.")
    else:
        print(f"Data file already exists at {download_path}, skipping download.")

    # File location
    datafile = "./data/uzh_sample_data/AGZ_subset/Log Files/GroundTruthAGL.csv"

    # Load data from CSV, skip header (startRow = 2 → skip first row)
    data = np.genfromtxt(datafile, delimiter=",", skip_header=1)

    # Assign columns to variables (each is a NumPy array)
    imgid = data[:, 0]
    x_gt = data[:, 1]
    y_gt = data[:, 2]
    z_gt = data[:, 3]
    omega_gt = data[:, 4]
    phi_gt = data[:, 5]
    kappa_gt = data[:, 6]
    x_gps = data[:, 7]
    y_gps = data[:, 8]
    z_gps = data[:, 9]

    # reformat data, assume ground truth begins at the origin
    data_interval = np.arange(2650)
    true_y = x_gt.reshape((-1,))[data_interval] - x_gt[0]
    true_x = -(y_gt.reshape((-1,))[data_interval] - y_gt[0])
    meas_y = x_gps.reshape((-1,))[data_interval] - x_gt[0]
    meas_x = -(y_gps.reshape((-1,))[data_interval] - y_gt[0])

    # np.random.seed(42)
    T_TOTAL = true_x.shape[0]  # number of time steps
    TIME = np.arange(T_TOTAL)
    mask = np.random.rand(T_TOTAL) > meas_drop_rate
    mask[0] = True
    mask[-1] = True

    # Insert NaNs where missing.
    meas_x[~mask] = np.nan
    meas_y[~mask] = np.nan

    # Save data to a .npz file
    np.savez(
        "./data/uzh_sample_data/data.npz",
        time=TIME,
        true_x=true_x,
        true_y=true_y,
        meas_x=meas_x,
        meas_y=meas_y,
        mask=mask,
    )

    # Plotting functionality
    if plotting:
        plt.figure(figsize=(10, 6))

        # Plot the true trajectory
        plt.plot(true_x, true_y, label="Ground Truth", color="b", linewidth=2)

        # Plot the noisy measurements (with NaNs excluded from the plot)
        plt.scatter(meas_x, meas_y, c="r", label="Noisy Measurements", s=10, alpha=0.7)

        # Adding labels and legend
        plt.title("2D Trajectory with Noisy Measurements")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()
