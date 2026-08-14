import numpy as np
from sklearn.decomposition import PCA

def compute_and_save_pca_basis(data, save_path, variance_threshold=0.95):
    """
    Compute PCA of the given data, find basis vectors capturing specified variance,
    and save the basis as a numpy file.

    Parameters:
    - data: numpy.ndarray, shape (n_samples, n_features)
    - save_path: str, path to save the PCA basis (.npy file)
    - variance_threshold: float, variance that should be retained (default 0.95)
    """

    # Fit PCA
    pca = PCA()
    pca.fit(data)

    # Find number of components to get at least variance_threshold total variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1

    # Refit PCA with reduced number of components
    pca = PCA(n_components=n_components)
    pca.fit(data)

    # Get the basis vectors
    pca_basis = pca.components_  # shape: (n_components, n_features)

    # Save the basis
    np.save(save_path, pca_basis)
    print(f"PCA basis (capturing {variance_threshold*100:.1f}% variance) saved to {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to input .npy file containing data matrix (samples x features)")
    parser.add_argument("output", type=str, help="Path to output .npy file for saving PCA basis")
    parser.add_argument("--variance", type=float, default=0.95, help="Variance threshold to capture (default=0.95)")
    args = parser.parse_args()

    # Load data
    data = np.load(args.input)
    compute_and_save_pca_basis(data, args.output, variance_threshold=args.variance)
