import torch
import torch.nn as nn
from typing import Tuple, Optional

class PCAFactorModel:
    “”“
    PCA-based dimensionality reduction for asset returns.
    
    Finds the optimal linear compression that maximizes preserved variance.
    Unlike economic factors, the compression basis is learned from data.
    “”“
    
    def __init__(self, n_components: int, device: str = ‘cpu’):
        self.n_components = n_components
        self.device = device
        self.components_ = None  # Principal directions (n_assets, n_components)
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, returns: torch.Tensor) -> ‘PCAFactorModel’:
        “”“
        Fit PCA to return matrix.
        
        Learns the K-dimensional subspace that captures maximum variance.
        
        Args:
            returns: (n_periods, n_assets) tensor of returns
            
        Returns:
            self
        “”“
        returns = returns.to(self.device)
        
        # Center the data (remove mean return per asset)
        self.mean_ = returns.mean(dim=0)
        centered = returns - self.mean_
        
        # Compute sample covariance matrix
        n_samples = centered.shape[0]
        cov = (centered.T @ centered) / (n_samples - 1)
        
        # Eigendecomposition gives principal directions
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Reverse to descending order (largest variance first)
        eigenvalues = torch.flip(eigenvalues, dims=[0])
        eigenvectors = torch.flip(eigenvectors, dims=[1])
        
        # Store top K components (the compression basis)
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / eigenvalues.sum()
        )
        
        return self
    
    def transform(self, returns: torch.Tensor) -> torch.Tensor:
        “”“
        Compress returns to K-dimensional factor space.
        
        Args:
            returns: (n_periods, n_assets) tensor
            
        Returns:
            factor_scores: (n_periods, n_components) compressed representation
        “”“
        centered = returns.to(self.device) - self.mean_
        return centered @ self.components_
    
    def inverse_transform(self, factor_scores: torch.Tensor) -> torch.Tensor:
        “”“
        Decompress from factor space back to asset space.
        
        Args:
            factor_scores: (n_periods, n_components) tensor
            
        Returns:
            reconstructed: (n_periods, n_assets) tensor
        “”“
        return factor_scores @ self.components_.T + self.mean_
    
    def get_reconstruction_error(self, returns: torch.Tensor) -> float:
        “”“
        Compute information loss from compression.
        
        Lower is better—means the K dimensions capture more structure.
        
        Args:
            returns: (n_periods, n_assets) tensor
            
        Returns:
            mse: Mean squared reconstruction error
        “”“
        reconstructed = self.inverse_transform(self.transform(returns))
        return ((returns.to(self.device) - reconstructed) ** 2).mean().item()
    
    def get_residuals(self, returns: torch.Tensor) -> torch.Tensor:
        “”“
        Extract the part of returns NOT explained by K factors.
        
        These residuals live in the (N-K)-dimensional subspace
        orthogonal to the principal components.
        
        Args:
            returns: (n_periods, n_assets) tensor
            
        Returns:
            residuals: (n_periods, n_assets) tensor
        “”“
        reconstructed = self.inverse_transform(self.transform(returns))
        return returns.to(self.device) - reconstructed
    
    def get_factor_covariance(self) -> torch.Tensor:
        “”“
        Return the covariance matrix in the compressed space.
        
        This is diagonal since PCA factors are orthogonal.
        
        Returns:
            factor_cov: (n_components, n_components) diagonal matrix
        “”“
        return torch.diag(self.explained_variance_)
    
    def reconstruct_covariance(self) -> torch.Tensor:
        “”“
        Reconstruct the full covariance matrix from K factors.
        
        Returns:
            cov: (n_assets, n_assets) approximation to true covariance
        “”“
        # Σ ≈ Λ Σ_F Λ^T where Σ_F is diagonal (eigenvalues)
        return self.components_ @ torch.diag(self.explained_variance_) @ self.components_.T


def demo_pca_dimensionality():
    “”“Demonstrate PCA as dimensionality reduction.”“”
    
    torch.manual_seed(42)
    
    # Simulate returns with known factor structure
    n_periods, n_assets, n_true_factors = 252, 100, 3
    
    # True low-dimensional structure
    true_factors = torch.randn(n_periods, n_true_factors) * 0.02
    true_loadings = torch.randn(n_assets, n_true_factors)
    
    # Full returns = low-rank structure + noise
    idio_vol = 0.03
    returns = true_factors @ true_loadings.T + torch.randn(n_periods, n_assets) * idio_vol
    
    print(”=== Dimensionality Reduction Analysis ===\n”)
    print(f”Original dimensionality: {n_assets} assets”)
    print(f”True latent dimensionality: {n_true_factors} factors”)
    print(f”Observations: {n_periods} periods\n”)
    
    # Fit PCA with varying K
    print(”Compression quality by number of dimensions:”)
    print(”-” * 50)
    
    for k in [1, 2, 3, 5, 10, 20]:
        pca = PCAFactorModel(n_components=k)
        pca.fit(returns)
        
        explained = pca.explained_variance_ratio_.sum().item()
        recon_error = pca.get_reconstruction_error(returns)
        
        compression_ratio = n_assets / k
        
        print(f”K={k:2d} | Variance explained: {explained:5.1%} | “
              f”MSE: {recon_error:.6f} | Compression: {compression_ratio:.0f}x”)
    
    print(”\n” + “=” * 50)
    print(”\nOptimal choice: K=3 captures true structure”)
    print(”K>3 adds dimensions that model noise, not signal”)
    
    # Show eigenvalue spectrum
    pca_full = PCAFactorModel(n_components=20)
    pca_full.fit(returns)
    
    print(”\nEigenvalue spectrum (scree plot data):”)
    for i in range(10):
        var = pca_full.explained_variance_ratio_[i].item()
        bar = “█” * int(var * 100)
        print(f”PC{i+1:2d}: {var:5.1%} {bar}”)


if __name__ == “__main__”:
    demo_pca_dimensionality()
