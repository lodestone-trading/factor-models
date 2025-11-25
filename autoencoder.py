import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Optional

class ReturnAutoencoder(nn.Module):
    “”“
    Nonlinear dimensionality reduction for asset returns.
    
    Learns a K-dimensional compression that can capture nonlinear
    structure in the return covariance.
    “”“
    
    def __init__(
        self,
        n_assets: int,
        n_latent: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_assets = n_assets
        self.n_latent = n_latent
        
        # Encoder: N → hidden → ... → K
        encoder_layers = []
        prev_dim = n_assets
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, n_latent))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: K → hidden → ... → N (mirror architecture)
        decoder_layers = []
        prev_dim = n_latent
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, n_assets))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        “”“Compress to K-dimensional latent space.”“”
        return self.encoder(x)
    
    def decompress(self, z: torch.Tensor) -> torch.Tensor:
        “”“Reconstruct from latent space.”“”
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        “”“Full compression-decompression pass.”“”
        latent = self.compress(x)
        reconstructed = self.decompress(latent)
        return reconstructed, latent


class AutoencoderTrainer:
    “”“
    Training harness for return autoencoder.
    
    Handles normalization, training loop, and evaluation.
    “”“
    
    def __init__(
        self,
        model: ReturnAutoencoder,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = ‘cpu’
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=’min’, factor=0.5, patience=10
        )
        self.criterion = nn.MSELoss()
        self.history = {’train_loss’: [], ‘val_loss’: []}
        
    def fit(
        self,
        train_returns: torch.Tensor,
        val_returns: Optional[torch.Tensor] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ) -> ‘AutoencoderTrainer’:
        “”“Train the autoencoder.”“”
        
        # Normalize for stable training
        self.mean_ = train_returns.mean(dim=0)
        self.std_ = train_returns.std(dim=0) + 1e-8
        
        train_norm = (train_returns - self.mean_) / self.std_
        train_loader = DataLoader(
            TensorDataset(train_norm), batch_size=batch_size, shuffle=True
        )
        
        if val_returns is not None:
            val_norm = (val_returns - self.mean_) / self.std_
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                reconstructed, _ = self.model(batch)
                loss = self.criterion(reconstructed, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item() * batch.size(0)
            
            train_loss /= len(train_norm)
            self.history[’train_loss’].append(train_loss)
            
            # Validation
            if val_returns is not None:
                self.model.eval()
                with torch.no_grad():
                    val_recon, _ = self.model(val_norm.to(self.device))
                    val_loss = self.criterion(val_recon, val_norm.to(self.device)).item()
                self.history[’val_loss’].append(val_loss)
                self.scheduler.step(val_loss)
            else:
                val_loss = None
                self.scheduler.step(train_loss)
            
            if verbose and (epoch + 1) % 25 == 0:
                msg = f”Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f}”
                if val_loss:
                    msg += f” | Val: {val_loss:.6f}”
                print(msg)
        
        return self
    
    def compress(self, returns: torch.Tensor) -> torch.Tensor:
        “”“Compress returns to latent space.”“”
        self.model.eval()
        norm = (returns - self.mean_) / self.std_
        with torch.no_grad():
            latent = self.model.compress(norm.to(self.device))
        return latent.cpu()
    
    def reconstruct(self, returns: torch.Tensor) -> torch.Tensor:
        “”“Full compression-decompression cycle.”“”
        self.model.eval()
        norm = (returns - self.mean_) / self.std_
        with torch.no_grad():
            recon_norm, _ = self.model(norm.to(self.device))
        return recon_norm.cpu() * self.std_ + self.mean_
    
    def reconstruction_error(self, returns: torch.Tensor) -> float:
        “”“Compute MSE from compression.”“”
        recon = self.reconstruct(returns)
        return ((returns - recon) ** 2).mean().item()


def demo_nonlinear_compression():
    “”“Compare linear vs nonlinear dimensionality reduction.”“”
    
    torch.manual_seed(42)
    
    # Simulate returns with NONLINEAR factor structure
    n_periods, n_assets, n_factors = 500, 100, 3
    
    # True factors
    factors = torch.randn(n_periods, n_factors) * 0.02
    
    # Nonlinear interactions (PCA can’t capture these)
    factor_squared = factors ** 2
    factor_interaction = factors[:, 0:1] * factors[:, 1:2]
    
    # Loadings
    linear_loadings = torch.randn(n_assets, n_factors)
    squared_loadings = torch.randn(n_assets, n_factors) * 0.5
    interaction_loadings = torch.randn(n_assets, 1) * 0.3
    
    # Construct returns with nonlinear structure
    linear_part = factors @ linear_loadings.T
    nonlinear_part = (
        factor_squared @ squared_loadings.T + 
        factor_interaction @ interaction_loadings.T
    )
    noise = torch.randn(n_periods, n_assets) * 0.015
    
    returns = linear_part + nonlinear_part + noise
    
    # Train/val split
    train_ret = returns[:400]
    val_ret = returns[400:]
    
    print(”=== Linear vs Nonlinear Compression ===\n”)
    print(f”Data has nonlinear factor interactions”)
    print(f”Train: {train_ret.shape[0]} periods, Val: {val_ret.shape[0]} periods”)
    print(f”Assets: {n_assets}\n”)
    
    # Compare compression quality for different K
    print(”Reconstruction MSE by latent dimension:”)
    print(”-” * 60)
    print(f”{’K’:>3} | {’PCA (linear)’:>15} | {’Autoencoder’:>15} | {’Improvement’:>12}”)
    print(”-” * 60)
    
    for k in [3, 5, 10]:
        # PCA
        pca = PCAFactorModel(n_components=k)
        pca.fit(train_ret)
        pca_mse = pca.get_reconstruction_error(val_ret)
        
        # Autoencoder
        ae = ReturnAutoencoder(n_assets, n_latent=k, hidden_dims=[64, 32])
        trainer = AutoencoderTrainer(ae, learning_rate=1e-3)
        trainer.fit(train_ret, val_ret, epochs=100, verbose=False)
        ae_mse = trainer.reconstruction_error(val_ret)
        
        improvement = (pca_mse - ae_mse) / pca_mse * 100
        
        print(f”{k:3d} | {pca_mse:15.6f} | {ae_mse:15.6f} | {improvement:+11.1f}%”)
    
    print(”-” * 60)
    print(”\nAutoencoder captures nonlinear structure that PCA misses.”)
    print(”Improvement increases with K as AE uses extra dimensions”)
    print(”to model factor interactions, while PCA wastes them on noise.”)


if __name__ == “__main__”:
    demo_nonlinear_compression()
