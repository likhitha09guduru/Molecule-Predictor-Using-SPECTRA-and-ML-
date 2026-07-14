"""
model_trainer.py - Training loop for NMR and IR spectral model.
Combined with model architecture for simplicity.
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Optional sklearn imports - handle gracefully
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

# XGBoost - optional, handle if not installed
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    XGBOOST_AVAILABLE = False
    # Create a placeholder class to avoid NameError
    class XGBRegressor:
        def __init__(self, *args, **kwargs):
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
        def fit(self, *args, **kwargs):
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
        def predict(self, *args, **kwargs):
            raise ImportError("xgboost not installed. Install with: pip install xgboost")

# CatBoost - optional
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CATBOOST_AVAILABLE = False
    # Create a placeholder class to avoid NameError
    class CatBoostRegressor:
        def __init__(self, *args, **kwargs):
            raise ImportError("catboost not installed. Install with: pip install catboost")
        def fit(self, *args, **kwargs):
            raise ImportError("catboost not installed. Install with: pip install catboost")
        def predict(self, *args, **kwargs):
            raise ImportError("catboost not installed. Install with: pip install catboost")


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class NMRIRModel(nn.Module):
    """
    Neural network model for NMR and IR spectral data.
    """
    
    def __init__(self,
                 nmr_input_dim: int = 3,
                 ir_input_dim: int = 2,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 fusion_type: str = 'concat',
                 target_type: str = 'regression'):
        """
        Initialize the NMR+IR model.
        
        Args:
            nmr_input_dim: NMR feature dimension (3 for shift, multiplicity, intensity)
            ir_input_dim: IR feature dimension (2 for wavenumber, intensity)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for regression)
            num_heads: Number of attention heads (for transformer)
            num_layers: Number of layers
            dropout: Dropout rate
            fusion_type: 'concat' or 'attention'
            target_type: 'regression' or 'classification'
        """
        super().__init__()
        
        try:
            self.hidden_dim = hidden_dim
            self.target_type = target_type
            
            # NMR Encoder - processes NMR peaks
            self.nmr_encoder = nn.Sequential(
                nn.Linear(nmr_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )
            
            # IR Encoder - processes IR peaks
            self.ir_encoder = nn.Sequential(
                nn.Linear(ir_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )
            
            # Fusion layer
            if fusion_type == 'concat':
                self.fusion = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            else:
                self.fusion = nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            
            # Output head
            if target_type == 'regression':
                self.head = nn.Linear(hidden_dim // 2, output_dim)
            else:
                self.head = nn.Linear(hidden_dim // 2, output_dim)
            
            logging.info(f"NMRIRModel initialized: hidden_dim={hidden_dim}, fusion={fusion_type}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def forward(self, nmr: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            nmr: NMR tensor (batch_size, max_peaks, 3)
            ir: IR tensor (batch_size, max_peaks, 2)
        
        Returns:
            Predictions (batch_size, output_dim)
        """
        try:
            # Flatten the peak dimensions
            nmr_flat = nmr.view(nmr.size(0), -1)
            ir_flat = ir.view(ir.size(0), -1)
            
            # Encode
            nmr_feat = self.nmr_encoder(nmr_flat)
            ir_feat = self.ir_encoder(ir_flat)
            
            # Fuse
            fused = torch.cat([nmr_feat, ir_feat], dim=-1)
            fused = self.fusion(fused)
            
            # Output
            output = self.head(fused)
            
            return output
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_config: Dict) -> NMRIRModel:
    """Create a model from configuration dictionary."""
    try:
        default_config = {
            'nmr_input_dim': 3,
            'ir_input_dim': 2,
            'hidden_dim': 128,
            'output_dim': 1,
            'num_heads': 4,
            'num_layers': 3,
            'dropout': 0.1,
            'fusion_type': 'concat',
            'target_type': 'regression'
        }
        
        config = {**default_config, **model_config}
        return NMRIRModel(**config)
        
    except Exception as e:
        raise CustomException(e, sys)


# ============================================================================
# DATASET CLASS
# ============================================================================

class SpectralDataset(Dataset):
    """PyTorch Dataset for NMR and IR spectral data."""
    
    def __init__(self, 
                 nmr_spectra: np.ndarray,
                 ir_spectra: np.ndarray,
                 targets: Optional[np.ndarray] = None,
                 compound_ids: Optional[List[str]] = None,
                 target_type: str = 'regression'):
        """
        Initialize SpectralDataset.
        
        Args:
            nmr_spectra: NMR spectral data (shape: N x max_peaks_nmr x 3)
            ir_spectra: IR spectral data (shape: N x max_peaks_ir x 2)
            targets: Target values
            compound_ids: List of compound identifiers
            target_type: 'regression' or 'classification'
        """
        try:
            self.nmr_spectra = torch.tensor(nmr_spectra, dtype=torch.float32)
            self.ir_spectra = torch.tensor(ir_spectra, dtype=torch.float32)
            
            if targets is not None:
                if target_type == 'regression':
                    self.targets = torch.tensor(targets, dtype=torch.float32)
                else:
                    self.targets = torch.tensor(targets, dtype=torch.long)
            else:
                self.targets = None
            
            self.compound_ids = compound_ids
            self.target_type = target_type
            self.length = len(nmr_spectra)
            
            logging.info(f"SpectralDataset initialized with {self.length} samples (NMR+IR)")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {
            'nmr': self.nmr_spectra[idx],
            'ir': self.ir_spectra[idx]
        }
        
        if self.targets is not None:
            sample['target'] = self.targets[idx]
        
        if self.compound_ids is not None:
            sample['compound_id'] = self.compound_ids[idx]
        
        return sample


# ============================================================================
# MODEL TRAINER
# ============================================================================

@dataclass
class ModelTrainerConfig:
    """Configuration for model trainer."""
    trained_model_file_path: str = os.path.join("artifacts", "model.pth")
    checkpoint_dir: str = os.path.join("artifacts", "models")
    training_history_path: str = os.path.join("artifacts", "training_history.json")


class ModelTrainer:
    """Trainer for NMR and IR spectral models."""
    
    def __init__(self):
        """Initialize model trainer with configuration."""
        self.model_trainer_config = ModelTrainerConfig()
        
        # Create directories
        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
        os.makedirs(self.model_trainer_config.checkpoint_dir, exist_ok=True)
        
        # Default hyperparameters
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.epochs = 100
        self.early_stopping_patience = 20
        self.scheduler_patience = 10
        self.scheduler_factor = 0.5
        self.clip_grad_norm = 1.0
        self.batch_size = 32
        self.target_type = 'regression'
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logging.info("=" * 60)
        logging.info("MODEL TRAINER INITIALIZED (NMR + IR)")
        logging.info("=" * 60)
        logging.info(f"  Device: {self.device}")
        logging.info(f"  Model save path: {self.model_trainer_config.trained_model_file_path}")
        logging.info(f"  Target type: {self.target_type}")
        logging.info(f"  XGBoost available: {XGBOOST_AVAILABLE}")
        logging.info(f"  CatBoost available: {CATBOOST_AVAILABLE}")
        logging.info("=" * 60)
    
    def initiate_model_trainer(self, 
                               train_loader: DataLoader,
                               test_loader: DataLoader,
                               val_loader: Optional[DataLoader] = None,
                               model: Optional[nn.Module] = None,
                               model_config: Optional[Dict] = None) -> float:
        """
        Initiate model training for NMR and IR data.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            val_loader: DataLoader for validation data (optional)
            model: PyTorch model (if None, creates default model)
            model_config: Configuration for model creation
        
        Returns:
            R2 score on test data
        """
        try:
            logging.info("Entered model trainer method")
            
            # Create model if not provided
            if model is None:
                if model_config is None:
                    model_config = {
                        'nmr_input_dim': 3,
                        'ir_input_dim': 2,
                        'hidden_dim': 128,
                        'output_dim': 1,
                        'num_heads': 4,
                        'num_layers': 3,
                        'dropout': 0.1,
                        'fusion_type': 'concat',
                        'target_type': self.target_type
                    }
                
                model = create_model(model_config)
                logging.info(f"✅ Created model with {model.count_parameters():,} parameters")
            
            model = model.to(self.device)
            
            # Define loss function and optimizer
            if self.target_type == 'regression':
                criterion = nn.HuberLoss(delta=1.0)
            else:
                criterion = nn.CrossEntropyLoss()
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=self.scheduler_patience,
                factor=self.scheduler_factor
            )
            
            # Training loop
            logging.info("Starting training...")
            best_val_loss = float('inf')
            early_stopping_counter = 0
            best_model_state = None
            
            for epoch in range(1, self.epochs + 1):
                epoch_start_time = time.time()
                
                # Training phase
                model.train()
                train_loss = 0.0
                for batch_idx, batch in enumerate(train_loader):
                    # Handle different batch formats
                    if isinstance(batch, dict):
                        nmr = batch['nmr'].to(self.device)
                        ir = batch['ir'].to(self.device)
                        targets = batch['target'].to(self.device)
                    else:
                        # Assume tuple format (nmr, ir, targets)
                        nmr, ir, targets = batch
                        nmr = nmr.to(self.device)
                        ir = ir.to(self.device)
                        targets = targets.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(nmr, ir)
                    
                    # Handle output shape
                    if outputs.dim() > 1 and outputs.size(-1) == 1:
                        outputs = outputs.squeeze(-1)
                    
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                
                # Validation phase
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch in val_loader:
                            if isinstance(batch, dict):
                                nmr = batch['nmr'].to(self.device)
                                ir = batch['ir'].to(self.device)
                                targets = batch['target'].to(self.device)
                            else:
                                nmr, ir, targets = batch
                                nmr = nmr.to(self.device)
                                ir = ir.to(self.device)
                                targets = targets.to(self.device)
                            
                            outputs = model(nmr, ir)
                            
                            if outputs.dim() > 1 and outputs.size(-1) == 1:
                                outputs = outputs.squeeze(-1)
                            
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                else:
                    avg_val_loss = 0.0
                
                epoch_time = time.time() - epoch_start_time
                
                # Log progress
                if epoch % 10 == 0 or epoch == 1:
                    log_msg = f"Epoch {epoch}/{self.epochs} ({epoch_time:.1f}s) - Train Loss: {avg_train_loss:.4f}"
                    if val_loader is not None:
                        log_msg += f", Val Loss: {avg_val_loss:.4f}"
                    logging.info(log_msg)
                
                # Learning rate scheduling
                if val_loader is not None:
                    scheduler.step(avg_val_loss)
                
                # Checkpoint and early stopping
                if val_loader is not None:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_state = model.state_dict().copy()
                        early_stopping_counter = 0
                        # Save best model
                        torch.save(best_model_state, self.model_trainer_config.trained_model_file_path)
                        logging.info(f"✅ New best model saved (Val Loss: {best_val_loss:.4f})")
                    else:
                        early_stopping_counter += 1
                    
                    if early_stopping_counter >= self.early_stopping_patience:
                        logging.info(f"⏹️ Early stopping at epoch {epoch}")
                        break
                else:
                    # No validation: save periodically
                    if epoch % 10 == 0:
                        torch.save(model.state_dict(), self.model_trainer_config.trained_model_file_path)
            
            # Load best model if validation was used
            if val_loader is not None and best_model_state is not None:
                model.load_state_dict(best_model_state)
                logging.info(f"✅ Loaded best model with Val Loss: {best_val_loss:.4f}")
            else:
                # Save final model
                torch.save(model.state_dict(), self.model_trainer_config.trained_model_file_path)
                logging.info(f"✅ Final model saved to {self.model_trainer_config.trained_model_file_path}")
            
            # Evaluate on test set
            logging.info("\n" + "=" * 60)
            logging.info("EVALUATING ON TEST SET")
            logging.info("=" * 60)
            
            test_loss, test_metrics = self.evaluate_model(model, test_loader, criterion)
            
            logging.info(f"Test Loss: {test_loss:.4f}")
            for name, value in test_metrics.items():
                logging.info(f"Test {name}: {value:.4f}")
            
            return test_metrics.get('r2', 0.0)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def train_from_arrays(self,
                          train_arr: np.ndarray,
                          test_arr: np.ndarray,
                          val_arr: Optional[np.ndarray] = None,
                          model_config: Optional[Dict] = None) -> float:
        """
        Train model directly from numpy arrays.
        
        This is the bridge method that connects DataTransformation output
        to the ModelTrainer.
        
        Args:
            train_arr: Training array (features + targets)
            test_arr: Test array (features + targets)
            val_arr: Validation array (features + targets)
            model_config: Model configuration
        
        Returns:
            R2 score on test data
        """
        try:
            logging.info("\n" + "=" * 60)
            logging.info("TRAINING FROM NUMPY ARRAYS")
            logging.info("=" * 60)
            
            # Split features and targets
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
            
            logging.info(f"Train features shape: {X_train.shape}")
            logging.info(f"Train targets shape: {y_train.shape}")
            logging.info(f"Test features shape: {X_test.shape}")
            logging.info(f"Test targets shape: {y_test.shape}")
            
            if val_arr is not None:
                X_val = val_arr[:, :-1]
                y_val = val_arr[:, -1]
                logging.info(f"Val features shape: {X_val.shape}")
                logging.info(f"Val targets shape: {y_val.shape}")
            else:
                X_val, y_val = None, None
                logging.info("No validation data provided")
            
            # Create DataLoaders
            from torch.utils.data import TensorDataset
            
            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            test_dataset = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32)
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
            
            if val_arr is not None:
                val_dataset = TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.float32)
                )
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=False
                )
            else:
                val_loader = None
            
            logging.info(f"✅ Created DataLoaders:")
            logging.info(f"   Train: {len(train_loader)} batches")
            logging.info(f"   Test: {len(test_loader)} batches")
            if val_loader:
                logging.info(f"   Val: {len(val_loader)} batches")
            
            # Use the existing training method
            return self.initiate_model_trainer(
                train_loader=train_loader,
                test_loader=test_loader,
                val_loader=val_loader,
                model_config=model_config
            )
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate_model(self, 
                       model: nn.Module, 
                       dataloader: DataLoader,
                       criterion: nn.Module) -> Tuple[float, Dict]:
        """Evaluate model on a dataloader."""
        try:
            model.eval()
            total_loss = 0.0
            all_targets = []
            all_preds = []
            
            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, dict):
                        nmr = batch['nmr'].to(self.device)
                        ir = batch['ir'].to(self.device)
                        targets = batch['target'].to(self.device)
                    else:
                        nmr, ir, targets = batch
                        nmr = nmr.to(self.device)
                        ir = ir.to(self.device)
                        targets = targets.to(self.device)
                    
                    outputs = model(nmr, ir)
                    
                    if outputs.dim() > 1 and outputs.size(-1) == 1:
                        outputs = outputs.squeeze(-1)
                    
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    
                    all_targets.extend(targets.detach().cpu().numpy())
                    all_preds.extend(outputs.detach().cpu().numpy())
            
            avg_loss = total_loss / len(dataloader)
            
            # Convert to numpy arrays
            all_targets = np.array(all_targets)
            all_preds = np.array(all_preds)
            
            # Ensure correct shapes
            if all_targets.ndim > 1:
                all_targets = all_targets.squeeze()
            if all_preds.ndim > 1:
                all_preds = all_preds.squeeze()
            
            # Compute metrics
            metrics = {
                'mse': float(mean_squared_error(all_targets, all_preds)),
                'mae': float(mean_absolute_error(all_targets, all_preds)),
                'rmse': float(np.sqrt(mean_squared_error(all_targets, all_preds))),
                'r2': float(r2_score(all_targets, all_preds))
            }
            
            return avg_loss, metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def train_with_sklearn_models(self, 
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  use_neural_net: bool = True) -> float:
        """
        Train using traditional ML models or neural network.
        Follows the pattern from the reference implementation.
        """
        try:
            logging.info("Split training and test input data")
            
            if use_neural_net:
                # Convert to PyTorch DataLoaders
                train_dataset = TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32)
                )
                test_dataset = TensorDataset(
                    torch.tensor(X_test, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.float32)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                # Create neural network model
                class SimpleNN(nn.Module):
                    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(hidden_dim, hidden_dim // 2),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(hidden_dim // 2, output_dim)
                        )
                    
                    def forward(self, x):
                        return self.net(x)
                
                model = SimpleNN(X_train.shape[1]).to(self.device)
                
                # Train neural network
                criterion = nn.HuberLoss()
                optimizer = optim.AdamW(model.parameters(), lr=0.001)
                
                for epoch in range(100):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate
                model.eval()
                all_preds = []
                with torch.no_grad():
                    for batch_X, _ in test_loader:
                        batch_X = batch_X.to(self.device)
                        preds = model(batch_X).cpu().numpy().squeeze()
                        all_preds.extend(preds)
                
                predicted = np.array(all_preds)
                
                # Save model
                torch.save(model.state_dict(), self.model_trainer_config.trained_model_file_path)
                
            else:
                # Use traditional ML models (like reference)
                if not SKLEARN_AVAILABLE:
                    logging.error("sklearn not available. Install with: pip install scikit-learn")
                    raise ImportError("sklearn not available")
                
                # Build models dictionary based on available packages
                models = {
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                }
                
                # Add XGBoost if available
                if XGBOOST_AVAILABLE:
                    models["XGBRegressor"] = XGBRegressor()
                else:
                    logging.warning("XGBoost not available. Skipping XGBRegressor.")
                
                # Add CatBoost if available
                if CATBOOST_AVAILABLE:
                    models["CatBoosting Regressor"] = CatBoostRegressor(verbose=False)
                else:
                    logging.warning("CatBoost not available. Skipping CatBoosting Regressor.")
                
                # Train and evaluate each model
                best_model = None
                best_score = -float('inf')
                best_model_name = ""
                
                for name, model in models.items():
                    logging.info(f"Training {name}...")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                    
                    logging.info(f"  {name} R2 score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_model_name = name
                
                if best_model is None:
                    raise ValueError("No models trained successfully")
                
                predicted = best_model.predict(X_test)
                
                # Save best traditional model
                import joblib
                joblib.dump(best_model, self.model_trainer_config.trained_model_file_path.replace('.pth', '.pkl'))
                logging.info(f"✅ Best traditional model ({best_model_name}) saved with R2: {best_score:.4f}")
            
            # Calculate metrics
            r2_square = r2_score(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            
            logging.info(f"\nFinal Test Metrics:")
            logging.info(f"  R2 Score: {r2_square:.4f}")
            logging.info(f"  MSE: {mse:.4f}")
            logging.info(f"  MAE: {mae:.4f}")
            
            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_model(self, model_path: Optional[str] = None) -> nn.Module:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to model file (uses default if None)
        
        Returns:
            Loaded PyTorch model
        """
        try:
            if model_path is None:
                model_path = self.model_trainer_config.trained_model_file_path
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            # Create model architecture
            model_config = {
                'nmr_input_dim': 3,
                'ir_input_dim': 2,
                'hidden_dim': 128,
                'output_dim': 1,
                'target_type': self.target_type
            }
            model = create_model(model_config)
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            logging.info(f"✅ Model loaded from {model_path}")
            return model
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print("🧪 TESTING MODEL TRAINER (NMR + IR)")
        print("=" * 60)
        
        trainer = ModelTrainer()
        print("✅ Model Trainer initialized successfully!")
        print(f"   Model class: NMRIRModel")
        print(f"   Sklearn available: {SKLEARN_AVAILABLE}")
        print(f"   XGBoost available: {XGBOOST_AVAILABLE}")
        print(f"   CatBoost available: {CATBOOST_AVAILABLE}")
        
        # Test model creation
        model = create_model({})
        print(f"   Model parameters: {model.count_parameters():,}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        raise CustomException(e, sys)