"""
model_trainer.py - Training loop for multi-modal spectral model.

PURPOSE: Handles the complete training pipeline including:
1. Training loop with batch processing
2. Validation after each epoch
3. Early stopping
4. Model checkpointing
5. Learning rate scheduling
6. Metrics tracking and logging

INPUT: DataLoaders from DataTransformation, Model from model_architecture
OUTPUT: Trained model weights, training history, saved checkpoints
"""
import sys
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from datetime import datetime
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import ensure_directory


class ModelTrainer:
    """
    Complete training pipeline for multi-modal spectral models.
    
    HOW TO USE:
    -----------
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.001,
        epochs=100
    )
    history = trainer.train()
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 epochs: int = 100,
                 early_stopping_patience: int = 20,
                 scheduler_patience: int = 10,
                 scheduler_factor: float = 0.5,
                 clip_grad_norm: Optional[float] = 1.0,
                 device: Optional[str] = None,
                 checkpoint_dir: str = "artifacts/models/",
                 log_interval: int = 10,
                 target_type: str = 'regression',
                 num_classes: int = 1):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader (optional)
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            epochs: Maximum number of epochs
            early_stopping_patience: Epochs to wait before early stopping
            scheduler_patience: Epochs to wait before reducing LR
            scheduler_factor: Factor to reduce LR by
            clip_grad_norm: Gradient clipping value (None to disable)
            device: Device to use ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            log_interval: Print loss every N batches
            target_type: 'regression' or 'classification'
            num_classes: Number of classes (for classification)
        """
        try:
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.epochs = epochs
            self.early_stopping_patience = early_stopping_patience
            self.scheduler_patience = scheduler_patience
            self.scheduler_factor = scheduler_factor
            self.clip_grad_norm = clip_grad_norm
            self.log_interval = log_interval
            self.target_type = target_type
            self.num_classes = num_classes
            
            # Set device
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            
            self.model = self.model.to(self.device)
            
            # Setup checkpoint directory
            self.checkpoint_dir = Path(checkpoint_dir)
            ensure_directory(self.checkpoint_dir)
            
            # Create timestamp for this training run
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.checkpoint_dir / self.timestamp
            ensure_directory(self.run_dir)
            
            # Setup optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # Setup loss function
            if target_type == 'regression':
                self.criterion = nn.MSELoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
            
            # ✅ FIXED: Removed verbose parameter
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_patience,
                factor=scheduler_factor
            )
            
            # Training history
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'train_metrics': [],
                'val_metrics': [],
                'learning_rates': [],
                'epoch_times': []
            }
            
            # Best model tracking
            self.best_val_loss = float('inf')
            self.best_epoch = 0
            self.early_stopping_counter = 0
            
            logging.info("=" * 60)
            logging.info("MODEL TRAINER INITIALIZED")
            logging.info("=" * 60)
            logging.info(f"  Device: {self.device}")
            logging.info(f"  Learning rate: {learning_rate}")
            logging.info(f"  Weight decay: {weight_decay}")
            logging.info(f"  Max epochs: {epochs}")
            logging.info(f"  Early stopping patience: {early_stopping_patience}")
            logging.info(f"  Target type: {target_type}")
            if hasattr(self.model, 'count_parameters'):
                logging.info(f"  Model parameters: {self.model.count_parameters():,}")
            logging.info(f"  Checkpoint dir: {self.run_dir}")
            logging.info("=" * 60)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # TRAINING METHODS
    # =========================================================================
    
    def train(self) -> Dict:
        """
        Run the complete training loop.
        
        Returns:
            Training history dictionary
        """
        try:
            logging.info("\n" + "=" * 60)
            logging.info("STARTING TRAINING")
            logging.info("=" * 60)
            
            for epoch in range(1, self.epochs + 1):
                epoch_start_time = time.time()
                
                # Train for one epoch
                train_loss, train_metrics = self._train_epoch(epoch)
                
                # Validate
                val_loss, val_metrics = self._validate(epoch)
                
                epoch_time = time.time() - epoch_start_time
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_metrics'].append(train_metrics)
                self.history['val_metrics'].append(val_metrics)
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                self.history['epoch_times'].append(epoch_time)
                
                # Log progress
                self._log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics, epoch_time)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save checkpoint if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.early_stopping_counter = 0
                    self._save_checkpoint(epoch, val_loss, is_best=True)
                else:
                    self.early_stopping_counter += 1
                
                # Save regular checkpoint
                if epoch % 10 == 0:
                    self._save_checkpoint(epoch, val_loss, is_best=False)
                
                # Early stopping
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logging.info(f"\n⏹️ Early stopping triggered at epoch {epoch}")
                    break
            
            # Save final training history
            self._save_history()
            
            # Load best model
            self._load_best_model()
            
            # Final evaluation on test set
            if self.test_loader is not None:
                test_loss, test_metrics = self.evaluate(self.test_loader)
                logging.info("\n" + "=" * 60)
                logging.info("FINAL TEST EVALUATION")
                logging.info("=" * 60)
                logging.info(f"  Test Loss: {test_loss:.4f}")
                for name, value in test_metrics.items():
                    logging.info(f"  Test {name}: {value:.4f}")
                self.history['test_metrics'] = test_metrics
                self.history['test_loss'] = test_loss
            
            logging.info("\n" + "=" * 60)
            logging.info("TRAINING COMPLETE!")
            logging.info("=" * 60)
            logging.info(f"  Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
            logging.info(f"  Checkpoint saved to: {self.run_dir}")
            logging.info("=" * 60)
            
            return self.history
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """
        Train for one epoch.
        
        Returns:
            (average_loss, metrics_dict)
        """
        try:
            self.model.train()
            total_loss = 0.0
            all_targets = []
            all_preds = []
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Move data to device
                ms = batch['ms'].to(self.device)
                nmr = batch['nmr'].to(self.device)
                ir = batch['ir'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(ms, nmr, ir)
                
                # Compute loss
                if self.target_type == 'regression':
                    loss = self.criterion(outputs.squeeze(), targets)
                else:
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad_norm
                    )
                
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(outputs.detach().cpu().numpy())
                
                # Log progress
                if batch_idx % self.log_interval == 0:
                    logging.info(
                        f"  Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | "
                        f"Loss: {loss.item():.4f}"
                    )
            
            # Calculate metrics
            avg_loss = total_loss / len(self.train_loader)
            metrics = self._compute_metrics(
                np.array(all_targets),
                np.array(all_preds).squeeze()
            )
            
            return avg_loss, metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _validate(self, epoch: int) -> Tuple[float, Dict]:
        """
        Validate the model.
        
        Returns:
            (average_loss, metrics_dict)
        """
        try:
            if self.val_loader is None:
                return 0.0, {}
            
            self.model.eval()
            total_loss = 0.0
            all_targets = []
            all_preds = []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    ms = batch['ms'].to(self.device)
                    nmr = batch['nmr'].to(self.device)
                    ir = batch['ir'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    outputs = self.model(ms, nmr, ir)
                    
                    if self.target_type == 'regression':
                        loss = self.criterion(outputs.squeeze(), targets)
                    else:
                        loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    all_targets.extend(targets.cpu().numpy())
                    all_preds.extend(outputs.detach().cpu().numpy())
            
            avg_loss = total_loss / len(self.val_loader)
            metrics = self._compute_metrics(
                np.array(all_targets),
                np.array(all_preds).squeeze()
            )
            
            return avg_loss, metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """
        Evaluate the model on a given dataloader.
        
        Args:
            dataloader: DataLoader to evaluate on
        
        Returns:
            (average_loss, metrics_dict)
        """
        try:
            self.model.eval()
            total_loss = 0.0
            all_targets = []
            all_preds = []
            
            with torch.no_grad():
                for batch in dataloader:
                    ms = batch['ms'].to(self.device)
                    nmr = batch['nmr'].to(self.device)
                    ir = batch['ir'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    outputs = self.model(ms, nmr, ir)
                    
                    if self.target_type == 'regression':
                        loss = self.criterion(outputs.squeeze(), targets)
                    else:
                        loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    all_targets.extend(targets.cpu().numpy())
                    all_preds.extend(outputs.detach().cpu().numpy())
            
            avg_loss = total_loss / len(dataloader)
            metrics = self._compute_metrics(
                np.array(all_targets),
                np.array(all_preds).squeeze()
            )
            
            return avg_loss, metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # METRICS COMPUTATION
    # =========================================================================
    
    def _compute_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """
        Compute evaluation metrics.
        
        Args:
            targets: Ground truth values
            predictions: Model predictions
        
        Returns:
            Dictionary of metrics
        """
        try:
            metrics = {}
            
            if self.target_type == 'regression':
                # Regression metrics
                mse = np.mean((targets - predictions) ** 2)
                mae = np.mean(np.abs(targets - predictions))
                rmse = np.sqrt(mse)
                
                # R² score
                ss_res = np.sum((targets - predictions) ** 2)
                ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
            else:
                # Classification metrics
                pred_classes = np.argmax(predictions, axis=1)
                accuracy = np.mean(pred_classes == targets)
                
                # Compute per-class metrics if multi-class
                num_classes = self.num_classes
                if num_classes > 2:
                    class_metrics = {}
                    for c in range(num_classes):
                        class_metrics[f'accuracy_class_{c}'] = np.mean(pred_classes[targets == c] == c)
                    metrics.update(class_metrics)
                
                metrics['accuracy'] = accuracy
            
            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # CHECKPOINTING
    # =========================================================================
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            loss: Current validation loss
            is_best: Whether this is the best model so far
        """
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': loss,
                'best_val_loss': self.best_val_loss,
                'history': self.history,
                'config': {
                    'target_type': self.target_type,
                    'num_classes': self.num_classes,
                    'learning_rate': self.learning_rate,
                    'weight_decay': self.weight_decay
                }
            }
            
            if is_best:
                path = self.run_dir / "best_model.pt"
            else:
                path = self.run_dir / f"checkpoint_epoch_{epoch}.pt"
            
            torch.save(checkpoint, path)
            
            if is_best:
                logging.info(f"  ✅ Best model saved to {path}")
                
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_best_model(self):
        """Load the best model checkpoint."""
        try:
            best_path = self.run_dir / "best_model.pt"
            if best_path.exists():
                checkpoint = torch.load(best_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"✅ Loaded best model from {best_path}")
            else:
                logging.warning("No best model found, using current model")
        except Exception as e:
            raise CustomException(e, sys)
    
    def _save_history(self):
        """Save training history to JSON."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            history_serializable = {}
            for key, value in self.history.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        # Convert list of dicts to list of strings
                        history_serializable[key] = [str(v) for v in value]
                    else:
                        history_serializable[key] = value
                else:
                    history_serializable[key] = value
            
            path = self.run_dir / "training_history.json"
            with open(path, 'w') as f:
                json.dump(history_serializable, f, indent=4)
            
            logging.info(f"✅ Training history saved to {path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    
    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                   train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """
        Log epoch progress.
        """
        try:
            log_msg = f"\nEpoch {epoch}/{self.epochs} ({epoch_time:.1f}s)"
            log_msg += f"\n  Train Loss: {train_loss:.4f}"
            log_msg += f" | Val Loss: {val_loss:.4f}"
            
            if train_metrics:
                log_msg += "\n  Train Metrics:"
                for name, value in train_metrics.items():
                    if name not in ['mse', 'mae', 'rmse']:  # Skip detailed metrics in log
                        log_msg += f" {name}: {value:.4f}"
            
            if val_metrics:
                log_msg += "\n  Val Metrics:"
                for name, value in val_metrics.items():
                    if name not in ['mse', 'mae', 'rmse']:
                        log_msg += f" {name}: {value:.4f}"
            
            log_msg += f"\n  LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            log_msg += f" | Best: {self.best_val_loss:.4f} (epoch {self.best_epoch})"
            
            logging.info(log_msg)
            
        except Exception as e:
            raise CustomException(e, sys)


# ============================================================================
# TRAINING PIPELINE (Orchestrates everything)
# ============================================================================

def run_training_pipeline(
    model_config: Dict,
    dataloaders: Dict,
    training_config: Dict,
    output_dir: str = "artifacts/models/"
) -> Dict:
    """
    Complete training pipeline orchestrator.
    
    Args:
        model_config: Model configuration dictionary
        dataloaders: Dictionary with 'train', 'validation', 'test' DataLoaders
        training_config: Training configuration dictionary
        output_dir: Directory to save models
    
    Returns:
        Training history dictionary
    """
    try:
        from src.components.model_architecture import create_model
        
        # Create model
        model = create_model(model_config)
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders.get('validation'),
            test_loader=dataloaders.get('test'),
            checkpoint_dir=output_dir,
            **training_config
        )
        
        # Train
        history = trainer.train()
        
        return history
        
    except Exception as e:
        raise CustomException(e, sys)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING MODEL TRAINER")
    print("=" * 60)
    
    try:
        from src.components.data_ingestion import DataIngestion
        from src.components.data_preprocessing import SpectralPreprocessor
        from src.components.data_transformation import DataTransformation
        from src.components.model_architecture import MultiModalSpectraModel
        
        # Step 1: Load and preprocess data
        print("\n📂 Loading data from CSVs...")
        ingestor = DataIngestion(data_dir="artifacts/data/converted")
        ingestion_data = ingestor.load_all()
        
        if not ingestion_data['compounds']:
            print("\n⚠️ No data loaded. Please run conversion scripts first.")
            print("   python data_conversion/run_all_conversions.py")
            sys.exit(1)
        
        preprocessor = SpectralPreprocessor(
            max_peaks_ms=50,  # Smaller for testing
            max_peaks_nmr=40,
            max_peaks_ir=30,
            chunk_size=100
        )
        processed_data = preprocessor.create_dataset(ingestion_data)
        
        transformer = DataTransformation(
            batch_size=4,  # Small batch for testing
            shuffle_train=True,
            num_workers=0,
            target_type='regression'
        )
        
        # Create dummy targets for testing
        dummy_targets = {}
        for split_name in ['train', 'validation', 'test']:
            if split_name in processed_data and processed_data[split_name] is not None:
                num_samples = len(processed_data[split_name]['compound_ids'])
                dummy_targets[split_name] = np.random.uniform(0, 10, num_samples).tolist()
        
        dataloaders = transformer.create_dataloaders(
            processed_data,
            target_values=dummy_targets
        )
        
        # Step 2: Create model
        print("\n🤖 Creating model...")
        model = MultiModalSpectraModel(
            hidden_dim=32,  # Small for testing
            output_dim=1,
            target_type='regression',
            fusion_type='concat'
        )
        
        # Step 3: Train
        print("\n🏋️ Training model (5 epochs for testing)...")
        trainer = ModelTrainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['validation'],
            epochs=5,
            learning_rate=0.01,
            log_interval=2,
            checkpoint_dir="artifacts/models/test/"
        )
        
        history = trainer.train()
        
        print("\n📊 TRAINING HISTORY SUMMARY:")
        if history and history.get('train_loss'):
            print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
            print(f"  Best val loss: {min(history['val_loss']):.4f}")
        else:
            print("  No training history available.")
        
        print("\n✅ Model trainer test complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please run conversion scripts first to create CSV files.")
        print("   python data_conversion/run_all_conversions.py")
    except Exception as e:
        raise CustomException(e, sys)