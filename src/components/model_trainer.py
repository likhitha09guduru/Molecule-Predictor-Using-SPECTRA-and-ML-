"""
model_trainer.py - Training loop for multi-modal spectral model.
"""
import sys
import json
import time
from sklearn.feature_selection import f_regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import ensure_directory


class ModelTrainer:
    """Complete training pipeline for multi-modal spectral models."""
    
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
            
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            
            self.model = self.model.to(self.device)
            
            self.checkpoint_dir = Path(checkpoint_dir)
            ensure_directory(self.checkpoint_dir)
            
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.checkpoint_dir / self.timestamp
            ensure_directory(self.run_dir)
            
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # ✅ Use HuberLoss for robustness to outliers
            if target_type == 'regression':
                self.criterion = nn.HuberLoss(delta=1.0)
            else:
                self.criterion = nn.CrossEntropyLoss()
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_patience,
                factor=scheduler_factor
            )
            
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'train_metrics': [],
                'val_metrics': [],
                'learning_rates': [],
                'epoch_times': []
            }
            
            self.best_val_loss = float('inf')
            self.best_epoch = 0
            self.early_stopping_counter = 0
            self.has_validation = val_loader is not None
            
            logging.info("=" * 60)
            logging.info("MODEL TRAINER INITIALIZED")
            logging.info("=" * 60)
            logging.info(f"  Device: {self.device}")
            logging.info(f"  Learning rate: {learning_rate}")
            logging.info(f"  Max epochs: {epochs}")
            logging.info(f"  Target type: {target_type}")
            logging.info(f"  Loss function: {'HuberLoss' if target_type == 'regression' else 'CrossEntropyLoss'}")
            logging.info(f"  Validation loader: {'✅ Available' if self.has_validation else '❌ None'}")
            logging.info("=" * 60)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def train(self) -> Dict:
        try:
            logging.info("\n" + "=" * 60)
            logging.info("STARTING TRAINING")
            logging.info("=" * 60)
            
            if not self.has_validation:
                logging.warning("⚠️ No validation loader. Training without validation.")
            
            for epoch in range(1, self.epochs + 1):
                epoch_start_time = time.time()
                
                # Train
                train_loss, train_metrics = self._train_epoch(epoch)
                
                # Validate (only if available)
                if self.has_validation:
                    val_loss, val_metrics = self._validate(epoch)
                else:
                    val_loss = 0.0
                    val_metrics = {}
                    if epoch % 5 == 0:
                        logging.info(f"  Epoch {epoch}: No validation (skipping)")
                
                epoch_time = time.time() - epoch_start_time
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_metrics'].append(train_metrics)
                self.history['val_metrics'].append(val_metrics)
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                self.history['epoch_times'].append(epoch_time)
                
                # Log
                self._log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics, epoch_time)
                
                # Learning rate scheduling (only if validation)
                if self.has_validation:
                    self.scheduler.step(val_loss)
                
                # Checkpoints
                if self.has_validation:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_epoch = epoch
                        self.early_stopping_counter = 0
                        self._save_checkpoint(epoch, val_loss, is_best=True)
                    else:
                        self.early_stopping_counter += 1
                    
                    if epoch % 10 == 0:
                        self._save_checkpoint(epoch, val_loss, is_best=False)
                    
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        logging.info(f"\n⏹️ Early stopping at epoch {epoch}")
                        break
                else:
                    # No validation: save checkpoint periodically
                    if epoch % 10 == 0:
                        self._save_checkpoint(epoch, 0.0, is_best=False)
                    self._save_checkpoint(epoch, 0.0, is_best=True)
            
            # Save history
            self._save_history()
            
            # Load best model
            if self.has_validation:
                self._load_best_model()
            else:
                logging.info("No validation loader. Using final model.")
            
            # Test evaluation
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
            if self.has_validation:
                logging.info(f"  Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
            logging.info(f"  Checkpoint saved to: {self.run_dir}")
            logging.info("=" * 60)
            
            return self.history
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        try:
            self.model.train()
            total_loss = 0.0
            all_targets = []
            all_preds = []
            
            for batch_idx, batch in enumerate(self.train_loader):
                ms = batch['ms'].to(self.device)
                nmr = batch['nmr'].to(self.device)
                ir = batch['ir'].to(self.device)
                targets = batch['target'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(ms, nmr, ir)
                
                # Handle output shape for loss computation
                if self.target_type == 'regression':
                    if outputs.dim() > 1 and outputs.size(-1) == 1:
                        outputs = outputs.squeeze(-1)
                    loss = self.criterion(outputs, targets)
                else:
                    loss = self.criterion(outputs, targets)
                
                loss.backward()
                
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                all_targets.extend(targets.detach().cpu().numpy())
                all_preds.extend(outputs.detach().cpu().numpy())
                
                if batch_idx % self.log_interval == 0:
                    logging.info(f"  Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(self.train_loader)
            
            # Convert to numpy arrays
            all_targets = np.array(all_targets)
            all_preds = np.array(all_preds)
            
            # Ensure predictions have correct shape
            if self.target_type == 'regression' and all_preds.ndim > 1:
                all_preds = all_preds.squeeze()
            
            metrics = self._compute_metrics(all_targets, all_preds)
            
            return avg_loss, metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _validate(self, epoch: int) -> Tuple[float, Dict]:
        try:
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
                    
                    # Handle output shape for loss computation
                    if self.target_type == 'regression':
                        if outputs.dim() > 1 and outputs.size(-1) == 1:
                            outputs = outputs.squeeze(-1)
                        loss = self.criterion(outputs, targets)
                    else:
                        loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    all_targets.extend(targets.detach().cpu().numpy())
                    all_preds.extend(outputs.detach().cpu().numpy())
            
            avg_loss = total_loss / len(self.val_loader)
            
            # Convert to numpy arrays
            all_targets = np.array(all_targets)
            all_preds = np.array(all_preds)
            
            # Ensure predictions have correct shape
            if self.target_type == 'regression' and all_preds.ndim > 1:
                all_preds = all_preds.squeeze()
            
            metrics = self._compute_metrics(all_targets, all_preds)
            
            return avg_loss, metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
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
                    
                    # Handle output shape for loss computation
                    if self.target_type == 'regression':
                        if outputs.dim() > 1 and outputs.size(-1) == 1:
                            outputs = outputs.squeeze(-1)
                        loss = self.criterion(outputs, targets)
                    else:
                        loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    all_targets.extend(targets.detach().cpu().numpy())
                    all_preds.extend(outputs.detach().cpu().numpy())
            
            avg_loss = total_loss / len(dataloader)
            
            # Convert to numpy arrays
            all_targets = np.array(all_targets)
            all_preds = np.array(all_preds)
            
            # Ensure predictions have correct shape
            if self.target_type == 'regression' and all_preds.ndim > 1:
                all_preds = all_preds.squeeze()
            
            metrics = self._compute_metrics(all_targets, all_preds)
            
            return avg_loss, metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _compute_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        try:
            metrics = {}
            
            # Ensure predictions have correct shape
            if predictions.ndim > 1 and predictions.shape[1] == 1:
                predictions = predictions.squeeze(-1)
            
            if self.target_type == 'regression':
                # Ensure both are 1D arrays
                if targets.ndim > 1:
                    targets = targets.squeeze()
                if predictions.ndim > 1:
                    predictions = predictions.squeeze()
                
                # Calculate metrics
                mse = np.mean((targets - predictions) ** 2)
                mae = np.mean(np.abs(targets - predictions))
                rmse = np.sqrt(mse)
                
                ss_res = np.sum((targets - predictions) ** 2)
                ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                
                metrics = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2)
                }
                
            else:
                # Classification metrics
                if predictions.ndim > 1:
                    pred_classes = np.argmax(predictions, axis=1)
                else:
                    pred_classes = predictions
                
                metrics = {
                    'accuracy': float(np.mean(pred_classes == targets))
                }
            
            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
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
        try:
            best_path = self.run_dir / "best_model.pt"
            if best_path.exists():
                checkpoint = torch.load(
                    best_path,
                    map_location=self.device,
                    weights_only=False
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"✅ Loaded best model from {best_path}")
            else:
                logging.warning("No best model found, using current model")
        except Exception as e:
            raise CustomException(e, sys)
    
    def _save_history(self):
        try:
            history_serializable = {}
            for key, value in self.history.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        # Convert dict metrics to serializable format
                        history_serializable[key] = [
                            {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                             for k, v in item.items()} 
                            for item in value
                        ]
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
    
    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                   train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        try:
            log_msg = f"\nEpoch {epoch}/{self.epochs} ({epoch_time:.1f}s)"
            log_msg += f"\n  Train Loss: {train_loss:.4f}"
            
            if self.has_validation:
                log_msg += f" | Val Loss: {val_loss:.4f}"
            else:
                log_msg += f" | Val Loss: N/A"
            
            if train_metrics:
                log_msg += "\n  Train Metrics:"
                for name, value in train_metrics.items():
                    log_msg += f" {name}: {value:.4f}"
            
            if val_metrics and self.has_validation:
                log_msg += "\n  Val Metrics:"
                for name, value in val_metrics.items():
                    log_msg += f" {name}: {value:.4f}"
            
            log_msg += f"\n  LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            
            if self.has_validation:
                log_msg += f" | Best: {self.best_val_loss:.4f} (epoch {self.best_epoch})"
            
            logging.info(log_msg)
            
        except Exception as e:
            raise CustomException(e, sys)


def run_training_pipeline(
    model_config: Dict,
    dataloaders: Dict,
    training_config: Dict,
    output_dir: str = "artifacts/models/"
) -> Dict:
    try:
        from src.components.model_architecture import create_model
        
        model = create_model(model_config)
        
        trainer = ModelTrainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders.get('validation'),
            test_loader=dataloaders.get('test'),
            checkpoint_dir=output_dir,
            **training_config
        )
        
        return trainer.train()
        
    except Exception as e:
        raise CustomException(e, sys)