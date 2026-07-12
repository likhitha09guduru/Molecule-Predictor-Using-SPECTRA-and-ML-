"""
train_pipeline.py - End-to-end training pipeline.

PURPOSE: Orchestrates the complete training workflow by connecting
all components together in the correct order.

WORKFLOW:
1. Data Ingestion → Load CSV files
2. Data Preprocessing → Clean and normalize spectra
3. Data Transformation → Convert to PyTorch DataLoaders
4. Model Creation → Build multi-modal GNN
5. Model Training → Train with early stopping and checkpointing
6. Model Evaluation → Evaluate on test set
7. Model Saving → Save final model and artifacts

HOW TO USE:
-----------
python -m src.pipeline.train_pipeline
"""
import sys
import os
import json
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, Optional, Any, List, Union

import numpy as np
import torch

from src.logger import logging
from src.exception import CustomException
from src.utils import ensure_directory, save_json, load_json, save_pickle

# ============================================================================
# IMPORT COMPONENTS
# ============================================================================

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import SpectralPreprocessor
from src.components.data_transformation import DataTransformation
from src.components.model_architecture import MultiModalSpectraModel, create_model
from src.components.model_trainer import ModelTrainer, run_training_pipeline


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # Data paths
    'data_dir': 'artifacts/data/converted',
    'output_dir': 'artifacts/models/',
    
    # Preprocessing
    'max_peaks_ms': 200,
    'max_peaks_nmr': 150,
    'max_peaks_ir': 100,
    'noise_threshold_ms': 0.01,
    'noise_threshold_nmr': 0.001,
    'noise_threshold_ir': 0.01,
    'normalize': True,
    
    # DataLoader
    'batch_size': 32,
    'shuffle_train': True,
    'num_workers': 0,
    
    # Model
    'hidden_dim': 128,
    'num_heads': 4,
    'num_layers': 3,
    'dropout': 0.1,
    'fusion_type': 'concat',
    'target_type': 'regression',
    'output_dim': 1,
    
    # Training
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 100,
    'early_stopping_patience': 20,
    'scheduler_patience': 10,
    'scheduler_factor': 0.5,
    'clip_grad_norm': 1.0,
    'log_interval': 10,
    
    # Data splits
    'test_size': 0.2,
    'val_size': 0.15,
    
    # Use all modalities or any
    'use_all_modalities': True
}


class TrainPipeline:
    """
    Complete training pipeline orchestrator.
    
    HOW TO USE:
    -----------
    pipeline = TrainPipeline(config_path='config.json')
    pipeline.run()
    
    OR
    
    pipeline = TrainPipeline()
    pipeline.run()
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to JSON config file
            config: Dictionary with configuration
        """
        try:
            # Load configuration
            if config_path is not None:
                self.config = self._load_config(config_path)
            elif config is not None:
                self.config = config
            else:
                self.config = DEFAULT_CONFIG.copy()
            
            # Create timestamp for this run
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(self.config['output_dir']) / self.timestamp
            ensure_directory(self.run_dir)
            
            # Save config for reproducibility
            self._save_config()
            
            logging.info("=" * 60)
            logging.info("TRAIN PIPELINE INITIALIZED")
            logging.info("=" * 60)
            logging.info(f"  Run directory: {self.run_dir}")
            logging.info(f"  Config: {self.config}")
            logging.info("=" * 60)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise CustomException(e, sys)
    
    def _save_config(self):
        """Save configuration to run directory."""
        try:
            config_path = self.run_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"  Config saved to {config_path}")
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def run(self) -> Dict:
        """
        Run the complete training pipeline.
        
        Returns:
            Dictionary with training results and artifact paths
        """
        try:
            logging.info("\n" + "=" * 60)
            logging.info("STARTING TRAINING PIPELINE")
            logging.info("=" * 60)
            
            results = {
                'run_timestamp': self.timestamp,
                'run_dir': str(self.run_dir),
                'config': self.config
            }
            
            # =================================================================
            # STEP 1: Data Ingestion
            # =================================================================
            
            logging.info("\n📂 STEP 1: Data Ingestion")
            logging.info("-" * 40)
            
            ingestor = DataIngestion(
                data_dir=self.config['data_dir'],
                chunk_size=10000
            )
            
            ingestion_data = ingestor.load_all()
            
            if not ingestion_data['compounds']:
                logging.error("No data loaded. Please run conversion scripts first.")
                logging.error("  python data_conversion/run_all_conversions.py")
                return results
            
            logging.info(f"  Loaded {len(ingestion_data['compounds'])} compounds")
            logging.info(f"  Multi-modal: {len(ingestion_data['multi_modal_compounds'])}")
            
            results['num_compounds'] = len(ingestion_data['compounds'])
            results['multi_modal_compounds'] = len(ingestion_data['multi_modal_compounds'])
            
            # =================================================================
            # STEP 2: Data Preprocessing
            # =================================================================
            
            logging.info("\n🔧 STEP 2: Data Preprocessing")
            logging.info("-" * 40)
            
            preprocessor = SpectralPreprocessor(
                max_peaks_ms=self.config['max_peaks_ms'],
                max_peaks_nmr=self.config['max_peaks_nmr'],
                max_peaks_ir=self.config['max_peaks_ir'],
                noise_threshold_ms=self.config['noise_threshold_ms'],
                noise_threshold_nmr=self.config['noise_threshold_nmr'],
                noise_threshold_ir=self.config['noise_threshold_ir'],
                normalize=self.config['normalize'],
                chunk_size=5000
            )
            
            processed_data = preprocessor.create_dataset(
                ingestion_data,
                use_all_modalities=self.config['use_all_modalities']
            )
            
            # Save preprocessed data
            preprocessor.save_processed_data(
                processed_data,
                output_dir=str(self.run_dir / 'processed_data/')
            )
            
            logging.info(f"  Saved processed data to {self.run_dir / 'processed_data/'}")
            
            # =================================================================
            # STEP 3: Data Transformation (DataLoaders)
            # =================================================================
            
            logging.info("\n🔄 STEP 3: Data Transformation")
            logging.info("-" * 40)
            
            # ✅ CORRECTED: Pass None for target_values (uses real data)
            # The DataTransformation class will look for target values in the CSV data
            transformer = DataTransformation(
                batch_size=self.config['batch_size'],
                shuffle_train=self.config['shuffle_train'],
                num_workers=self.config['num_workers'],
                target_type=self.config['target_type']
            )
            
            dataloaders = transformer.create_dataloaders(
                processed_data,
                target_values=None  # ← Use REAL data, no random targets!
            )
            
            # Save dataloader statistics
            transformer.save_dataloader_state(
                dataloaders,
                output_dir=str(self.run_dir / 'dataloaders/')
            )
            
            logging.info(f"  Saved DataLoader stats to {self.run_dir / 'dataloaders/'}")
            
            # =================================================================
            # STEP 4: Model Creation
            # =================================================================
            
            logging.info("\n🤖 STEP 4: Model Creation")
            logging.info("-" * 40)
            
            model_config = {
                'ms_input_dim': 2,
                'nmr_input_dim': 3,
                'ir_input_dim': 2,
                'hidden_dim': self.config['hidden_dim'],
                'output_dim': self.config['output_dim'],
                'num_heads': self.config['num_heads'],
                'num_layers': self.config['num_layers'],
                'dropout': self.config['dropout'],
                'fusion_type': self.config['fusion_type'],
                'target_type': self.config['target_type']
            }
            
            model = create_model(model_config)
            
            logging.info(f"  Model created with {model.count_parameters():,} parameters")
            
            # =================================================================
            # STEP 5: Model Training
            # =================================================================
            
            logging.info("\n🏋️ STEP 5: Model Training")
            logging.info("-" * 40)
            
            # Create trainer
            trainer = ModelTrainer(
                model=model,
                train_loader=dataloaders['train'],
                val_loader=dataloaders.get('validation'),
                test_loader=dataloaders.get('test'),
                learning_rate=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                epochs=self.config['epochs'],
                early_stopping_patience=self.config['early_stopping_patience'],
                scheduler_patience=self.config['scheduler_patience'],
                scheduler_factor=self.config['scheduler_factor'],
                clip_grad_norm=self.config['clip_grad_norm'],
                log_interval=self.config['log_interval'],
                checkpoint_dir=str(self.run_dir / 'checkpoints/'),
                target_type=self.config['target_type'],
                num_classes=self.config['output_dim'] if self.config['target_type'] == 'classification' else 1
            )
            
            # Train
            history = trainer.train()
            
            results['history'] = history
            results['best_val_loss'] = min(history['val_loss'])
            results['best_epoch'] = history['val_loss'].index(min(history['val_loss'])) + 1
            
            logging.info(f"\n  ✅ Training complete!")
            logging.info(f"  Best validation loss: {results['best_val_loss']:.4f} at epoch {results['best_epoch']}")
            
            # =================================================================
            # STEP 6: Final Evaluation
            # =================================================================
            
            if dataloaders.get('test') is not None:
                logging.info("\n📊 STEP 6: Final Test Evaluation")
                logging.info("-" * 40)
                
                test_loss, test_metrics = trainer.evaluate(dataloaders['test'])
                results['test_loss'] = test_loss
                results['test_metrics'] = test_metrics
                
                logging.info(f"  Test Loss: {test_loss:.4f}")
                for name, value in test_metrics.items():
                    logging.info(f"  Test {name}: {value:.4f}")
            
            # =================================================================
            # STEP 7: Save Final Model and Results
            # =================================================================
            
            logging.info("\n💾 STEP 7: Saving Final Model and Results")
            logging.info("-" * 40)
            
            # Save final model
            final_model_path = self.run_dir / 'final_model.pt'
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'config': model_config,
                'history': history,
                'best_val_loss': results['best_val_loss'],
                'best_epoch': results['best_epoch']
            }, final_model_path)
            
            logging.info(f"  Final model saved to {final_model_path}")
            
            # Save results
            results_path = self.run_dir / 'results.json'
            # Convert numpy values to Python types for JSON
            results_serializable = {}
            for key, value in results.items():
                if key == 'history':
                    # Convert history to serializable format
                    history_serializable = {}
                    for h_key, h_value in history.items():
                        if isinstance(h_value, list):
                            history_serializable[h_key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in h_value]
                        else:
                            history_serializable[h_key] = float(h_value) if isinstance(h_value, (np.floating, float)) else h_value
                    results_serializable[key] = history_serializable
                elif isinstance(value, dict):
                    results_serializable[key] = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in value.items()}
                elif isinstance(value, (np.floating, float)):
                    results_serializable[key] = float(value)
                else:
                    results_serializable[key] = value
            
            save_json(results_serializable, results_path)
            logging.info(f"  Results saved to {results_path}")
            
            # =================================================================
            # COMPLETE
            # =================================================================
            
            logging.info("\n" + "=" * 60)
            logging.info("✅ TRAINING PIPELINE COMPLETE!")
            logging.info("=" * 60)
            logging.info(f"  Run directory: {self.run_dir}")
            logging.info(f"  Best validation loss: {results['best_val_loss']:.4f}")
            if 'test_loss' in results:
                logging.info(f"  Test loss: {results['test_loss']:.4f}")
            logging.info(f"  Model saved to: {final_model_path}")
            logging.info("=" * 60)
            
            return results
            
        except Exception as e:
            raise CustomException(e, sys)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description='Run the training pipeline')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--data_dir', type=str, default='artifacts/data/converted',
                        help='Directory with converted CSV files')
    parser.add_argument('--output_dir', type=str, default='artifacts/models/',
                        help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'attention', 'gated'],
                        help='Fusion type for multi-modal model')
    parser.add_argument('--target_type', type=str, default='regression',
                        choices=['regression', 'classification'],
                        help='Type of prediction task')
    parser.add_argument('--no_cache', action='store_true',
                        help='Do not use cached data')
    
    args = parser.parse_args()
    
    try:
        # Build config from arguments
        config = {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'hidden_dim': args.hidden_dim,
            'fusion_type': args.fusion_type,
            'target_type': args.target_type,
            **{k: v for k, v in DEFAULT_CONFIG.items() 
               if k not in ['data_dir', 'output_dir', 'epochs', 'batch_size', 
                           'hidden_dim', 'fusion_type', 'target_type']}
        }
        
        # Run pipeline
        pipeline = TrainPipeline(config=config)
        results = pipeline.run()
        
        print(f"\n✅ Training complete! Results saved to {results['run_dir']}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()