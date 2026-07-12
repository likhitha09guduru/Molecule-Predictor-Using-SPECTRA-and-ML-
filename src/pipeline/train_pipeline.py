"""
train_pipeline.py - End-to-end training pipeline.
"""
import sys
import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, Optional

import torch

from src.logger import logging
from src.exception import CustomException
from src.utils import ensure_directory, save_json

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import SpectralPreprocessor
from src.components.data_transformation import DataTransformation
from src.components.model_architecture import create_model
from src.components.model_trainer import ModelTrainer


DEFAULT_CONFIG = {
    'data_dir': 'artifacts/data/converted',
    'output_dir': 'artifacts/models/',
    'max_peaks_ms': 200,
    'max_peaks_nmr': 150,
    'max_peaks_ir': 100,
    'noise_threshold_ms': 0.01,
    'noise_threshold_nmr': 0.001,
    'noise_threshold_ir': 0.01,
    'normalize': True,
    'batch_size': 32,
    'shuffle_train': True,
    'num_workers': 0,
    'hidden_dim': 128,
    'num_heads': 4,
    'num_layers': 3,
    'dropout': 0.1,
    'fusion_type': 'concat',
    'target_type': 'regression',
    'output_dim': 1,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 100,
    'early_stopping_patience': 20,
    'scheduler_patience': 10,
    'scheduler_factor': 0.5,
    'clip_grad_norm': 1.0,
    'log_interval': 10,
    'use_all_modalities': True
}


class TrainPipeline:
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        try:
            if config_path:
                self.config = self._load_config(config_path)
            elif config:
                self.config = config
            else:
                self.config = DEFAULT_CONFIG.copy()
            
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(self.config['output_dir']) / self.timestamp
            ensure_directory(self.run_dir)
            self._save_config()
            
            logging.info("=" * 60)
            logging.info("TRAIN PIPELINE INITIALIZED")
            logging.info("=" * 60)
            logging.info(f"  Run directory: {self.run_dir}")
            logging.info("=" * 60)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _save_config(self):
        config_path = self.run_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logging.info(f"  Config saved to {config_path}")
    
    def run(self) -> Dict:
        try:
            logging.info("\n" + "=" * 60)
            logging.info("STARTING TRAINING PIPELINE")
            logging.info("=" * 60)
            
            results = {
                'run_timestamp': self.timestamp,
                'run_dir': str(self.run_dir),
                'config': self.config
            }
            
            # ================================================================
            # STEP 1: Data Ingestion
            # ================================================================
            
            logging.info("\n📂 STEP 1: Data Ingestion")
            logging.info("-" * 40)
            
            ingestor = DataIngestion(data_dir=self.config['data_dir'], chunk_size=10000)
            ingestion_data = ingestor.load_all()
            
            if not ingestion_data['compounds']:
                logging.error("No data loaded. Please run conversion scripts first.")
                return results
            
            logging.info(f"  Loaded {len(ingestion_data['compounds'])} compounds")
            logging.info(f"  Multi-modal: {len(ingestion_data['multi_modal_compounds'])}")
            
            results['num_compounds'] = len(ingestion_data['compounds'])
            results['multi_modal_compounds'] = len(ingestion_data['multi_modal_compounds'])
            
            # ================================================================
            # STEP 2: Data Preprocessing
            # ================================================================
            
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
            
            preprocessor.save_processed_data(
                processed_data,
                output_dir=str(self.run_dir / 'processed_data/')
            )
            
            logging.info(f"  Saved processed data to {self.run_dir / 'processed_data/'}")
            
            # ================================================================
            # STEP 3: Data Transformation
            # ================================================================
            
            logging.info("\n🔄 STEP 3: Data Transformation")
            logging.info("-" * 40)
            
            # Load target values
            targets_file = Path(self.config['data_dir']) / "ms_spectra.csv"
            if not targets_file.exists():
                raise CustomException(f"Target file not found: {targets_file}", sys)
            
            targets_df = pd.read_csv(targets_file)
            
            if 'target' not in targets_df.columns:
                raise CustomException(
                    f"No 'target' column found in {targets_file}. "
                    "Please add a 'target' column with the property you want to predict.",
                    sys
                )
            
            target_dict = dict(zip(targets_df['compound_id'], targets_df['target']))
            logging.info(f"  Loaded {len(target_dict)} target values")
            
            # Build target_values for each split
            target_values = {}
            for split_name in ['train', 'validation', 'test']:
                if split_name in processed_data and processed_data[split_name] is not None:
                    compound_ids = processed_data[split_name]['compound_ids']
                    targets = []
                    missing = []
                    for cid in compound_ids:
                        if cid in target_dict and pd.notna(target_dict[cid]):
                            targets.append(float(target_dict[cid]))
                        else:
                            missing.append(cid)
                            targets.append(0.0)
                    
                    if missing:
                        logging.warning(f"Missing targets for {len(missing)} compounds in {split_name}")
                    
                    target_values[split_name] = targets
                    logging.info(f"  ✅ Loaded {len(targets)} targets for {split_name}")
                else:
                    target_values[split_name] = None
                    logging.warning(f"  No data for {split_name} split")
            
            transformer = DataTransformation(
                batch_size=self.config['batch_size'],
                shuffle_train=self.config['shuffle_train'],
                num_workers=self.config['num_workers'],
                target_type=self.config['target_type']
            )
            
            dataloaders = transformer.create_dataloaders(
                processed_data,
                target_values=target_values
            )
            
            transformer.save_dataloader_state(
                dataloaders,
                output_dir=str(self.run_dir / 'dataloaders/')
            )
            
            # ✅ Log what we have
            logging.info(f"  DataLoaders created:")
            for split_name in ['train', 'validation', 'test']:
                loader = dataloaders.get(split_name)
                if loader is not None:
                    logging.info(f"    {split_name}: {len(loader.dataset)} samples")
                else:
                    logging.info(f"    {split_name}: None")
            
            # ================================================================
            # STEP 4: Model Creation
            # ================================================================
            
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
            
            # ================================================================
            # STEP 5: Model Training
            # ================================================================
            
            logging.info("\n🏋️ STEP 5: Model Training")
            logging.info("-" * 40)
            
            trainer = ModelTrainer(
                model=model,
                train_loader=dataloaders['train'],
                val_loader=dataloaders.get('validation'),  # ✅ This will be None if not available
                test_loader=dataloaders.get('test'),       # ✅ This will be None if not available
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
            
            history = trainer.train()
            results['history'] = history
            
            if history.get('val_loss') and len(history['val_loss']) > 0:
                valid_losses = [l for l in history['val_loss'] if l > 0]
                if valid_losses:
                    results['best_val_loss'] = min(valid_losses)
                    results['best_epoch'] = history['val_loss'].index(min(valid_losses)) + 1
                    logging.info(f"\n  ✅ Best validation loss: {results['best_val_loss']:.4f} at epoch {results['best_epoch']}")
            else:
                logging.info("\n  ✅ Training completed without validation")
            
            # ================================================================
            # STEP 6: Save Final Model
            # ================================================================
            
            logging.info("\n💾 STEP 6: Saving Final Model")
            logging.info("-" * 40)
            
            final_model_path = self.run_dir / 'final_model.pt'
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'config': model_config,
                'history': history,
                'best_val_loss': results.get('best_val_loss'),
                'best_epoch': results.get('best_epoch')
            }, final_model_path)
            
            logging.info(f"  Final model saved to {final_model_path}")
            
            # Save results
            save_json(results, self.run_dir / 'results.json')
            logging.info(f"  Results saved to {self.run_dir / 'results.json'}")
            
            logging.info("\n" + "=" * 60)
            logging.info("✅ TRAINING PIPELINE COMPLETE!")
            logging.info("=" * 60)
            logging.info(f"  Run directory: {self.run_dir}")
            if 'best_val_loss' in results:
                logging.info(f"  Best validation loss: {results['best_val_loss']:.4f}")
            logging.info(f"  Model saved to: {final_model_path}")
            logging.info("=" * 60)
            
            return results
            
        except Exception as e:
            raise CustomException(e, sys)


def main():
    parser = argparse.ArgumentParser(description='Run the training pipeline')
    parser.add_argument('--data_dir', type=str, default='artifacts/data/converted')
    parser.add_argument('--output_dir', type=str, default='artifacts/models/')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'attention', 'gated'])
    parser.add_argument('--target_type', type=str, default='regression',
                        choices=['regression', 'classification'])
    
    args = parser.parse_args()
    
    try:
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
        
        pipeline = TrainPipeline(config=config)
        results = pipeline.run()
        print(f"\n✅ Training complete! Results saved to {results['run_dir']}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()