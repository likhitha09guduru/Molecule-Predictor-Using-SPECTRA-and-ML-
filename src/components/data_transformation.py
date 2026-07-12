"""
data_transformation.py - Converts preprocessed data to PyTorch tensors.

PURPOSE: Takes preprocessed numpy arrays and converts them to PyTorch tensors
with DataLoader support for efficient training.

WHAT THIS DOES:
1. Converts numpy arrays → PyTorch tensors
2. Creates PyTorch Dataset classes
3. Creates DataLoader for batching and shuffling
4. Handles multi-modal data (MS, NMR, IR)
5. Supports chunking for large datasets

INPUT: Output from data_preprocessing.py (dictionaries with numpy arrays)
OUTPUT: PyTorch DataLoader objects ready for model training
"""
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import gc

from src.logger import logging
from src.exception import CustomException
from src.utils import ensure_directory


class SpectralDataset(Dataset):
    """
    PyTorch Dataset for multi-modal spectral data.
    
    This dataset handles:
    - MS spectra: Shape (max_peaks_ms, 2)
    - NMR spectra: Shape (max_peaks_nmr, 3)
    - IR spectra: Shape (max_peaks_ir, 2)
    - Targets: Single value (regression) or class (classification)
    """
    
    def __init__(self, 
                 ms_spectra: np.ndarray,
                 nmr_spectra: np.ndarray,
                 ir_spectra: np.ndarray,
                 targets: Optional[np.ndarray] = None,
                 compound_ids: Optional[List[str]] = None,
                 target_type: str = 'regression'):
        """
        Initialize the dataset.
        
        Args:
            ms_spectra: Shape (N, max_peaks_ms, 2)
            nmr_spectra: Shape (N, max_peaks_nmr, 3)
            ir_spectra: Shape (N, max_peaks_ir, 2)
            targets: Shape (N,) for regression or (N, num_classes) for classification
            compound_ids: List of compound IDs (for debugging)
            target_type: 'regression' or 'classification'
        """
        try:
            self.ms_spectra = torch.tensor(ms_spectra, dtype=torch.float32)
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
            self.length = len(ms_spectra)
            
            logging.info(f"SpectralDataset initialized with {self.length} samples")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - 'ms': MS spectrum tensor
            - 'nmr': NMR spectrum tensor
            - 'ir': IR spectrum tensor
            - 'target': Target value (if available)
            - 'compound_id': Compound ID (if available)
        """
        try:
            sample = {
                'ms': self.ms_spectra[idx],
                'nmr': self.nmr_spectra[idx],
                'ir': self.ir_spectra[idx]
            }
            
            if self.targets is not None:
                sample['target'] = self.targets[idx]
            
            if self.compound_ids is not None:
                sample['compound_id'] = self.compound_ids[idx]
            
            return sample
            
        except Exception as e:
            raise CustomException(e, sys)


class DataTransformation:
    """
    Transforms preprocessed data into PyTorch tensors and DataLoaders.
    
    HOW TO USE:
    -----------
    transformer = DataTransformation(batch_size=32)
    loaders = transformer.create_dataloaders(processed_data, target_values=targets)
    
    # Access loaders
    train_loader = loaders['train']
    val_loader = loaders['validation']
    test_loader = loaders['test']
    """
    
    def __init__(self, 
                 batch_size: int = 32,
                 shuffle_train: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 target_type: str = 'regression',
                 drop_last: bool = False):
        """
        Initialize the data transformer.
        
        Args:
            batch_size: Number of samples per batch
            shuffle_train: Whether to shuffle training data
            num_workers: Number of workers for DataLoader (0 = main process)
            pin_memory: Whether to pin memory for faster GPU transfer
            target_type: 'regression' or 'classification'
            drop_last: Whether to drop the last incomplete batch
        """
        try:
            self.batch_size = batch_size
            self.shuffle_train = shuffle_train
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.target_type = target_type
            self.drop_last = drop_last
            
            logging.info("=" * 60)
            logging.info("DATA TRANSFORMATION INITIALIZED")
            logging.info("=" * 60)
            logging.info(f"  Batch size: {batch_size}")
            logging.info(f"  Shuffle train: {shuffle_train}")
            logging.info(f"  Num workers: {num_workers}")
            logging.info(f"  Target type: {target_type}")
            logging.info("=" * 60)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # MAIN METHOD: Create DataLoaders
    # =========================================================================
    
    def create_dataloaders(self, 
                           processed_data: Dict,
                           target_property: Optional[str] = None,
                           target_values: Optional[Dict] = None) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for train, validation, and test splits.
        
        Args:
            processed_data: Output from SpectralPreprocessor.create_dataset()
            target_property: Name of target property (e.g., 'logP', 'solubility')
            target_values: Dictionary mapping split_name -> list of target values
        
        Returns:
            Dictionary with 'train', 'validation', 'test' DataLoaders
        
        Raises:
            CustomException: If no targets are provided
        """
        try:
            logging.info("\n" + "=" * 60)
            logging.info("CREATING DATALOADERS")
            logging.info("=" * 60)
            
            # ✅ REQUIRED: Check if targets are provided
            if target_values is None:
                error_msg = (
                    "❌ FATAL: No target values provided!\n"
                    "   target_values parameter is required.\n"
                    "   Please provide target_values as a dictionary:\n"
                    "   target_values = {\n"
                    "       'train': [0.5, 1.2, 0.8, ...],\n"
                    "       'validation': [0.6, 1.1, ...],\n"
                    "       'test': [0.7, 0.9, ...]\n"
                    "   }\n"
                    "   The number of targets must match the number of compounds in each split."
                )
                logging.error(error_msg)
                raise CustomException(error_msg, sys)
            
            dataloaders = {}
            
            for split_name in ['train', 'validation', 'test']:
                if split_name not in processed_data or processed_data[split_name] is None:
                    logging.warning(f"No data available for {split_name} split")
                    dataloaders[split_name] = None
                    continue
                
                data = processed_data[split_name]
                compound_ids = data.get('compound_ids', [])
                num_samples = len(compound_ids)
                
                # ✅ REQUIRED: Check if targets exist for this split
                if split_name not in target_values:
                    error_msg = (
                        f"❌ FATAL: No targets provided for {split_name} split!\n"
                        f"   Expected {num_samples} targets for {split_name}.\n"
                        f"   Available splits in target_values: {list(target_values.keys())}\n"
                        f"   Please add targets for '{split_name}' split."
                    )
                    logging.error(error_msg)
                    raise CustomException(error_msg, sys)
                
                targets_array = target_values[split_name]
                
                # ✅ REQUIRED: Check if target count matches compound count
                if len(targets_array) != num_samples:
                    error_msg = (
                        f"❌ FATAL: Target count mismatch for {split_name} split!\n"
                        f"   Number of compounds: {num_samples}\n"
                        f"   Number of targets: {len(targets_array)}\n"
                        f"   These must match. Please check your target values."
                    )
                    logging.error(error_msg)
                    raise CustomException(error_msg, sys)
                
                targets = np.array(targets_array, dtype=np.float32)
                logging.info(f"  ✅ Using provided targets for {split_name}: {len(targets)} samples")
                
                # Create dataset
                dataset = SpectralDataset(
                    ms_spectra=data['ms'],
                    nmr_spectra=data['nmr'],
                    ir_spectra=data['ir'],
                    targets=targets,
                    compound_ids=compound_ids,
                    target_type=self.target_type
                )
                
                # Create dataloader
                shuffle = self.shuffle_train if split_name == 'train' else False
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=self.drop_last and split_name == 'train'
                )
                
                dataloaders[split_name] = dataloader
                logging.info(f"  {split_name}: {len(dataset)} samples, {len(dataloader)} batches")
            
            logging.info("\n" + "=" * 60)
            logging.info("✅ DATALOADERS CREATED SUCCESSFULLY!")
            logging.info("=" * 60)
            
            return dataloaders
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # BATCH PROCESSING HELPERS
    # =========================================================================
    
    def collate_multi_modal(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for multi-modal data.
        
        Args:
            batch: List of samples from SpectralDataset
        
        Returns:
            Dictionary with batched tensors
        """
        try:
            ms = torch.stack([item['ms'] for item in batch])
            nmr = torch.stack([item['nmr'] for item in batch])
            ir = torch.stack([item['ir'] for item in batch])
            
            result = {
                'ms': ms,
                'nmr': nmr,
                'ir': ir
            }
            
            if 'target' in batch[0]:
                if self.target_type == 'regression':
                    targets = torch.tensor([item['target'] for item in batch], dtype=torch.float32)
                else:
                    targets = torch.tensor([item['target'] for item in batch], dtype=torch.long)
                result['target'] = targets
            
            if 'compound_id' in batch[0]:
                result['compound_ids'] = [item['compound_id'] for item in batch]
            
            return result
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_batch_statistics(self, dataloader: DataLoader) -> Dict:
        """
        Get statistics for a dataloader (for debugging).
        
        Args:
            dataloader: PyTorch DataLoader
        
        Returns:
            Dictionary with batch statistics
        """
        try:
            if dataloader is None:
                return {}
            
            batch = next(iter(dataloader))
            stats = {
                'num_batches': len(dataloader),
                'batch_size': self.batch_size,
                'ms_shape': batch['ms'].shape,
                'nmr_shape': batch['nmr'].shape,
                'ir_shape': batch['ir'].shape
            }
            
            if 'target' in batch:
                stats['target_shape'] = batch['target'].shape
                if self.target_type == 'regression':
                    stats['target_min'] = batch['target'].min().item()
                    stats['target_max'] = batch['target'].max().item()
                    stats['target_mean'] = batch['target'].mean().item()
                    stats['target_std'] = batch['target'].std().item()
                else:
                    stats['num_classes'] = len(torch.unique(batch['target']))
            
            return stats
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # SAVE AND LOAD METHODS
    # =========================================================================
    
    def save_dataloader_state(self, 
                              dataloaders: Dict[str, DataLoader],
                              output_dir: str = "artifacts/data/dataloaders/") -> None:
        """
        Save dataloader statistics (not the actual data).
        
        Args:
            dataloaders: Dictionary of DataLoaders
            output_dir: Directory to save statistics
        """
        try:
            output_path = Path(output_dir)
            ensure_directory(output_path)
            
            stats = {}
            for split_name, loader in dataloaders.items():
                if loader is not None:
                    stats[split_name] = {
                        'num_samples': len(loader.dataset),
                        'num_batches': len(loader),
                        'batch_size': self.batch_size,
                        'ms_shape': loader.dataset.ms_spectra.shape,
                        'nmr_shape': loader.dataset.nmr_spectra.shape,
                        'ir_shape': loader.dataset.ir_spectra.shape
                    }
            
            import json
            with open(output_path / 'dataloader_stats.json', 'w') as f:
                json.dump(stats, f, indent=4)
            
            logging.info(f"DataLoader statistics saved to {output_path}")
            
        except Exception as e:
            raise CustomException(e, sys)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING DATA TRANSFORMATION")
    print("=" * 60)
    
    try:
        from src.components.data_ingestion import DataIngestion
        from src.components.data_preprocessing import SpectralPreprocessor
        
        # Step 1: Ingest real data
        print("\n📂 Loading real data from CSVs...")
        ingestor = DataIngestion(data_dir="artifacts/data/converted")
        ingestion_data = ingestor.load_all()
        
        if not ingestion_data['compounds']:
            print("\n⚠️ No data loaded. Please run conversion scripts first:")
            print("   python data_conversion/run_all_conversions.py")
            sys.exit(1)
        
        # Step 2: Preprocess data
        print("\n🔧 Preprocessing data...")
        preprocessor = SpectralPreprocessor(
            max_peaks_ms=200,
            max_peaks_nmr=150,
            max_peaks_ir=100,
            chunk_size=5000
        )
        processed_data = preprocessor.create_dataset(ingestion_data)
        
        # Step 3: Create dummy targets for testing ONLY
        # ✅ In production, replace this with real target values from your CSVs
        print("\n🎯 Creating dummy targets for testing...")
        print("   ⚠️  In production, replace these with real target values!")
        
        dummy_targets = {}
        for split_name in ['train', 'validation', 'test']:
            if split_name in processed_data and processed_data[split_name] is not None:
                num_samples = len(processed_data[split_name]['compound_ids'])
                dummy_targets[split_name] = np.random.uniform(0, 10, num_samples).tolist()
                print(f"   {split_name}: {num_samples} dummy targets generated")
        
        # Step 4: Transform to DataLoaders
        print("\n🔄 Creating DataLoaders with real data...")
        transformer = DataTransformation(
            batch_size=16,
            shuffle_train=True,
            num_workers=0
        )
        
        # ✅ Pass target_values (required)
        dataloaders = transformer.create_dataloaders(
            processed_data,
            target_values=dummy_targets  # ← In production, use real targets
        )
        
        # Step 5: Show statistics
        print("\n📊 DATALOADER STATISTICS:")
        for split_name, loader in dataloaders.items():
            if loader is not None:
                stats = transformer.get_batch_statistics(loader)
                print(f"\n{split_name.upper()}:")
                print(f"  Batches: {stats['num_batches']}")
                print(f"  Batch size: {stats['batch_size']}")
                print(f"  MS shape: {stats['ms_shape']}")
                print(f"  NMR shape: {stats['nmr_shape']}")
                print(f"  IR shape: {stats['ir_shape']}")
                if 'target_shape' in stats:
                    print(f"  Target shape: {stats['target_shape']}")
                    print(f"  Target range: {stats['target_min']:.2f} - {stats['target_max']:.2f}")
        
        # Step 6: Sample a batch
        if dataloaders.get('train') is not None:
            print("\n🔬 SAMPLE BATCH (first batch from train):")
            batch = next(iter(dataloaders['train']))
            print(f"  MS tensor shape: {batch['ms'].shape}")
            print(f"  NMR tensor shape: {batch['nmr'].shape}")
            print(f"  IR tensor shape: {batch['ir'].shape}")
            if 'target' in batch:
                print(f"  Target tensor shape: {batch['target'].shape}")
                print(f"  Target values (first 5): {batch['target'][:5].tolist()}")
        
        print("\n✅ Data transformation test complete!")
        print("\n📌 REMINDER: Replace dummy_targets with real target values from your CSVs!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please run conversion scripts first to create CSV files.")
        print("   python data_conversion/run_all_conversions.py")
    except Exception as e:
        raise CustomException(e, sys)