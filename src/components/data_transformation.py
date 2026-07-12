"""
data_transformation.py - Converts preprocessed data to PyTorch tensors.
"""
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from src.logger import logging
from src.exception import CustomException
from src.utils import ensure_directory


class SpectralDataset(Dataset):
    """PyTorch Dataset for multi-modal spectral data."""
    
    def __init__(self, 
                 ms_spectra: np.ndarray,
                 nmr_spectra: np.ndarray,
                 ir_spectra: np.ndarray,
                 targets: Optional[np.ndarray] = None,
                 compound_ids: Optional[List[str]] = None,
                 target_type: str = 'regression'):
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
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
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


class DataTransformation:
    """Transforms preprocessed data into PyTorch tensors and DataLoaders."""
    
    def __init__(self, 
                 batch_size: int = 32,
                 shuffle_train: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 target_type: str = 'regression',
                 drop_last: bool = False):
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
            logging.info(f"  Target type: {target_type}")
            logging.info("=" * 60)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_dataloaders(self, 
                           processed_data: Dict,
                           target_values: Optional[Dict] = None) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for train, validation, and test splits.
        
        Args:
            processed_data: Output from SpectralPreprocessor.create_dataset()
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
                if split_name not in target_values or target_values[split_name] is None:
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
    
    def save_dataloader_state(self, 
                              dataloaders: Dict[str, DataLoader],
                              output_dir: str = "artifacts/data/dataloaders/") -> None:
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
                        'ms_shape': loader.dataset.ms_spectra.shape if hasattr(loader.dataset, 'ms_spectra') else None,
                        'nmr_shape': loader.dataset.nmr_spectra.shape if hasattr(loader.dataset, 'nmr_spectra') else None,
                        'ir_shape': loader.dataset.ir_spectra.shape if hasattr(loader.dataset, 'ir_spectra') else None
                    }
            
            with open(output_path / 'dataloader_stats.json', 'w') as f:
                json.dump(stats, f, indent=4)
            
            logging.info(f"DataLoader statistics saved to {output_path}")
            
        except Exception as e:
            raise CustomException(e, sys)