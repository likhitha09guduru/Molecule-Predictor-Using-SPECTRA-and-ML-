"""
data_transformation.py - Transforms NMR and IR spectral data for molecular prediction.
"""
import sys
import os
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, ensure_directory


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation."""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    train_arr_path: str = os.path.join('artifacts', 'train_arr.npy')
    test_arr_path: str = os.path.join('artifacts', 'test_arr.npy')
    val_arr_path: str = os.path.join('artifacts', 'val_arr.npy')


class SpectralDataset(Dataset):
    """PyTorch Dataset for NMR and IR spectral data."""
    
    def __init__(self, 
                 nmr_spectra: np.ndarray,
                 ir_spectra: np.ndarray,
                 targets: Optional[np.ndarray] = None,
                 compound_ids: Optional[List[str]] = None,
                 target_type: str = 'regression'):
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


class DataTransformation:
    """
    Data transformation class for NMR and IR spectral data.
    """
    
    def __init__(self, max_peaks_nmr: int = 150, max_peaks_ir: int = 100):
        """Initialize data transformation with configuration."""
        self.data_transformation_config = DataTransformationConfig()
        self.batch_size = 32
        self.shuffle_train = True
        self.num_workers = 0
        self.pin_memory = False
        self.target_type = 'regression'
        self.max_peaks_nmr = max_peaks_nmr
        self.max_peaks_ir = max_peaks_ir
        
        logging.info("=" * 60)
        logging.info("DATA TRANSFORMATION INITIALIZED (NMR + IR)")
        logging.info("=" * 60)
        logging.info(f"  Batch size: {self.batch_size}")
        logging.info(f"  Target type: {self.target_type}")
        logging.info(f"  Max NMR peaks: {self.max_peaks_nmr}")
        logging.info(f"  Max IR peaks: {self.max_peaks_ir}")
        logging.info("=" * 60)
    
    def _process_spectral_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process spectral data from DataFrame into numpy arrays.
        
        Args:
            df: DataFrame with NMR and IR data
        
        Returns:
            Tuple of (nmr_array, ir_array, targets)
        """
        try:
            nmr_list = []
            ir_list = []
            target_list = []
            
            for idx, row in df.iterrows():
                # Process NMR peaks
                nmr_data = np.zeros((self.max_peaks_nmr, 3), dtype=np.float32)
                
                # Get proton peaks (1H) - format: [(shift, multiplicity, intensity), ...]
                proton_peaks = row.get('proton_peaks', [])
                if isinstance(proton_peaks, str):
                    # Parse string if needed
                    try:
                        proton_peaks = eval(proton_peaks)
                    except:
                        proton_peaks = []
                
                # Fill NMR data
                for i, peak in enumerate(proton_peaks[:self.max_peaks_nmr]):
                    if isinstance(peak, (tuple, list)) and len(peak) >= 3:
                        nmr_data[i, 0] = float(peak[0])  # chemical shift
                        # Encode multiplicity
                        mult = str(peak[1]).lower()
                        mult_map = {'s': 0, 'd': 1, 't': 2, 'q': 3, 'm': 4, 'dd': 5, 'dt': 6, 'td': 7, 'brs': 8}
                        nmr_data[i, 1] = mult_map.get(mult, 4)  # default to 'm'
                        nmr_data[i, 2] = float(peak[2])  # intensity
                
                nmr_list.append(nmr_data)
                
                # Process IR peaks
                ir_data = np.zeros((self.max_peaks_ir, 2), dtype=np.float32)
                
                ir_peaks = row.get('ir_peaks', [])
                if isinstance(ir_peaks, str):
                    try:
                        ir_peaks = eval(ir_peaks)
                    except:
                        ir_peaks = []
                
                for i, peak in enumerate(ir_peaks[:self.max_peaks_ir]):
                    if isinstance(peak, (tuple, list)) and len(peak) >= 2:
                        ir_data[i, 0] = float(peak[0])  # wavenumber
                        ir_data[i, 1] = float(peak[1])  # intensity
                
                ir_list.append(ir_data)
                
                # Get target (using SMILES as target for now)
                target = row.get('smiles', '')
                # For regression, use a numeric target - you can replace with actual target
                # For now, use proton_count as a placeholder target
                target_val = row.get('proton_count', 0)
                target_list.append(float(target_val))
            
            return np.array(nmr_list), np.array(ir_list), np.array(target_list)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, 
                                     train_path: str,
                                     test_path: str,
                                     val_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
        """
        Initiate data transformation for NMR and IR data.
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
            val_path: Path to validation data CSV (optional)
            
        Returns:
            Tuple of (train_arr, test_arr, val_arr, preprocessor_path)
        """
        try:
            logging.info("Entered data transformation method")
            
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            if val_path:
                val_df = pd.read_csv(val_path)
                logging.info(f"Validation data loaded: {len(val_df)} rows")
            
            logging.info(f"Train data loaded: {len(train_df)} rows")
            logging.info(f"Test data loaded: {len(test_df)} rows")
            
            # Process spectral data
            logging.info("Processing spectral data into numerical arrays...")
            
            # Process train data
            train_nmr, train_ir, train_targets = self._process_spectral_data(train_df)
            logging.info(f"Train NMR shape: {train_nmr.shape}, Train IR shape: {train_ir.shape}")
            
            # Process test data
            test_nmr, test_ir, test_targets = self._process_spectral_data(test_df)
            logging.info(f"Test NMR shape: {test_nmr.shape}, Test IR shape: {test_ir.shape}")
            
            # Process validation data if available
            if val_path:
                val_nmr, val_ir, val_targets = self._process_spectral_data(val_df)
                logging.info(f"Val NMR shape: {val_nmr.shape}, Val IR shape: {val_ir.shape}")
            else:
                val_nmr, val_ir, val_targets = None, None, None
            
            # For sklearn compatibility, flatten the spectral data
            # This creates feature vectors for sklearn models
            train_nmr_flat = train_nmr.reshape(train_nmr.shape[0], -1)
            train_ir_flat = train_ir.reshape(train_ir.shape[0], -1)
            train_features = np.concatenate([train_nmr_flat, train_ir_flat], axis=1)
            
            test_nmr_flat = test_nmr.reshape(test_nmr.shape[0], -1)
            test_ir_flat = test_ir.reshape(test_ir.shape[0], -1)
            test_features = np.concatenate([test_nmr_flat, test_ir_flat], axis=1)
            
            # Create arrays with targets
            train_arr = np.c_[train_features, train_targets.reshape(-1, 1)]
            test_arr = np.c_[test_features, test_targets.reshape(-1, 1)]
            
            if val_path:
                val_nmr_flat = val_nmr.reshape(val_nmr.shape[0], -1)
                val_ir_flat = val_ir.reshape(val_ir.shape[0], -1)
                val_features = np.concatenate([val_nmr_flat, val_ir_flat], axis=1)
                val_arr = np.c_[val_features, val_targets.reshape(-1, 1)]
            else:
                val_arr = None
            
            # Save arrays
            np.save(self.data_transformation_config.train_arr_path, train_arr)
            np.save(self.data_transformation_config.test_arr_path, test_arr)
            if val_arr is not None:
                np.save(self.data_transformation_config.val_arr_path, val_arr)
            
            logging.info(f"Train array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")
            if val_arr is not None:
                logging.info(f"Validation array shape: {val_arr.shape}")
            
            logging.info("Data transformation completed successfully")
            
            return (
                train_arr,
                test_arr,
                val_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_dataloaders(self, 
                          train_nmr: np.ndarray,
                          train_ir: np.ndarray,
                          train_targets: np.ndarray,
                          test_nmr: np.ndarray,
                          test_ir: np.ndarray,
                          test_targets: np.ndarray,
                          val_nmr: Optional[np.ndarray] = None,
                          val_ir: Optional[np.ndarray] = None,
                          val_targets: Optional[np.ndarray] = None) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders from spectral data.
        
        Args:
            train_nmr: Training NMR data
            train_ir: Training IR data
            train_targets: Training targets
            test_nmr: Test NMR data
            test_ir: Test IR data
            test_targets: Test targets
            val_nmr: Validation NMR data (optional)
            val_ir: Validation IR data (optional)
            val_targets: Validation targets (optional)
        
        Returns:
            Dictionary with 'train', 'validation', 'test' DataLoaders
        """
        try:
            logging.info("\n" + "=" * 60)
            logging.info("CREATING PYTORCH DATALOADERS")
            logging.info("=" * 60)
            
            dataloaders = {}
            
            # Training dataloader
            train_dataset = SpectralDataset(
                nmr_spectra=train_nmr,
                ir_spectra=train_ir,
                targets=train_targets,
                target_type=self.target_type
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            dataloaders['train'] = train_loader
            logging.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
            
            # Test dataloader
            test_dataset = SpectralDataset(
                nmr_spectra=test_nmr,
                ir_spectra=test_ir,
                targets=test_targets,
                target_type=self.target_type
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            dataloaders['test'] = test_loader
            logging.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
            
            # Validation dataloader if available
            if val_nmr is not None:
                val_dataset = SpectralDataset(
                    nmr_spectra=val_nmr,
                    ir_spectra=val_ir,
                    targets=val_targets,
                    target_type=self.target_type
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory
                )
                dataloaders['validation'] = val_loader
                logging.info(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
            
            logging.info("=" * 60)
            logging.info("✅ DATALOADERS CREATED SUCCESSFULLY!")
            logging.info("=" * 60)
            
            return dataloaders
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print("🧪 TESTING DATA TRANSFORMATION (NMR + IR)")
        print("=" * 60)
        
        data_transformation = DataTransformation()
        print("\n✅ Data transformation test completed!")
        
    except Exception as e:
        raise CustomException(e, sys)