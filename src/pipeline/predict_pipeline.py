"""
predict_pipeline.py - Prediction pipeline for trained models.

PURPOSE: Loads a trained model and makes predictions on new spectral data.

WORKFLOW:
1. Load trained model from checkpoint
2. Load preprocessing parameters
3. Accept new spectra (MS, NMR, IR)
4. Preprocess spectra (same as training)
5. Run model inference
6. Return predictions

HOW TO USE:
-----------
from src.pipeline.predict_pipeline import PredictPipeline

predictor = PredictPipeline(model_path="artifacts/models/best_model.pt")
prediction = predictor.predict(ms_spectrum, nmr_spectrum, ir_spectrum)
"""
import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import (
    ensure_directory,
    parse_peak_string,
    parse_nmr_peaks,
    load_json,
    load_pickle,
    encode_multiplicity
)

# ============================================================================
# IMPORT COMPONENTS
# ============================================================================

from src.components.model_architecture import MultiModalSpectraModel, create_model


# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================

@dataclass
class PredictionConfig:
    """Configuration for prediction pipeline."""
    
    # Model paths
    model_path: str = "artifacts/models/best_model.pt"
    config_path: Optional[str] = None
    
    # Preprocessing parameters (should match training)
    max_peaks_ms: int = 200
    max_peaks_nmr: int = 150
    max_peaks_ir: int = 100
    noise_threshold_ms: float = 0.01
    noise_threshold_nmr: float = 0.001
    noise_threshold_ir: float = 0.01
    normalize: bool = True
    pad_value: float = 0.0
    
    # Inference settings
    device: str = "cpu"  # "cuda" or "cpu"
    batch_size: int = 32


class PredictPipeline:
    """
    Prediction pipeline for trained multi-modal models.
    
    HOW TO USE:
    -----------
    # Initialize with model path
    predictor = PredictPipeline(model_path="artifacts/models/best_model.pt")
    
    # Predict on single sample
    prediction = predictor.predict_single(ms_peaks, nmr_peaks, ir_peaks)
    
    # Predict on multiple samples
    predictions = predictor.predict_batch(ms_list, nmr_list, ir_list)
    """
    
    def __init__(self, config: Optional[PredictionConfig] = None, **kwargs):
        """
        Initialize the prediction pipeline.
        
        Args:
            config: PredictionConfig object
            **kwargs: Override config parameters
        """
        try:
            # Setup config
            if config is None:
                self.config = PredictionConfig()
            else:
                self.config = config
            
            # Override with kwargs
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Set device
            if self.config.device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            
            # Load model
            self.model = None
            self.model_config = None
            self.target_type = None
            self._load_model()
            
            logging.info("=" * 60)
            logging.info("PREDICT PIPELINE INITIALIZED")
            logging.info("=" * 60)
            logging.info(f"  Device: {self.device}")
            logging.info(f"  Target type: {self.target_type}")
            logging.info(f"  Model path: {self.config.model_path}")
            logging.info("=" * 60)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # MODEL LOADING
    # =========================================================================
    
    def _load_model(self):
        """Load the trained model from checkpoint."""
        try:
            model_path = Path(self.config.model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model config
            if 'config' in checkpoint:
                self.model_config = checkpoint['config']
                self.target_type = self.model_config.get('target_type', 'regression')
            elif self.config.config_path and Path(self.config.config_path).exists():
                self.model_config = load_json(self.config.config_path)
                self.target_type = self.model_config.get('target_type', 'regression')
            else:
                # Use default config from model
                logging.warning("No config found, using default model configuration")
                self.model_config = {
                    'ms_input_dim': 2,
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
                self.target_type = 'regression'
            
            # Create model
            self.model = create_model(self.model_config)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logging.info(f"  Model loaded successfully: {self.model_config['hidden_dim']} hidden dim")
            logging.info(f"  Output dimension: {self.model_config['output_dim']}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # SINGLE SPECTRUM PROCESSING
    # =========================================================================
    
    def _process_ms_peaks(self, peaks: List[Tuple[float, float]]) -> np.ndarray:
        """
        Process MS peaks for model input.
        
        Args:
            peaks: List of (m/z, intensity) tuples
        
        Returns:
            Numpy array of shape (max_peaks_ms, 2)
        """
        try:
            # Filter noise
            filtered = [(p[0], p[1]) for p in peaks if p[1] > self.config.noise_threshold_ms]
            
            if not filtered:
                return np.full((self.config.max_peaks_ms, 2), self.config.pad_value, dtype=np.float32)
            
            # Sort by intensity (descending)
            filtered.sort(key=lambda x: x[1], reverse=True)
            
            # Normalize
            if self.config.normalize:
                max_intensity = max(p[1] for p in filtered)
                if max_intensity > 0:
                    filtered = [(p[0], p[1] / max_intensity) for p in filtered]
            
            # Pad/truncate
            values = np.array([p[0] for p in filtered[:self.config.max_peaks_ms]])
            intensities = np.array([p[1] for p in filtered[:self.config.max_peaks_ms]])
            
            if len(values) < self.config.max_peaks_ms:
                values = np.pad(values, (0, self.config.max_peaks_ms - len(values)), 
                               constant_values=self.config.pad_value)
                intensities = np.pad(intensities, (0, self.config.max_peaks_ms - len(intensities)),
                                    constant_values=self.config.pad_value)
            
            return np.column_stack([values, intensities])
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _process_nmr_peaks(self, peaks: List[Tuple[float, str, float]]) -> np.ndarray:
        """
        Process NMR peaks for model input.
        
        Args:
            peaks: List of (shift, multiplicity, integral) tuples
        
        Returns:
            Numpy array of shape (max_peaks_nmr, 3)
        """
        try:
            # Filter noise and encode multiplicity
            filtered = []
            for shift, mult, intensity in peaks:
                if intensity > self.config.noise_threshold_nmr:
                    mult_code = encode_multiplicity(mult)
                    filtered.append((float(shift), int(mult_code), float(intensity)))
            
            if not filtered:
                return np.full((self.config.max_peaks_nmr, 3), self.config.pad_value, dtype=np.float32)
            
            # Sort by intensity (descending)
            filtered.sort(key=lambda x: x[2], reverse=True)
            
            # Normalize
            if self.config.normalize:
                max_intensity = max(p[2] for p in filtered)
                if max_intensity > 0:
                    filtered = [(p[0], p[1], p[2] / max_intensity) for p in filtered]
            
            # Pad/truncate
            shifts = np.array([p[0] for p in filtered[:self.config.max_peaks_nmr]])
            mult_codes = np.array([p[1] for p in filtered[:self.config.max_peaks_nmr]])
            intensities = np.array([p[2] for p in filtered[:self.config.max_peaks_nmr]])
            
            if len(shifts) < self.config.max_peaks_nmr:
                shifts = np.pad(shifts, (0, self.config.max_peaks_nmr - len(shifts)),
                               constant_values=self.config.pad_value)
                mult_codes = np.pad(mult_codes, (0, self.config.max_peaks_nmr - len(mult_codes)),
                                  constant_values=0)
                intensities = np.pad(intensities, (0, self.config.max_peaks_nmr - len(intensities)),
                                   constant_values=self.config.pad_value)
            
            return np.column_stack([shifts, mult_codes, intensities])
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _process_ir_peaks(self, peaks: List[Tuple[float, float]]) -> np.ndarray:
        """
        Process IR peaks for model input.
        
        Args:
            peaks: List of (wavenumber, intensity) tuples
        
        Returns:
            Numpy array of shape (max_peaks_ir, 2)
        """
        try:
            # Filter noise
            filtered = [(p[0], p[1]) for p in peaks if p[1] > self.config.noise_threshold_ir]
            
            if not filtered:
                return np.full((self.config.max_peaks_ir, 2), self.config.pad_value, dtype=np.float32)
            
            # Sort by intensity (descending)
            filtered.sort(key=lambda x: x[1], reverse=True)
            
            # Normalize
            if self.config.normalize:
                max_intensity = max(p[1] for p in filtered)
                if max_intensity > 0:
                    filtered = [(p[0], p[1] / max_intensity) for p in filtered]
            
            # Pad/truncate
            values = np.array([p[0] for p in filtered[:self.config.max_peaks_ir]])
            intensities = np.array([p[1] for p in filtered[:self.config.max_peaks_ir]])
            
            if len(values) < self.config.max_peaks_ir:
                values = np.pad(values, (0, self.config.max_peaks_ir - len(values)),
                               constant_values=self.config.pad_value)
                intensities = np.pad(intensities, (0, self.config.max_peaks_ir - len(intensities)),
                                    constant_values=self.config.pad_value)
            
            return np.column_stack([values, intensities])
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # PREDICTION METHODS
    # =========================================================================
    
    def predict_single(self, 
                       ms_peaks: List[Tuple[float, float]],
                       nmr_peaks: List[Tuple[float, str, float]],
                       ir_peaks: List[Tuple[float, float]]) -> Union[float, Dict]:
        """
        Predict for a single set of spectra.
        
        Args:
            ms_peaks: List of (m/z, intensity) tuples
            nmr_peaks: List of (shift, multiplicity, integral) tuples
            ir_peaks: List of (wavenumber, intensity) tuples
        
        Returns:
            Prediction value (regression) or class probabilities (classification)
        """
        try:
            # Process individual spectra
            ms_processed = self._process_ms_peaks(ms_peaks)
            nmr_processed = self._process_nmr_peaks(nmr_peaks)
            ir_processed = self._process_ir_peaks(ir_peaks)
            
            # Add batch dimension
            ms_tensor = torch.tensor(ms_processed, dtype=torch.float32).unsqueeze(0).to(self.device)
            nmr_tensor = torch.tensor(nmr_processed, dtype=torch.float32).unsqueeze(0).to(self.device)
            ir_tensor = torch.tensor(ir_processed, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(ms_tensor, nmr_tensor, ir_tensor)
                prediction = output.squeeze().cpu().numpy()
            
            if self.target_type == 'regression':
                return float(prediction)
            else:
                # Return probabilities for each class
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                return {
                    'class': int(np.argmax(probs)),
                    'probabilities': probs.tolist(),
                    'confidence': float(np.max(probs))
                }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_batch(self,
                      ms_list: List[List[Tuple[float, float]]],
                      nmr_list: List[List[Tuple[float, str, float]]],
                      ir_list: List[List[Tuple[float, float]]]) -> np.ndarray:
        """
        Predict for multiple sets of spectra.
        
        Args:
            ms_list: List of MS peak lists
            nmr_list: List of NMR peak lists
            ir_list: List of IR peak lists
        
        Returns:
            Array of predictions
        """
        try:
            if not (len(ms_list) == len(nmr_list) == len(ir_list)):
                raise ValueError("All input lists must have the same length")
            
            all_predictions = []
            
            # Process in batches
            for i in range(0, len(ms_list), self.config.batch_size):
                batch_end = min(i + self.config.batch_size, len(ms_list))
                
                ms_batch = []
                nmr_batch = []
                ir_batch = []
                
                for j in range(i, batch_end):
                    ms_processed = self._process_ms_peaks(ms_list[j])
                    nmr_processed = self._process_nmr_peaks(nmr_list[j])
                    ir_processed = self._process_ir_peaks(ir_list[j])
                    
                    ms_batch.append(ms_processed)
                    nmr_batch.append(nmr_processed)
                    ir_batch.append(ir_processed)
                
                # Convert to tensors
                ms_tensor = torch.tensor(np.array(ms_batch), dtype=torch.float32).to(self.device)
                nmr_tensor = torch.tensor(np.array(nmr_batch), dtype=torch.float32).to(self.device)
                ir_tensor = torch.tensor(np.array(ir_batch), dtype=torch.float32).to(self.device)
                
                # Make predictions
                with torch.no_grad():
                    outputs = self.model(ms_tensor, nmr_tensor, ir_tensor)
                    predictions = outputs.squeeze().cpu().numpy()
                
                if self.target_type == 'regression':
                    all_predictions.extend(predictions.tolist())
                else:
                    # For classification, return class probabilities
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_predictions.extend(probs.tolist())
            
            return np.array(all_predictions)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_from_strings(self,
                             ms_peak_string: str,
                             nmr_peak_string: str,
                             ir_peak_string: str) -> Union[float, Dict]:
        """
        Predict from peak strings (CSV format).
        
        Args:
            ms_peak_string: MS peak string like "45.0:80, 29.0:100"
            nmr_peak_string: NMR peak string like "1.2:3:t, 3.6:2:q"
            ir_peak_string: IR peak string like "3300:100, 2900:80"
        
        Returns:
            Prediction value
        """
        try:
            # Parse strings
            ms_peaks = parse_peak_string(ms_peak_string)
            nmr_peaks = parse_nmr_peaks(nmr_peak_string)
            ir_peaks = parse_peak_string(ir_peak_string)
            
            return self.predict_single(ms_peaks, nmr_peaks, ir_peaks)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # BULK PREDICTION FROM CSV
    # =========================================================================
    
    def predict_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Predict from a CSV file with spectra data.
        
        Expected CSV columns:
            - compound_id: Optional
            - ms_peaks: MS peak string
            - nmr_peaks: NMR peak string
            - ir_peaks: IR peak string
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            DataFrame with predictions added
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Check required columns
            required_cols = ['ms_peaks', 'nmr_peaks', 'ir_peaks']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            predictions = []
            
            for idx, row in df.iterrows():
                try:
                    ms_peaks = parse_peak_string(row['ms_peaks'])
                    nmr_peaks = parse_nmr_peaks(row['nmr_peaks'])
                    ir_peaks = parse_peak_string(row['ir_peaks'])
                    
                    pred = self.predict_single(ms_peaks, nmr_peaks, ir_peaks)
                    
                    if self.target_type == 'regression':
                        predictions.append(float(pred))
                    else:
                        predictions.append(pred['class'])
                        
                except Exception as e:
                    logging.warning(f"Failed to predict for row {idx}: {e}")
                    predictions.append(None)
            
            df['prediction'] = predictions
            
            # Add confidence for classification
            if self.target_type == 'classification':
                df['confidence'] = [
                    self.predict_single(
                        parse_peak_string(row['ms_peaks']),
                        parse_nmr_peaks(row['nmr_peaks']),
                        parse_peak_string(row['ir_peaks'])
                    )['confidence'] if pd.notna(row['prediction']) else None
                    for idx, row in df.iterrows()
                ]
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # MODEL INFORMATION
    # =========================================================================
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'config': self.model_config,
            'target_type': self.target_type,
            'device': str(self.device),
            'model_path': self.config.model_path,
            'num_parameters': self.model.count_parameters() if self.model else 0
        }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line entry point for prediction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run prediction pipeline')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--ms', type=str, help='MS peak string')
    parser.add_argument('--nmr', type=str, help='NMR peak string')
    parser.add_argument('--ir', type=str, help='IR peak string')
    parser.add_argument('--csv', type=str, help='Path to CSV with spectra')
    parser.add_argument('--output', type=str, help='Output CSV path (with predictions)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = PredictPipeline(
            model_path=args.model,
            device=args.device
        )
        
        # Show model info
        info = predictor.get_model_info()
        print(f"\n📊 Model: {args.model}")
        print(f"  Target type: {info['target_type']}")
        print(f"  Hidden dim: {info['config']['hidden_dim']}")
        print(f"  Parameters: {info['num_parameters']:,}")
        
        if args.csv:
            # Predict from CSV
            print(f"\n📂 Predicting from CSV: {args.csv}")
            df = predictor.predict_from_csv(args.csv)
            
            output_path = args.output or args.csv.replace('.csv', '_predictions.csv')
            df.to_csv(output_path, index=False)
            print(f"✅ Predictions saved to: {output_path}")
            
        elif args.ms and args.nmr and args.ir:
            # Predict from strings
            print("\n🔬 Predicting from input strings...")
            prediction = predictor.predict_from_strings(args.ms, args.nmr, args.ir)
            
            print(f"\n📊 Prediction:")
            if isinstance(prediction, dict):
                print(f"  Class: {prediction['class']}")
                print(f"  Confidence: {prediction['confidence']:.4f}")
                print(f"  Probabilities: {prediction['probabilities']}")
            else:
                print(f"  Value: {prediction:.4f}")
        else:
            print("\n⚠️ Please provide either --csv or all three peak strings (--ms, --nmr, --ir)")
            print("Example:")
            print("  python -m src.pipeline.predict_pipeline --model model.pt --ms '45:80,29:100' --nmr '1.2:3:t' --ir '3300:100'")
            print("  python -m src.pipeline.predict_pipeline --model model.pt --csv data.csv --output predictions.csv")
        
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()