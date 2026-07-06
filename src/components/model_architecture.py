"""
model_architecture.py - Multi-modal GNN for spectral data.

PURPOSE: Defines a deep learning model that processes MS, NMR, and IR spectra
simultaneously using separate encoders and a fusion mechanism.

ARCHITECTURE:
1. MS Encoder: 1D CNN or Transformer for mass spec peaks
2. NMR Encoder: Transformer for NMR peak sequences
3. IR Encoder: 1D CNN for IR spectra
4. Fusion Layer: Cross-attention or concatenation
5. Prediction Head: Regression or classification

INPUT: Preprocessed tensors from DataTransformation
OUTPUT: Predictions (regression value or class probabilities)
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from src.logger import logging
from src.exception import CustomException


# ============================================================================
# ENCODER MODULES
# ============================================================================

class MSEncoder(nn.Module):
    """
    Encoder for Mass Spectrometry data.
    
    Input: (batch_size, max_peaks_ms, 2) where last dim is [m/z, intensity]
    Output: (batch_size, hidden_dim)
    """
    
    def __init__(self, 
                 input_dim: int = 2,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize MS Encoder.
        
        Args:
            input_dim: Input feature dimension (2 for m/z and intensity)
            hidden_dim: Hidden dimension size
            num_layers: Number of CNN/Transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        try:
            self.hidden_dim = hidden_dim
            
            # Option 1: 1D CNN (for sequential peaks)
            self.conv_layers = nn.ModuleList()
            in_channels = input_dim
            
            for i in range(num_layers):
                out_channels = hidden_dim if i == num_layers - 1 else hidden_dim // 2
                self.conv_layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
                )
                self.conv_layers.append(nn.BatchNorm1d(out_channels))
                self.conv_layers.append(nn.ReLU())
                self.conv_layers.append(nn.Dropout(dropout))
                in_channels = out_channels
            
            # Global pooling
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            
            # Final projection
            self.projection = nn.Linear(hidden_dim, hidden_dim)
            
            logging.info(f"MSEncoder initialized: input_dim={input_dim}, hidden_dim={hidden_dim}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, max_peaks_ms, 2)
        
        Returns:
            (batch_size, hidden_dim)
        """
        try:
            # Transpose for Conv1d: (batch, features, sequence)
            x = x.permute(0, 2, 1)  # (batch, 2, max_peaks)
            
            # Apply conv layers
            for layer in self.conv_layers:
                x = layer(x)
            
            # Global pooling
            x = self.global_pool(x)  # (batch, hidden_dim, 1)
            x = x.squeeze(-1)  # (batch, hidden_dim)
            
            # Final projection
            x = self.projection(x)
            
            return x
            
        except Exception as e:
            raise CustomException(e, sys)


class NMREncoder(nn.Module):
    """
    Encoder for NMR data with multiplicity encoding.
    
    Input: (batch_size, max_peaks_nmr, 3) where last dim is [shift, mult_code, intensity]
    Output: (batch_size, hidden_dim)
    """
    
    def __init__(self,
                 input_dim: int = 3,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize NMR Encoder.
        
        Args:
            input_dim: Input feature dimension (3 for shift, mult, intensity)
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        try:
            self.hidden_dim = hidden_dim
            
            # Input projection
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Global pooling
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            
            logging.info(f"NMREncoder initialized: input_dim={input_dim}, hidden_dim={hidden_dim}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, max_peaks_nmr, 3)
        
        Returns:
            (batch_size, hidden_dim)
        """
        try:
            # Input projection
            x = self.input_projection(x)  # (batch, max_peaks, hidden_dim)
            
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Transformer
            x = self.transformer(x)  # (batch, max_peaks, hidden_dim)
            
            # Global pooling
            x = x.mean(dim=1)  # (batch, hidden_dim)
            
            return x
            
        except Exception as e:
            raise CustomException(e, sys)


class IREncoder(nn.Module):
    """
    Encoder for IR spectroscopy data.
    
    Input: (batch_size, max_peaks_ir, 2) where last dim is [wavenumber, intensity]
    Output: (batch_size, hidden_dim)
    """
    
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize IR Encoder.
        
        Args:
            input_dim: Input feature dimension (2 for wavenumber and intensity)
            hidden_dim: Hidden dimension size
            num_layers: Number of CNN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        try:
            self.hidden_dim = hidden_dim
            
            # 1D CNN for IR
            self.conv_layers = nn.ModuleList()
            in_channels = input_dim
            
            for i in range(num_layers):
                out_channels = hidden_dim if i == num_layers - 1 else hidden_dim // 2
                self.conv_layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
                )
                self.conv_layers.append(nn.BatchNorm1d(out_channels))
                self.conv_layers.append(nn.ReLU())
                self.conv_layers.append(nn.Dropout(dropout))
                in_channels = out_channels
            
            # Global pooling
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            
            # Final projection
            self.projection = nn.Linear(hidden_dim, hidden_dim)
            
            logging.info(f"IREncoder initialized: input_dim={input_dim}, hidden_dim={hidden_dim}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, max_peaks_ir, 2)
        
        Returns:
            (batch_size, hidden_dim)
        """
        try:
            # Transpose for Conv1d
            x = x.permute(0, 2, 1)  # (batch, 2, max_peaks)
            
            # Apply conv layers
            for layer in self.conv_layers:
                x = layer(x)
            
            # Global pooling
            x = self.global_pool(x)  # (batch, hidden_dim, 1)
            x = x.squeeze(-1)  # (batch, hidden_dim)
            
            # Final projection
            x = self.projection(x)
            
            return x
            
        except Exception as e:
            raise CustomException(e, sys)


# ============================================================================
# POSITIONAL ENCODING (For NMR Transformer)
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-based encoders.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


# ============================================================================
# FUSION MODULE
# ============================================================================

class CrossModalFusion(nn.Module):
    """
    Fuses features from MS, NMR, and IR encoders.
    
    Options:
    - 'concat': Simple concatenation
    - 'attention': Cross-attention between modalities
    - 'gated': Gated fusion with learned weights
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 fusion_type: str = 'concat',
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize fusion module.
        
        Args:
            hidden_dim: Hidden dimension size
            fusion_type: 'concat', 'attention', or 'gated'
            num_heads: Number of attention heads (for 'attention')
            dropout: Dropout rate
        """
        super().__init__()
        
        try:
            self.fusion_type = fusion_type
            self.hidden_dim = hidden_dim
            
            if fusion_type == 'concat':
                self.fusion_dim = hidden_dim * 3
                self.projection = nn.Linear(self.fusion_dim, hidden_dim)
                
            elif fusion_type == 'attention':
                # Cross-attention layers
                self.ms_to_nmr = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                self.ms_to_ir = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                self.nmr_to_ir = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                
                # Final projection
                self.projection = nn.Linear(hidden_dim * 3, hidden_dim)
                
            elif fusion_type == 'gated':
                # Learnable gates for each modality
                self.gate_ms = nn.Linear(hidden_dim, 1)
                self.gate_nmr = nn.Linear(hidden_dim, 1)
                self.gate_ir = nn.Linear(hidden_dim, 1)
                self.projection = nn.Linear(hidden_dim, hidden_dim)
                
            else:
                raise ValueError(f"Unknown fusion_type: {fusion_type}")
            
            self.dropout = nn.Dropout(dropout)
            
            logging.info(f"CrossModalFusion initialized: fusion_type={fusion_type}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def forward(self, ms_feat: torch.Tensor, nmr_feat: torch.Tensor, ir_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            ms_feat: (batch_size, hidden_dim)
            nmr_feat: (batch_size, hidden_dim)
            ir_feat: (batch_size, hidden_dim)
        
        Returns:
            (batch_size, hidden_dim)
        """
        try:
            if self.fusion_type == 'concat':
                # Simple concatenation
                fused = torch.cat([ms_feat, nmr_feat, ir_feat], dim=-1)
                fused = self.projection(fused)
                
            elif self.fusion_type == 'attention':
                # Add sequence dimension for attention
                ms = ms_feat.unsqueeze(1)  # (batch, 1, hidden)
                nmr = nmr_feat.unsqueeze(1)  # (batch, 1, hidden)
                ir = ir_feat.unsqueeze(1)  # (batch, 1, hidden)
                
                # Cross-attention
                ms_to_nmr, _ = self.ms_to_nmr(ms, nmr, nmr)
                ms_to_ir, _ = self.ms_to_ir(ms, ir, ir)
                nmr_to_ir, _ = self.nmr_to_ir(nmr, ir, ir)
                
                # Combine
                fused = torch.cat([
                    ms_to_nmr.squeeze(1),
                    ms_to_ir.squeeze(1),
                    nmr_to_ir.squeeze(1)
                ], dim=-1)
                fused = self.projection(fused)
                
            elif self.fusion_type == 'gated':
                # Learned gates
                gate_ms = torch.sigmoid(self.gate_ms(ms_feat))
                gate_nmr = torch.sigmoid(self.gate_nmr(nmr_feat))
                gate_ir = torch.sigmoid(self.gate_ir(ir_feat))
                
                # Weighted sum
                fused = (gate_ms * ms_feat + gate_nmr * nmr_feat + gate_ir * ir_feat) / 3
                fused = self.projection(fused)
            
            return self.dropout(fused)
            
        except Exception as e:
            raise CustomException(e, sys)


# ============================================================================
# MAIN MODEL
# ============================================================================

class MultiModalSpectraModel(nn.Module):
    """
    Complete multi-modal model for spectral data analysis.
    
    Architecture:
    1. MS Encoder (CNN)
    2. NMR Encoder (Transformer)
    3. IR Encoder (CNN)
    4. Cross-modal Fusion
    5. Prediction Head (Regression or Classification)
    """
    
    def __init__(self,
                 ms_input_dim: int = 2,
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
        Initialize the multi-modal model.
        
        Args:
            ms_input_dim: MS input feature dimension
            nmr_input_dim: NMR input feature dimension
            ir_input_dim: IR input feature dimension
            hidden_dim: Hidden dimension size
            output_dim: Output dimension (1 for regression, num_classes for classification)
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            dropout: Dropout rate
            fusion_type: 'concat', 'attention', or 'gated'
            target_type: 'regression' or 'classification'
        """
        super().__init__()
        
        try:
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.target_type = target_type
            
            # Encoders
            self.ms_encoder = MSEncoder(
                input_dim=ms_input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
            
            self.nmr_encoder = NMREncoder(
                input_dim=nmr_input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
            
            self.ir_encoder = IREncoder(
                input_dim=ir_input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
            
            # Fusion
            self.fusion = CrossModalFusion(
                hidden_dim=hidden_dim,
                fusion_type=fusion_type,
                num_heads=num_heads,
                dropout=dropout
            )
            
            # Prediction head
            if target_type == 'regression':
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, output_dim)
                )
            else:
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, output_dim)
                )
            
            logging.info("=" * 60)
            logging.info("MULTI-MODAL SPECTRA MODEL INITIALIZED")
            logging.info("=" * 60)
            logging.info(f"  MS encoder: CNN (input_dim={ms_input_dim})")
            logging.info(f"  NMR encoder: Transformer (input_dim={nmr_input_dim})")
            logging.info(f"  IR encoder: CNN (input_dim={ir_input_dim})")
            logging.info(f"  Fusion type: {fusion_type}")
            logging.info(f"  Hidden dim: {hidden_dim}")
            logging.info(f"  Output dim: {output_dim}")
            logging.info(f"  Target type: {target_type}")
            logging.info("=" * 60)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def forward(self, 
                ms: torch.Tensor,
                nmr: torch.Tensor,
                ir: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            ms: (batch_size, max_peaks_ms, 2)
            nmr: (batch_size, max_peaks_nmr, 3)
            ir: (batch_size, max_peaks_ir, 2)
        
        Returns:
            (batch_size, output_dim)
        """
        try:
            # Encode each modality
            ms_feat = self.ms_encoder(ms)    # (batch, hidden)
            nmr_feat = self.nmr_encoder(nmr)  # (batch, hidden)
            ir_feat = self.ir_encoder(ir)    # (batch, hidden)
            
            # Fuse features
            fused = self.fusion(ms_feat, nmr_feat, ir_feat)  # (batch, hidden)
            
            # Prediction
            output = self.head(fused)  # (batch, output_dim)
            
            return output
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_embeddings(self,
                       ms: torch.Tensor,
                       nmr: torch.Tensor,
                       ir: torch.Tensor) -> torch.Tensor:
        """
        Get fused embeddings (for visualization or downstream tasks).
        
        Args:
            ms: (batch_size, max_peaks_ms, 2)
            nmr: (batch_size, max_peaks_nmr, 3)
            ir: (batch_size, max_peaks_ir, 2)
        
        Returns:
            (batch_size, hidden_dim)
        """
        try:
            ms_feat = self.ms_encoder(ms)
            nmr_feat = self.nmr_encoder(nmr)
            ir_feat = self.ir_encoder(ir)
            return self.fusion(ms_feat, nmr_feat, ir_feat)
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_modality_embeddings(self,
                                ms: torch.Tensor,
                                nmr: torch.Tensor,
                                ir: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get individual modality embeddings (for analysis).
        
        Returns:
            Dictionary with 'ms', 'nmr', 'ir' embeddings
        """
        try:
            return {
                'ms': self.ms_encoder(ms),
                'nmr': self.nmr_encoder(nmr),
                'ir': self.ir_encoder(ir)
            }
        except Exception as e:
            raise CustomException(e, sys)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_config: Dict) -> MultiModalSpectraModel:
    """
    Create a model from configuration dictionary.
    
    Args:
        model_config: Dictionary with model parameters
        
    Returns:
        MultiModalSpectraModel instance
    """
    try:
        default_config = {
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
        
        # Merge with provided config
        config = {**default_config, **model_config}
        
        model = MultiModalSpectraModel(**config)
        return model
        
    except Exception as e:
        raise CustomException(e, sys)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING MODEL ARCHITECTURE")
    print("=" * 60)
    
    try:
        # Create sample tensors
        batch_size = 4
        max_peaks_ms = 200
        max_peaks_nmr = 150
        max_peaks_ir = 100
        
        ms = torch.randn(batch_size, max_peaks_ms, 2)
        nmr = torch.randn(batch_size, max_peaks_nmr, 3)
        ir = torch.randn(batch_size, max_peaks_ir, 2)
        
        # Create model
        model = MultiModalSpectraModel(
            hidden_dim=128,
            output_dim=1,
            target_type='regression',
            fusion_type='concat'
        )
        
        print("\n📊 MODEL SUMMARY:")
        print(f"  Total parameters: {model.count_parameters():,}")
        
        # Forward pass
        output = model(ms, nmr, ir)
        print(f"  Output shape: {output.shape}")
        
        # Get embeddings
        embeddings = model.get_embeddings(ms, nmr, ir)
        print(f"  Embedding shape: {embeddings.shape}")
        
        # Test different fusion types
        for fusion_type in ['concat', 'attention', 'gated']:
            model = MultiModalSpectraModel(
                hidden_dim=64,
                output_dim=1,
                fusion_type=fusion_type
            )
            output = model(ms, nmr, ir)
            print(f"  {fusion_type} output shape: {output.shape}")
        
        print("\n✅ Model architecture test complete!")
        
    except Exception as e:
        raise CustomException(e, sys)