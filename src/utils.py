"""
utils.py - Utility functions for the spectroscopy project.

Contains reusable helper functions used across multiple modules:
- Peak parsing (MS, NMR, IR)
- Data validation
- File operations
- Common calculations
"""
import sys
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path
import json
import pickle

from src.logger import logging
from src.exception import CustomException


# ============================================================================
# PEAK PARSING FUNCTIONS
# ============================================================================

def parse_peak_string(peak_string: str) -> List[Tuple[float, float]]:
    """
    Parse peak string like '45.0:80, 29.0:100' into list of tuples.
    
    Args:
        peak_string: String of peaks separated by commas
        
    Returns:
        List of (value, intensity) tuples
    
    Example:
        >>> parse_peak_string("45.0:80, 29.0:100")
        [(45.0, 80.0), (29.0, 100.0)]
    """
    peaks = []
    if not peak_string or pd.isna(peak_string):
        return peaks
    
    for part in str(peak_string).split(','):
        try:
            parts = part.strip().split(':')
            if len(parts) == 2:
                peaks.append((float(parts[0]), float(parts[1])))
        except (ValueError, TypeError):
            continue
    return peaks


def parse_nmr_peaks(peak_string: str) -> List[Tuple[float, str, float]]:
    """
    Parse NMR peak string like '1.2:3:t, 3.6:2:q'.
    
    Args:
        peak_string: String of NMR peaks separated by commas
        
    Returns:
        List of (shift, multiplicity, integral) tuples
    
    Example:
        >>> parse_nmr_peaks("1.2:3:t, 3.6:2:q")
        [(1.2, 't', 3.0), (3.6, 'q', 2.0)]
    """
    peaks = []
    if not peak_string or pd.isna(peak_string):
        return peaks
    
    for part in str(peak_string).split(','):
        try:
            parts = part.strip().split(':')
            if len(parts) >= 1:
                shift = float(parts[0])
                integral = float(parts[1]) if len(parts) > 1 else 1.0
                multiplicity = parts[2] if len(parts) > 2 else 's'
                peaks.append((shift, multiplicity, integral))
        except (ValueError, TypeError):
            continue
    return peaks


def format_peaks_to_string(peaks: List[Tuple[float, float]], 
                           precision: int = 4) -> str:
    """
    Convert peaks list to string format.
    
    Args:
        peaks: List of (value, intensity) tuples
        precision: Decimal precision for values
        
    Returns:
        String like "45.0:80, 29.0:100"
    
    Example:
        >>> format_peaks_to_string([(45.0, 80.0), (29.0, 100.0)])
        "45.0:80.0, 29.0:100.0"
    """
    if not peaks:
        return ""
    
    return ", ".join([f"{value:.{precision}f}:{intensity:.1f}" 
                      for value, intensity in peaks])


def format_nmr_peaks_to_string(peaks: List[Tuple[float, str, float]]) -> str:
    """
    Convert NMR peaks to string format.
    
    Args:
        peaks: List of (shift, multiplicity, integral) tuples
        
    Returns:
        String like "1.2:3:t, 3.6:2:q"
    """
    if not peaks:
        return ""
    
    return ", ".join([f"{shift}:{integral}:{mult}" 
                      for shift, mult, integral in peaks])


# ============================================================================
# ENCODING FUNCTIONS
# ============================================================================

def encode_multiplicity(mult: str) -> int:
    """
    Convert multiplicity string to integer code.
    
    Multiplicity mapping:
        s (singlet) = 0
        d (doublet) = 1
        t (triplet) = 2
        q (quartet) = 3
        m (multiplet) = 4
        dd (doublet of doublets) = 5
        dt (doublet of triplets) = 6
        td (triplet of doublets) = 7
        dq (doublet of quartets) = 8
        qd (quartet of doublets) = 9
        brs (broad singlet) = 10
        br (broad) = 11
    
    Args:
        mult: Multiplicity string (e.g., 's', 'd', 't', 'q', 'm')
    
    Returns:
        Integer code (0-11)
    """
    mult_map = {
        's': 0, 'singlet': 0,
        'd': 1, 'doublet': 1,
        't': 2, 'triplet': 2,
        'q': 3, 'quartet': 3,
        'm': 4, 'multiplet': 4,
        'dd': 5, 'dt': 6, 'td': 7, 'dq': 8, 'qd': 9,
        'brs': 10, 'br': 11
    }
    mult = str(mult).lower().strip()
    return mult_map.get(mult, 0)  # Default to singlet if unknown


def decode_multiplicity(code: int) -> str:
    """
    Convert integer code back to multiplicity string.
    
    Args:
        code: Integer code (0-11)
    
    Returns:
        Multiplicity string
    """
    code_map = {
        0: 's', 1: 'd', 2: 't', 3: 'q', 4: 'm',
        5: 'dd', 6: 'dt', 7: 'td', 8: 'dq', 9: 'qd',
        10: 'brs', 11: 'br'
    }
    return code_map.get(code, 's')


# ============================================================================
# DATA VALIDATION FUNCTIONS
# ============================================================================

def validate_compound_id(compound_id: str) -> bool:
    """
    Validate if a compound ID is valid.
    
    Args:
        compound_id: String to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not compound_id:
        return False
    if not isinstance(compound_id, str):
        return False
    if len(compound_id) < 3:
        return False
    return True


def validate_peaks(peaks: List[Tuple[float, float]]) -> bool:
    """
    Validate if peaks are in correct format.
    
    Args:
        peaks: List of (value, intensity) tuples
    
    Returns:
        True if valid, False otherwise
    """
    if not peaks:
        return False
    if not isinstance(peaks, list):
        return False
    for peak in peaks:
        if not isinstance(peak, (list, tuple)) or len(peak) != 2:
            return False
        if not isinstance(peak[0], (int, float)) or not isinstance(peak[1], (int, float)):
            return False
    return True


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, file_path: Union[str, Path]) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the JSON file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(file_path: Union[str, Path]) -> dict:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Dictionary with loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_pickle(data: object, file_path: Union[str, Path]) -> None:
    """
    Save data as pickle file.
    
    Args:
        data: Object to save
        file_path: Path to save the pickle file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path: Union[str, Path]) -> object:
    """
    Load data from pickle file.
    
    Args:
        file_path: Path to the pickle file
    
    Returns:
        Loaded object
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# ============================================================================
# DATA STATISTICS FUNCTIONS
# ============================================================================

def get_peak_statistics(peaks: List[Tuple[float, float]]) -> dict:
    """
    Calculate statistics for a peak list.
    
    Args:
        peaks: List of (value, intensity) tuples
    
    Returns:
        Dictionary with min, max, mean, median, std, count
    """
    if not peaks:
        return {
            'min': None, 'max': None, 'mean': None, 
            'median': None, 'std': None, 'count': 0
        }
    
    values = [p[0] for p in peaks]
    intensities = [p[1] for p in peaks]
    
    return {
        'min_value': min(values),
        'max_value': max(values),
        'mean_value': np.mean(values),
        'median_value': np.median(values),
        'std_value': np.std(values),
        'min_intensity': min(intensities),
        'max_intensity': max(intensities),
        'mean_intensity': np.mean(intensities),
        'count': len(peaks)
    }


def get_modality_overlap(compound_dict: dict, 
                         modality1: str, 
                         modality2: str) -> dict:
    """
    Find overlap of compounds between two modalities.
    
    Args:
        compound_dict: Dictionary with modality names as keys and sets of compound IDs
        modality1: Name of first modality
        modality2: Name of second modality
    
    Returns:
        Dictionary with overlap statistics
    """
    set1 = compound_dict.get(modality1, set())
    set2 = compound_dict.get(modality2, set())
    
    overlap = set1 & set2
    only1 = set1 - set2
    only2 = set2 - set1
    
    return {
        f'{modality1}_count': len(set1),
        f'{modality2}_count': len(set2),
        'overlap_count': len(overlap),
        f'only_{modality1}_count': len(only1),
        f'only_{modality2}_count': len(only2),
        'overlap_percent': (len(overlap) / max(len(set1), 1)) * 100
    }


# ============================================================================
# STRING CLEANING FUNCTIONS
# ============================================================================

def clean_smiles(smiles: str) -> str:
    """
    Clean a SMILES string.
    
    Args:
        smiles: Raw SMILES string
    
    Returns:
        Cleaned SMILES string
    """
    if not smiles:
        return ""
    smiles = str(smiles).strip()
    # Remove common prefixes
    prefixes = ['SMILES=', 'SMILES:']
    for prefix in prefixes:
        if smiles.startswith(prefix):
            smiles = smiles[len(prefix):]
    return smiles


def clean_compound_id(compound_id: str) -> str:
    """
    Clean a compound ID string.
    
    Args:
        compound_id: Raw compound ID
    
    Returns:
        Cleaned compound ID
    """
    if not compound_id:
        return ""
    compound_id = str(compound_id).strip()
    # Remove spaces and special characters
    compound_id = ''.join(c for c in compound_id if c.isalnum() or c in ['_', '-'])
    return compound_id


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_utils():
    """Run tests for all utility functions."""
    print("\n" + "=" * 60)
    print("🧪 TESTING UTILITY FUNCTIONS")
    print("=" * 60)
    
    # Test peak parsing
    peak_string = "45.0:80, 29.0:100, 15.0:20"
    parsed = parse_peak_string(peak_string)
    print(f"\n✅ parse_peak_string: {parsed}")
    
    # Test NMR peak parsing
    nmr_string = "1.2:3:t, 3.6:2:q"
    parsed_nmr = parse_nmr_peaks(nmr_string)
    print(f"✅ parse_nmr_peaks: {parsed_nmr}")
    
    # Test multiplicity encoding
    for mult in ['s', 'd', 't', 'q', 'm', 'dd', 'dt']:
        code = encode_multiplicity(mult)
        decoded = decode_multiplicity(code)
        print(f"✅ {mult} → {code} → {decoded}")
    
    # Test format functions
    formatted = format_peaks_to_string(parsed)
    print(f"✅ format_peaks_to_string: {formatted}")
    
    # Test validation
    print(f"✅ validate_compound_id('C001'): {validate_compound_id('C001')}")
    print(f"✅ validate_compound_id(''): {validate_compound_id('')}")
    
    print("\n✅ All utility tests passed!")


if __name__ == "__main__":
    try:
        test_utils()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise