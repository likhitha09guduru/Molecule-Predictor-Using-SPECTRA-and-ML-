"""
convert_ms.py - Converts raw mass spectrometry files (.mzML, .mgf, .msp)
to a standardized CSV format with chunking to handle large files.
Uses the matchms library to read the raw files and extract relevant information.

NOTE: This file also extracts the 'smiles' field from spectrum metadata when
present. SMILES represents the actual molecular structure - if your raw MS
files include this field, it is your real training label (the thing the
model should predict), rather than a randomly generated placeholder.
"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.logger import logging
from src.exception import CustomException

try:
    from matchms.importing import load_from_mzml, load_from_mgf, load_from_msp
    HAS_MATCHMS = True
    logging.info("matchms loaded successfully")
except ImportError:
    HAS_MATCHMS = False
    logging.warning("matchms not installed. Install: pip install matchms")


def convert_ms_files(input_dir: str,
                     output_csv: str = "artifacts/data/converted/ms_spectra.csv",
                     chunk_size: int = 1000):
    """
    Convert MS files to CSV with chunking to handle large files.

    Args:
        input_dir: Folder containing your mass spectrometry files (.mzml, .mgf, .msp)
        output_csv: Path to save the CSV file
        chunk_size: Number of spectra to buffer before writing to CSV
    """
    try:
        if not HAS_MATCHMS:
            logging.error("matchms not installed. Cannot convert MS files.")
            return None

        input_path = Path(input_dir)
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove any existing output so re-running doesn't create duplicate rows
        if output_path.exists():
            output_path.unlink()

        # Collect all MS files
        ms_files = []
        for ext in ['*.mzml', '*.mgf', '*.msp']:
            ms_files.extend(input_path.rglob(ext))

        if not ms_files:
            logging.error(f"No MS files found in {input_dir}")
            return None

        logging.info(f"Found {len(ms_files)} MS files to convert")

        first_write = True
        num_converted = 0
        num_with_smiles = 0
        num_failed = 0

        for file_path in tqdm(ms_files, desc="Converting MS files"):
            try:
                ext = file_path.suffix.lower()
                if ext == '.mzml':
                    spectra_generator = load_from_mzml(str(file_path))
                elif ext == '.mgf':
                    spectra_generator = load_from_mgf(str(file_path))
                elif ext == '.msp':
                    spectra_generator = load_from_msp(str(file_path))
                else:
                    logging.warning(f"Skipping unsupported file type: {file_path.name}")
                    continue

                chunk = []

                for spec in spectra_generator:
                    try:
                        compound_id = spec.metadata.get(
                            'compound_id',
                            spec.metadata.get('name', spec.metadata.get('title', file_path.stem))
                        )

                        smiles = spec.metadata.get('smiles', spec.metadata.get('SMILES', ''))
                        if smiles:
                            num_with_smiles += 1

                        precursor_mz = spec.metadata.get('precursor_mz', None)
                        charge = spec.metadata.get('charge', 1)

                        if spec.peaks.mz is None or len(spec.peaks.mz) == 0:
                            logging.warning(f"  {compound_id}: no peaks found, skipping")
                            continue

                        peaks_str = ", ".join(
                            f"{mz:.4f}:{intensity:.1f}"
                            for mz, intensity in zip(spec.peaks.mz, spec.peaks.intensities)
                        )

                        chunk.append({
                            'compound_id': str(compound_id),
                            'smiles': str(smiles),
                            'precursor_mz': precursor_mz,
                            'charge': charge,
                            'peaks': peaks_str,
                            'source_file': str(file_path)
                        })
                        num_converted += 1

                    except Exception as e:
                        logging.warning(f"  Failed to process one spectrum in {file_path.name}: {str(e)}")
                        continue

                    # Flush buffer once it reaches chunk_size
                    if len(chunk) >= chunk_size:
                        _append_to_csv(chunk, output_path, first_write)
                        first_write = False
                        logging.info(f"Written {chunk_size} spectra to CSV")
                        chunk = []

                # Flush whatever's left from this file
                if chunk:
                    _append_to_csv(chunk, output_path, first_write)
                    first_write = False
                    logging.info(f"Written final {len(chunk)} spectra from {file_path.name}")

            except Exception as e:
                logging.error(f"Failed to convert {file_path}: {str(e)}")
                num_failed += 1
                continue

        logging.info("=" * 50)
        logging.info("MS conversion complete!")
        logging.info(f"  Spectra converted: {num_converted}")
        logging.info(f"  Spectra with SMILES label: {num_with_smiles}")
        logging.info(f"  Files failed: {num_failed}")
        logging.info(f"  Data saved to {output_path}")
        logging.info("=" * 50)

        if num_converted > 0 and num_with_smiles == 0:
            logging.warning(
                "No spectra had a SMILES label. If you need real molecule "
                "identities for training, check whether your raw files "
                "actually contain this metadata field."
            )

        return {
            'converted': num_converted,
            'with_smiles': num_with_smiles,
            'failed': num_failed,
            'output_path': str(output_path)
        }

    except Exception as e:
        raise CustomException(e, sys)


def _append_to_csv(chunk: list, output_path: Path, first_write: bool):
    """Append a chunk of records to the CSV file."""
    df = pd.DataFrame(chunk)
    df.to_csv(output_path, mode='a', index=False, header=first_write)


if __name__ == "__main__":
    try:
        convert_ms_files(
            input_dir="artifacts/data/raw/ms_files/",
            output_csv="artifacts/data/converted/ms_spectra.csv",
            chunk_size=1000
        )
    except Exception as e:
        raise CustomException(e, sys)