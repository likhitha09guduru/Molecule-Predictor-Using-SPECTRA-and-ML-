"""
run_all_conversions.py - Run this script ONCE to convert all raw files to CSV format.

1. Place raw files in:
   - artifacts/data/raw/ms_files/  (.mzml, .mgf, .msp)
   - artifacts/data/raw/nmr_files/ (.jdx, .dx)
   - artifacts/data/raw/ir_files/  (.jdx, .dx)
2. Run either:
   - python -m data_conversion.run_all_conversions   (recommended, works as a package)
   - python data_conversion/run_all_conversions.py   (also works, run directly)
3. Find CSV files in:
   - artifacts/data/converted/ms_spectra.csv
   - artifacts/data/converted/nmr_spectra.csv
   - artifacts/data/converted/ir_spectra.csv
"""

import sys
from pathlib import Path
from src.logger import logging
from src.exception import CustomException

# Support running both as a package (relative import) and as a standalone
# script (direct import) - these behave differently depending on how the
# file is invoked, so we try both.
try:
    from .convert_ms import convert_ms_files
    from .convert_nmr import convert_nmr_files
    from .convert_ir import convert_ir_files
except ImportError:
    from convert_ms import convert_ms_files
    from convert_nmr import convert_nmr_files
    from convert_ir import convert_ir_files


def run_all_conversions(
    ms_input_dir: str = "artifacts/data/raw/ms_files/",
    nmr_input_dir: str = "artifacts/data/raw/nmr_files/",
    ir_input_dir: str = "artifacts/data/raw/ir_files/",
    output_dir: str = "artifacts/data/converted/",
    chunk_size: int = 1000):
    """Convert all raw spectral files to CSV format."""
    try:
        logging.info("=" * 50)
        logging.info("STARTING BULK CONVERSION: RAW -> CSV")
        logging.info("=" * 50)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # 1. Convert MS files
        logging.info("\nConverting MS files...")
        results['ms'] = convert_ms_files(
            input_dir=ms_input_dir,
            output_csv=str(output_path / "ms_spectra.csv"),
            chunk_size=chunk_size
        )

        # 2. Convert NMR files
        logging.info("\nConverting NMR files...")
        results['nmr'] = convert_nmr_files(
            input_dir=nmr_input_dir,
            output_csv=str(output_path / "nmr_spectra.csv"),
            chunk_size=chunk_size
        )

        # 3. Convert IR files
        logging.info("\nConverting IR files...")
        results['ir'] = convert_ir_files(
            input_dir=ir_input_dir,
            output_csv=str(output_path / "ir_spectra.csv"),
            chunk_size=chunk_size
        )

        logging.info("\n" + "=" * 50)
        logging.info("ALL CONVERSIONS COMPLETE!")
        logging.info(f"CSV files saved to: {output_dir}")
        logging.info("=" * 50)

        # Report which modalities actually succeeded (a converter returns
        # None if its dependency is missing or no input files were found)
        for modality, result in results.items():
            if result is None:
                logging.warning(f"  {modality.upper()}: conversion did not complete - check earlier logs")
            else:
                logging.info(f"  {modality.upper()}: {result}")

        # Flag the SMILES situation clearly, since that's the real training label
        ms_result = results.get('ms')
        if ms_result and ms_result.get('with_smiles', 0) == 0:
            logging.warning(
                "\n  IMPORTANT: No MS spectra had a SMILES label after conversion. "
                "This means there is currently no real molecule-identity label "
                "available for training from the MS data."
            )

        # Show what was actually created on disk
        files = list(output_path.glob("*.csv"))
        if files:
            logging.info("\nCreated CSV files:")
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                logging.info(f"  - {f.name} ({size_mb:.2f} MB)")
        else:
            logging.warning("\nNo CSV files were created. Please check your input directories.")

        return results

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        run_all_conversions(
            ms_input_dir="artifacts/data/raw/ms_files/",
            nmr_input_dir="artifacts/data/raw/nmr_files/",
            ir_input_dir="artifacts/data/raw/ir_files/",
            output_dir="artifacts/data/converted/",
            chunk_size=1000
        )
    except Exception as e:
        raise CustomException(e, sys)