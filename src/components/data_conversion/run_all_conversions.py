"""Run this script ONCE to convert all raw files to CSV format.
1. Place raw files in:
   - artifacts/data/raw/ms_files/  (.mzml, .mgf, .msp)
   - artifacts/data/raw/nmr_files/ (.jdx, .dx)
   - artifacts/data/raw/ir_files/  (.jdx, .dx)
2. Run: python data_conversion/run_all_conversions.py
3. Find CSV files in:
   - artifacts/data/converted/ms_spectra.csv
   - artifacts/data/converted/nmr_spectra.csv
   - artifacts/data/converted/ir_spectra.csv"""

import sys
from pathlib import Path
from src.logger import logging
from src.exception import CustomException
from convert_ms import convert_ms_files
from convert_nmr import convert_nmr_files
from convert_ir import convert_ir_files

def run_all_conversions(
    ms_input_dir: str = "artifacts/data/raw/ms_files/",
    nmr_input_dir: str = "artifacts/data/raw/nmr_files/",
    ir_input_dir: str = "artifacts/data/raw/ir_files/",
    output_dir: str = "artifacts/data/converted/",
    chunk_size: int = 1000):
    #Convert all raw spectral files to CSV format
    try:
        logging.info("=" * 10)
        logging.info("STARTING BULK CONVERSION: RAW → CSV")
        logging.info("=" * 10)
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Convert MS files
        logging.info("\n Converting MS files...")
        convert_ms_files(
            input_dir=ms_input_dir,
            output_csv=f"{output_dir}/ms_spectra.csv",
            chunk_size=chunk_size
        )

        # 2. Convert NMR files
        logging.info("\nConverting NMR files...")
        convert_nmr_files(
            input_dir=nmr_input_dir,
            output_csv=f"{output_dir}/nmr_spectra.csv",
            chunk_size=chunk_size
        )

        # 3. Convert IR files
        logging.info("\n Converting IR files...")
        convert_ir_files(
            input_dir=ir_input_dir,
            output_csv=f"{output_dir}/ir_spectra.csv",
            chunk_size=chunk_size
        )

        logging.info("\n" + "=" * 10)
        logging.info(" ALL CONVERSIONS COMPLETE!")
        logging.info(f"CSV files saved to: {output_dir}")
        logging.info("=" * 10)

        # Show what was created 
        files = list(output_path.glob("*.csv"))
        if files:
            logging.info("\nCreated CSV files:")
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                logging.info(f"  - {f.name} ({size_mb:.2f} MB)")
        else:
            logging.warning("\n No CSV files were created. Please check your input directories.")

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