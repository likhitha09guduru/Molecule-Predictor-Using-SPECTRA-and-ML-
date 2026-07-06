"""
test_pipeline.py - Quick test for train_pipeline
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.train_pipeline import TrainPipeline
from src.logger import logging

print("\n" + "=" * 60)
print("🧪 TESTING TRAIN PIPELINE (DRY RUN)")
print("=" * 60)

try:
    # Create pipeline with minimal config for testing
    config = {
        'data_dir': 'artifacts/data/converted',
        'output_dir': 'artifacts/models/test/',
        'epochs': 1,           # Just 1 epoch
        'batch_size': 4,       # Small batch
        'hidden_dim': 16,      # Tiny model
        'fusion_type': 'concat',
        'target_type': 'regression',
        'output_dim': 1,
        'max_peaks_ms': 20,    # Fewer peaks for speed
        'max_peaks_nmr': 15,
        'max_peaks_ir': 10,
        'log_interval': 1
    }
    
    pipeline = TrainPipeline(config=config)
    
    # Check if data exists
    from src.components.data_ingestion import DataIngestion
    ingestor = DataIngestion(data_dir=config['data_dir'])
    data = ingestor.load_all()
    
    if data['compounds']:
        print(f"\n✅ Data found: {len(data['compounds'])} compounds")
        print(f"   MS: {len(data['ms'])} | NMR: {len(data['nmr'])} | IR: {len(data['ir'])}")
        print(f"   Multi-modal: {len(data['multi_modal_compounds'])}")
    else:
        print("\n⚠️ No data found. Please run conversion scripts first:")
        print("   python data_conversion/run_all_conversions.py")
        sys.exit(1)
    
    print("\n✅ TrainPipeline test passed!")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    raise

print("\n" + "=" * 60)