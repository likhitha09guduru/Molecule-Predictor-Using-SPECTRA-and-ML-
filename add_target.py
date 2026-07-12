"""
add_target.py - Add a target column (Molecular Weight) to your CSV file.
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path

# Define the path to your CSV file
csv_path = Path("artifacts/data/converted/ms_spectra.csv")

# 1. Load the CSV file
print(f"📂 Loading data from: {csv_path}")
df = pd.read_csv(csv_path)

# 2. Check if the 'target' column already exists
if 'target' in df.columns:
    print("⚠️ A 'target' column already exists. It will be overwritten.")

# 3. Define a function to calculate molecular weight from SMILES
def get_molecular_weight(smiles):
    """Calculate exact molecular weight from a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Using ExactMolWt for precision. You can use Descriptors.MolWt for average mass.
            return Descriptors.ExactMolWt(mol)
        else:
            return None
    except:
        return None

# 4. Apply the function to the 'smiles' column and create the 'target' column
print("🧪 Calculating molecular weights...")
df['target'] = df['smiles'].apply(get_molecular_weight)

# 5. Check for any failures (compounds where calculation returned None)
failed_count = df['target'].isna().sum()
if failed_count > 0:
    print(f"⚠️ Warning: Could not calculate MW for {failed_count} compounds. Please check their SMILES strings.")
    # Print the first few problematic SMILES for debugging
    print(df[df['target'].isna()]['smiles'].head())

# 6. Save the updated DataFrame back to the CSV file
df.to_csv(csv_path, index=False)
print(f"✅ Success! Added a 'target' column (Molecular Weight) to {csv_path}")
print(f"   Number of compounds processed: {len(df)}")
print(f"   Target values range: {df['target'].min():.2f} - {df['target'].max():.2f} Da")