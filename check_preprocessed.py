# quick_check_preprocessed.py
import pandas as pd

print("\n📊 PREPROCESSED DATA SUMMARY\n" + "="*60)

# Check expression data
expr = pd.read_csv('data/processed/combined/geo_expression_combined.csv', index_col=0)
print(f"\n✅ Combined Expression Matrix:")
print(f"   Genes: {expr.shape[0]:,}")
print(f"   Samples: {expr.shape[1]:,}")

# Check labels
labels = pd.read_csv('data/processed/combined/geo_response_labels.csv', index_col=0)
print(f"\n✅ Response Labels:")
print(f"   Total: {len(labels)}")
print(f"   pCR: {(labels['response']==1).sum()}")
print(f"   RD: {(labels['response']==0).sum()}")

# Check TCGA
tcga = pd.read_csv('data/processed/combined/tcga_clinical_features.csv')
print(f"\n✅ TCGA Clinical Features:")
print(f"   Patients: {len(tcga):,}")
print(f"   Features: {tcga.shape[1]}")

print("\n" + "="*60 + "\n")