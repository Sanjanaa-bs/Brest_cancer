# File: verify_downloads.py

import os
import pandas as pd

def verify_data():
    """Verify all data has been downloaded correctly"""
    
    print("\n🔍 VERIFYING DOWNLOADED DATA")
    print("=" * 60)
    
    # Check GEO datasets
    geo_datasets = ['GSE25066', 'GSE20271', 'GSE20194', 'GSE32646']
    
    print("\n📊 GEO Datasets:")
    for gse in geo_datasets:
        expr_path = f"data/processed/{gse}/{gse}_expression.csv"
        clin_path = f"data/processed/{gse}/{gse}_clinical.csv"
        
        if os.path.exists(expr_path) and os.path.exists(clin_path):
            expr_df = pd.read_csv(expr_path, index_col=0, nrows=5)
            clin_df = pd.read_csv(clin_path, index_col=0)
            
            print(f"   ✅ {gse}:")
            print(f"      Expression: {expr_df.shape[1]} samples")
            print(f"      Clinical: {clin_df.shape[0]} samples")
        else:
            print(f"   ❌ {gse}: Missing files")
    
    # Check TCGA data
    print("\n📊 TCGA Data:")
    tcga_master = "data/raw/TCGA/TCGA_BRCA_clinical_master.csv"
    
    if os.path.exists(tcga_master):
        tcga_df = pd.read_csv(tcga_master)
        print(f"   ✅ TCGA-BRCA:")
        print(f"      Patients: {tcga_df.shape[0]}")
        print(f"      Variables: {tcga_df.shape[1]}")
    else:
        print(f"   ❌ TCGA-BRCA: Missing files")
    
    print("\n" + "=" * 60)
    print("✅ Verification complete!")

if __name__ == "__main__":
    verify_data()