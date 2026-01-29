"""
Quick Data Verification
========================
Check what data has been downloaded so far
"""

import os
import pandas as pd
import glob

def check_geo_data():
    """Check GEO datasets"""
    
    print("\n" + "=" * 70)
    print("GEO DATASETS STATUS")
    print("=" * 70)
    
    geo_datasets = {
        'GSE25066': {'expected_samples': 170, 'pcr_expected': 57, 'rd_expected': 113},
        'GSE20271': {'expected_samples': 58, 'pcr_expected': 13, 'rd_expected': 45},
        'GSE20194': {'expected_samples': 71, 'pcr_expected': 25, 'rd_expected': 46},
        'GSE32646': {'expected_samples': 26, 'pcr_expected': 10, 'rd_expected': 16}
    }
    
    total_samples = 0
    available_datasets = 0
    
    for gse_id, info in geo_datasets.items():
        expr_path = f"data/processed/{gse_id}/{gse_id}_expression.csv"
        clin_path = f"data/processed/{gse_id}/{gse_id}_clinical.csv"
        
        print(f"\n📊 {gse_id}:")
        
        if os.path.exists(expr_path) and os.path.exists(clin_path):
            # Read data
            expr_df = pd.read_csv(expr_path, index_col=0, nrows=5)
            clin_df = pd.read_csv(clin_path, index_col=0)
            
            n_samples = expr_df.shape[1]
            n_probes = expr_df.shape[0]
            
            print(f"   ✅ Status: Downloaded")
            print(f"   📈 Samples: {n_samples} (expected: {info['expected_samples']})")
            print(f"   🧬 Probes: ~{n_probes:,}")
            print(f"   📋 Clinical features: {clin_df.shape[1]}")
            
            # Check for pCR labels
            pcr_cols = [col for col in clin_df.columns 
                       if any(term in col.lower() for term in ['pcr', 'response', 'residual'])]
            
            if pcr_cols:
                print(f"   🏷️  Response labels: Available ({pcr_cols[0]})")
                
                # Try to show distribution
                try:
                    label_col = clin_df[pcr_cols[0]]
                    if label_col.dtype in ['int64', 'float64']:
                        n_pcr = (label_col == 1).sum()
                        n_rd = (label_col == 0).sum()
                        print(f"       pCR: {n_pcr}, RD: {n_rd}")
                    else:
                        print(f"       Distribution: {label_col.value_counts().to_dict()}")
                except:
                    pass
            else:
                print(f"   ⚠️  Response labels: Not found in clinical data")
            
            total_samples += n_samples
            available_datasets += 1
            
        else:
            print(f"   ❌ Status: Not downloaded")
            print(f"   Expected: {info['expected_samples']} samples")
    
    print("\n" + "=" * 70)
    print(f"✅ Available datasets: {available_datasets}/4")
    print(f"📊 Total samples: {total_samples}")
    print("=" * 70)
    
    return available_datasets >= 3  # We need at least 3 for good analysis


def check_tcga_data():
    """Check TCGA data"""
    
    print("\n" + "=" * 70)
    print("TCGA DATASET STATUS")
    print("=" * 70)
    
    tcga_files = {
        'Master clinical': 'data/raw/TCGA/TCGA_BRCA_clinical_master.csv',
        'Key features': 'data/raw/TCGA/TCGA_BRCA_key_features.csv',
        'Treatment info': 'data/raw/TCGA/TCGA_BRCA_treatment.csv',
        'Xena clinical': 'data/raw/TCGA/xena_clinical.txt'
    }
    
    tcga_available = False
    
    for file_type, file_path in tcga_files.items():
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, nrows=5)
                else:
                    df = pd.read_csv(file_path, sep='\t', nrows=5)
                
                full_df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_csv(file_path, sep='\t')
                
                print(f"\n✅ {file_type}:")
                print(f"   Shape: {full_df.shape}")
                print(f"   Patients: {full_df.shape[0]}")
                print(f"   Features: {full_df.shape[1]}")
                
                if 'Master clinical' in file_type:
                    tcga_available = True
                    
            except Exception as e:
                print(f"\n⚠️  {file_type}: File exists but error reading: {str(e)}")
        else:
            print(f"\n❌ {file_type}: Not downloaded")
    
    if not tcga_available:
        print("\n⚠️  TCGA data not yet downloaded")
        print("   Run: python 02_download_tcga_data.py")
    
    return tcga_available


def check_disk_space():
    """Check data folder size"""
    
    print("\n" + "=" * 70)
    print("DISK SPACE USAGE")
    print("=" * 70)
    
    data_dir = 'data'
    
    if os.path.exists(data_dir):
        total_size = 0
        
        for dirpath, dirnames, filenames in os.walk(data_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        # Convert to MB
        size_mb = total_size / (1024 * 1024)
        
        print(f"\n📦 Total data size: {size_mb:.2f} MB")
        
        # Count files
        csv_files = glob.glob('data/**/*.csv', recursive=True)
        txt_files = glob.glob('data/**/*.txt', recursive=True)
        pkl_files = glob.glob('data/**/*.pkl', recursive=True)
        
        print(f"📄 Files:")
        print(f"   CSV files: {len(csv_files)}")
        print(f"   TXT files: {len(txt_files)}")
        print(f"   PKL files: {len(pkl_files)}")


def main():
    """Main verification"""
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "DATA DOWNLOAD STATUS" + " " * 28 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Check GEO data
    geo_ok = check_geo_data()
    
    # Check TCGA data
    tcga_ok = check_tcga_data()
    
    # Check disk space
    check_disk_space()
    
    # Overall status
    print("\n" + "=" * 70)
    print("OVERALL STATUS")
    print("=" * 70)
    
    if geo_ok:
        print("✅ GEO data: Ready for preprocessing")
    else:
        print("⚠️  GEO data: Need at least 3 datasets")
    
    if tcga_ok:
        print("✅ TCGA data: Ready for preprocessing")
    else:
        print("⚠️  TCGA data: Not yet downloaded")
    
    print("\n" + "=" * 70)
    
    if geo_ok and tcga_ok:
        print("\n🎉 ALL DATA DOWNLOADED! Ready for next step:")
        print("   → Run: python 03_preprocess_data.py")
    elif geo_ok:
        print("\n✅ GEO data ready!")
        print("⏳ Next: Download TCGA data")
        print("   → Run: python 02_download_tcga_data.py")
    else:
        print("\n⏳ Continue downloading datasets")
        print("   → Fix GSE32646 (optional): python fix_GSE32646_download.py")
        print("   → Download TCGA: python 02_download_tcga_data.py")
    
    print("\n")


if __name__ == "__main__":
    main()