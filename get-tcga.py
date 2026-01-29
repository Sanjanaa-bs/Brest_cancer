"""
Step 1.4: Download TCGA Breast Cancer Clinical Data
===================================================

This script downloads TCGA-BRCA clinical data from multiple sources:
1. GDC Data Portal (primary source)
2. UCSC Xena Browser (backup/supplementary)
3. cBioPortal (additional clinical features)

Total expected: ~1,198 patients with ~1,481 clinical variables
"""

import pandas as pd
import numpy as np
import requests
import os
from tqdm import tqdm
import json
import zipfile
import io

class TCGADataDownloader:
    """Download TCGA breast cancer clinical data"""
    
    def __init__(self, output_dir='data/raw/TCGA'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # TCGA-BRCA project information
        self.project_id = 'TCGA-BRCA'
        
        # GDC API endpoints
        self.gdc_api = 'https://api.gdc.cancer.gov'
        
        # UCSC Xena URLs
        self.xena_base = 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/'
        
    def download_from_xena(self):
        """
        Download clinical data from UCSC Xena Browser
        This is the easiest and most reliable method
        """
        
        print("\n📥 Downloading TCGA-BRCA clinical data from UCSC Xena...")
        
        # Xena files for TCGA-BRCA
        xena_files = {
            'clinical': 'TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix',
            'survival': 'TCGA.BRCA.sampleMap%2FBRCASurvival.txt',
            'phenotype': 'TCGA.BRCA.sampleMap%2FBRCA_phenotype'
        }
        
        downloaded_data = {}
        
        for data_type, filename in xena_files.items():
            url = self.xena_base + filename
            save_path = os.path.join(self.output_dir, f'xena_{data_type}.txt')
            
            try:
                print(f"\n   Downloading {data_type} data...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                # Save file
                with open(save_path, 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192), 
                                     desc=f"   {data_type}"):
                        f.write(chunk)
                
                # Load into pandas
                df = pd.read_csv(save_path, sep='\t', low_memory=False)
                downloaded_data[data_type] = df
                
                print(f"   ✅ Downloaded {data_type}: {df.shape}")
                
            except Exception as e:
                print(f"   ⚠️  Could not download {data_type}: {str(e)}")
                downloaded_data[data_type] = None
        
        return downloaded_data
    
    def download_from_gdc_portal(self):
        """
        Download clinical data from GDC Data Portal using API
        More comprehensive but requires more processing
        """
        
        print("\n📥 Downloading from GDC Data Portal...")
        
        # Query for clinical files
        files_endpoint = f"{self.gdc_api}/files"
        
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": ["TCGA-BRCA"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_category",
                        "value": ["Clinical"]
                    }
                }
            ]
        }
        
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.case_id",
            "format": "JSON",
            "size": "2000"
        }
        
        try:
            response = requests.get(files_endpoint, params=params)
            response.raise_for_status()
            
            file_data = response.json()
            
            print(f"   Found {len(file_data['data']['hits'])} clinical files")
            
            # Download clinical supplement files
            clinical_files = []
            
            for hit in tqdm(file_data['data']['hits'][:5], desc="   Downloading files"):
                file_id = hit['file_id']
                file_name = hit['file_name']
                
                # Download file
                data_endpoint = f"{self.gdc_api}/data/{file_id}"
                file_response = requests.get(data_endpoint)
                
                if file_response.status_code == 200:
                    save_path = os.path.join(self.output_dir, f"gdc_{file_name}")
                    with open(save_path, 'wb') as f:
                        f.write(file_response.content)
                    clinical_files.append(save_path)
            
            print(f"   ✅ Downloaded {len(clinical_files)} files from GDC")
            return clinical_files
            
        except Exception as e:
            print(f"   ⚠️  Error downloading from GDC: {str(e)}")
            return []
    
    def create_combined_clinical_dataset(self, xena_data):
        """
        Combine multiple clinical data sources into one master file
        Following the structure from bc_1 paper (Table 1)
        """
        
        print("\n🔧 Creating combined clinical dataset...")
        
        if xena_data['clinical'] is None:
            print("   ❌ No clinical data available")
            return None
        
        clinical_df = xena_data['clinical'].copy()
        
        # Add survival data if available
        if xena_data['survival'] is not None:
            survival_df = xena_data['survival']
            clinical_df = clinical_df.merge(
                survival_df,
                left_on='sampleID',
                right_on='sample',
                how='left',
                suffixes=('', '_survival')
            )
        
        print(f"   Combined dataset shape: {clinical_df.shape}")
        print(f"   Total patients: {clinical_df.shape[0]}")
        print(f"   Total variables: {clinical_df.shape[1]}")
        
        # Save master clinical file
        output_path = os.path.join(self.output_dir, 'TCGA_BRCA_clinical_master.csv')
        clinical_df.to_csv(output_path, index=False)
        print(f"   💾 Saved master clinical data: {output_path}")
        
        return clinical_df
    
    def extract_key_clinical_features(self, clinical_df):
        """
        Extract the key clinical features needed for our study
        Based on bc_1 paper Table 1 and Table 2
        """
        
        print("\n🔍 Extracting key clinical features...")
        
        # Define important features based on bc_1 paper
        important_features = [
            # Demographics
            'age_at_diagnosis', 'age_at_initial_pathologic_diagnosis',
            
            # Receptor status
            'er_status_by_ihc', 'pr_status_by_ihc', 'her2_status_by_ihc',
            'ER_Status_nature2012', 'PR_Status_nature2012', 'HER2_Final_Status_nature2012',
            
            # Pathologic stage
            'pathologic_stage', 'ajcc_pathologic_tumor_stage',
            'pathologic_T', 'pathologic_N', 'pathologic_M',
            
            # Tumor characteristics
            'histological_type', 'tumor_grade',
            'lymph_nodes_examined_count', 'lymph_node_examined_count',
            
            # Treatment
            'chemotherapy', 'hormone_therapy', 'radiation_therapy',
            'pharmaceutical_tx_adjuvant',
            
            # Menopause
            'menopause_status',
            
            # Surgery
            'surgical_procedure_first',
            
            # Survival
            'OS', 'OS.time', 'DSS', 'DSS.time', 'DFI', 'DFI.time', 'PFI', 'PFI.time',
            
            # Additional
            'necrosis_percent', 'tumor_nuclei_percent',
            'breast_carcinoma_estrogen_receptor_status',
            'breast_carcinoma_progesterone_receptor_status',
            'lab_proc_her2_neu_immunohistochemistry_receptor_status'
        ]
        
        # Find available features
        available_features = [col for col in important_features if col in clinical_df.columns]
        
        if len(available_features) == 0:
            print("   ⚠️  Using all available columns")
            print(f"   Available columns: {list(clinical_df.columns)[:20]}...")
            return clinical_df
        
        print(f"   Found {len(available_features)} key features")
        print(f"   Features: {available_features[:10]}...")
        
        # Extract relevant features
        key_df = clinical_df[['sampleID'] + available_features].copy()
        
        # Save key features
        output_path = os.path.join(self.output_dir, 'TCGA_BRCA_key_features.csv')
        key_df.to_csv(output_path, index=False)
        print(f"   💾 Saved key features: {output_path}")
        
        return key_df
    
    def get_treatment_information(self, clinical_df):
        """
        Extract treatment information (chemotherapy vs hormone therapy)
        This is our target variable from bc_1 paper
        """
        
        print("\n💊 Extracting treatment information...")
        
        # Look for treatment columns
        treatment_cols = [col for col in clinical_df.columns 
                         if any(term in col.lower() for term in 
                               ['treatment', 'chemotherapy', 'hormone', 'therapy', 'chemo'])]
        
        print(f"   Found treatment columns: {treatment_cols}")
        
        if len(treatment_cols) > 0:
            treatment_df = clinical_df[['sampleID'] + treatment_cols].copy()
            
            # Save treatment info
            output_path = os.path.join(self.output_dir, 'TCGA_BRCA_treatment.csv')
            treatment_df.to_csv(output_path, index=False)
            print(f"   💾 Saved treatment info: {output_path}")
            
            return treatment_df
        else:
            print("   ⚠️  No treatment columns found")
            return None
    
    def generate_download_summary(self, clinical_df):
        """Generate a summary report of downloaded data"""
        
        print("\n" + "=" * 70)
        print("TCGA DATA DOWNLOAD SUMMARY")
        print("=" * 70)
        
        print(f"\n✅ Successfully downloaded TCGA-BRCA clinical data")
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total patients: {clinical_df.shape[0]}")
        print(f"   Total variables: {clinical_df.shape[1]}")
        print(f"   Missing values: {clinical_df.isnull().sum().sum()}")
        print(f"   Completeness: {(1 - clinical_df.isnull().sum().sum() / clinical_df.size) * 100:.2f}%")
        
        print(f"\n📋 Sample columns:")
        for col in list(clinical_df.columns)[:15]:
            non_null = clinical_df[col].notna().sum()
            print(f"   - {col}: {non_null} non-null values")
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'download_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"TCGA-BRCA Download Summary\n")
            f.write(f"==========================\n\n")
            f.write(f"Total patients: {clinical_df.shape[0]}\n")
            f.write(f"Total variables: {clinical_df.shape[1]}\n")
            f.write(f"\nColumns:\n")
            for col in clinical_df.columns:
                f.write(f"  - {col}\n")
        
        print(f"\n💾 Summary saved to: {summary_path}")


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("TCGA BREAST CANCER CLINICAL DATA DOWNLOAD")
    print("=" * 70)
    
    # Initialize downloader
    downloader = TCGADataDownloader()
    
    # Try downloading from UCSC Xena (easiest method)
    xena_data = downloader.download_from_xena()
    
    if xena_data['clinical'] is not None:
        # Create combined dataset
        clinical_df = downloader.create_combined_clinical_dataset(xena_data)
        
        if clinical_df is not None:
            # Extract key features
            key_features = downloader.extract_key_clinical_features(clinical_df)
            
            # Extract treatment info
            treatment_df = downloader.get_treatment_information(clinical_df)
            
            # Generate summary
            downloader.generate_download_summary(clinical_df)
            
            print("\n✅ TCGA data download complete!")
            print("\nNext step: Run 03_preprocess_data.py to preprocess the data")
    else:
        print("\n❌ Could not download TCGA data")
        print("   Please try downloading manually from:")
        print("   https://xenabrowser.net/datapages/?cohort=TCGA%20Breast%20Cancer%20(BRCA)")


if __name__ == "__main__":
    main()