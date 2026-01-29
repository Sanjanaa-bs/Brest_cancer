"""
Step 1.3: Download GEO Gene Expression Datasets
================================================

This script downloads the following datasets:
- GSE25066 (Development dataset: 170 TNBC patients)
- GSE20271 (Validation 1: 58 TNBC patients)
- GSE20194 (Validation 2: 71 TNBC patients)
- GSE32646 (Validation 3: 26 TNBC patients)

Platform: Affymetrix HG U133A / U133 Plus 2.0
"""

import GEOparse
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle

class GEODataDownloader:
    """Download and process GEO datasets for breast cancer analysis"""
    
    def __init__(self, output_dir='data/raw/GEO'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset information from bc_2 paper (Table 2)
        self.datasets = {
            'GSE25066': {
                'platform': 'GPL96',
                'samples': 170,
                'pcr': 57,
                'rd': 113,
                'description': 'Development dataset - TNBC patients'
            },
            'GSE20271': {
                'platform': 'GPL96',
                'samples': 58,
                'pcr': 13,
                'rd': 45,
                'description': 'Validation 1 - TNBC patients'
            },
            'GSE20194': {
                'platform': 'GPL96',
                'samples': 71,
                'pcr': 25,
                'rd': 46,
                'description': 'Validation 2 - TNBC patients'
            },
            'GSE32646': {
                'platform': 'GPL570',
                'samples': 26,
                'pcr': 10,
                'rd': 16,
                'description': 'Validation 3 - TNBC patients'
            }
        }
    
    def download_dataset(self, gse_id, force_download=False):
        """
        Download a single GEO dataset
        
        Parameters:
        -----------
        gse_id : str
            GEO Series ID (e.g., 'GSE25066')
        force_download : bool
            If True, re-download even if file exists
        """
        
        save_path = os.path.join(self.output_dir, f"{gse_id}.pkl")
        
        # Check if already downloaded
        if os.path.exists(save_path) and not force_download:
            print(f"✓ {gse_id} already downloaded. Loading from cache...")
            with open(save_path, 'rb') as f:
                return pickle.load(f)
        
        print(f"\n📥 Downloading {gse_id}...")
        print(f"   Description: {self.datasets[gse_id]['description']}")
        print(f"   Expected samples: {self.datasets[gse_id]['samples']}")
        
        try:
            # Download from GEO
            gse = GEOparse.get_GEO(geo=gse_id, destdir=self.output_dir, silent=False)
            
            # Save to pickle for faster loading later
            with open(save_path, 'wb') as f:
                pickle.dump(gse, f)
            
            print(f"✅ Successfully downloaded {gse_id}")
            return gse
            
        except Exception as e:
            print(f"❌ Error downloading {gse_id}: {str(e)}")
            return None
    
    def extract_expression_matrix(self, gse, gse_id):
        """
        Extract gene expression matrix from GEO object
        
        Returns:
        --------
        expression_df : pandas DataFrame
            Rows = genes/probes, Columns = samples
        """
        
        print(f"\n🔍 Extracting expression data from {gse_id}...")
        
        # Get the platform (GPL)
        platform_id = self.datasets[gse_id]['platform']
        
        # Get expression matrix
        # GEO data structure: gse.gsms contains all samples
        expression_data = []
        sample_ids = []
        
        for gsm_name, gsm in tqdm(gse.gsms.items(), desc="Processing samples"):
            # Get expression values
            if 'VALUE' in gsm.table.columns:
                expression_data.append(gsm.table['VALUE'].values)
            elif 'value' in gsm.table.columns:
                expression_data.append(gsm.table['value'].values)
            else:
                # Try to find any numeric column
                numeric_cols = gsm.table.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    expression_data.append(gsm.table[numeric_cols[0]].values)
                else:
                    print(f"⚠️  Warning: No expression values found for {gsm_name}")
                    continue
            
            sample_ids.append(gsm_name)
        
        # Create DataFrame
        expression_df = pd.DataFrame(
            expression_data,
            columns=gse.gsms[list(gse.gsms.keys())[0]].table['ID_REF'].values,
            index=sample_ids
        ).T  # Transpose so rows = probes, columns = samples
        
        # Convert to numeric
        expression_df = expression_df.apply(pd.to_numeric, errors='coerce')
        
        print(f"   Shape: {expression_df.shape}")
        print(f"   Samples: {len(sample_ids)}")
        print(f"   Probes: {expression_df.shape[0]}")
        
        return expression_df
    
    def extract_clinical_info(self, gse, gse_id):
        """
        Extract clinical/phenotype information
        
        Returns:
        --------
        clinical_df : pandas DataFrame
            Clinical information for each sample
        """
        
        print(f"\n📋 Extracting clinical data from {gse_id}...")
        
        clinical_data = []
        
        for gsm_name, gsm in gse.gsms.items():
            sample_info = {
                'sample_id': gsm_name,
                'title': gsm.metadata.get('title', [''])[0],
                'source': gsm.metadata.get('source_name_ch1', [''])[0],
            }
            
            # Extract characteristics (contains clinical info)
            if 'characteristics_ch1' in gsm.metadata:
                for char in gsm.metadata['characteristics_ch1']:
                    if ':' in char:
                        key, value = char.split(':', 1)
                        sample_info[key.strip()] = value.strip()
            
            clinical_data.append(sample_info)
        
        clinical_df = pd.DataFrame(clinical_data)
        clinical_df.set_index('sample_id', inplace=True)
        
        print(f"   Clinical features: {clinical_df.shape[1]}")
        print(f"   Available columns: {list(clinical_df.columns)[:5]}...")
        
        return clinical_df
    
    def extract_response_labels(self, clinical_df):
        """
        Extract pCR (pathological complete response) labels
        
        Returns:
        --------
        labels : pandas Series
            1 = pCR, 0 = RD (Residual Disease)
        """
        
        print("\n🏷️  Extracting treatment response labels (pCR/RD)...")
        
        # Look for pCR information in various column names
        pcr_columns = [col for col in clinical_df.columns 
                      if any(term in col.lower() for term in 
                            ['pcr', 'response', 'residual', 'pathologic'])]
        
        if len(pcr_columns) > 0:
            print(f"   Found potential pCR columns: {pcr_columns}")
            # Use the first one
            labels = clinical_df[pcr_columns[0]]
            
            # Try to convert to binary (1=pCR, 0=RD)
            if labels.dtype == 'object':
                labels = labels.map({
                    'pCR': 1, 'PCR': 1, 'complete response': 1,
                    'RD': 0, 'residual disease': 0, 'no response': 0
                })
            
            print(f"   pCR samples: {labels.sum()}")
            print(f"   RD samples: {(labels == 0).sum()}")
            
            return labels
        else:
            print("   ⚠️  No pCR labels found in metadata")
            return None
    
    def save_processed_data(self, expression_df, clinical_df, gse_id):
        """Save processed data to CSV files"""
        
        output_dir = f"data/processed/{gse_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save expression data
        expression_path = f"{output_dir}/{gse_id}_expression.csv"
        expression_df.to_csv(expression_path)
        print(f"   💾 Saved expression data: {expression_path}")
        
        # Save clinical data
        clinical_path = f"{output_dir}/{gse_id}_clinical.csv"
        clinical_df.to_csv(clinical_path)
        print(f"   💾 Saved clinical data: {clinical_path}")
    
    def download_all_datasets(self):
        """Download all datasets required for the study"""
        
        print("=" * 70)
        print("DOWNLOADING ALL GEO DATASETS FOR BREAST CANCER STUDY")
        print("=" * 70)
        
        results = {}
        
        for gse_id in self.datasets.keys():
            print("\n" + "=" * 70)
            
            # Download dataset
            gse = self.download_dataset(gse_id)
            
            if gse is not None:
                # Extract data
                expression_df = self.extract_expression_matrix(gse, gse_id)
                clinical_df = self.extract_clinical_info(gse, gse_id)
                labels = self.extract_response_labels(clinical_df)
                
                # Save processed data
                self.save_processed_data(expression_df, clinical_df, gse_id)
                
                results[gse_id] = {
                    'expression': expression_df,
                    'clinical': clinical_df,
                    'labels': labels,
                    'status': 'success'
                }
            else:
                results[gse_id] = {'status': 'failed'}
        
        print("\n" + "=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)
        
        for gse_id, result in results.items():
            status = "✅" if result['status'] == 'success' else "❌"
            print(f"{status} {gse_id}: {result['status']}")
        
        return results


def main():
    """Main execution function"""
    
    # Initialize downloader
    downloader = GEODataDownloader()
    
    # Download all datasets
    results = downloader.download_all_datasets()
    
    print("\n✅ Data download complete!")
    print("\nNext step: Run get-tcga.py to download TCGA clinical data")


if __name__ == "__main__":
    main()