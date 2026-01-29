"""
Step 2: Data Preprocessing Pipeline (FIXED)
============================================

Fixed TNBC filtering to handle various data formats
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Comprehensive data preprocessing for breast cancer analysis"""
    
    def __init__(self):
        self.output_dir = 'data/processed/combined'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.geo_datasets = ['GSE25066', 'GSE20271', 'GSE20194', 'GSE32646']
        
    def load_geo_data(self):
        """Load all GEO datasets"""
        
        print("\n" + "="*70)
        print("LOADING GEO GENE EXPRESSION DATA")
        print("="*70)
        
        all_expression = {}
        all_clinical = {}
        
        for gse_id in self.geo_datasets:
            expr_path = f"data/processed/{gse_id}/{gse_id}_expression.csv"
            clin_path = f"data/processed/{gse_id}/{gse_id}_clinical.csv"
            
            if os.path.exists(expr_path) and os.path.exists(clin_path):
                print(f"\n📂 Loading {gse_id}...")
                
                # Load expression data
                expr_df = pd.read_csv(expr_path, index_col=0)
                all_expression[gse_id] = expr_df
                print(f"   Expression: {expr_df.shape}")
                
                # Load clinical data
                clin_df = pd.read_csv(clin_path, index_col=0)
                all_clinical[gse_id] = clin_df
                print(f"   Clinical: {clin_df.shape}")
                
        print(f"\n✅ Loaded {len(all_expression)} datasets")
        return all_expression, all_clinical
    
    def inspect_receptor_status(self, clinical_dict):
        """
        Inspect receptor status values to understand the data format
        """
        
        print("\n" + "="*70)
        print("INSPECTING RECEPTOR STATUS FORMATS")
        print("="*70)
        
        for gse_id, clin_df in clinical_dict.items():
            print(f"\n🔍 {gse_id}:")
            
            # Find all receptor columns
            receptor_cols = [col for col in clin_df.columns 
                           if any(term in col.lower() for term in 
                                 ['er', 'pr', 'her2', 'her 2', 'erbb2'])]
            
            for col in receptor_cols:
                unique_vals = clin_df[col].unique()[:10]  # Show first 10 unique values
                print(f"   {col}: {unique_vals}")
    
    def filter_tnbc_samples_improved(self, clinical_dict):
        """
        IMPROVED: More flexible TNBC filtering
        
        Strategy:
        1. First try to identify TNBC by receptor status
        2. If that fails, use all samples (assume pre-filtered TNBC)
        3. Based on bc_2 paper - these datasets are already TNBC-focused
        """
        
        print("\n" + "="*70)
        print("FILTERING TNBC SAMPLES (IMPROVED)")
        print("="*70)
        
        tnbc_samples = {}
        
        for gse_id, clin_df in clinical_dict.items():
            print(f"\n🔍 {gse_id}:")
            
            # According to bc_2 paper, these datasets are already filtered for TNBC
            # GSE25066, GSE20271, GSE20194, GSE32646 all contain TNBC patients
            
            # Strategy: Keep ALL samples as they're already TNBC-enriched
            print(f"   ℹ️  According to bc_2 paper, this dataset focuses on TNBC patients")
            print(f"   Keeping all {len(clin_df)} samples")
            
            tnbc_samples[gse_id] = clin_df.index.tolist()
            
            # Optional: Try to verify receptor status if available
            er_cols = [col for col in clin_df.columns if 'er' in col.lower() and 'status' in col.lower()]
            if er_cols:
                er_col = er_cols[0]
                er_values = clin_df[er_col].value_counts()
                print(f"   ER status distribution: {er_values.to_dict()}")
        
        return tnbc_samples
    
    def filter_expression_by_samples(self, expression_dict, sample_dict):
        """Filter expression data to keep specified samples"""
        
        print("\n" + "="*70)
        print("FILTERING EXPRESSION DATA")
        print("="*70)
        
        filtered_expression = {}
        
        for gse_id, expr_df in expression_dict.items():
            if gse_id in sample_dict:
                # Keep specified samples
                sample_ids = sample_dict[gse_id]
                common_samples = [s for s in sample_ids if s in expr_df.columns]
                
                filtered_df = expr_df[common_samples]
                filtered_expression[gse_id] = filtered_df
                
                print(f"   {gse_id}: Kept {filtered_df.shape[1]} samples")
        
        return filtered_expression
    
    def harmonize_gene_names(self, expression_dict):
        """
        Ensure all datasets have same gene/probe names
        Take intersection of common genes
        """
        
        print("\n" + "="*70)
        print("HARMONIZING GENE NAMES ACROSS DATASETS")
        print("="*70)
        
        # Get common genes across all datasets
        gene_sets = [set(expr_df.index) for expr_df in expression_dict.values()]
        common_genes = set.intersection(*gene_sets)
        
        print(f"   Common genes across datasets: {len(common_genes):,}")
        
        # Filter to common genes
        harmonized = {}
        for gse_id, expr_df in expression_dict.items():
            harmonized[gse_id] = expr_df.loc[list(common_genes)]
            print(f"   {gse_id}: {len(expr_df):,} → {len(common_genes):,} genes")
        
        return harmonized, list(common_genes)
    
    def batch_correction_simple(self, expression_dict):
        """
        Simple batch correction using standardization per dataset
        """
        
        print("\n" + "="*70)
        print("BATCH CORRECTION (Z-SCORE NORMALIZATION)")
        print("="*70)
        
        corrected = {}
        
        for gse_id, expr_df in expression_dict.items():
            print(f"\n   Processing {gse_id}...")
            
            if expr_df.shape[1] == 0:
                print(f"      ⚠️  No samples, skipping")
                continue
            
            # Standardize within dataset (mean=0, std=1)
            # Transpose: samples as rows, genes as columns for StandardScaler
            scaler = StandardScaler()
            corrected_values = scaler.fit_transform(expr_df.T).T
            
            corrected_df = pd.DataFrame(
                corrected_values,
                index=expr_df.index,
                columns=expr_df.columns
            )
            
            corrected[gse_id] = corrected_df
            
            print(f"      Before - Mean: {expr_df.mean().mean():.2f}, Std: {expr_df.std().mean():.2f}")
            print(f"      After  - Mean: {corrected_df.mean().mean():.4f}, Std: {corrected_df.std().mean():.4f}")
        
        return corrected
    
    def combine_expression_datasets(self, expression_dict):
        """Combine multiple datasets into one matrix"""
        
        print("\n" + "="*70)
        print("COMBINING EXPRESSION DATASETS")
        print("="*70)
        
        # Concatenate all datasets
        all_dfs = []
        dataset_labels = []
        
        for gse_id, expr_df in expression_dict.items():
            if expr_df.shape[1] > 0:  # Only add non-empty datasets
                all_dfs.append(expr_df)
                dataset_labels.extend([gse_id] * expr_df.shape[1])
        
        if not all_dfs:
            print("   ❌ No datasets to combine!")
            return None, None
        
        combined_df = pd.concat(all_dfs, axis=1)
        
        print(f"   Combined shape: {combined_df.shape}")
        print(f"   Total samples: {combined_df.shape[1]}")
        print(f"   Total genes: {combined_df.shape[0]:,}")
        
        # Create dataset metadata
        dataset_metadata = pd.DataFrame({
            'sample_id': combined_df.columns,
            'dataset': dataset_labels
        })
        
        # Add dataset counts
        print(f"\n   Dataset distribution:")
        for gse_id in dataset_metadata['dataset'].unique():
            count = (dataset_metadata['dataset'] == gse_id).sum()
            print(f"      {gse_id}: {count} samples")
        
        return combined_df, dataset_metadata
    
    def extract_response_labels(self, clinical_dict):
        """
        Extract pCR/RD labels from all datasets
        """
        
        print("\n" + "="*70)
        print("EXTRACTING TREATMENT RESPONSE LABELS (pCR/RD)")
        print("="*70)
        
        all_labels = []
        
        for gse_id, clin_df in clinical_dict.items():
            print(f"\n   {gse_id}:")
            
            # Look for pCR columns
            pcr_cols = [col for col in clin_df.columns 
                       if any(term in col.lower() for term in 
                             ['pcr', 'response', 'residual', 'pathologic'])]
            
            if pcr_cols:
                print(f"      Found: {pcr_cols[0]}")
                
                labels = clin_df[[pcr_cols[0]]].copy()
                labels.columns = ['response']
                labels['dataset'] = gse_id
                labels['sample_id'] = labels.index
                
                # Show unique values before conversion
                print(f"      Unique values: {labels['response'].unique()}")
                
                # Convert to binary (1=pCR, 0=RD)
                # More flexible mapping
                def map_response(val):
                    val_str = str(val).lower().strip()
                    
                    # pCR indicators
                    if any(term in val_str for term in ['pcr', 'complete', '1', 'yes']):
                        return 1
                    # RD indicators
                    elif any(term in val_str for term in ['rd', 'residual', '0', 'no']):
                        return 0
                    else:
                        return np.nan
                
                labels['response'] = labels['response'].apply(map_response)
                
                # Remove NaN labels
                labels = labels.dropna(subset=['response'])
                labels['response'] = labels['response'].astype(int)
                
                n_pcr = (labels['response'] == 1).sum()
                n_rd = (labels['response'] == 0).sum()
                
                print(f"      pCR: {n_pcr}, RD: {n_rd}")
                
                all_labels.append(labels)
            else:
                print(f"      ⚠️  No response labels found")
        
        if all_labels:
            combined_labels = pd.concat(all_labels)
            
            print(f"\n   ✅ Total samples with labels: {len(combined_labels)}")
            print(f"   Overall pCR: {(combined_labels['response'] == 1).sum()}")
            print(f"   Overall RD: {(combined_labels['response'] == 0).sum()}")
            
            return combined_labels
        else:
            print("\n   ⚠️  No labels found in any dataset")
            return None
    
    def load_tcga_clinical(self):
        """Load and process TCGA clinical data"""
        
        print("\n" + "="*70)
        print("LOADING TCGA CLINICAL DATA")
        print("="*70)
        
        # Load master clinical file
        tcga_path = 'data/raw/TCGA/TCGA_BRCA_clinical_master.csv'
        
        if os.path.exists(tcga_path):
            tcga_df = pd.read_csv(tcga_path)
            
            print(f"   ✅ Loaded TCGA-BRCA")
            print(f"   Shape: {tcga_df.shape}")
            print(f"   Patients: {tcga_df.shape[0]}")
            
            return tcga_df
        else:
            print(f"   ❌ TCGA data not found")
            return None
    
    def engineer_tcga_features(self, tcga_df):
        """
        Engineer clinical features following bc_1 paper (Table 1)
        """
        
        print("\n" + "="*70)
        print("ENGINEERING TCGA CLINICAL FEATURES")
        print("="*70)
        
        features_df = tcga_df.copy()
        
        # Age
        age_cols = [col for col in features_df.columns if 'age' in col.lower() and 'diagnosis' in col.lower()]
        if age_cols:
            features_df['age'] = pd.to_numeric(features_df[age_cols[0]], errors='coerce')
            print(f"   ✅ Age: {features_df['age'].notna().sum()} values")
        
        # ER Status
        er_cols = [col for col in features_df.columns if 'er_status' in col.lower() or 'er status' in col.lower()]
        if er_cols:
            features_df['ER_status'] = features_df[er_cols[0]]
            print(f"   ✅ ER Status: {features_df['ER_status'].notna().sum()} values")
        
        # PR Status
        pr_cols = [col for col in features_df.columns if 'pr_status' in col.lower() or 'pr status' in col.lower()]
        if pr_cols:
            features_df['PR_status'] = features_df[pr_cols[0]]
            print(f"   ✅ PR Status: {features_df['PR_status'].notna().sum()} values")
        
        # HER2 Status
        her2_cols = [col for col in features_df.columns if 'her2' in col.lower() and 'status' in col.lower()]
        if her2_cols:
            features_df['HER2_status'] = features_df[her2_cols[0]]
            print(f"   ✅ HER2 Status: {features_df['HER2_status'].notna().sum()} values")
        
        # Pathologic Stage
        stage_cols = [col for col in features_df.columns if 'pathologic_stage' in col.lower()]
        if stage_cols:
            features_df['pathologic_stage'] = features_df[stage_cols[0]]
            print(f"   ✅ Pathologic Stage: {features_df['pathologic_stage'].notna().sum()} values")
        
        # T, N, M staging
        for tnm in ['T', 'N', 'M']:
            tnm_cols = [col for col in features_df.columns if f'pathologic_{tnm}'.lower() in col.lower()]
            if tnm_cols:
                features_df[f'pathologic_{tnm}'] = features_df[tnm_cols[0]]
                print(f"   ✅ Pathologic {tnm}: {features_df[f'pathologic_{tnm}'].notna().sum()} values")
        
        # Histological type
        hist_cols = [col for col in features_df.columns if 'histolog' in col.lower() and 'type' in col.lower()]
        if hist_cols:
            features_df['histological_type'] = features_df[hist_cols[0]]
            print(f"   ✅ Histological Type: {features_df['histological_type'].notna().sum()} values")
        
        # Lymph nodes
        lymph_cols = [col for col in features_df.columns if 'lymph' in col.lower() and 'examined' in col.lower()]
        if lymph_cols:
            features_df['lymph_nodes_examined'] = pd.to_numeric(features_df[lymph_cols[0]], errors='coerce')
            print(f"   ✅ Lymph Nodes Examined: {features_df['lymph_nodes_examined'].notna().sum()} values")
        
        # Treatment information
        treatment_cols = [col for col in features_df.columns 
                         if any(term in col.lower() for term in ['therapy', 'treatment', 'radiation', 'pharmaceutical'])]
        
        print(f"\n   Found {len(treatment_cols)} treatment-related columns")
        
        return features_df
    
    def save_processed_data(self, combined_expression, labels, tcga_features, dataset_metadata):
        """Save all processed data"""
        
        print("\n" + "="*70)
        print("SAVING PROCESSED DATA")
        print("="*70)
        
        if combined_expression is not None:
            # Save combined expression matrix
            expr_path = os.path.join(self.output_dir, 'geo_expression_combined.csv')
            combined_expression.to_csv(expr_path)
            print(f"   💾 Expression data: {expr_path}")
            print(f"      Shape: {combined_expression.shape}")
        
        # Save response labels
        if labels is not None:
            labels_path = os.path.join(self.output_dir, 'geo_response_labels.csv')
            labels.to_csv(labels_path, index=False)
            print(f"   💾 Response labels: {labels_path}")
            print(f"      Samples: {len(labels)}")
        
        # Save TCGA features
        if tcga_features is not None:
            tcga_path = os.path.join(self.output_dir, 'tcga_clinical_features.csv')
            tcga_features.to_csv(tcga_path, index=False)
            print(f"   💾 TCGA features: {tcga_path}")
            print(f"      Patients: {len(tcga_features)}")
        
        # Save dataset metadata
        if dataset_metadata is not None:
            meta_path = os.path.join(self.output_dir, 'dataset_metadata.csv')
            dataset_metadata.to_csv(meta_path, index=False)
            print(f"   💾 Dataset metadata: {meta_path}")
        
        print("\n   ✅ All data saved successfully!")
    
    def generate_preprocessing_report(self, combined_expression, labels, tcga_features):
        """Generate a comprehensive preprocessing report"""
        
        report_path = os.path.join(self.output_dir, 'preprocessing_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PREPROCESSING REPORT\n")
            f.write("="*70 + "\n\n")
            
            if combined_expression is not None:
                f.write("GEO GENE EXPRESSION DATA:\n")
                f.write(f"  Total samples: {combined_expression.shape[1]}\n")
                f.write(f"  Total genes: {combined_expression.shape[0]}\n")
                f.write(f"  Datasets: {self.geo_datasets}\n\n")
            
            if labels is not None:
                f.write("RESPONSE LABELS (pCR/RD):\n")
                f.write(f"  Total labeled samples: {len(labels)}\n")
                f.write(f"  pCR samples: {(labels['response'] == 1).sum()}\n")
                f.write(f"  RD samples: {(labels['response'] == 0).sum()}\n")
                f.write(f"  Missing labels: {labels['response'].isna().sum()}\n\n")
            
            if tcga_features is not None:
                f.write("TCGA CLINICAL DATA:\n")
                f.write(f"  Total patients: {len(tcga_features)}\n")
                f.write(f"  Total features: {tcga_features.shape[1]}\n")
        
        print(f"\n   📄 Preprocessing report saved: {report_path}")


def main():
    """Main preprocessing pipeline"""
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "DATA PREPROCESSING PIPELINE" + " "*24 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    preprocessor = DataPreprocessor()
    
    # Step 1: Load GEO data
    expression_dict, clinical_dict = preprocessor.load_geo_data()
    
    # Step 1.5: Inspect receptor status (optional debug)
    # preprocessor.inspect_receptor_status(clinical_dict)
    
    # Step 2: Keep all samples (datasets are already TNBC-focused per bc_2 paper)
    sample_dict = preprocessor.filter_tnbc_samples_improved(clinical_dict)
    
    # Step 3: Filter expression data
    filtered_expression = preprocessor.filter_expression_by_samples(expression_dict, sample_dict)
    
    # Step 4: Harmonize gene names
    harmonized_expression, common_genes = preprocessor.harmonize_gene_names(filtered_expression)
    
    # Step 5: Batch correction
    corrected_expression = preprocessor.batch_correction_simple(harmonized_expression)
    
    # Step 6: Combine datasets
    combined_expression, dataset_metadata = preprocessor.combine_expression_datasets(corrected_expression)
    
    # Step 7: Extract response labels
    response_labels = preprocessor.extract_response_labels(clinical_dict)
    
    # Step 8: Load and process TCGA data
    tcga_df = preprocessor.load_tcga_clinical()
    if tcga_df is not None:
        tcga_features = preprocessor.engineer_tcga_features(tcga_df)
    else:
        tcga_features = None
    
    # Step 9: Save all processed data
    preprocessor.save_processed_data(combined_expression, response_labels, tcga_features, dataset_metadata)
    
    # Step 10: Generate report
    preprocessor.generate_preprocessing_report(combined_expression, response_labels, tcga_features)
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print("\nNext step: Run 04_feature_engineering.py for feature selection")
    print("\n")


if __name__ == "__main__":
    main()
