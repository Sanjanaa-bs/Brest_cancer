"""
Step 3: Feature Engineering & Selection
========================================

This script performs:
1. Prior marker collection (from bc_2 paper)
2. Differential expression analysis (pCR vs RD)
3. Feature selection using Random Forest RFE
4. Feature stability validation

Following bc_2 paper methodology
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Feature engineering and selection for breast cancer prediction"""
    
    def __init__(self):
        self.output_dir = 'data/processed/features'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load preprocessed data
        self.expression = None
        self.labels = None
        self.metadata = None
        
    def load_preprocessed_data(self):
        """Load preprocessed expression and labels"""
        
        print("\n" + "="*70)
        print("LOADING PREPROCESSED DATA")
        print("="*70)
        
        # Load expression
        expr_path = 'data/processed/combined/geo_expression_combined.csv'
        self.expression = pd.read_csv(expr_path, index_col=0)
        print(f"   ✅ Expression: {self.expression.shape}")
        
        # Load labels
        labels_path = 'data/processed/combined/geo_response_labels.csv'
        labels_df = pd.read_csv(labels_path)
        
        # Check column names
        print(f"   Label columns: {list(labels_df.columns)}")
        
        # Find response column (it might be unnamed or have different name)
        response_col = None
        for col in labels_df.columns:
            if 'response' in col.lower() or labels_df[col].dtype in ['int64', 'float64']:
                if labels_df[col].nunique() == 2:  # Binary column
                    response_col = col
                    break
        
        if response_col is None:
            # Try first numeric column
            numeric_cols = labels_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                response_col = numeric_cols[0]
        
        if response_col:
            self.labels = labels_df[[response_col, 'sample_id']].copy()
            self.labels.columns = ['response', 'sample_id']
            self.labels.set_index('sample_id', inplace=True)
            
            print(f"   ✅ Labels: {len(self.labels)} samples")
            print(f"      pCR: {(self.labels['response']==1).sum()}")
            print(f"      RD: {(self.labels['response']==0).sum()}")
        else:
            print(f"   ❌ Could not find response column")
            return False
        
        # Load metadata
        meta_path = 'data/processed/combined/dataset_metadata.csv'
        self.metadata = pd.read_csv(meta_path)
        print(f"   ✅ Metadata: {len(self.metadata)} samples")
        
        return True
    
    def collect_prior_markers(self):
        """
        Collect known breast cancer marker genes
        Based on bc_2 paper Table 3 and methodology
        """
        
        print("\n" + "="*70)
        print("COLLECTING PRIOR BREAST CANCER MARKERS")
        print("="*70)
        
        # Key markers from bc_2 paper (Table 5 - their 86-gene model)
        # And from various breast cancer gene panels
        
        prior_markers = {
            # DNA Repair genes (from bc_2 enrichment analysis)
            'DNA_Repair': [
                'MSH2', 'MSH6', 'RAD51', 'XRCC5', 'BRCA1', 'BRCA2', 
                'ATM', 'ATR', 'CHEK1', 'CHEK2', 'PARP1'
            ],
            
            # Cell Cycle genes (from bc_2 enrichment analysis)
            'Cell_Cycle': [
                'CDK2', 'E2F3', 'MCM2', 'MCM3', 'CCND1', 'CCNE1',
                'MKI67', 'AURKA', 'AURKB', 'PLK1', 'BUB1'
            ],
            
            # Oncotype DX genes
            'Oncotype_DX': [
                'ESR1', 'PGR', 'ERBB2', 'MKI67', 'AURKA', 'BIRC5',
                'CCNB1', 'MYBL2', 'MMP11', 'CTSL2', 'GRB7', 'GSTM1',
                'CD68', 'BAG1', 'BCL2', 'SCUBE2', 'ACTB', 'GAPDH',
                'RPLP0', 'GUS', 'TFRC'
            ],
            
            # MammaPrint genes (key genes from 70-gene signature)
            'MammaPrint': [
                'BBC3', 'TGFB3', 'IGFBP5', 'CENPA', 'MELK', 'CCNE2',
                'EXT1', 'EGLN1', 'GNAZ', 'SERF1A', 'FLT1', 'DIAPH3',
                'NMU', 'SCUBE2', 'EGFR', 'VEGF'
            ],
            
            # PAM50 genes
            'PAM50': [
                'ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA',
                'CCNB1', 'CCNE1', 'CDC20', 'CDC6', 'CDH3', 'CENPF',
                'CEP55', 'CXXC5', 'EGFR', 'ERBB2', 'ESR1', 'EXO1',
                'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C',
                'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK',
                'MIA', 'MKI67', 'MLPH', 'MMP11', 'MYBL2', 'MYC',
                'NAT1', 'NDC80', 'NUF2', 'ORC6', 'PGR', 'PHGDH',
                'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B',
                'TYMS', 'UBE2C', 'UBE2T'
            ],
            
            # Proliferation genes
            'Proliferation': [
                'MKI67', 'PCNA', 'TOP2A', 'BIRC5', 'CCNB1', 'CCNB2',
                'CDK1', 'CENPA', 'CENPF', 'PLK1', 'AURKA', 'AURKB'
            ],
            
            # Immune genes (relevant for NAC response)
            'Immune': [
                'CD8A', 'CD8B', 'CD4', 'GZMA', 'GZMB', 'PRF1',
                'IFNG', 'PDCD1', 'CD274', 'CTLA4', 'LAG3'
            ],
            
            # Triple negative specific
            'TNBC_Specific': [
                'EGFR', 'KRT5', 'KRT6B', 'KRT14', 'KRT17', 'TP63',
                'FOXC1', 'SOX10', 'VIM', 'CDH2'
            ],
            
            # Chemotherapy response genes
            'Chemo_Response': [
                'GSTP1', 'ABCB1', 'ABCG2', 'TYMS', 'TOP1', 'TOP2A',
                'TUBB3', 'ERCC1', 'RRM1', 'TS'
            ]
        }
        
        # Flatten to unique genes
        all_markers = set()
        for category, genes in prior_markers.items():
            all_markers.update(genes)
            print(f"   {category}: {len(genes)} genes")
        
        print(f"\n   ✅ Total unique prior markers: {len(all_markers)}")
        
        # Save prior markers
        markers_df = pd.DataFrame({
            'gene': list(all_markers),
            'source': 'prior_knowledge'
        })
        
        markers_path = os.path.join(self.output_dir, 'prior_markers.csv')
        markers_df.to_csv(markers_path, index=False)
        print(f"   💾 Saved to: {markers_path}")
        
        return list(all_markers)
    
    def differential_expression_analysis(self):
        """
        Perform differential expression analysis (pCR vs RD)
        Using t-test (bc_2 uses bootstrapped t-test, we'll use standard)
        """
        
        print("\n" + "="*70)
        print("DIFFERENTIAL EXPRESSION ANALYSIS")
        print("="*70)
        
        # Get samples with labels
        common_samples = list(set(self.expression.columns) & set(self.labels.index))
        
        print(f"   Samples with both expression & labels: {len(common_samples)}")
        
        # Filter expression to labeled samples
        expr_labeled = self.expression[common_samples]
        labels_aligned = self.labels.loc[common_samples]
        
        # Separate pCR and RD groups
        pcr_samples = labels_aligned[labels_aligned['response'] == 1].index
        rd_samples = labels_aligned[labels_aligned['response'] == 0].index
        
        print(f"   pCR samples: {len(pcr_samples)}")
        print(f"   RD samples: {len(rd_samples)}")
        
        # Perform t-test for each gene
        print(f"\n   Performing t-tests for {len(expr_labeled)} genes...")
        
        degs = []
        
        for gene in expr_labeled.index:
            pcr_vals = expr_labeled.loc[gene, pcr_samples]
            rd_vals = expr_labeled.loc[gene, rd_samples]
            
            # t-test
            t_stat, p_val = stats.ttest_ind(pcr_vals, rd_vals, equal_var=False)
            
            # Fold change
            pcr_mean = pcr_vals.mean()
            rd_mean = rd_vals.mean()
            fold_change = pcr_mean - rd_mean  # Difference in means (already z-scored)
            
            degs.append({
                'gene': gene,
                'p_value': p_val,
                't_statistic': t_stat,
                'fold_change': fold_change,
                'pcr_mean': pcr_mean,
                'rd_mean': rd_mean
            })
        
        degs_df = pd.DataFrame(degs)
        
        # Benjamini-Hochberg correction
        from statsmodels.stats.multitest import fdrcorrection
        _, degs_df['fdr'] = fdrcorrection(degs_df['p_value'])
        
        # Identify significant DEGs
        sig_degs = degs_df[degs_df['fdr'] < 0.05].copy()
        sig_degs = sig_degs.sort_values('fdr')
        
        print(f"   ✅ Significant DEGs (FDR < 0.05): {len(sig_degs)}")
        print(f"      Up-regulated (pCR > RD): {(sig_degs['fold_change'] > 0).sum()}")
        print(f"      Down-regulated (pCR < RD): {(sig_degs['fold_change'] < 0).sum()}")
        
        # Save results
        degs_path = os.path.join(self.output_dir, 'differential_expression.csv')
        degs_df.to_csv(degs_path, index=False)
        print(f"   💾 All DEG results: {degs_path}")
        
        sig_degs_path = os.path.join(self.output_dir, 'significant_degs.csv')
        sig_degs.to_csv(sig_degs_path, index=False)
        print(f"   💾 Significant DEGs: {sig_degs_path}")
        
        return sig_degs['gene'].tolist()
    
    def create_candidate_feature_set(self, prior_markers, degs):
        """
        Combine prior markers and DEGs to create candidate feature set
        Following bc_2 methodology
        """
        
        print("\n" + "="*70)
        print("CREATING CANDIDATE FEATURE SET")
        print("="*70)
        
        # Union of prior markers and DEGs
        candidate_genes = set(prior_markers) | set(degs)
        
        print(f"   Prior markers: {len(prior_markers)}")
        print(f"   DEGs: {len(degs)}")
        print(f"   Union: {len(candidate_genes)}")
        
        # Filter to genes actually in expression data
        available_genes = set(self.expression.index)
        candidate_genes_available = list(candidate_genes & available_genes)
        
        print(f"   Available in data: {len(candidate_genes_available)}")
        
        # Save candidate genes
        candidates_df = pd.DataFrame({
            'gene': candidate_genes_available,
            'in_prior': [g in prior_markers for g in candidate_genes_available],
            'in_degs': [g in degs for g in candidate_genes_available]
        })
        
        candidates_path = os.path.join(self.output_dir, 'candidate_genes.csv')
        candidates_df.to_csv(candidates_path, index=False)
        print(f"   💾 Saved to: {candidates_path}")
        
        return candidate_genes_available
    
    def random_forest_rfe(self, candidate_genes, target_n_features=100):
        """
        Random Forest Recursive Feature Elimination
        Select optimal gene signature (bc_2 uses 86 genes)
        """
        
        print("\n" + "="*70)
        print("RANDOM FOREST RECURSIVE FEATURE ELIMINATION")
        print("="*70)
        
        # Get samples with labels
        common_samples = list(set(self.expression.columns) & set(self.labels.index))
        
        # Prepare data
        X = self.expression.loc[candidate_genes, common_samples].T
        y = self.labels.loc[common_samples, 'response'].values
        
        print(f"   Starting features: {len(candidate_genes)}")
        print(f"   Samples: {len(X)}")
        print(f"   Target features: {target_n_features}")
        
        # Random Forest for feature importance
        print(f"\n   Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importances
        importances = pd.DataFrame({
            'gene': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   Top 10 genes by importance:")
        for idx, row in importances.head(10).iterrows():
            print(f"      {row['gene']}: {row['importance']:.4f}")
        
        # Select top N features
        selected_genes = importances.head(target_n_features)['gene'].tolist()
        
        print(f"\n   ✅ Selected {len(selected_genes)} genes")
        
        # Cross-validation with selected genes
        X_selected = X[selected_genes]
        
        cv_scores = cross_val_score(
            rf, X_selected, y, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print(f"   Cross-validation AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Save selected genes with importances
        selected_df = importances.head(target_n_features).copy()
        selected_df['rank'] = range(1, len(selected_df) + 1)
        
        selected_path = os.path.join(self.output_dir, f'selected_genes_top{target_n_features}.csv')
        selected_df.to_csv(selected_path, index=False)
        print(f"   💾 Saved to: {selected_path}")
        
        # Save all importances
        all_imp_path = os.path.join(self.output_dir, 'all_gene_importances.csv')
        importances.to_csv(all_imp_path, index=False)
        
        return selected_genes, selected_df
    
    def validate_feature_stability(self, selected_genes):
        """
        Validate feature stability across different train/test splits
        """
        
        print("\n" + "="*70)
        print("VALIDATING FEATURE STABILITY")
        print("="*70)
        
        # Get samples with labels
        common_samples = list(set(self.expression.columns) & set(self.labels.index))
        
        X = self.expression.loc[selected_genes, common_samples].T
        y = self.labels.loc[common_samples, 'response'].values
        
        # Multiple train/test splits
        n_splits = 10
        gene_selection_counts = {gene: 0 for gene in selected_genes}
        
        print(f"   Testing stability over {n_splits} random splits...")
        
        from sklearn.model_selection import ShuffleSplit
        
        ss = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
        
        for i, (train_idx, test_idx) in enumerate(ss.split(X)):
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]
            
            # Train RF
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)
            
            # Get top 50% most important genes
            importances = rf.feature_importances_
            top_genes_idx = np.argsort(importances)[-len(selected_genes)//2:]
            top_genes = X_train.columns[top_genes_idx]
            
            for gene in top_genes:
                gene_selection_counts[gene] += 1
        
        # Calculate stability score
        stability_df = pd.DataFrame({
            'gene': list(gene_selection_counts.keys()),
            'selection_count': list(gene_selection_counts.values()),
            'stability_score': [count / n_splits for count in gene_selection_counts.values()]
        }).sort_values('stability_score', ascending=False)
        
        print(f"\n   Genes selected in >80% of splits: {(stability_df['stability_score'] > 0.8).sum()}")
        print(f"   Genes selected in >50% of splits: {(stability_df['stability_score'] > 0.5).sum()}")
        
        # Save stability scores
        stability_path = os.path.join(self.output_dir, 'feature_stability.csv')
        stability_df.to_csv(stability_path, index=False)
        print(f"   💾 Saved to: {stability_path}")
        
        return stability_df
    
    def generate_feature_report(self, selected_genes, selected_df, stability_df):
        """Generate comprehensive feature engineering report"""
        
        report_path = os.path.join(self.output_dir, 'feature_engineering_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FEATURE ENGINEERING REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"SELECTED FEATURES:\n")
            f.write(f"  Total genes: {len(selected_genes)}\n\n")
            
            f.write(f"TOP 20 GENES BY IMPORTANCE:\n")
            for idx, row in selected_df.head(20).iterrows():
                f.write(f"  {row['rank']:2d}. {row['gene']:15s} (importance: {row['importance']:.4f})\n")
            
            f.write(f"\n\nFEATURE STABILITY:\n")
            stable_genes = stability_df[stability_df['stability_score'] > 0.8]
            f.write(f"  Highly stable genes (>80%): {len(stable_genes)}\n")
            
            f.write(f"\n  Top 10 most stable:\n")
            for idx, row in stable_genes.head(10).iterrows():
                f.write(f"    {row['gene']:15s} ({row['stability_score']:.1%})\n")
        
        print(f"\n   📄 Feature report saved: {report_path}")


def main():
    """Main feature engineering pipeline"""
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*12 + "FEATURE ENGINEERING & SELECTION" + " "*23 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    engineer = FeatureEngineer()
    
    # Step 1: Load preprocessed data
    if not engineer.load_preprocessed_data():
        print("\n❌ Failed to load preprocessed data")
        return
    
    # Step 2: Collect prior markers
    prior_markers = engineer.collect_prior_markers()
    
    # Step 3: Differential expression analysis
    degs = engineer.differential_expression_analysis()
    
    # Step 4: Create candidate feature set
    candidate_genes = engineer.create_candidate_feature_set(prior_markers, degs)
    
    # Step 5: Random Forest RFE
    # Try different feature counts: 50, 100, 150
    for n_features in [50, 100, 150]:
        print(f"\n{'='*70}")
        print(f"FEATURE SELECTION: TOP {n_features} GENES")
        print(f"{'='*70}")
        
        selected_genes, selected_df = engineer.random_forest_rfe(
            candidate_genes, 
            target_n_features=n_features
        )
        
        # Step 6: Validate stability
        stability_df = engineer.validate_feature_stability(selected_genes)
        
        # Step 7: Generate report
        engineer.generate_feature_report(selected_genes, selected_df, stability_df)
    
    print("\n" + "="*70)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print("\nGenerated feature sets:")
    print("  • 50 genes (compact model)")
    print("  • 100 genes (balanced model)")
    print("  • 150 genes (comprehensive model)")
    print("\nNext step: Run 05_model_training.py to train ML models")
    print("\n")


if __name__ == "__main__":
    main()