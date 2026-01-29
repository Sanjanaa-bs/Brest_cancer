"""
Step 5: Model Interpretation with SHAP
=======================================

Use SHAP (SHapley Additive exPlanations) to:
1. Identify most important genes
2. Visualize feature contributions
3. Understand model predictions
4. Create publication-ready figures

Following bc_1 methodology
"""

import pandas as pd
import numpy as np
import os
import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelInterpreter:
    """SHAP-based model interpretation"""
    
    def __init__(self, n_features=100):
        self.n_features = n_features
        self.output_dir = f'results/interpretation_top{n_features}'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f'{self.output_dir}/figures', exist_ok=True)
        
    def load_data(self):
        """Load preprocessed data and selected features"""
        
        print("\n" + "="*70)
        print(f"LOADING DATA (TOP {self.n_features} GENES)")
        print("="*70)
        
        # Load expression
        expr_path = 'data/processed/combined/geo_expression_combined.csv'
        self.expression = pd.read_csv(expr_path, index_col=0)
        
        # Load labels
        labels_path = 'data/processed/combined/geo_response_labels.csv'
        labels_df = pd.read_csv(labels_path)
        self.labels = labels_df[['response', 'sample_id']].copy()
        self.labels.set_index('sample_id', inplace=True)
        
        # Load selected genes
        genes_path = f'data/processed/features/selected_genes_top{self.n_features}.csv'
        genes_df = pd.read_csv(genes_path)
        self.selected_genes = genes_df['gene'].tolist()
        
        print(f"   ✅ Loaded {len(self.selected_genes)} genes")
        
    def prepare_data(self):
        """Prepare data for modeling"""
        
        # Get common samples
        common_samples = list(set(self.expression.columns) & set(self.labels.index))
        
        # Prepare X and y
        X = self.expression.loc[self.selected_genes, common_samples].T
        y = self.labels.loc[common_samples, 'response'].values
        
        return X, y
    
    def train_best_models(self, X, y):
        """Train the best performing models"""
        
        print("\n" + "="*70)
        print("TRAINING BEST MODELS FOR INTERPRETATION")
        print("="*70)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"   Original: {X.shape}")
        print(f"   After SMOTE: {X_resampled.shape}")
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric='logloss', use_label_encoder=False
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
        }
        
        trained_models = {}
        
        for name, model in models.items():
            print(f"\n   Training {name}...")
            model.fit(X_resampled, y_resampled)
            
            # Score on original data
            score = model.score(X, y)
            print(f"   Accuracy on original data: {score:.4f}")
            
            trained_models[name] = model
        
        return trained_models, X_resampled, y_resampled
    
    def compute_shap_values(self, models, X, X_resampled):
        """Compute SHAP values for all models"""
        
        print("\n" + "="*70)
        print("COMPUTING SHAP VALUES")
        print("="*70)
        
        shap_values_dict = {}
        explainers_dict = {}
        
        for name, model in models.items():
            print(f"\n   Computing SHAP for {name}...")
            
            try:
                # Create explainer (use subset for speed)
                if name == 'XGBoost':
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)
                else:
                    explainer = shap.TreeExplainer(model)
                    # For Random Forest and GBM, get values for class 1 (pCR)
                    shap_vals = explainer.shap_values(X)
                    if isinstance(shap_vals, list):
                        shap_values = shap_vals[1]  # pCR class
                    else:
                        shap_values = shap_vals
                
                shap_values_dict[name] = shap_values
                explainers_dict[name] = explainer
                
                print(f"      ✅ SHAP values computed: {shap_values.shape}")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
        
        return shap_values_dict, explainers_dict
    
    def plot_shap_summary(self, shap_values_dict, X):
        """Create SHAP summary plots"""
        
        print("\n" + "="*70)
        print("CREATING SHAP SUMMARY PLOTS")
        print("="*70)
        
        for name, shap_values in shap_values_dict.items():
            print(f"\n   Creating plot for {name}...")
            
            # Summary plot (beeswarm)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, X, 
                plot_type="dot",
                max_display=20,
                show=False
            )
            plt.title(f'SHAP Summary Plot - {name} ({self.n_features} genes)', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            plot_path = os.path.join(
                self.output_dir, 'figures', 
                f'shap_summary_{name.replace(" ", "_").lower()}.png'
            )
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"      💾 Saved: {plot_path}")
            plt.close()
            
            # Bar plot (feature importance)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X,
                plot_type="bar",
                max_display=20,
                show=False
            )
            plt.title(f'Feature Importance - {name} ({self.n_features} genes)', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            bar_path = os.path.join(
                self.output_dir, 'figures',
                f'shap_importance_{name.replace(" ", "_").lower()}.png'
            )
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            print(f"      💾 Saved: {bar_path}")
            plt.close()
    
    def extract_top_features(self, shap_values_dict, X):
        """Extract top features based on SHAP values"""
        
        print("\n" + "="*70)
        print("EXTRACTING TOP FEATURES")
        print("="*70)
        
        all_feature_importance = {}
        
        for name, shap_values in shap_values_dict.items():
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'gene': X.columns,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False)
            
            all_feature_importance[name] = importance_df
            
            print(f"\n   {name} - Top 10 genes:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"      {row['gene']:20s}: {row['mean_abs_shap']:.4f}")
            
            # Save full ranking
            save_path = os.path.join(
                self.output_dir,
                f'feature_importance_{name.replace(" ", "_").lower()}.csv'
            )
            importance_df.to_csv(save_path, index=False)
            print(f"      💾 Saved: {save_path}")
        
        return all_feature_importance
    
    def create_consensus_ranking(self, all_feature_importance):
        """Create consensus feature ranking across models"""
        
        print("\n" + "="*70)
        print("CREATING CONSENSUS FEATURE RANKING")
        print("="*70)
        
        # Combine rankings from all models
        consensus_scores = {}
        
        for gene in all_feature_importance[list(all_feature_importance.keys())[0]]['gene']:
            scores = []
            for model_name, importance_df in all_feature_importance.items():
                gene_importance = importance_df[importance_df['gene'] == gene]['mean_abs_shap'].values
                if len(gene_importance) > 0:
                    scores.append(gene_importance[0])
            
            if scores:
                consensus_scores[gene] = np.mean(scores)
        
        # Create consensus DataFrame
        consensus_df = pd.DataFrame({
            'gene': list(consensus_scores.keys()),
            'consensus_importance': list(consensus_scores.values())
        }).sort_values('consensus_importance', ascending=False)
        
        # Add individual model rankings
        for model_name, importance_df in all_feature_importance.items():
            model_col = f'{model_name.replace(" ", "_").lower()}_importance'
            consensus_df = consensus_df.merge(
                importance_df[['gene', 'mean_abs_shap']].rename(
                    columns={'mean_abs_shap': model_col}
                ),
                on='gene',
                how='left'
            )
        
        # Save consensus ranking
        consensus_path = os.path.join(self.output_dir, 'consensus_feature_ranking.csv')
        consensus_df.to_csv(consensus_path, index=False)
        
        print(f"   ✅ Consensus ranking created")
        print(f"   💾 Saved: {consensus_path}")
        
        print(f"\n   Top 20 genes (consensus):")
        for idx, row in consensus_df.head(20).iterrows():
            print(f"      {idx+1:2d}. {row['gene']:20s}: {row['consensus_importance']:.4f}")
        
        return consensus_df
    
    def plot_top_genes_heatmap(self, consensus_df, X, y):
        """Create heatmap of top genes"""
        
        print("\n" + "="*70)
        print("CREATING TOP GENES HEATMAP")
        print("="*70)
        
        # Get top 30 genes
        top_genes = consensus_df.head(30)['gene'].tolist()
        
        # Prepare data
        X_top = X[top_genes].copy()
        X_top['response'] = y
        X_top = X_top.sort_values('response')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Heatmap
        sns.heatmap(
            X_top[top_genes].T,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Expression (Z-score)'},
            ax=ax,
            xticklabels=False
        )
        
        # Add response annotation
        response_colors = ['blue' if r == 0 else 'red' for r in X_top['response']]
        for i, color in enumerate(response_colors):
            ax.axvline(x=i, color=color, alpha=0.3, linewidth=0.1)
        
        ax.set_xlabel('Samples (Blue=RD, Red=pCR)', fontsize=12)
        ax.set_ylabel('Genes', fontsize=12)
        ax.set_title(f'Top 30 Genes Expression Heatmap ({self.n_features} gene model)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        heatmap_path = os.path.join(self.output_dir, 'figures', 'top_genes_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {heatmap_path}")
        plt.close()
    
    def generate_interpretation_report(self, consensus_df):
        """Generate interpretation report"""
        
        report_path = os.path.join(self.output_dir, 'interpretation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"MODEL INTERPRETATION REPORT - TOP {self.n_features} GENES\n")
            f.write("="*70 + "\n\n")
            
            f.write("SHAP ANALYSIS SUMMARY:\n")
            f.write("-"*70 + "\n")
            f.write("SHAP (SHapley Additive exPlanations) values quantify the\n")
            f.write("contribution of each gene to the model's predictions.\n\n")
            
            f.write("TOP 30 MOST IMPORTANT GENES (CONSENSUS RANKING):\n")
            f.write("-"*70 + "\n")
            for idx, row in consensus_df.head(30).iterrows():
                f.write(f"  {idx+1:2d}. {row['gene']:20s} "
                       f"(importance: {row['consensus_importance']:.4f})\n")
            
            f.write("\n\nINTERPRETATION:\n")
            f.write("-"*70 + "\n")
            f.write("- Higher SHAP values indicate stronger contribution to pCR prediction\n")
            f.write("- Top genes represent key molecular signatures of treatment response\n")
            f.write("- These genes can serve as potential biomarkers for personalized treatment\n")
        
        print(f"\n   📄 Report saved: {report_path}")


def main():
    """Main interpretation pipeline"""
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "MODEL INTERPRETATION (SHAP)" + " "*24 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    # Analyze all feature sets
    for n_features in [50, 100, 150]:
        print("\n" + "█"*70)
        print(f"{'█'*5} INTERPRETING {n_features}-GENE MODEL {' '*(55-len(str(n_features)))}{'█'*5}")
        print("█"*70)
        
        interpreter = ModelInterpreter(n_features=n_features)
        
        # Load data
        interpreter.load_data()
        X, y = interpreter.prepare_data()
        
        # Train models
        models, X_resampled, y_resampled = interpreter.train_best_models(X, y)
        
        # Compute SHAP values
        shap_values_dict, explainers_dict = interpreter.compute_shap_values(
            models, X, X_resampled
        )
        
        # Create visualizations
        interpreter.plot_shap_summary(shap_values_dict, X)
        
        # Extract top features
        all_feature_importance = interpreter.extract_top_features(shap_values_dict, X)
        
        # Consensus ranking
        consensus_df = interpreter.create_consensus_ranking(all_feature_importance)
        
        # Heatmap
        interpreter.plot_top_genes_heatmap(consensus_df, X, y)
        
        # Report
        interpreter.generate_interpretation_report(consensus_df)
    
    print("\n" + "="*70)
    print("✅ MODEL INTERPRETATION COMPLETE!")
    print("="*70)
    print("\nGenerated SHAP analysis for:")
    print("  • 50-gene model")
    print("  • 100-gene model")
    print("  • 150-gene model")
    print("\nAll results saved in results/interpretation_topXXX/")
    print("\n🎉 PROJECT COMPLETE! Ready for paper writing!")
    print("\n")


if __name__ == "__main__":
    main()