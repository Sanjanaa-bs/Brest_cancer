"""
Step 4: Model Training & Evaluation
====================================

Train multiple ML models with:
- Random Forest, XGBoost, GBM, AdaBoost, LR, SVM
- 5-fold cross-validation
- SMOTE for class imbalance
- Bootstrap confidence intervals
- Comprehensive performance metrics

Following bc_1 and bc_2 paper methodologies
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, matthews_corrcoef
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Train and evaluate multiple ML models"""
    
    def __init__(self, n_features=100):
        self.n_features = n_features
        self.output_dir = f'results/models_top{n_features}'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f'{self.output_dir}/figures', exist_ok=True)
        
        self.expression = None
        self.labels = None
        self.selected_genes = None
        
    def load_data(self):
        """Load preprocessed data and selected features"""
        
        print("\n" + "="*70)
        print(f"LOADING DATA (TOP {self.n_features} GENES)")
        print("="*70)
        
        # Load expression
        expr_path = 'data/processed/combined/geo_expression_combined.csv'
        self.expression = pd.read_csv(expr_path, index_col=0)
        print(f"   ✅ Expression: {self.expression.shape}")
        
        # Load labels
        labels_path = 'data/processed/combined/geo_response_labels.csv'
        labels_df = pd.read_csv(labels_path)
        self.labels = labels_df[['response', 'sample_id']].copy()
        self.labels.set_index('sample_id', inplace=True)
        print(f"   ✅ Labels: {len(self.labels)} samples")
        
        # Load selected genes
        genes_path = f'data/processed/features/selected_genes_top{self.n_features}.csv'
        genes_df = pd.read_csv(genes_path)
        self.selected_genes = genes_df['gene'].tolist()
        print(f"   ✅ Selected genes: {len(self.selected_genes)}")
        
        return True
    
    def prepare_data(self):
        """Prepare X and y for modeling"""
        
        print("\n" + "="*70)
        print("PREPARING DATA FOR MODELING")
        print("="*70)
        
        # Get common samples
        common_samples = list(set(self.expression.columns) & set(self.labels.index))
        
        # Prepare X (features) and y (labels)
        X = self.expression.loc[self.selected_genes, common_samples].T
        y = self.labels.loc[common_samples, 'response'].values
        
        print(f"   Features (X): {X.shape}")
        print(f"   Labels (y): {y.shape}")
        print(f"   Class distribution:")
        print(f"      pCR (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
        print(f"      RD (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
        
        return X, y
    
    def define_models(self):
        """Define all ML models to train"""
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=(763/208)  # Adjust for class imbalance
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.5,
                random_state=42
            ),
            
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                solver='liblinear'
            ),
            
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        return models
    
    def train_with_cv(self, X, y, use_smote=True):
        """
        Train all models with 5-fold cross-validation
        Optional SMOTE for handling class imbalance
        """
        
        print("\n" + "="*70)
        print("TRAINING MODELS WITH 5-FOLD CROSS-VALIDATION")
        if use_smote:
            print("Using SMOTE for class imbalance")
        print("="*70)
        
        models = self.define_models()
        results = {}
        
        # Define cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'auroc': 'roc_auc',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        
        for model_name, model in models.items():
            print(f"\n   Training {model_name}...")
            
            try:
                if use_smote and model_name not in ['XGBoost']:  # XGBoost handles imbalance internally
                    # Create pipeline with SMOTE
                    pipeline = ImbPipeline([
                        ('smote', SMOTE(random_state=42)),
                        ('classifier', model)
                    ])
                else:
                    pipeline = model
                
                # Cross-validation
                cv_results = cross_validate(
                    pipeline, X, y,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=-1
                )
                
                # Store results
                results[model_name] = {
                    'accuracy': cv_results['test_accuracy'],
                    'auroc': cv_results['test_auroc'],
                    'precision': cv_results['test_precision'],
                    'recall': cv_results['test_recall'],
                    'f1': cv_results['test_f1']
                }
                
                # Print summary
                print(f"      Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
                print(f"      AUROC:     {cv_results['test_auroc'].mean():.4f} ± {cv_results['test_auroc'].std():.4f}")
                print(f"      Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
                print(f"      Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
                print(f"      F1-Score:  {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
                results[model_name] = None
        
        return results
    
    def bootstrap_confidence_intervals(self, X, y, model_name, model, n_iterations=100):
        """
        Calculate bootstrap confidence intervals for model performance
        Following bc_1 methodology (1000 iterations simplified to 100 for speed)
        """
        
        print(f"\n   Bootstrapping {model_name} ({n_iterations} iterations)...")
        
        metrics_bootstrap = {
            'accuracy': [],
            'auroc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        from sklearn.utils import resample
        
        for i in tqdm(range(n_iterations), desc=f"   {model_name}"):
            # Bootstrap sample
            X_boot, y_boot = resample(X, y, random_state=i, stratify=y)
            
            # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_boot, y_boot, test_size=0.3, random_state=i, stratify=y_boot
            )
            
            # Apply SMOTE to training data
            smote = SMOTE(random_state=i)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            
            # Train model
            model.fit(X_train_sm, y_train_sm)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics_bootstrap['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics_bootstrap['auroc'].append(roc_auc_score(y_test, y_pred_proba))
            metrics_bootstrap['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics_bootstrap['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics_bootstrap['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        
        # Calculate confidence intervals
        ci_results = {}
        for metric_name, values in metrics_bootstrap.items():
            ci_results[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5)
            }
        
        return ci_results
    
    def train_final_models(self, X, y):
        """Train final models on full dataset with bootstrap CI"""
        
        print("\n" + "="*70)
        print("TRAINING FINAL MODELS WITH BOOTSTRAP CI")
        print("="*70)
        
        models = self.define_models()
        final_results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*70}")
            print(f"{model_name}")
            print('='*70)
            
            try:
                # Bootstrap confidence intervals
                ci_results = self.bootstrap_confidence_intervals(X, y, model_name, model, n_iterations=100)
                
                final_results[model_name] = ci_results
                
                # Print results
                print(f"\n   Bootstrap Results (100 iterations):")
                for metric_name, stats in ci_results.items():
                    print(f"      {metric_name.capitalize():10s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                          f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
                final_results[model_name] = None
        
        return final_results
    
    def create_results_summary(self, cv_results, bootstrap_results):
        """Create comprehensive results summary table"""
        
        print("\n" + "="*70)
        print("CREATING RESULTS SUMMARY")
        print("="*70)
        
        # CV results
        cv_summary = []
        for model_name, metrics in cv_results.items():
            if metrics:
                cv_summary.append({
                    'Model': model_name,
                    'CV_Accuracy': f"{metrics['accuracy'].mean():.4f} ± {metrics['accuracy'].std():.4f}",
                    'CV_AUROC': f"{metrics['auroc'].mean():.4f} ± {metrics['auroc'].std():.4f}",
                    'CV_Precision': f"{metrics['precision'].mean():.4f} ± {metrics['precision'].std():.4f}",
                    'CV_Recall': f"{metrics['recall'].mean():.4f} ± {metrics['recall'].std():.4f}",
                    'CV_F1': f"{metrics['f1'].mean():.4f} ± {metrics['f1'].std():.4f}"
                })
        
        cv_df = pd.DataFrame(cv_summary)
        cv_path = os.path.join(self.output_dir, 'cv_results_summary.csv')
        cv_df.to_csv(cv_path, index=False)
        print(f"   💾 CV results: {cv_path}")
        
        # Bootstrap results
        bootstrap_summary = []
        for model_name, metrics in bootstrap_results.items():
            if metrics:
                bootstrap_summary.append({
                    'Model': model_name,
                    'Bootstrap_Accuracy': f"{metrics['accuracy']['mean']:.4f} [{metrics['accuracy']['ci_lower']:.4f}, {metrics['accuracy']['ci_upper']:.4f}]",
                    'Bootstrap_AUROC': f"{metrics['auroc']['mean']:.4f} [{metrics['auroc']['ci_lower']:.4f}, {metrics['auroc']['ci_upper']:.4f}]",
                    'Bootstrap_Precision': f"{metrics['precision']['mean']:.4f} [{metrics['precision']['ci_lower']:.4f}, {metrics['precision']['ci_upper']:.4f}]",
                    'Bootstrap_Recall': f"{metrics['recall']['mean']:.4f} [{metrics['recall']['ci_lower']:.4f}, {metrics['recall']['ci_upper']:.4f}]",
                    'Bootstrap_F1': f"{metrics['f1']['mean']:.4f} [{metrics['f1']['ci_lower']:.4f}, {metrics['f1']['ci_upper']:.4f}]"
                })
        
        bootstrap_df = pd.DataFrame(bootstrap_summary)
        bootstrap_path = os.path.join(self.output_dir, 'bootstrap_results_summary.csv')
        bootstrap_df.to_csv(bootstrap_path, index=False)
        print(f"   💾 Bootstrap results: {bootstrap_path}")
        
        return cv_df, bootstrap_df
    
    def plot_performance_comparison(self, bootstrap_results):
        """Create performance comparison plots"""
        
        print("\n" + "="*70)
        print("CREATING PERFORMANCE VISUALIZATIONS")
        print("="*70)
        
        # Extract data for plotting
        models = []
        aurocs = []
        auroc_cis = []
        
        for model_name, metrics in bootstrap_results.items():
            if metrics:
                models.append(model_name)
                aurocs.append(metrics['auroc']['mean'])
                auroc_cis.append([
                    metrics['auroc']['mean'] - metrics['auroc']['ci_lower'],
                    metrics['auroc']['ci_upper'] - metrics['auroc']['mean']
                ])
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: AUROC comparison
        ax1 = axes[0]
        y_pos = np.arange(len(models))
        ax1.barh(y_pos, aurocs, xerr=np.array(auroc_cis).T, capsize=5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(models)
        ax1.set_xlabel('AUROC')
        ax1.set_title('Model Performance Comparison (AUROC)')
        ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.3)
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Multi-metric comparison
        ax2 = axes[1]
        metrics_to_plot = ['accuracy', 'auroc', 'precision', 'recall', 'f1']
        metric_data = {metric: [] for metric in metrics_to_plot}
        
        for model_name in models:
            for metric in metrics_to_plot:
                metric_data[metric].append(bootstrap_results[model_name][metric]['mean'])
        
        x = np.arange(len(metrics_to_plot))
        width = 0.12
        
        for i, model_name in enumerate(models):
            values = [metric_data[m][i] for m in metrics_to_plot]
            ax2.bar(x + i*width, values, width, label=model_name)
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Multi-Metric Performance Comparison')
        ax2.set_xticks(x + width * len(models) / 2)
        ax2.set_xticklabels([m.capitalize() for m in metrics_to_plot])
        ax2.legend(loc='lower right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'figures', 'performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Performance plot: {plot_path}")
        plt.close()
    
    def generate_final_report(self, cv_df, bootstrap_df):
        """Generate comprehensive training report"""
        
        report_path = os.path.join(self.output_dir, 'training_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"MODEL TRAINING REPORT - TOP {self.n_features} GENES\n")
            f.write("="*70 + "\n\n")
            
            f.write("CROSS-VALIDATION RESULTS (5-FOLD):\n")
            f.write("-"*70 + "\n")
            f.write(cv_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("BOOTSTRAP RESULTS (100 ITERATIONS, 95% CI):\n")
            f.write("-"*70 + "\n")
            f.write(bootstrap_df.to_string(index=False))
            f.write("\n\n")
            
            # Find best model
            best_model = bootstrap_df.loc[bootstrap_df['Bootstrap_AUROC'].apply(
                lambda x: float(x.split('[')[0])
            ).idxmax(), 'Model']
            
            f.write(f"BEST PERFORMING MODEL: {best_model}\n")
        
        print(f"\n   📄 Training report: {report_path}")


def main():
    """Main training pipeline"""
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*18 + "MODEL TRAINING PIPELINE" + " "*25 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    # Train models for different feature sets
    for n_features in [50, 100, 150]:
        print("\n" + "█"*70)
        print(f"{'█'*5} TRAINING MODELS WITH TOP {n_features} GENES {' '*(55-len(str(n_features)))}{'█'*5}")
        print("█"*70)
        
        trainer = ModelTrainer(n_features=n_features)
        
        # Load data
        if not trainer.load_data():
            continue
        
        # Prepare data
        X, y = trainer.prepare_data()
        
        # Cross-validation
        cv_results = trainer.train_with_cv(X, y, use_smote=True)
        
        # Bootstrap confidence intervals
        bootstrap_results = trainer.train_final_models(X, y)
        
        # Create summary
        cv_df, bootstrap_df = trainer.create_results_summary(cv_results, bootstrap_results)
        
        # Visualizations
        trainer.plot_performance_comparison(bootstrap_results)
        
        # Final report
        trainer.generate_final_report(cv_df, bootstrap_df)
    
    print("\n" + "="*70)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*70)
    print("\nTrained models with:")
    print("  • 50 genes")
    print("  • 100 genes")
    print("  • 150 genes")
    print("\nNext step: Run 06_model_interpretation.py for SHAP analysis")
    print("\n")


if __name__ == "__main__":
    main()