"""
Model explainability module for fraud detection.
Implements SHAP analysis and feature importance visualization.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')


class FraudModelExplainer:
    """
    Model explainability for fraud detection using SHAP and feature importance.
    """
    
    def __init__(self, model, feature_names: List[str], random_state: int = 42):
        """
        Initialize model explainer.
        
        Args:
            model: Trained model
            feature_names (list): List of feature names
            random_state (int): Random seed for reproducibility
        """
        self.model = model
        self.feature_names = feature_names
        self.random_state = random_state
        self.explainer = None
        self.shap_values = None
        self.insights = {}
        
    def extract_builtin_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Extract built-in feature importance from model.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        print("\n" + "="*80)
        print("BUILT-IN FEATURE IMPORTANCE")
        print("="*80)
        
        importance_df = None
        
        try:
            # Try different methods based on model type
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based models (Random Forest, XGBoost, etc.)
                importances = self.model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_type = "Gini Importance"
                
            elif hasattr(self.model, 'coef_'):
                # Linear models (Logistic Regression)
                if len(self.model.coef_.shape) == 2:
                    # Multi-class
                    importances = np.abs(self.model.coef_[0])
                else:
                    # Binary
                    importances = np.abs(self.model.coef_)
                
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_type = "Coefficient Magnitude"
                
            else:
                print("⚠️  Model doesn't have built-in feature importance")
                return None
            
            # Plot top features
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel(f'Feature Importance ({importance_type})', fontsize=12)
            plt.title(f'Top {top_n} Features - Built-in Importance', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add importance values
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{importance:.4f}', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.show()
            
            # Display importance table
            print(f"\n📊 {importance_type} (Top {top_n} features):")
            print(top_features[['feature', 'importance']].to_string(index=False))
            
            self.insights['builtin_importance'] = {
                'type': importance_type,
                'top_features': top_features.to_dict('records'),
                'all_features': importance_df.to_dict('records')
            }
            
            return importance_df
            
        except Exception as e:
            print(f"✗ Error extracting built-in importance: {e}")
            return None
    
    def compute_shap_values(self, X: pd.DataFrame, 
                           sample_size: int = 1000) -> None:
        """
        Compute SHAP values for model explanations.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            sample_size (int): Number of samples for SHAP computation
        """
        print("\n" + "="*80)
        print("COMPUTING SHAP VALUES")
        print("="*80)
        print("SHAP (SHapley Additive exPlanations) values explain individual predictions")
        
        try:
            # Sample data for faster computation
            if len(X) > sample_size:
                X_sample = X.sample(sample_size, random_state=self.random_state)
                print(f"📊 Using {sample_size:,} samples for SHAP computation "
                      f"({sample_size/len(X)*100:.1f}% of data)")
            else:
                X_sample = X
                print(f"📊 Using all {len(X_sample):,} samples for SHAP computation")
            
            # Create explainer based on model type
            model_type = str(type(self.model)).lower()
            
            if 'xgboost' in model_type or 'lgbm' in model_type or 'randomforest' in model_type:
                # Tree-based models
                print("🌳 Using TreeExplainer for tree-based model")
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(X_sample)
                
                # Handle multi-output SHAP values
                if isinstance(self.shap_values, list):
                    # For binary classification, get SHAP values for positive class
                    self.shap_values = self.shap_values[1]
                    
            elif 'linear' in model_type or 'logistic' in model_type:
                # Linear models
                print("📈 Using LinearExplainer for linear model")
                self.explainer = shap.LinearExplainer(self.model, X_sample)
                self.shap_values = self.explainer.shap_values(X_sample)
                
            else:
                # Kernel SHAP for any model (slower)
                print("🔧 Using KernelExplainer (generic, may be slow)")
                self.explainer = shap.KernelExplainer(self.model.predict_proba, X_sample)
                self.shap_values = self.explainer.shap_values(X_sample)
                
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
            
            print(f"✅ SHAP values computed successfully")
            print(f"   Shape: {self.shap_values.shape}")
            
        except Exception as e:
            print(f"✗ Error computing SHAP values: {e}")
            raise
    
    def plot_shap_summary(self, X: pd.DataFrame, 
                         max_display: int = 20) -> None:
        """
        Plot SHAP summary plot (global feature importance).
        
        Args:
            X (pd.DataFrame): Feature dataframe
            max_display (int): Maximum features to display
        """
        if self.shap_values is None:
            print("⚠️  SHAP values not computed. Run compute_shap_values first.")
            return
        
        print("\n" + "="*80)
        print("SHAP SUMMARY PLOT (Global Feature Importance)")
        print("="*80)
        print("Shows feature importance and impact direction across all predictions")
        
        try:
            # Sample for visualization
            if len(X) > 1000:
                X_sample = X.sample(1000, random_state=self.random_state)
                shap_values_sample = self.shap_values[X_sample.index]
            else:
                X_sample = X
                shap_values_sample = self.shap_values
            
            # Create summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values_sample, X_sample, 
                            feature_names=self.feature_names,
                            max_display=max_display, show=False)
            
            plt.title('SHAP Summary Plot - Global Feature Importance', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Calculate global SHAP importance
            shap_importance = np.abs(self.shap_values).mean(axis=0)
            shap_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)
            
            # Display top features
            print(f"\n📊 SHAP Feature Importance (Top {max_display}):")
            print(shap_importance_df.head(max_display)[['feature', 'shap_importance']]
                  .to_string(index=False))
            
            # Store insights
            self.insights['shap_summary'] = {
                'top_features': shap_importance_df.head(max_display).to_dict('records'),
                'global_importance': shap_importance_df.to_dict('records')
            }
            
        except Exception as e:
            print(f"✗ Error creating SHAP summary plot: {e}")
    
    def plot_shap_force_plots(self, X: pd.DataFrame, y: pd.Series,
                             n_cases: int = 3) -> None:
        """
        Plot SHAP force plots for individual predictions.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.Series): True labels
            n_cases (int): Number of cases to visualize
        """
        if self.shap_values is None:
            print("⚠️  SHAP values not computed. Run compute_shap_values first.")
            return
        
        print("\n" + "="*80)
        print("SHAP FORCE PLOTS (Individual Predictions)")
        print("="*80)
        print("Shows how each feature contributes to individual predictions")
        
        try:
            # Get model predictions
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            
            # Find interesting cases
            cases = self._find_interesting_cases(X, y, y_pred, y_pred_proba, n_cases)
            
            for i, case in enumerate(cases):
                print(f"\n{'='*60}")
                print(f"CASE {i+1}: {case['type'].upper()}")
                print(f"{'='*60}")
                print(f"Predicted probability: {case['pred_prob']:.4f}")
                print(f"Actual class: {case['actual']} ({'Fraud' if case['actual'] == 1 else 'Legitimate'})")
                print(f"Predicted class: {case['pred']} ({'Fraud' if case['pred'] == 1 else 'Legitimate'})")
                print(f"Sample index: {case['index']}")
                
                # Create force plot
                plt.figure(figsize=(14, 4))
                shap.force_plot(
                    self.explainer.expected_value[1] if hasattr(self.explainer.expected_value, '__len__') 
                    else self.explainer.expected_value,
                    self.shap_values[case['index']],
                    X.iloc[case['index']],
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )
                
                plt.title(f"SHAP Force Plot - {case['type']} (Index: {case['index']})", 
                         fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.show()
                
                # Print feature contributions
                print("\n📋 Top contributing features:")
                feature_contributions = pd.DataFrame({
                    'feature': self.feature_names,
                    'value': X.iloc[case['index']].values,
                    'shap_value': self.shap_values[case['index']]
                }).sort_values('shap_value', key=abs, ascending=False)
                
                print(feature_contributions.head(10).to_string(index=False))
                
                # Store case insights
                case_key = f"case_{i+1}_{case['type']}"
                self.insights[case_key] = {
                    'type': case['type'],
                    'index': case['index'],
                    'actual': case['actual'],
                    'predicted': case['pred'],
                    'probability': case['pred_prob'],
                    'top_features': feature_contributions.head(10).to_dict('records')
                }
                
        except Exception as e:
            print(f"✗ Error creating force plots: {e}")
    
    def _find_interesting_cases(self, X: pd.DataFrame, y: pd.Series,
                               y_pred: np.ndarray, y_pred_proba: np.ndarray,
                               n_cases: int) -> List[Dict[str, Any]]:
        """
        Find interesting cases for SHAP visualization.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.Series): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities
            n_cases (int): Number of cases to find
            
        Returns:
            list: List of case dictionaries
        """
        cases = []
        
        # Find true positives (correctly predicted fraud)
        tp_mask = (y == 1) & (y_pred == 1)
        if tp_mask.any():
            tp_indices = X[tp_mask].index
            tp_probs = y_pred_proba[tp_mask]
            # Choose TP with highest confidence
            tp_idx = tp_indices[tp_probs.argmax()]
            cases.append({
                'type': 'true_positive',
                'index': tp_idx,
                'actual': y[tp_idx],
                'pred': y_pred[tp_idx],
                'pred_prob': y_pred_proba[tp_idx]
            })
        
        # Find false positives (legitimate flagged as fraud)
        fp_mask = (y == 0) & (y_pred == 1)
        if fp_mask.any():
            fp_indices = X[fp_mask].index
            fp_probs = y_pred_proba[fp_mask]
            # Choose FP with highest confidence (most confusing)
            fp_idx = fp_indices[fp_probs.argmax()]
            cases.append({
                'type': 'false_positive',
                'index': fp_idx,
                'actual': y[fp_idx],
                'pred': y_pred[fp_idx],
                'pred_prob': y_pred_proba[fp_idx]
            })
        
        # Find false negatives (missed fraud)
        fn_mask = (y == 1) & (y_pred == 0)
        if fn_mask.any():
            fn_indices = X[fn_mask].index
            fn_probs = y_pred_proba[fn_mask]
            # Choose FN with highest probability (closest to threshold)
            fn_idx = fn_indices[fn_probs.argmax()]
            cases.append({
                'type': 'false_negative',
                'index': fn_idx,
                'actual': y[fn_idx],
                'pred': y_pred[fn_idx],
                'pred_prob': y_pred_proba[fn_idx]
            })
        
        # Find true negatives (correctly predicted legitimate)
        tn_mask = (y == 0) & (y_pred == 0)
        if tn_mask.any() and len(cases) < n_cases:
            tn_indices = X[tn_mask].index
            tn_probs = y_pred_proba[tn_mask]
            # Choose TN with lowest probability (clearly legitimate)
            tn_idx = tn_indices[tn_probs.argmin()]
            cases.append({
                'type': 'true_negative',
                'index': tn_idx,
                'actual': y[tn_idx],
                'pred': y_pred[tn_idx],
                'pred_prob': y_pred_proba[tn_idx]
            })
        
        # If still need more cases, add borderline predictions
        if len(cases) < n_cases:
            borderline_mask = (y_pred_proba > 0.4) & (y_pred_proba < 0.6)
            if borderline_mask.any():
                borderline_indices = X[borderline_mask].index
                for idx in borderline_indices[:n_cases - len(cases)]:
                    cases.append({
                        'type': 'borderline',
                        'index': idx,
                        'actual': y[idx],
                        'pred': y_pred[idx],
                        'pred_prob': y_pred_proba[idx]
                    })
        
        return cases[:n_cases]
    
    def compare_feature_importance(self, builtin_importance: pd.DataFrame = None,
                                  top_n: int = 10) -> pd.DataFrame:
        """
        Compare built-in feature importance with SHAP importance.
        
        Args:
            builtin_importance (pd.DataFrame): Built-in importance dataframe
            top_n (int): Number of top features to compare
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        if self.shap_values is None:
            print("⚠️  SHAP values not computed. Run compute_shap_values first.")
            return None
        
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE COMPARISON")
        print("="*80)
        print("Comparing built-in importance with SHAP importance")
        
        try:
            # Calculate SHAP importance
            shap_importance = np.abs(self.shap_values).mean(axis=0)
            shap_df = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)
            
            comparison = None
            
            if builtin_importance is not None:
                # Merge with built-in importance
                comparison = pd.merge(
                    builtin_importance[['feature', 'importance']].rename(
                        columns={'importance': 'builtin_importance'}
                    ),
                    shap_df[['feature', 'shap_importance']],
                    on='feature',
                    how='inner'
                )
                
                # Normalize importances for comparison
                comparison['builtin_normalized'] = (
                    comparison['builtin_importance'] / comparison['builtin_importance'].sum()
                )
                comparison['shap_normalized'] = (
                    comparison['shap_importance'] / comparison['shap_importance'].sum()
                )
                
                # Calculate rank difference
                comparison['builtin_rank'] = comparison['builtin_normalized'].rank(
                    ascending=False
                )
                comparison['shap_rank'] = comparison['shap_normalized'].rank(
                    ascending=False
                )
                comparison['rank_difference'] = abs(
                    comparison['builtin_rank'] - comparison['shap_rank']
                )
                
                # Get top features by SHAP importance
                top_comparison = comparison.head(top_n).copy()
                
                # Plot comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Built-in importance
                top_builtin = top_comparison.sort_values('builtin_normalized', ascending=True)
                ax1.barh(range(len(top_builtin)), top_builtin['builtin_normalized'])
                ax1.set_yticks(range(len(top_builtin)))
                ax1.set_yticklabels(top_builtin['feature'])
                ax1.set_xlabel('Normalized Importance', fontsize=12)
                ax1.set_title('Built-in Feature Importance', fontsize=14, fontweight='bold')
                
                # SHAP importance
                top_shap = top_comparison.sort_values('shap_normalized', ascending=True)
                ax2.barh(range(len(top_shap)), top_shap['shap_normalized'])
                ax2.set_yticks(range(len(top_shap)))
                ax2.set_yticklabels(top_shap['feature'])
                ax2.set_xlabel('Normalized Importance', fontsize=12)
                ax2.set_title('SHAP Feature Importance', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                plt.show()
                
                # Display comparison table
                print(f"\n📊 Feature Importance Comparison (Top {top_n} by SHAP):")
                display_cols = ['feature', 'builtin_rank', 'shap_rank', 
                              'rank_difference', 'builtin_normalized', 'shap_normalized']
                print(top_comparison[display_cols].to_string(index=False))
                
                # Identify discrepancies
                large_differences = top_comparison[top_comparison['rank_difference'] > 5]
                if not large_differences.empty:
                    print(f"\n⚠️  Large importance discrepancies detected:")
                    for _, row in large_differences.iterrows():
                        print(f"  {row['feature']}: Built-in rank {int(row['builtin_rank'])}, "
                              f"SHAP rank {int(row['shap_rank'])}")
                
                self.insights['importance_comparison'] = {
                    'comparison': comparison.to_dict('records'),
                    'top_features': top_comparison.to_dict('records'),
                    'discrepancies': large_differences.to_dict('records') if not large_differences.empty else []
                }
            
            return comparison
            
        except Exception as e:
            print(f"✗ Error comparing feature importance: {e}")
            return None
    
    def generate_business_recommendations(self, X: pd.DataFrame, y: pd.Series,
                                         threshold: float = 0.5) -> Dict[str, List[str]]:
        """
        Generate actionable business recommendations from SHAP analysis.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.Series): True labels
            threshold (float): Classification threshold
            
        Returns:
            dict: Business recommendations
        """
        if self.shap_values is None:
            print("⚠️  SHAP values not computed. Run compute_shap_values first.")
            return {}
        
        print("\n" + "="*80)
        print("BUSINESS RECOMMENDATIONS")
        print("="*80)
        print("Actionable insights derived from model explainability")
        
        try:
            recommendations = {
                'risk_factors': [],
                'verification_suggestions': [],
                'monitoring_recommendations': [],
                'feature_engineering_suggestions': []
            }
            
            # Get model predictions
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # 1. Analyze top risk factors
            shap_importance = np.abs(self.shap_values).mean(axis=0)
            top_risk_factors = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False).head(5)
            
            print("\n🔴 TOP 5 RISK FACTORS (based on SHAP analysis):")
            for _, row in top_risk_factors.iterrows():
                feature = row['feature']
                importance = row['shap_importance']
                
                # Get feature statistics for fraud vs non-fraud
                fraud_mean = X[y == 1][feature].mean()
                nonfraud_mean = X[y == 0][feature].mean()
                
                print(f"  {feature}:")
                print(f"    Importance: {importance:.4f}")
                print(f"    Fraud avg: {fraud_mean:.2f}, Non-fraud avg: {nonfraud_mean:.2f}")
                
                # Generate recommendation based on feature
                if 'hour' in feature.lower():
                    recommendations['risk_factors'].append(
                        f"Transactions during hour {int(fraud_mean)} have highest fraud risk"
                    )
                    recommendations['verification_suggestions'].append(
                        f"Implement additional verification for transactions between {int(fraud_mean)}:00"
                    )
                elif 'country' in feature.lower():
                    recommendations['risk_factors'].append(
                        f"Country-based features are among top risk indicators"
                    )
                    recommendations['monitoring_recommendations'].append(
                        f"Monitor transactions from countries with high fraud rates"
                    )
                elif 'transaction' in feature.lower():
                    recommendations['risk_factors'].append(
                        f"Transaction frequency/velocity is key fraud indicator"
                    )
                    recommendations['verification_suggestions'].append(
                        f"Flag users with >3 transactions per hour for manual review"
                    )
            
            # 2. Analyze feature interactions
            print("\n🔍 KEY INSIGHTS FROM FEATURE INTERACTIONS:")
            
            # Analyze time since signup for new users
            if 'time_since_signup_hours' in X.columns:
                new_user_fraud_rate = y[X['time_since_signup_hours'] <= 24].mean()
                if new_user_fraud_rate > 0.1:
                    print(f"  New users (≤24h) have {new_user_fraud_rate*100:.1f}% fraud rate")
                    recommendations['verification_suggestions'].append(
                        "Implement enhanced verification for new users (first 24 hours)"
                    )
            
            # Analyze device sharing
            if 'unique_users_per_device' in X.columns:
                device_sharing_fraud = X[y == 1]['unique_users_per_device'].mean()
                if device_sharing_fraud > 1.5:
                    print(f"  Fraudulent accounts average {device_sharing_fraud:.1f} users per device")
                    recommendations['monitoring_recommendations'].append(
                        "Monitor devices with multiple user accounts"
                    )
            
            # 3. Threshold optimization analysis
            print("\n🎯 PREDICTION THRESHOLD ANALYSIS:")
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            print(f"  Current threshold: {threshold}")
            print(f"  False positives: {fp:,} ({fp/len(y)*100:.2f}%)")
            print(f"  False negatives: {fn:,} ({fn/len(y)*100:.2f}%)")
            
            if fp > fn * 3:
                recommendations['feature_engineering_suggestions'].append(
                    "Consider lowering prediction threshold to reduce false negatives"
                )
            elif fn > fp * 3:
                recommendations['feature_engineering_suggestions'].append(
                    "Consider increasing prediction threshold to reduce false positives"
                )
            
            # 4. Generate specific recommendations
            print("\n💡 ACTIONABLE RECOMMENDATIONS:")
            
            # Recommendation 1: Real-time monitoring
            recommendations['monitoring_recommendations'].append(
                "Implement real-time monitoring dashboard with top 5 risk factors"
            )
            print("  1. Implement real-time monitoring dashboard with top 5 risk factors")
            
            # Recommendation 2: Rule-based system
            top_feature = top_risk_factors.iloc[0]['feature']
            recommendations['verification_suggestions'].append(
                f"Create rule: Additional verification when {top_feature} exceeds 90th percentile"
            )
            print(f"  2. Create rule: Additional verification when {top_feature} exceeds 90th percentile")
            
            # Recommendation 3: Feature enhancement
            recommendations['feature_engineering_suggestions'].append(
                "Add geolocation velocity features (distance/time between transactions)"
            )
            print("  3. Add geolocation velocity features (distance/time between transactions)")
            
            # Store recommendations
            self.insights['business_recommendations'] = recommendations
            
            return recommendations
            
        except Exception as e:
            print(f"✗ Error generating business recommendations: {e}")
            return {}
    
    def generate_explainability_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive explainability report.
        
        Returns:
            dict: Complete explainability insights
        """
        report = {
            'insights': self.insights,
            'summary': {
                'shap_values_computed': self.shap_values is not None,
                'explainer_available': self.explainer is not None,
                'total_features': len(self.feature_names),
                'analysis_completed': bool(self.insights)
            }
        }
        
        print("\n" + "="*80)
        print("EXPLAINABILITY REPORT SUMMARY")
        print("="*80)
        
        for key, value in report['summary'].items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        if 'business_recommendations' in self.insights:
            print("\n📋 BUSINESS RECOMMENDATIONS SUMMARY:")
            for category, recs in self.insights['business_recommendations'].items():
                if recs:
                    print(f"\n{category.replace('_', ' ').title()}:")
                    for i, rec in enumerate(recs[:3], 1):  
                        print(f"  {i}. {rec}")
        
        return report