"""
Exploratory Data Analysis module for fraud detection.
Implements comprehensive univariate and bivariate analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FraudEDA:
    """
    Comprehensive EDA for fraud detection datasets.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize EDA analyzer.
        
        Args:
            figsize (tuple): Default figure size
        """
        self.figsize = figsize
        self.insights = {}
        
    def analyze_class_distribution(self, df: pd.DataFrame, 
                                  target_col: str = 'class') -> Dict[str, Any]:
        """
        Analyze class distribution for imbalance.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            
        Returns:
            dict: Class distribution statistics
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        class_counts = df[target_col].value_counts()
        class_percentages = df[target_col].value_counts(normalize=True) * 100
        
        stats = {
            'class_counts': class_counts.to_dict(),
            'class_percentages': class_percentages.to_dict(),
            'total_samples': len(df),
            'imbalance_ratio': class_counts[0] / class_counts[1] if len(class_counts) > 1 else 0
        }
        
        # Plot class distribution
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        sns.countplot(data=df, x=target_col, ax=axes[0])
        axes[0].set_title(f'Class Distribution (Count)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Class (0=Legitimate, 1=Fraud)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        
        # Add count labels
        for i, count in enumerate(class_counts):
            axes[0].text(i, count + count*0.01, f'{count:,}', 
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Pie chart
        axes[1].pie(class_counts, labels=[f'Legitimate\n({class_counts[0]:,})', 
                                         f'Fraud\n({class_counts[1]:,})'],
                   autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
        axes[1].set_title(f'Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*60)
        print(f"Total samples: {stats['total_samples']:,}")
        print(f"Legitimate transactions: {class_counts[0]:,} ({class_percentages[0]:.2f}%)")
        print(f"Fraudulent transactions: {class_counts[1]:,} ({class_percentages[1]:.2f}%)")
        print(f"Imbalance ratio: {stats['imbalance_ratio']:.2f}:1")
        print(f"Fraud rate: {class_percentages[1]:.4f}%")
        
        if class_percentages[1] < 1:
            print("\n⚠️  WARNING: Highly imbalanced dataset (<1% fraud)")
            print("   Consider using specialized techniques for imbalanced data")
        
        self.insights['class_distribution'] = stats
        return stats
    
    def univariate_analysis(self, df: pd.DataFrame, 
                           numerical_cols: List[str] = None,
                           categorical_cols: List[str] = None,
                           max_categories: int = 10):
        """
        Perform univariate analysis on features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            numerical_cols (list): List of numerical columns
            categorical_cols (list): List of categorical columns
            max_categories (int): Max categories to display
        """
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print("\n" + "="*80)
        print("UNIVARIATE ANALYSIS")
        print("="*80)
        
        # Numerical features
        if numerical_cols:
            print(f"\n📊 NUMERICAL FEATURES ({len(numerical_cols)}):")
            
            # Calculate statistics
            num_stats = df[numerical_cols].describe().T
            num_stats['skewness'] = df[numerical_cols].skew()
            num_stats['kurtosis'] = df[numerical_cols].kurtosis()
            
            print(num_stats[['count', 'mean', 'std', 'min', '50%', 'max', 'skewness', 'kurtosis']]
                  .round(3))
            
            # Plot distributions
            n_cols = min(4, len(numerical_cols))
            n_rows = int(np.ceil(len(numerical_cols) / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
            axes = axes.flatten()
            
            for idx, col in enumerate(numerical_cols):
                if idx < len(axes):
                    # Histogram with KDE
                    sns.histplot(data=df, x=col, kde=True, ax=axes[idx], bins=50)
                    axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
                    axes[idx].set_xlabel(col, fontsize=10)
                    axes[idx].set_ylabel('Frequency', fontsize=10)
                    
                    # Add statistics text
                    stats_text = (f"Mean: {df[col].mean():.2f}\n"
                                 f"Std: {df[col].std():.2f}\n"
                                 f"Skew: {df[col].skew():.2f}")
                    axes[idx].text(0.95, 0.95, stats_text, 
                                  transform=axes[idx].transAxes,
                                  fontsize=9, verticalalignment='top',
                                  horizontalalignment='right',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Hide empty subplots
            for idx in range(len(numerical_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            self.insights['numerical_stats'] = num_stats
        
        # Categorical features
        if categorical_cols:
            print(f"\n📈 CATEGORICAL FEATURES ({len(categorical_cols)}):")
            
            cat_stats = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                cat_stats[col] = {
                    'unique_values': df[col].nunique(),
                    'top_category': value_counts.index[0] if len(value_counts) > 0 else None,
                    'top_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'top_percentage': (value_counts.iloc[0] / len(df) * 100) if len(value_counts) > 0 else 0
                }
            
            cat_stats_df = pd.DataFrame(cat_stats).T
            print(cat_stats_df.round(3))
            
            # Plot top categories
            n_cat = min(6, len(categorical_cols))
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(categorical_cols[:n_cat]):
                # Get top categories
                top_categories = df[col].value_counts().head(max_categories)
                
                # Bar plot
                bars = axes[idx].bar(range(len(top_categories)), top_categories.values)
                axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(col, fontsize=10)
                axes[idx].set_ylabel('Count', fontsize=10)
                axes[idx].set_xticks(range(len(top_categories)))
                axes[idx].set_xticklabels([str(x)[:15] for x in top_categories.index], 
                                         rotation=45, ha='right')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                  f'{height:,}', ha='center', va='bottom', fontsize=9)
            
            # Hide empty subplots
            for idx in range(len(categorical_cols[:n_cat]), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            self.insights['categorical_stats'] = cat_stats_df
    
    def bivariate_analysis(self, df: pd.DataFrame, 
                          target_col: str = 'class',
                          numerical_cols: List[str] = None,
                          categorical_cols: List[str] = None):
        """
        Perform bivariate analysis with target variable.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            numerical_cols (list): List of numerical columns
            categorical_cols (list): List of categorical columns
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        print("\n" + "="*80)
        print("BIVARIATE ANALYSIS WITH TARGET")
        print("="*80)
        
        # Numerical features vs target
        if numerical_cols is None:
            numerical_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                            if col != target_col]
        
        if numerical_cols:
            print(f"\n📈 NUMERICAL FEATURES vs TARGET ({len(numerical_cols)} features):")
            
            # Calculate statistics by class
            bivariate_stats = []
            for col in numerical_cols:
                stats = df.groupby(target_col)[col].agg(['mean', 'std', 'median', 'min', 'max'])
                stats['diff_means'] = stats.loc[1, 'mean'] - stats.loc[0, 'mean']
                stats['diff_pct'] = (stats.loc[1, 'mean'] - stats.loc[0, 'mean']) / stats.loc[0, 'mean'] * 100
                bivariate_stats.append(stats)
            
            # Plot distributions by class
            n_cols = min(3, len(numerical_cols))
            n_rows = int(np.ceil(len(numerical_cols) / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for idx, col in enumerate(numerical_cols):
                if idx < len(axes):
                    # Box plot by class
                    sns.boxplot(data=df, x=target_col, y=col, ax=axes[idx])
                    axes[idx].set_title(f'{col} by Class', fontsize=12, fontweight='bold')
                    axes[idx].set_xlabel('Class (0=Legitimate, 1=Fraud)', fontsize=10)
                    axes[idx].set_ylabel(col, fontsize=10)
            
            # Hide empty subplots
            for idx in range(len(numerical_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            # Correlation with target
            print("\n📊 CORRELATION WITH TARGET:")
            correlations = {}
            for col in numerical_cols:
                corr = df[[col, target_col]].corr().iloc[0, 1]
                correlations[col] = corr
            
            corr_df = pd.DataFrame({'correlation': correlations}).sort_values('correlation', 
                                                                             key=abs, 
                                                                             ascending=False)
            print(corr_df.round(4).head(10))
            
            self.insights['numerical_correlations'] = corr_df
        
        # Categorical features vs target
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            print(f"\n📊 CATEGORICAL FEATURES vs TARGET ({len(categorical_cols)} features):")
            
            # Plot categorical features
            n_cat = min(6, len(categorical_cols))
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(categorical_cols[:n_cat]):
                # Create contingency table
                contingency = pd.crosstab(df[col], df[target_col], normalize='index') * 100
                
                # Plot stacked bar chart
                contingency.plot(kind='bar', stacked=True, ax=axes[idx], 
                               color=['#2ecc71', '#e74c3c'])
                axes[idx].set_title(f'{col} vs Fraud Rate', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(col, fontsize=10)
                axes[idx].set_ylabel('Percentage (%)', fontsize=10)
                axes[idx].legend(['Legitimate', 'Fraud'])
                axes[idx].tick_params(axis='x', rotation=45)
                
                # Calculate and display fraud rate
                fraud_rate = df.groupby(col)[target_col].mean() * 100
                max_fraud = fraud_rate.max()
                min_fraud = fraud_rate.min()
                
                if max_fraud > 0:
                    axes[idx].text(0.02, 0.98, 
                                  f"Max fraud: {max_fraud:.1f}%\nMin fraud: {min_fraud:.1f}%",
                                  transform=axes[idx].transAxes,
                                  fontsize=9, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Hide empty subplots
            for idx in range(len(categorical_cols[:n_cat]), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            # Calculate chi-square statistics (simplified)
            print("\n📈 FRAUD RATE BY CATEGORY (Top 5 per feature):")
            for col in categorical_cols[:min(5, len(categorical_cols))]:
                fraud_rates = df.groupby(col)[target_col].mean().sort_values(ascending=False)
                print(f"\n{col}:")
                for category, rate in fraud_rates.head().items():
                    print(f"  {category}: {rate*100:.2f}% fraud")
    
    def correlation_analysis(self, df: pd.DataFrame, 
                            numerical_cols: List[str] = None):
        """
        Perform correlation analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            numerical_cols (list): List of numerical columns
        """
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        if len(numerical_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = df[numerical_cols].corr()
            
            # Plot heatmap
            plt.figure(figsize=(14, 12))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                       square=True, linewidths=.5, 
                       cbar_kws={"shrink": .8}, annot=True, fmt='.2f',
                       annot_kws={'size': 8})
            
            plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Identify high correlations
            print("\n🔍 HIGHLY CORRELATED FEATURES (|correlation| > 0.8):")
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))
            
            if high_corr_pairs:
                for feat1, feat2, corr in high_corr_pairs:
                    print(f"  {feat1} ↔ {feat2}: {corr:.3f}")
                print("\n⚠️  WARNING: Highly correlated features detected")
                print("   Consider feature selection or dimensionality reduction")
            else:
                print("  No highly correlated features found")
            
            self.insights['correlation_matrix'] = corr_matrix
            self.insights['high_correlation_pairs'] = high_corr_pairs
    
    def outlier_analysis(self, df: pd.DataFrame, 
                        numerical_cols: List[str] = None,
                        threshold: float = 3.0):
        """
        Detect and analyze outliers.
        
        Args:
            df (pd.DataFrame): Input dataframe
            numerical_cols (list): List of numerical columns
            threshold (float): Z-score threshold for outliers
        """
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print("\n" + "="*80)
        print("OUTLIER ANALYSIS")
        print("="*80)
        
        outlier_stats = {}
        
        for col in numerical_cols:
            # Calculate Z-scores
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > threshold
            
            if outliers.any():
                outlier_count = outliers.sum()
                outlier_pct = outlier_count / len(df) * 100
                
                outlier_stats[col] = {
                    'outlier_count': outlier_count,
                    'outlier_percentage': outlier_pct,
                    'min_value': df[col].min(),
                    'max_value': df[col].max(),
                    'q1': df[col].quantile(0.25),
                    'q3': df[col].quantile(0.75),
                    'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
                }
        
        if outlier_stats:
            outlier_df = pd.DataFrame(outlier_stats).T
            print("\nOutlier Statistics:")
            print(outlier_df[['outlier_count', 'outlier_percentage', 'min_value', 
                            'max_value', 'q1', 'q3', 'iqr']].round(3))
            
            # Plot boxplots for columns with outliers
            outlier_cols = list(outlier_stats.keys())[:min(6, len(outlier_stats))]
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(outlier_cols):
                sns.boxplot(y=df[col], ax=axes[idx], color='lightblue')
                axes[idx].set_title(f'{col} - Outliers: {outlier_stats[col]["outlier_count"]} '
                                  f'({outlier_stats[col]["outlier_percentage"]:.1f}%)', 
                                  fontsize=12, fontweight='bold')
                axes[idx].set_ylabel(col, fontsize=10)
                
                # Mark outliers
                q1 = outlier_stats[col]['q1']
                q3 = outlier_stats[col]['q3']
                iqr = outlier_stats[col]['iqr']
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                axes[idx].axhspan(lower_bound, upper_bound, alpha=0.2, color='green')
            
            # Hide empty subplots
            for idx in range(len(outlier_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            print("\n⚠️  WARNING: Outliers detected")
            print("   Consider robust scaling or outlier treatment")
        else:
            print("✓ No significant outliers detected (Z-score > 3)")
        
        self.insights['outlier_stats'] = outlier_stats if outlier_stats else {}
    
    def generate_eda_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive EDA report.
        
        Returns:
            dict: EDA insights and statistics
        """
        report = {
            'insights': self.insights,
            'summary': {
                'total_features_analyzed': len(self.insights.get('numerical_stats', [])) + 
                                         len(self.insights.get('categorical_stats', [])),
                'imbalance_ratio': self.insights.get('class_distribution', {}).get('imbalance_ratio', 0),
                'fraud_rate': self.insights.get('class_distribution', {}).get('class_percentages', {}).get(1, 0),
                'outlier_features': len(self.insights.get('outlier_stats', {})),
                'high_correlation_pairs': len(self.insights.get('high_correlation_pairs', []))
            }
        }
        
        print("\n" + "="*80)
        print("EDA SUMMARY REPORT")
        print("="*80)
        for key, value in report['summary'].items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return report