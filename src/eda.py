"""
Exploratory Data Analysis that actually explores.
No superficial plots allowed.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class FraudEDA:
    """Comprehensive EDA that leaves no stone unturned."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.set_visualization_style()
        
    def set_visualization_style(self):
        """Professional plotting style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
    def class_distribution_analysis(self) -> Dict[str, Any]:
        """Quantify the imbalance like a statistician."""
        class_counts = self.df['class'].value_counts()
        class_percent = self.df['class'].value_counts(normalize=True) * 100
        
        print("=== CLASS DISTRIBUTION ANALYSIS ===")
        print(f"Total samples: {len(self.df)}")
        print(f"Non-fraud (0): {class_counts[0]:,} ({class_percent[0]:.2f}%)")
        print(f"Fraud (1): {class_counts[1]:,} ({class_percent[1]:.2f}%)")
        print(f"Imbalance ratio: {class_counts[0]/class_counts[1]:.1f}:1")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        sns.countplot(data=self.df, x='class', ax=axes[0])
        axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Class (0=Non-Fraud, 1=Fraud)')
        axes[0].set_ylabel('Count')
        
        # Add percentage labels
        total = len(self.df)
        for p in axes[0].patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height() + total * 0.01
            axes[0].annotate(percentage, (x, y), ha='center')
        
        # Pie chart
        axes[1].pie(class_counts, labels=['Non-Fraud', 'Fraud'], 
                   autopct='%1.1f%%', startangle=90, explode=(0.05, 0.1))
        axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('notebooks/class_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return {
            'counts': class_counts.to_dict(),
            'percentages': class_percent.to_dict(),
            'imbalance_ratio': class_counts[0] / class_counts[1]
        }
    
    def univariate_analysis(self, numerical_cols: list = None):
        """Analyze distributions of key variables."""
        if numerical_cols is None:
            numerical_cols = ['purchase_value', 'age']
        
        print("=== UNIVARIATE ANALYSIS ===")
        
        # Numerical features
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols):
            if idx >= len(axes):
                break
                
            # Distribution plot
            sns.histplot(data=self.df, x=col, kde=True, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
            axes[idx].set_xlabel(col)
            
            # Stats
            stats = self.df[col].describe()
            print(f"\n{col} Statistics:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std: {stats['std']:.2f}")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  25%: {stats['25%']:.2f}")
            print(f"  50%: {stats['50%']:.2f}")
            print(f"  75%: {stats['75%']:.2f}")
            print(f"  Max: {stats['max']:.2f}")
            
            # Box plot
            sns.boxplot(data=self.df, y=col, ax=axes[idx+3])
            axes[idx+3].set_title(f'Box Plot of {col}', fontweight='bold')
            axes[idx+3].set_ylabel(col)
        
        plt.tight_layout()
        plt.savefig('notebooks/univariate_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Categorical features
        cat_cols = ['source', 'browser', 'sex']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, col in enumerate(cat_cols):
            value_counts = self.df[col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add count labels
            for p in axes[idx].patches:
                axes[idx].annotate(f'{int(p.get_height()):,}', 
                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha='center', va='center', xytext=(0, 10), 
                                 textcoords='offset points')
            
            print(f"\n{col} Distribution:")
            for val, count in value_counts.items():
                print(f"  {val}: {count:,} ({count/len(self.df)*100:.1f}%)")
        
        plt.tight_layout()
        plt.savefig('notebooks/categorical_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def bivariate_analysis(self):
        """Analyze relationships between features and target."""
        print("=== BIVARIATE ANALYSIS ===")
        
        # Numerical vs Target
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Purchase Value by Class
        sns.boxplot(data=self.df, x='class', y='purchase_value', ax=axes[0,0])
        axes[0,0].set_title('Purchase Value by Fraud Status', fontweight='bold')
        axes[0,0].set_xlabel('Class (0=Non-Fraud, 1=Fraud)')
        axes[0,0].set_ylabel('Purchase Value')
        
        # Age by Class
        sns.boxplot(data=self.df, x='class', y='age', ax=axes[0,1])
        axes[0,1].set_title('Age by Fraud Status', fontweight='bold')
        axes[0,1].set_xlabel('Class (0=Non-Fraud, 1=Fraud)')
        axes[0,1].set_ylabel('Age')
        
        # Categorical vs Target - Source
        source_fraud_rate = self.df.groupby('source')['class'].mean().sort_values()
        sns.barplot(x=source_fraud_rate.index, y=source_fraud_rate.values, ax=axes[1,0])
        axes[1,0].set_title('Fraud Rate by Source', fontweight='bold')
        axes[1,0].set_xlabel('Source')
        axes[1,0].set_ylabel('Fraud Rate')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Browser vs Target
        browser_fraud_rate = self.df.groupby('browser')['class'].mean().sort_values()
        sns.barplot(x=browser_fraud_rate.index, y=browser_fraud_rate.values, ax=axes[1,1])
        axes[1,1].set_title('Fraud Rate by Browser', fontweight='bold')
        axes[1,1].set_xlabel('Browser')
        axes[1,1].set_ylabel('Fraud Rate')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('notebooks/bivariate_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Correlation matrix (numerical only)
        numerical_df = self.df.select_dtypes(include=[np.number])
        if len(numerical_df.columns) > 1:
            corr_matrix = numerical_df.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=1, 
                       cbar_kws={"shrink": 0.8})
            plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('notebooks/correlation_matrix.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print("\nCorrelation with target (class):")
            for col in corr_matrix.columns:
                if col != 'class':
                    corr = corr_matrix.loc[col, 'class']
                    print(f"  {col}: {corr:.4f}")
    
    def temporal_analysis(self):
        """Analyze time-based patterns."""
        print("=== TEMPORAL ANALYSIS ===")
        
        # Extract hour and day
        self.df['purchase_hour'] = self.df['purchase_time'].dt.hour
        self.df['purchase_day'] = self.df['purchase_time'].dt.dayofweek
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Fraud by hour
        hour_fraud_rate = self.df.groupby('purchase_hour')['class'].mean()
        sns.lineplot(x=hour_fraud_rate.index, y=hour_fraud_rate.values, 
                    marker='o', ax=axes[0])
        axes[0].fill_between(hour_fraud_rate.index, 0, hour_fraud_rate.values, alpha=0.3)
        axes[0].set_title('Fraud Rate by Hour of Day', fontweight='bold')
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Fraud Rate')
        axes[0].grid(True, alpha=0.3)
        
        # Fraud by day of week
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_fraud_rate = self.df.groupby('purchase_day')['class'].mean()
        sns.barplot(x=day_names, y=day_fraud_rate.values, ax=axes[1])
        axes[1].set_title('Fraud Rate by Day of Week', fontweight='bold')
        axes[1].set_xlabel('Day of Week')
        axes[1].set_ylabel('Fraud Rate')
        
        plt.tight_layout()
        plt.savefig('notebooks/temporal_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Remove temporary columns
        self.df.drop(['purchase_hour', 'purchase_day'], axis=1, inplace=True)
    
    def comprehensive_eda(self):
        """Execute full EDA pipeline."""
        print("=== STARTING COMPREHENSIVE EDA ===")
        
        # 1. Class distribution
        class_info = self.class_distribution_analysis()
        
        # 2. Univariate analysis
        self.univariate_analysis()
        
        # 3. Bivariate analysis
        self.bivariate_analysis()
        
        # 4. Temporal analysis
        self.temporal_analysis()
        
        print("\n=== EDA COMPLETE ===")
        return class_info