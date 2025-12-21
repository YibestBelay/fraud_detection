"""
Feature engineering that creates value without exploding memory.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')


class FraudFeatureEngineer:
    """Create meaningful features for fraud detection."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.optimize_memory()

    def optimize_memory(self):
        """Convert types to save memory."""
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].astype('category')
        for col in self.df.select_dtypes(include='int64').columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
        for col in self.df.select_dtypes(include='float64').columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')

    def create_time_based_features(self) -> pd.DataFrame:
        df = self.df.copy()
        # Ensure datetime
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        # Time-based features
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype('int8')
        df['time_since_signup'] = ((df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600).astype('float32')
        df['purchase_month'] = df['purchase_time'].dt.month.astype('int8')
        df['purchase_day'] = df['purchase_time'].dt.day.astype('int8')
        self.df = df
        return df

    def create_frequency_features(self) -> pd.DataFrame:
        df = self.df.copy()
        df = df.sort_values(['user_id', 'purchase_time'])
        df['transactions_last_1h'] = 0
        df['transactions_last_24h'] = 0
        df['transactions_last_7d'] = 0
        # Use rolling for efficiency if needed (here simple placeholder)
        self.df = df
        return df

    def create_aggregate_features(self) -> pd.DataFrame:
        df = self.df.copy()
        user_stats = (
            df.groupby('user_id')
            .agg({
                'purchase_value': ['mean', 'std', 'max', 'min'],
                'age': 'first',
                'source': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
                'browser': lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
            })
        )
        user_stats.columns = ['avg_purchase', 'std_purchase', 'max_purchase', 'min_purchase', 'user_age', 'common_source', 'common_browser']
        df = df.merge(user_stats, on='user_id', how='left')
        df['purchase_deviation'] = (df['purchase_value'] - df['avg_purchase']).astype('float32')
        df['purchase_zscore'] = ((df['purchase_value'] - df['avg_purchase']) / df['std_purchase'].replace(0, 1)).astype('float32')
        self.df = df
        return df

    def create_interaction_features(self) -> pd.DataFrame:
        df = self.df.copy()
        df['device_browser'] = (df['device_id'].astype(str) + '_' + df['browser'].astype(str)).astype('category')
        df['source_browser'] = (df['source'].astype(str) + '_' + df['browser'].astype(str)).astype('category')
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 50, 65, 100],
                                 labels=['0-18', '19-25', '26-35', '36-50', '51-65', '65+']).astype('category')
        df['purchase_group'] = pd.qcut(df['purchase_value'], q=5, labels=False).astype('int8')
        self.df = df
        return df

    def feature_engineering_pipeline(self) -> pd.DataFrame:
        """Execute complete feature engineering pipeline."""
        self.create_time_based_features()
        self.create_frequency_features()
        self.create_aggregate_features()
        self.create_interaction_features()
        return self.df


# """
# Feature engineering that creates value, not just columns.

# """
# import pandas as pd
# import numpy as np
# from datetime import timedelta
# import warnings

# warnings.filterwarnings('ignore')


# class FraudFeatureEngineer:
#     """Create meaningful features for fraud detection."""

#     def __init__(self, df: pd.DataFrame):
#         self.df = df.copy()

#     def create_time_based_features(self) -> pd.DataFrame:
#         """Engineer time-based features with domain knowledge."""
#         df = self.df.copy()

#         print("=== CREATING TIME-BASED FEATURES ===")

#         # 1. Hour of day
#         df['hour_of_day'] = df['purchase_time'].dt.hour

#         # 2. Day of week
#         df['day_of_week'] = df['purchase_time'].dt.dayofweek

#         # 3. Weekend flag
#         df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

#         # 4. Time since signup (in hours)
#         df['time_since_signup'] = (
#             df['purchase_time'] - df['signup_time']
#         ).dt.total_seconds() / 3600

#         # 5. Month and day
#         df['purchase_month'] = df['purchase_time'].dt.month
#         df['purchase_day'] = df['purchase_time'].dt.day

#         time_features = [
#             'hour_of_day',
#             'day_of_week',
#             'is_weekend',
#             'time_since_signup',
#             'purchase_month',
#             'purchase_day'
#         ]

#         print(f"Created {len(time_features)} time-based features: {time_features}")

#         # Validate
#         print("\nTime feature statistics:")
#         print(f"Average time since signup: {df['time_since_signup'].mean():.2f} hours")
#         print(f"Std time since signup: {df['time_since_signup'].std():.2f} hours")
#         print(f"Min time since signup: {df['time_since_signup'].min():.2f} hours")
#         print(f"Max time since signup: {df['time_since_signup'].max():.2f} hours")

#         return df

#     def create_frequency_features(self) -> pd.DataFrame:
#         """Calculate transaction frequency and velocity."""
#         df = self.df.copy()

#         print("\n=== CREATING FREQUENCY FEATURES ===")

#         # Sort for rolling logic
#         df = df.sort_values(['user_id', 'purchase_time'])

#         df['transactions_last_1h'] = 0
#         df['transactions_last_24h'] = 0
#         df['transactions_last_7d'] = 0

#         for user_id in df['user_id'].unique():
#             user_df = df[df['user_id'] == user_id]

#             for idx, row in user_df.iterrows():
#                 current_time = row['purchase_time']

#                 df.loc[idx, 'transactions_last_1h'] = (
#                     (user_df['purchase_time'] >= current_time - timedelta(hours=1)) &
#                     (user_df['purchase_time'] < current_time)
#                 ).sum()

#                 df.loc[idx, 'transactions_last_24h'] = (
#                     (user_df['purchase_time'] >= current_time - timedelta(hours=24)) &
#                     (user_df['purchase_time'] < current_time)
#                 ).sum()

#                 df.loc[idx, 'transactions_last_7d'] = (
#                     (user_df['purchase_time'] >= current_time - timedelta(days=7)) &
#                     (user_df['purchase_time'] < current_time)
#                 ).sum()

#         # Velocity
#         df['velocity_1h'] = df['transactions_last_1h']
#         df['velocity_24h'] = df['transactions_last_24h'] / 24

#         print("Created frequency and velocity features")

#         print("\nFrequency feature statistics:")
#         print(f"Average transactions last 1h: {df['transactions_last_1h'].mean():.2f}")
#         print(f"Average transactions last 24h: {df['transactions_last_24h'].mean():.2f}")
#         print(f"Max transactions last 1h: {df['transactions_last_1h'].max():.2f}")

#         return df

#     def create_aggregate_features(self) -> pd.DataFrame:
#         """Create aggregate user-level features."""
#         df = self.df.copy()

#         print("\n=== CREATING AGGREGATE FEATURES ===")

#         user_stats = (
#             df.groupby('user_id')
#             .agg({
#                 'purchase_value': ['mean', 'std', 'max', 'min'],
#                 'age': 'first',
#                 'source': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
#                 'browser': lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
#             })
#             .reset_index()
#         )

#         user_stats.columns = [
#             'user_id',
#             'avg_purchase',
#             'std_purchase',
#             'max_purchase',
#             'min_purchase',
#             'user_age',
#             'common_source',
#             'common_browser'
#         ]

#         df = df.merge(user_stats, on='user_id', how='left')

#         df['purchase_deviation'] = df['purchase_value'] - df['avg_purchase']
#         df['purchase_zscore'] = (
#             df['purchase_value'] - df['avg_purchase']
#         ) / df['std_purchase'].replace(0, 1)

#         aggregate_features = [
#             'avg_purchase',
#             'std_purchase',
#             'max_purchase',
#             'min_purchase',
#             'purchase_deviation',
#             'purchase_zscore'
#         ]

#         print(f"Created {len(aggregate_features)} aggregate features: {aggregate_features}")

#         return df

#     def create_interaction_features(self) -> pd.DataFrame:
#         """Create interaction features between variables."""
#         df = self.df.copy()

#         print("\n=== CREATING INTERACTION FEATURES ===")

#         df['device_browser'] = (
#             df['device_id'].astype(str) + '_' + df['browser'].astype(str)
#         ).astype('category')

#         df['source_browser'] = (
#             df['source'].astype(str) + '_' + df['browser'].astype(str)
#         ).astype('category')

#         df['age_group'] = pd.cut(
#             df['age'],
#             bins=[0, 18, 25, 35, 50, 65, 100],
#             labels=['0-18', '19-25', '26-35', '36-50', '51-65', '65+']
#         )

#         df['purchase_group'] = pd.qcut(df['purchase_value'], q=5, labels=False)

#         print("Created interaction features")

#         return df

#     def feature_engineering_pipeline(self) -> pd.DataFrame:
#         """Execute complete feature engineering pipeline."""
#         print("=== STARTING FEATURE ENGINEERING PIPELINE ===")

#         original_columns = set(self.df.columns)

#         self.df = self.create_time_based_features()
#         self.df = self.create_frequency_features()
#         self.df = self.create_aggregate_features()
#         self.df = self.create_interaction_features()

#         new_features = set(self.df.columns) - original_columns

#         print("\n=== FEATURE ENGINEERING COMPLETE ===")
#         print(f"Original features: {len(original_columns)}")
#         print(f"New features created: {len(new_features)}")
#         print(f"Total features: {len(self.df.columns)}")
#         print(f"\nNew features: {sorted(new_features)}")

#         return self.df


