"""
IP address geolocation module for fraud detection.
Implements efficient IP-to-country mapping using range-based lookup.
"""

import pandas as pd
import numpy as np
import struct
import socket
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class IPGeolocationMapper:
    """
    Efficient IP-to-country mapping using range-based lookup.
    Optimized for large datasets.
    """
    
    def __init__(self):
        """Initialize geolocation mapper."""
        self.ip_ranges = None
        self.country_codes = None
        self.is_loaded = False
        
    def ip_to_int(self, ip_address: str) -> int:
        """
        Convert IP address to integer.
        
        Args:
            ip_address (str): IP address in string format
            
        Returns:
            int: Integer representation of IP
            
        Raises:
            ValueError: If IP address is invalid
        """
        try:
            # Handle both IPv4 and potential string representations
            if isinstance(ip_address, str):
                # Remove any quotes or extra characters
                ip_address = ip_address.strip().strip('"').strip("'")
                
                # Check if it's already an integer string
                if ip_address.replace('.', '').isdigit():
                    # It's an IPv4 address
                    return struct.unpack("!I", socket.inet_aton(ip_address))[0]
                elif ip_address.isdigit():
                    # It's already an integer
                    return int(ip_address)
                else:
                    # Try to parse as integer
                    try:
                        return int(float(ip_address))
                    except:
                        raise ValueError(f"Invalid IP address format: {ip_address}")
            elif isinstance(ip_address, (int, float)):
                # Already numeric
                return int(ip_address)
            else:
                raise ValueError(f"Unsupported IP address type: {type(ip_address)}")
                
        except Exception as e:
            raise ValueError(f"Error converting IP {ip_address}: {e}")
    
    def load_ip_country_mapping(self, filepath: str) -> None:
        """
        Load IP address to country mapping data.
        
        Args:
            filepath (str): Path to IP-to-Country CSV file
        """
        try:
            print(f"📥 Loading IP-to-country mapping from {filepath}...")
            
            # Load the mapping data
            df_mapping = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
            missing_cols = [col for col in required_cols if col not in df_mapping.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert IP addresses to integers
            print("🔄 Converting IP addresses to integers...")
            df_mapping['lower_bound_int'] = df_mapping['lower_bound_ip_address'].apply(self.ip_to_int)
            df_mapping['upper_bound_int'] = df_mapping['upper_bound_ip_address'].apply(self.ip_to_int)
            
            # Sort by lower bound for efficient lookup
            df_mapping = df_mapping.sort_values('lower_bound_int').reset_index(drop=True)
            
            # Store as numpy arrays for faster lookup
            self.ip_ranges = df_mapping[['lower_bound_int', 'upper_bound_int']].values
            self.country_codes = df_mapping['country'].values
            
            self.is_loaded = True
            
            print(f"✓ IP-to-country mapping loaded: {len(df_mapping):,} ranges")
            print(f"✓ IP range: {df_mapping['lower_bound_int'].min():,} to {df_mapping['upper_bound_int'].max():,}")
            
            # Show country distribution
            country_counts = df_mapping['country'].value_counts()
            print(f"\n🌍 Top 10 countries by IP range count:")
            for country, count in country_counts.head(10).items():
                print(f"  {country}: {count:,} ranges")
            
        except Exception as e:
            print(f"✗ Error loading IP mapping: {e}")
            raise
    
    def find_country_for_ip(self, ip_int: int) -> Optional[str]:
        """
        Find country for a single IP address using binary search.
        
        Args:
            ip_int (int): IP address as integer
            
        Returns:
            str or None: Country code if found, None otherwise
        """
        if not self.is_loaded:
            raise ValueError("IP mapping not loaded. Call load_ip_country_mapping first.")
        
        # Binary search for the IP range
        left, right = 0, len(self.ip_ranges) - 1
        
        while left <= right:
            mid = (left + right) // 2
            lower, upper = self.ip_ranges[mid]
            
            if lower <= ip_int <= upper:
                return self.country_codes[mid]
            elif ip_int < lower:
                right = mid - 1
            else:
                left = mid + 1
        
        return None
    
    def map_ips_to_countries(self, ip_series: pd.Series, 
                            batch_size: int = 10000) -> pd.Series:
        """
        Map IP addresses to countries efficiently.
        
        Args:
            ip_series (pd.Series): Series of IP addresses
            batch_size (int): Batch size for processing
            
        Returns:
            pd.Series: Series of country codes
        """
        if not self.is_loaded:
            raise ValueError("IP mapping not loaded. Call load_ip_country_mapping first.")
        
        print(f"🗺️  Mapping {len(ip_series):,} IP addresses to countries...")
        
        # Convert IPs to integers
        print("🔄 Converting IPs to integers...")
        ip_ints = []
        failed_ips = 0
        
        for ip in ip_series:
            try:
                ip_ints.append(self.ip_to_int(ip))
            except:
                ip_ints.append(None)
                failed_ips += 1
        
        if failed_ips > 0:
            print(f"⚠️  Failed to convert {failed_ips:,} IP addresses ({failed_ips/len(ip_series)*100:.2f}%)")
        
        # Create DataFrame for batch processing
        df_batch = pd.DataFrame({
            'ip_original': ip_series.values,
            'ip_int': ip_ints
        })
        
        # Remove rows with invalid IPs
        valid_mask = df_batch['ip_int'].notnull()
        df_valid = df_batch[valid_mask].copy()
        
        if len(df_valid) == 0:
            print("✗ No valid IP addresses to map")
            return pd.Series([None] * len(ip_series), index=ip_series.index)
        
        # Vectorized lookup using searchsorted
        print("🔍 Performing vectorized IP lookup...")
        
        # Sort IP ranges for searchsorted
        lower_bounds = self.ip_ranges[:, 0]
        
        # Find insertion points
        positions = np.searchsorted(lower_bounds, df_valid['ip_int'].values) - 1
        
        # Check if IP falls within range
        country_results = []
        for idx, pos in enumerate(positions):
            if pos >= 0 and pos < len(self.ip_ranges):
                lower, upper = self.ip_ranges[pos]
                ip_int = df_valid.iloc[idx]['ip_int']
                if lower <= ip_int <= upper:
                    country_results.append(self.country_codes[pos])
                else:
                    country_results.append(None)
            else:
                country_results.append(None)
        
        # Create result series
        df_valid['country'] = country_results
        
        # Merge back with original
        df_result = pd.merge(df_batch, df_valid[['ip_original', 'country']], 
                           on='ip_original', how='left')
        
        # Calculate statistics
        mapped_count = df_result['country'].notnull().sum()
        mapping_rate = mapped_count / len(df_result) * 100
        
        print(f"✓ IP mapping completed: {mapped_count:,}/{len(df_result):,} mapped ({mapping_rate:.2f}%)")
        
        if mapped_count > 0:
            top_countries = df_result['country'].value_counts().head(10)
            print(f"\n🌍 Top countries in dataset:")
            for country, count in top_countries.items():
                print(f"  {country}: {count:,} IPs ({count/len(df_result)*100:.1f}%)")
        
        return df_result['country']
    
    def analyze_fraud_by_country(self, df: pd.DataFrame, 
                                ip_col: str = 'ip_address',
                                fraud_col: str = 'class') -> pd.DataFrame:
        """
        Analyze fraud patterns by country.
        
        Args:
            df (pd.DataFrame): DataFrame with IP addresses and fraud labels
            ip_col (str): Column name for IP addresses
            fraud_col (str): Column name for fraud labels
            
        Returns:
            pd.DataFrame: Fraud statistics by country
        """
        if ip_col not in df.columns:
            raise ValueError(f"IP column '{ip_col}' not found in dataframe")
        if fraud_col not in df.columns:
            raise ValueError(f"Fraud column '{fraud_col}' not found in dataframe")
        
        print("\n" + "="*80)
        print("FRAUD ANALYSIS BY COUNTRY")
        print("="*80)
        
        # Map IPs to countries
        df_country = df.copy()
        df_country['country'] = self.map_ips_to_countries(df_country[ip_col])
        
        # Analyze fraud by country
        fraud_by_country = df_country.groupby('country').agg({
            fraud_col: ['count', 'sum', 'mean']
        }).round(4)
        
        fraud_by_country.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
        fraud_by_country['fraud_rate'] = fraud_by_country['fraud_rate'] * 100
        
        # Sort by fraud rate
        fraud_by_country = fraud_by_country.sort_values('fraud_rate', ascending=False)
        
        print(f"\n📊 Fraud statistics by country ({len(fraud_by_country)} countries):")
        print(fraud_by_country.head(15))
        
        # Plot top countries by fraud rate (with sufficient transactions)
        min_transactions = 100  # Minimum transactions to consider
        significant_countries = fraud_by_country[fraud_by_country['total_transactions'] >= min_transactions]
        
        if len(significant_countries) > 0:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 8))
            top_countries = significant_countries.head(10)
            
            # Create bar plot
            bars = plt.bar(range(len(top_countries)), top_countries['fraud_rate'])
            plt.title('Top 10 Countries by Fraud Rate', fontsize=14, fontweight='bold')
            plt.xlabel('Country', fontsize=12)
            plt.ylabel('Fraud Rate (%)', fontsize=12)
            plt.xticks(range(len(top_countries)), top_countries.index, rotation=45, ha='right')
            
            # Add value labels
            for i, (bar, (_, row)) in enumerate(zip(bars, top_countries.iterrows())):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}%\n({int(row["total_transactions"]):,} txns)',
                        ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
            # Identify high-risk countries
            high_risk_threshold = fraud_by_country['fraud_rate'].quantile(0.95)
            high_risk_countries = fraud_by_country[fraud_by_country['fraud_rate'] > high_risk_threshold]
            
            if len(high_risk_countries) > 0:
                print(f"\n⚠️  HIGH-RISK COUNTRIES (fraud rate > {high_risk_threshold:.2f}%):")
                for country, row in high_risk_countries.iterrows():
                    print(f"  {country}: {row['fraud_rate']:.2f}% fraud "
                          f"({int(row['fraud_count']):,} frauds in {int(row['total_transactions']):,} transactions)")
        
        return fraud_by_country
    
    def validate_mapping(self, test_ips: List[str] = None) -> Dict[str, str]:
        """
        Validate IP mapping with test IPs.
        
        Args:
            test_ips (list): List of test IP addresses
            
        Returns:
            dict: Mapping results
        """
        if test_ips is None:
            # Some test IPs from different countries
            test_ips = [
                '8.8.8.8',        # Google DNS (US)
                '1.1.1.1',        # Cloudflare (US)
                '5.62.61.252',    # Germany
                '103.10.197.50',  # Australia
                '200.58.127.250', # Brazil
                '41.77.96.0',     # South Africa
                '110.34.183.1',   # Thailand
                '14.63.166.126',  # South Korea
                '195.154.183.109',# France
                '92.38.148.60'    # Russia
            ]
        
        results = {}
        print("\n" + "="*60)
        print("IP MAPPING VALIDATION")
        print("="*60)
        
        for ip in test_ips:
            try:
                ip_int = self.ip_to_int(ip)
                country = self.find_country_for_ip(ip_int)
                results[ip] = country
                print(f"  {ip:20} → {country if country else 'Not found'}")
            except Exception as e:
                results[ip] = f"Error: {e}"
                print(f"  {ip:20} → Error: {e}")
        
        success_rate = sum(1 for v in results.values() if v and 'Error' not in str(v)) / len(results) * 100
        print(f"\n✓ Validation success rate: {success_rate:.1f}%")
        
        return results