"""
IP to Country mapping with range-based lookup.
Optimized for performance.
"""
import pandas as pd
import numpy as np
from typing import Optional
import ipaddress

class IPCountryMapper:
    """Convert IP addresses to countries using range-based lookup."""
    
    @staticmethod
    def ip_to_int(ip_str: str) -> Optional[int]:
        """Convert IP address to integer."""
        try:
            return int(ipaddress.IPv4Address(ip_str))
        except:
            return None
    
    def prepare_ip_ranges(self, ip_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare IP ranges for efficient lookup."""
        ip_ranges = ip_df.copy()
        
        # Convert bounds to integers
        ip_ranges['lower_int'] = ip_ranges['lower_bound_ip_address'].apply(self.ip_to_int)
        ip_ranges['upper_int'] = ip_ranges['upper_bound_ip_address'].apply(self.ip_to_int)
        
        # Sort for binary search
        ip_ranges = ip_ranges.sort_values('lower_int').reset_index(drop=True)
        
        return ip_ranges[['lower_int', 'upper_int', 'country']]
    
    def map_ip_to_country(self, fraud_df: pd.DataFrame, ip_ranges: pd.DataFrame) -> pd.DataFrame:
        """Merge fraud data with country mapping using range lookup."""
        df = fraud_df.copy()
        
        # Convert fraud IPs to integers
        df['ip_int'] = df['ip_address'].apply(self.ip_to_int)
        
        # Binary search for each IP
        countries = []
        
        for ip_int in df['ip_int']:
            if pd.isna(ip_int):
                countries.append(None)
                continue
                
            # Binary search in sorted ranges
            left, right = 0, len(ip_ranges) - 1
            country = None
            
            while left <= right:
                mid = (left + right) // 2
                lower = ip_ranges.iloc[mid]['lower_int']
                upper = ip_ranges.iloc[mid]['upper_int']
                
                if lower <= ip_int <= upper:
                    country = ip_ranges.iloc[mid]['country']
                    break
                elif ip_int < lower:
                    right = mid - 1
                else:
                    left = mid + 1
            
            countries.append(country)
        
        df['country'] = countries
        df['country'] = df['country'].astype('category')
        
        # Remove helper column
        df.drop('ip_int', axis=1, inplace=True)
        
        # Analyze fraud patterns by country
        if df['country'].notna().any():
            country_fraud_rate = df.groupby('country')['class'].agg(['count', 'mean'])
            country_fraud_rate.columns = ['transaction_count', 'fraud_rate']
            country_fraud_rate = country_fraud_rate.sort_values('fraud_rate', ascending=False)
            
            print("\n=== COUNTRY FRAUD ANALYSIS ===")
            print("Top 10 countries by fraud rate:")
            print(country_fraud_rate.head(10).to_string())
            
            print("\nTop 10 countries by transaction volume:")
            print(country_fraud_rate.sort_values('transaction_count', ascending=False).head(10).to_string())
        
        return df