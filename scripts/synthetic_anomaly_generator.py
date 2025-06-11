"""
Synthetic Anomaly Generator for Public Procurement Data

This module creates synthetic anomalies by either adding new rows or replacing
existing rows in procurement datasets to test the effectiveness of anomaly 
detection models.

The anomalies are based on red flags identified in procurement literature
and real-world corruption patterns.

Author: RonanB400
Date: January 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
import random
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class OriginalAnomalyAnalyzer:
    """Analyze original dataset for existing patterns similar to synthetic anomalies."""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_all_anomaly_types(self, df: pd.DataFrame) -> Dict:
        """Analyze all anomaly types in the original dataset.
        
        Args:
            df: Original dataframe (before synthetic anomalies)
            
        Returns:
            Dictionary with analysis results for each anomaly type
        """
        logger.info("Analyzing original dataset for existing anomaly patterns...")
        
        results = {
            'total_contracts': len(df),
            'anomaly_analysis': {}
        }
        
        # Analyze each anomaly type
        anomaly_functions = {
            'single_bid_competitive': self._analyze_single_bid_competitive,
            'price_inflation': self._analyze_price_inflation,
            'price_deflation': self._analyze_price_deflation,
            'procedure_manipulation': self._analyze_procedure_manipulation,
            'suspicious_modifications': self._analyze_suspicious_modifications,
            'high_market_concentration': self._analyze_high_market_concentration,
            'temporal_clustering': self._analyze_temporal_clustering,
            'excessive_subcontracting': self._analyze_excessive_subcontracting,
            'short_contract_duration': self._analyze_short_contract_duration,
            'suspicious_buyer_supplier_pairs': self._analyze_suspicious_pairs
        }
        
        for anomaly_type, analyze_func in anomaly_functions.items():
            try:
                analysis = analyze_func(df)
                results['anomaly_analysis'][anomaly_type] = analysis
                logger.info(f"{anomaly_type}: {analysis['count']} contracts "
                           f"({analysis['percentage']:.2f}%)")
            except Exception as e:
                logger.error(f"Error analyzing {anomaly_type}: {str(e)}")
                results['anomaly_analysis'][anomaly_type] = {
                    'count': 0, 'percentage': 0.0, 'error': str(e)
                }
        
        self.analysis_results = results
        return results
    
    def _analyze_single_bid_competitive(self, df: pd.DataFrame) -> Dict:
        """Analyze single bid competitive anomalies in original data."""
        competitive_procedures = ["Appel d'offres ouvert", 
                                "Appel d'offres restreint"]
        
        mask = (df['procedure'].isin(competitive_procedures) & 
                (df['offresRecues'] == 1) & 
                df['offresRecues'].notna())
        
        count = np.sum(mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': 'Competitive procedures with exactly 1 bid',
            'threshold_used': 'offresRecues == 1 AND competitive procedure'
        }
    
    def _analyze_price_inflation(self, df: pd.DataFrame) -> Dict:
        """Analyze price inflation anomalies in original data."""
        # Define inflation as >3 standard deviations from mean per CPV category
        required_cols = ['montant', 'codeCPV_3']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {'count': 0, 'percentage': 0.0, 
                    'error': f'Missing columns: {missing_cols}'}
        
        if df['montant'].isna().all():
            return {'count': 0, 'percentage': 0.0, 'error': 'No amount data'}
        
        # Filter to valid data
        valid_data = df[df['montant'].notna() & df['codeCPV_3'].notna()].copy()
        if len(valid_data) == 0:
            return {'count': 0, 'percentage': 0.0, 'error': 'No valid amount/CPV data'}
        
        # Calculate statistics per CPV category
        cpv_stats = valid_data.groupby('codeCPV_3')['montant'].agg(['mean', 'std', 'count']).reset_index()
        cpv_stats = cpv_stats[cpv_stats['count'] >= 200]  # Need at least 200 contracts for reliable stats
        
        if len(cpv_stats) == 0:
            return {'count': 0, 'percentage': 0.0, 'error': 'No CPV categories with sufficient data'}
        
        # Identify inflated contracts (>3 std dev from mean)
        inflated_mask = pd.Series(False, index=df.index)
        analyzed_categories = []
        
        for _, row in cpv_stats.iterrows():
            cpv = row['codeCPV_3']
            mean_amount = row['mean']
            std_amount = row['std']
            
            if pd.notna(std_amount) and std_amount > 0:
                threshold = mean_amount + (3 * std_amount)
                cpv_mask = (df['codeCPV_3'] == cpv) & (df['montant'] > threshold) & df['montant'].notna()
                inflated_mask |= cpv_mask
                analyzed_categories.append(cpv)
        
        count = np.sum(inflated_mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': f'Contracts >3 std dev above mean within CPV category ({len(analyzed_categories)} categories analyzed)',
            'threshold_used': 'montant > mean + 3*std per codeCPV_3 category'
        }
    
    def _analyze_price_deflation(self, df: pd.DataFrame) -> Dict:
        """Analyze price deflation anomalies in original data."""
        # Define deflation as <3 standard deviations from mean per CPV category
        required_cols = ['montant', 'codeCPV_3']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {'count': 0, 'percentage': 0.0, 'error': f'Missing columns: {missing_cols}'}
        
        if df['montant'].isna().all():
            return {'count': 0, 'percentage': 0.0, 'error': 'No amount data'}
        
        # Filter to valid data
        valid_data = df[df['montant'].notna() & df['codeCPV_3'].notna()].copy()
        if len(valid_data) == 0:
            return {'count': 0, 'percentage': 0.0, 'error': 'No valid amount/CPV data'}
        
        # Calculate statistics per CPV category
        cpv_stats = valid_data.groupby('codeCPV_3')['montant'].agg(['mean', 'std', 'count']).reset_index()
        cpv_stats = cpv_stats[cpv_stats['count'] >= 200]  # Need at least 200 contracts for reliable stats
        
        if len(cpv_stats) == 0:
            return {'count': 0, 'percentage': 0.0, 'error': 'No CPV categories with sufficient data'}
        
        # Identify deflated contracts (<3 std dev from mean)
        deflated_mask = pd.Series(False, index=df.index)
        analyzed_categories = []
        
        for _, row in cpv_stats.iterrows():
            cpv = row['codeCPV_3']
            mean_amount = row['mean']
            std_amount = row['std']
            
            if pd.notna(std_amount) and std_amount > 0:
                threshold = mean_amount - (3 * std_amount)
                # Only consider positive thresholds to avoid negative amounts
                threshold = max(threshold, 0)
                cpv_mask = (df['codeCPV_3'] == cpv) & (df['montant'] < threshold) & (df['montant'] > 0) & df['montant'].notna()
                deflated_mask |= cpv_mask
                analyzed_categories.append(cpv)
        
        count = np.sum(deflated_mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': f'Contracts <3 std dev below mean within CPV category ({len(analyzed_categories)} categories analyzed)',
            'threshold_used': 'montant < max(0, mean - 3*std) per codeCPV_3 category'
        }
    
    def _analyze_procedure_manipulation(self, df: pd.DataFrame) -> Dict:
        """Analyze procedure manipulation in original data."""
        # High-value contracts using non-competitive procedures
        if 'montant' not in df.columns or df['montant'].isna().all():
            return {'count': 0, 'percentage': 0.0, 'error': 'No amount data'}
        
        # Define high-value as top 25% of contracts
        high_value_threshold = np.percentile(df['montant'].dropna(), 75)
        
        non_competitive = ['Procédure adaptée', 'Marché négocié sans publicité']
        
        mask = (df['montant'] > high_value_threshold) & \
               df['procedure'].isin(non_competitive) & \
               df['montant'].notna()
        
        count = np.sum(mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': f'High-value contracts (>{high_value_threshold:,.0f}) using non-competitive procedures',
            'threshold_used': f'montant > {high_value_threshold:,.0f} AND non-competitive procedure'
        }
    
    def _analyze_suspicious_modifications(self, df: pd.DataFrame) -> Dict:
        """Analyze suspicious contract modifications in original data."""
        if 'dureeMois' not in df.columns or df['dureeMois'].isna().all():
            return {'count': 0, 'percentage': 0.0, 'error': 'No duration data'}
        
        valid_durations = df['dureeMois'].dropna()
        if len(valid_durations) == 0:
            return {'count': 0, 'percentage': 0.0, 'error': 'No valid durations'}
        
        # Define suspicious as top 5% of durations (very long contracts)
        long_duration_threshold = np.percentile(valid_durations, 95)
        mask = (df['dureeMois'] > long_duration_threshold) & df['dureeMois'].notna()
        
        count = np.sum(mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': f'Contracts with very long duration (>{long_duration_threshold:.1f} months)',
            'threshold_used': f'dureeMois > {long_duration_threshold:.1f}'
        }
    
    def _analyze_high_market_concentration(self, df: pd.DataFrame) -> Dict:
        """Analyze high market concentration in original data."""
        if 'acheteur_id' not in df.columns or 'codeCPV_3' not in df.columns or 'titulaire_id' not in df.columns:
            return {'count': 0, 'percentage': 0.0, 'error': 'Missing required columns'}
        
        # Find buyer-CPV combinations where one supplier has >70% of contracts
        buyer_cpv_supplier_counts = df.groupby(['acheteur_id', 'codeCPV_3', 'titulaire_id']).size()
        buyer_cpv_total_counts = df.groupby(['acheteur_id', 'codeCPV_3']).size()
        
        # Calculate supplier market share within each buyer-CPV combination
        supplier_shares = buyer_cpv_supplier_counts / buyer_cpv_total_counts
        
        # Find cases where supplier has >70% market share and >2 total contracts
        high_concentration_mask = (supplier_shares > 0.7) & (buyer_cpv_total_counts > 2)
        
        # Count contracts in these high-concentration scenarios
        high_concentration_combinations = high_concentration_mask[high_concentration_mask].index
        
        count = 0
        for (buyer, cpv, supplier) in high_concentration_combinations:
            count += len(df[(df['acheteur_id'] == buyer) & 
                           (df['codeCPV_3'] == cpv) & 
                           (df['titulaire_id'] == supplier)])
        
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': 'Contracts in buyer-CPV combinations with >70% supplier concentration',
            'threshold_used': 'supplier_share > 0.7 AND total_contracts > 2'
        }
    
    def _analyze_temporal_clustering(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal clustering in original data."""
        if 'dateNotification' not in df.columns:
            return {'count': 0, 'percentage': 0.0, 'error': 'No date data'}
        
        # Convert dates
        try:
            df_temp = df.copy()
            df_temp['date_parsed'] = pd.to_datetime(df_temp['dateNotification'], errors='coerce')
            df_temp = df_temp.dropna(subset=['date_parsed', 'acheteur_id', 'titulaire_id', 'codeCPV_3'])
            
            if len(df_temp) == 0:
                return {'count': 0, 'percentage': 0.0, 'error': 'No valid dates'}
            
            # Group by buyer-supplier pairs with same CPV code
            buyer_supplier_cpv_groups = df_temp.groupby(['acheteur_id', 'titulaire_id', 'codeCPV_3'])
            
            clustered_contracts = 0
            for (buyer, supplier, cpv), group in buyer_supplier_cpv_groups:
                if len(group) >= 4:  # Need at least 4 contracts to detect clustering
                    dates = sorted(group['date_parsed'])
                    
                    # Check for clustering: 4+ contracts within 30 days
                    for i in range(len(dates) - 3):
                        if (dates[i+3] - dates[i]).days <= 30:
                            clustered_contracts += 4
                            break
            
            percentage = (clustered_contracts / len(df)) * 100
            
            return {
                'count': clustered_contracts,
                'percentage': percentage,
                'description': 'Contracts in buyer-supplier-CPV groups with 4+ contracts within 30 days',
                'threshold_used': '4+ contracts within 30 days for same buyer-supplier-CPV combination'
            }
        except Exception as e:
            return {'count': 0, 'percentage': 0.0, 'error': f'Date parsing error: {str(e)}'}
    
    def _analyze_excessive_subcontracting(self, df: pd.DataFrame) -> Dict:
        """Analyze excessive subcontracting in original data."""
        if 'sousTraitanceDeclaree' not in df.columns:
            return {'count': 0, 'percentage': 0.0, 'error': 'No subcontracting data'}
        
        # Contracts declaring subcontracting
        mask = (df['sousTraitanceDeclaree'] == 1) & df['sousTraitanceDeclaree'].notna()
        count = np.sum(mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': 'Contracts declaring subcontracting',
            'threshold_used': 'sousTraitanceDeclaree == 1'
        }
    
    def _analyze_short_contract_duration(self, df: pd.DataFrame) -> Dict:
        """Analyze short contract duration in original data."""
        if 'dureeMois' not in df.columns or df['dureeMois'].isna().all():
            return {'count': 0, 'percentage': 0.0, 'error': 'No duration data'}
        
        # Very short contracts (< 1 month)
        mask = (df['dureeMois'] < 1) & (df['dureeMois'] > 0) & df['dureeMois'].notna()
        count = np.sum(mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': 'Contracts with duration < 1 month',
            'threshold_used': 'dureeMois < 1'
        }
    
    def _analyze_suspicious_pairs(self, df: pd.DataFrame) -> Dict:
        """Analyze suspicious buyer-supplier pairs in original data."""
        if 'montant' not in df.columns or df['montant'].isna().all():
            return {'count': 0, 'percentage': 0.0, 'error': 'No amount data'}
        
        # Calculate total amounts per buyer-supplier pair
        pair_totals = df.groupby(['acheteur_id', 'titulaire_id'])['montant'].agg(['sum', 'count'])
        
        # Define suspicious as mean + 2 standard deviations of total amounts with 2+ contracts
        pair_sums = pair_totals['sum'].dropna()
        if len(pair_sums) == 0:
            return {'count': 0, 'percentage': 0.0, 'error': 'No valid pair amounts'}
        
        mean_amount = pair_sums.mean()
        std_amount = pair_sums.std()
        
        if pd.isna(std_amount) or std_amount == 0:
            return {'count': 0, 'percentage': 0.0, 'error': 'Cannot calculate standard deviation'}
        
        suspicious_threshold = mean_amount + (2 * std_amount)
        suspicious_pairs = pair_totals[(pair_totals['sum'] > suspicious_threshold) & 
                                     (pair_totals['count'] >= 2)]
        
        # Count contracts in suspicious pairs
        count = 0
        for (buyer, supplier), _ in suspicious_pairs.iterrows():
            count += len(df[(df['acheteur_id'] == buyer) & 
                           (df['titulaire_id'] == supplier)])
        
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': f'Contracts in buyer-supplier pairs with total amount >{suspicious_threshold:,.0f} (mean + 2σ)',
            'threshold_used': f'total_pair_amount > {mean_amount:,.0f} + 2*{std_amount:,.0f} AND contract_count >= 2'
        }
    
    def print_analysis_summary(self, results: Dict = None):
        """Print a comprehensive summary of the original dataset analysis."""
        if results is None:
            results = self.analysis_results
        
        if not results:
            logger.error("No analysis results available. Run analyze_all_anomaly_types first.")
            return
        
        print("\n" + "="*80)
        print("ORIGINAL DATASET ANOMALY PATTERN ANALYSIS")
        print("="*80)
        print(f"Total contracts analyzed: {results['total_contracts']:,}")
        print(f"\nAnalyzing prevalence of synthetic anomaly patterns in original data:")
        print("-" * 80)
        
        # Sort by percentage descending
        anomaly_data = []
        for anomaly_type, analysis in results['anomaly_analysis'].items():
            if 'error' not in analysis:
                anomaly_data.append((anomaly_type, analysis))
        
        anomaly_data.sort(key=lambda x: x[1]['percentage'], reverse=True)
        
        print(f"{'Anomaly Type':<35} {'Count':<10} {'Percentage':<12} {'Status':<15}")
        print("-" * 80)
        
        total_flagged = 0
        for anomaly_type, analysis in anomaly_data:
            count = analysis['count']
            percentage = analysis['percentage']
            total_flagged += count
            
            # Determine status based on prevalence
            if percentage > 10:
                status = "🔴 Very High"
            elif percentage > 5:
                status = "🟡 High" 
            elif percentage > 1:
                status = "🟠 Medium"
            elif percentage > 0.1:
                status = "🟢 Low"
            else:
                status = "✅ Very Low"
            
            type_name = anomaly_type.replace('_', ' ').title()
            print(f"{type_name:<35} {count:<10,} {percentage:<12.2f}% {status:<15}")
        
        print("-" * 80)
        print(f"{'TOTAL FLAGGED (with overlap)':<35} {total_flagged:<10,} {(total_flagged/results['total_contracts']*100):<12.1f}%")
        
        # Print detailed descriptions
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS")
        print(f"{'='*80}")
        
        for anomaly_type, analysis in anomaly_data:
            print(f"\n{anomaly_type.replace('_', ' ').title()}:")
            print(f"  Count: {analysis['count']:,} contracts ({analysis['percentage']:.2f}%)")
            print(f"  Description: {analysis['description']}")
            print(f"  Threshold: {analysis['threshold_used']}")
        
        # Print errors if any
        errors = [(k, v) for k, v in results['anomaly_analysis'].items() if 'error' in v]
        if errors:
            print(f"\n{'='*80}")
            print("ANALYSIS ERRORS")
            print(f"{'='*80}")
            for anomaly_type, analysis in errors:
                print(f"{anomaly_type}: {analysis['error']}")
        
        # Recommendations
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        high_prevalence = [name for name, analysis in anomaly_data 
                          if analysis['percentage'] > 5]
        medium_prevalence = [name for name, analysis in anomaly_data 
                            if 1 < analysis['percentage'] <= 5]
        
        if high_prevalence:
            print("🔴 HIGH PREVALENCE PATTERNS (>5%):")
            for pattern in high_prevalence:
                print(f"  • {pattern.replace('_', ' ').title()}")
            print("   → These patterns are very common in your dataset.")
            print("   → Synthetic anomalies of these types may not be distinguishable.")
            print("   → Consider removing these from synthetic anomaly generation.")
        
        if medium_prevalence:
            print("\n🟠 MEDIUM PREVALENCE PATTERNS (1-5%):")
            for pattern in medium_prevalence:
                print(f"  • {pattern.replace('_', ' ').title()}")
            print("   → These patterns exist but are less common.")
            print("   → Synthetic anomalies might still be learnable but with reduced signal.")
        
        low_prevalence = [name for name, analysis in anomaly_data 
                         if analysis['percentage'] <= 1]
        if low_prevalence:
            print("\n✅ LOW PREVALENCE PATTERNS (<1%):")
            for pattern in low_prevalence:
                print(f"  • {pattern.replace('_', ' ').title()}")
            print("   → These patterns are rare in your dataset.")
            print("   → Synthetic anomalies of these types should be most learnable.")
        
        print(f"\n💡 SUMMARY:")
        total_percentage = sum(analysis['percentage'] for _, analysis in anomaly_data)
        print(f"   • Total coverage of anomaly patterns: {total_percentage:.1f}% (with overlap)")
        print(f"   • High prevalence patterns: {len(high_prevalence)}/{len(anomaly_data)}")
        print(f"   → Focus synthetic training on low-prevalence patterns for best results.")


def analyze_original_dataset_anomalies(df: pd.DataFrame) -> Dict:
    """Convenience function to analyze original dataset for anomaly patterns.
    
    Args:
        df: Original dataframe (before synthetic anomalies)
        
    Returns:
        Dictionary with analysis results
        
    Usage:
        # Load your original dataset
        df_original = pd.read_csv('your_data.csv')
        
        # Analyze existing anomaly patterns
        results = analyze_original_dataset_anomalies(df_original)
    """
    analyzer = OriginalAnomalyAnalyzer()
    results = analyzer.analyze_all_anomaly_types(df)
    analyzer.print_analysis_summary(results)
    return results



class OriginalAnomalyRemover:
    """Enhanced analyzer to identify and remove original anomalies from data."""
    
    def __init__(self):
        self.analyzer = OriginalAnomalyAnalyzer()
        self.anomalous_indices = set()
        
    def identify_original_anomalies(self, df: pd.DataFrame, 
                                   anomaly_types: List[str] = None,
                                   strict_threshold: bool = True) -> set:
        """Identify indices of rows that contain original anomalies.
        
        Args:
            df: DataFrame to analyze
            anomaly_types: List of anomaly types to check for
            strict_threshold: If True, use stricter thresholds for detection
            
        Returns:
            Set of indices of anomalous rows to remove
        """
        if anomaly_types is None:
            anomaly_types = [
                'single_bid_competitive',
                'price_inflation', 
                'price_deflation',
                'high_market_concentration',
                'temporal_clustering',
                'suspicious_buyer_supplier_pairs'
            ]
        
        logger.info("Identifying original anomalies for removal...")
        logger.info(f"Checking for: {', '.join(anomaly_types)}")
        
        anomalous_indices = set()
        
        for anomaly_type in anomaly_types:
            try:
                indices = self._get_anomaly_indices(df, anomaly_type, 
                                                    strict_threshold)
                anomalous_indices.update(indices)
                logger.info(f"{anomaly_type}: found {len(indices)} anomalies")
            except Exception as e:
                logger.warning(f"Failed to analyze {anomaly_type}: {str(e)}")
                
        logger.info(f"Total unique anomalous rows identified: "
                    f"{len(anomalous_indices)} "
                    f"({len(anomalous_indices)/len(df)*100:.2f}%)")
        
        self.anomalous_indices = anomalous_indices
        return anomalous_indices
    
    def _get_anomaly_indices(self, df: pd.DataFrame, anomaly_type: str,
                             strict_threshold: bool = True) -> set:
        """Get indices of specific anomaly type."""
        
        if anomaly_type == 'single_bid_competitive':
            return self._get_single_bid_indices(df, strict_threshold)
        elif anomaly_type == 'price_inflation':
            return self._get_price_inflation_indices(df, strict_threshold)
        elif anomaly_type == 'price_deflation':
            return self._get_price_deflation_indices(df, strict_threshold)
        elif anomaly_type == 'high_market_concentration':
            return self._get_market_concentration_indices(df, strict_threshold)
        elif anomaly_type == 'temporal_clustering':
            return self._get_temporal_clustering_indices(df, strict_threshold)
        elif anomaly_type == 'suspicious_buyer_supplier_pairs':
            return self._get_suspicious_pairs_indices(df, strict_threshold)
        else:
            return set()
    
    def _get_single_bid_indices(self, df: pd.DataFrame, 
                                strict: bool = True) -> set:
        """Get indices of single bid competitive anomalies."""
        competitive_procedures = ["Appel d'offres ouvert", 
                                  "Appel d'offres restreint"]
        
        mask = (df['procedure'].isin(competitive_procedures) & 
                (df['offresRecues'] == 1) & 
                df['offresRecues'].notna())
        
        return set(df[mask].index.tolist())
    
    def _get_price_inflation_indices(self, df: pd.DataFrame, 
                                     strict: bool = True) -> set:
        """Get indices of price inflation anomalies."""
        required_cols = ['montant', 'codeCPV_3']
        if not all(col in df.columns for col in required_cols):
            return set()
        
        valid_data = df[df['montant'].notna() & df['codeCPV_3'].notna()]
        if len(valid_data) == 0:
            return set()
        
        cpv_stats = (valid_data.groupby('codeCPV_3')['montant']
                     .agg(['mean', 'std', 'count']).reset_index())
        
        # Use same thresholds as _analyze_price_inflation
        cpv_stats = cpv_stats[cpv_stats['count'] >= 200]
        
        inflated_mask = pd.Series(False, index=df.index)
        
        for _, row in cpv_stats.iterrows():
            cpv = row['codeCPV_3']
            mean_amount = row['mean']
            std_amount = row['std']
            
            if pd.notna(std_amount) and std_amount > 0:
                threshold = mean_amount + (3 * std_amount)
                cpv_mask = ((df['codeCPV_3'] == cpv) & 
                            (df['montant'] > threshold) & 
                            df['montant'].notna())
                inflated_mask |= cpv_mask
        
        return set(df[inflated_mask].index.tolist())
    
    def _get_price_deflation_indices(self, df: pd.DataFrame, 
                                     strict: bool = True) -> set:
        """Get indices of price deflation anomalies."""
        required_cols = ['montant', 'codeCPV_3']
        if not all(col in df.columns for col in required_cols):
            return set()
        
        valid_data = df[df['montant'].notna() & df['codeCPV_3'].notna()]
        if len(valid_data) == 0:
            return set()
        
        cpv_stats = (valid_data.groupby('codeCPV_3')['montant']
                     .agg(['mean', 'std', 'count']).reset_index())
        
        # Use same thresholds as _analyze_price_deflation
        cpv_stats = cpv_stats[cpv_stats['count'] >= 200]
        
        deflated_mask = pd.Series(False, index=df.index)
        
        for _, row in cpv_stats.iterrows():
            cpv = row['codeCPV_3']
            mean_amount = row['mean']
            std_amount = row['std']
            
            if pd.notna(std_amount) and std_amount > 0:
                threshold = mean_amount - (3 * std_amount)
                threshold = max(threshold, 0)  # Only consider positive thresholds
                cpv_mask = ((df['codeCPV_3'] == cpv) & 
                            (df['montant'] < threshold) & 
                            (df['montant'] > 0) &  # Exclude zero amounts
                            df['montant'].notna())
                deflated_mask |= cpv_mask
        
        return set(df[deflated_mask].index.tolist())
    
    def _get_market_concentration_indices(self, df: pd.DataFrame, 
                                          strict: bool = True) -> set:
        """Get indices of high market concentration anomalies."""
        required_cols = ['acheteur_id', 'titulaire_id', 'codeCPV_3']
        if not all(col in df.columns for col in required_cols):
            return set()
        
        # Use same logic as _analyze_high_market_concentration
        buyer_cpv_supplier_counts = df.groupby(['acheteur_id', 'codeCPV_3', 'titulaire_id']).size()
        buyer_cpv_total_counts = df.groupby(['acheteur_id', 'codeCPV_3']).size()
        
        # Calculate supplier market share within each buyer-CPV combination
        supplier_shares = buyer_cpv_supplier_counts / buyer_cpv_total_counts
        
        # Find cases where supplier has >70% market share and >2 total contracts
        high_concentration_mask = (supplier_shares > 0.7) & (buyer_cpv_total_counts > 2)
        
        # Get the high-concentration combinations
        high_concentration_combinations = high_concentration_mask[high_concentration_mask].index
        
        # Collect indices of contracts in these high-concentration scenarios
        concentration_indices = set()
        for (buyer, cpv, supplier) in high_concentration_combinations:
            mask = ((df['acheteur_id'] == buyer) & 
                   (df['codeCPV_3'] == cpv) & 
                   (df['titulaire_id'] == supplier))
            concentration_indices.update(df[mask].index.tolist())
        
        return concentration_indices
    
    def _get_temporal_clustering_indices(self, df: pd.DataFrame, 
                                         strict: bool = True) -> set:
        """Get indices of temporal clustering anomalies."""
        if 'dateNotification' not in df.columns:
            return set()
        
        # Convert date column
        try:
            df_temp = df.copy()
            df_temp['date'] = pd.to_datetime(df_temp['dateNotification'], 
                                             errors='coerce')
            valid_dates = df_temp['date'].notna()
            df_temp = df_temp[valid_dates]
        except:
            return set()
        
        if len(df_temp) == 0:
            return set()
        
        clustering_indices = set()
        window_days = 15 if strict else 30
        min_cluster_size = 5 if strict else 3
        
        # Group by buyer-supplier-CPV combinations
        group_cols = ['acheteur_id', 'titulaire_id', 'codeCPV_3']
        if all(col in df_temp.columns for col in group_cols):
            
            for group_keys, group in df_temp.groupby(group_cols):
                if len(group) >= min_cluster_size:
                    # Sort by date
                    group_sorted = group.sort_values('date')
                    dates = group_sorted['date'].dt.date.tolist()
                    
                    # Find clusters of contracts within window_days
                    for i in range(len(dates) - min_cluster_size + 1):
                        window_end_date = dates[i] + timedelta(days=window_days)
                        
                        # Count contracts in window
                        contracts_in_window = sum(1 for d in dates[i:] 
                                                  if d <= window_end_date)
                        
                        if contracts_in_window >= min_cluster_size:
                            # Add indices of contracts in this cluster
                            cluster_indices = []
                            for j, date in enumerate(dates[i:], i):
                                if date <= window_end_date:
                                    cluster_indices.append(
                                        group_sorted.iloc[j].name)
                                else:
                                    break
                            clustering_indices.update(cluster_indices)
        
        return clustering_indices
    
    def _get_suspicious_pairs_indices(self, df: pd.DataFrame, 
                                      strict: bool = True) -> set:
        """Get indices of suspicious buyer-supplier pairs."""
        required_cols = ['acheteur_id', 'titulaire_id', 'montant']
        if not all(col in df.columns for col in required_cols):
            return set()
        
        if df['montant'].isna().all():
            return set()
        
        # Calculate total amounts per buyer-supplier pair
        pair_totals = df.groupby(['acheteur_id', 'titulaire_id'])['montant'].agg(['sum', 'count'])
        
        # Use mean + 2 standard deviations of total amounts with 2+ contracts
        pair_sums = pair_totals['sum'].dropna()
        if len(pair_sums) == 0:
            return set()
        
        mean_amount = pair_sums.mean()
        std_amount = pair_sums.std()
        
        if pd.isna(std_amount) or std_amount == 0:
            return set()
        
        # For strict mode, use 2 std dev; for non-strict, use 1.5 std dev
        multiplier = 2.0 if strict else 1.5
        suspicious_threshold = mean_amount + (multiplier * std_amount)
        suspicious_pairs = pair_totals[(pair_totals['sum'] > suspicious_threshold) & 
                                     (pair_totals['count'] >= 2)]
        
        # Collect indices of contracts in suspicious pairs
        suspicious_indices = set()
        for (buyer, supplier), _ in suspicious_pairs.iterrows():
            pair_mask = ((df['acheteur_id'] == buyer) & 
                         (df['titulaire_id'] == supplier))
            suspicious_indices.update(df[pair_mask].index.tolist())
        
        return suspicious_indices
    
    def clean_dataset(self, df: pd.DataFrame, 
                     anomaly_types: List[str] = None,
                     strict_threshold: bool = True) -> pd.DataFrame:
        """Remove identified anomalies from dataset.
        
        Args:
            df: DataFrame to clean
            anomaly_types: List of anomaly types to remove
            strict_threshold: Use strict detection thresholds
            
        Returns:
            Cleaned DataFrame with anomalies removed
        """
        anomalous_indices = self.identify_original_anomalies(
            df, anomaly_types, strict_threshold)
        
        if len(anomalous_indices) == 0:
            logger.info("No original anomalies found to remove")
            return df.copy()
        
        # Remove anomalous rows
        df_clean = df.drop(index=anomalous_indices).copy()
        
        logger.info(f"Removed {len(anomalous_indices)} anomalous rows "
                   f"({len(anomalous_indices)/len(df)*100:.2f}%)")
        logger.info(f"Clean dataset: {len(df_clean)} rows "
                   f"(was {len(df)} rows)")
        
        return df_clean



class SyntheticAnomalyGenerator:
    """Generate synthetic anomalies by adding new rows or replacing existing 
    rows in procurement data."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the anomaly generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Track which rows have been modified for each anomaly type
        self.anomaly_labels = {}
        self.replaced_indices = set()
        
    def generate_anomalies(self,
                           df: pd.DataFrame,
                           anomaly_types: List[str] = None,
                           anomaly_percentage: float = 0.05,
                           replace_rows: bool = True) -> pd.DataFrame:
        """Generate synthetic anomalies by adding new rows or replacing 
        existing ones.
        
        Args:
            df: Original procurement dataframe
            anomaly_types: List of anomaly types to generate
            anomaly_percentage: Percentage of synthetic anomalies relative to 
                original data
            replace_rows: If True, replace existing rows instead of adding 
                new ones
            
        Returns:
            DataFrame with anomalous rows and is_synthetic_anomaly column
        """
        
        if anomaly_types is None:
            anomaly_types = [
                'single_bid_competitive',
                'price_inflation',
                'price_deflation',
                'procedure_manipulation',
                'suspicious_modifications',
                'high_market_concentration',
                'temporal_clustering',
                'excessive_subcontracting',
                'short_contract_duration',
                'suspicious_buyer_supplier_pairs'
            ]
        
        # Start with original dataframe
        df_original = df.copy()
        
        # Calculate number of anomalies to create per type
        if replace_rows:
            # When replacing, we can't exceed the original dataset size
            total_anomalies = min(int(len(df_original) * anomaly_percentage),
                                len(df_original))
        else:
            total_anomalies = int(len(df_original) * anomaly_percentage)
            
        anomalies_per_type = max(1, total_anomalies // len(anomaly_types))
        
        mode_str = "replacing" if replace_rows else "adding"
        logger.info(f"Generating {total_anomalies} total synthetic anomaly "
                    f"rows by {mode_str}")
        logger.info(f"Approximately {anomalies_per_type} anomalies per type")
        
        # Store all new anomalous rows and their corresponding indices
        all_anomalous_data = []
        self.replaced_indices = set()
        
        # Method mapping for cleaner code
        method_map = {
            'single_bid_competitive': self._generate_single_bid_anomalies,
            'price_inflation': self._generate_price_inflation_anomalies,
            'price_deflation': self._generate_price_deflation_anomalies,
            'procedure_manipulation': (
                self._generate_procedure_manipulation_anomalies),
            'suspicious_modifications': (
                self._generate_suspicious_modification_anomalies),
            'high_market_concentration': (
                self._generate_high_market_concentration_anomalies),
            'temporal_clustering': (
                self._generate_temporal_clustering_anomalies),
            'excessive_subcontracting': (
                self._generate_excessive_subcontracting_anomalies),
            'short_contract_duration': self._generate_short_duration_anomalies,
            'suspicious_buyer_supplier_pairs': (
                self._generate_suspicious_pairs_anomalies)
        }
        
        # Generate each type of anomaly
        for anomaly_type in anomaly_types:
            if anomaly_type in method_map:
                logger.info(f"Generating {anomaly_type} anomalies...")
                new_rows, template_indices = method_map[anomaly_type](
                    df_original, anomalies_per_type, anomaly_type)
                
                if len(new_rows) > 0:
                    for row, template_idx in zip(new_rows, template_indices):
                        all_anomalous_data.append({
                            'row_data': row,
                            'template_index': template_idx,
                            'anomaly_type': anomaly_type
                        })
                        if replace_rows:
                            self.replaced_indices.add(template_idx)
        
        # Create the final dataset
        if replace_rows:
            df_result = self._create_replacement_dataset(
                df_original, all_anomalous_data, anomaly_types)
        else:
            df_result = self._create_additive_dataset(
                df_original, all_anomalous_data, anomaly_types)
        
        # Log summary
        total_synthetic_anomalies = np.sum(
            df_result['is_synthetic_anomaly'] > 0)
        logger.info(f"Generated {total_synthetic_anomalies} total synthetic "
                    f"anomaly rows")
        logger.info(f"Original dataset: {len(df_original)} rows")
        logger.info(f"Final dataset: {len(df_result)} rows")
        
        if replace_rows:
            logger.info(f"Replaced {len(self.replaced_indices)} original "
                       f"rows")
        else:
            percentage = (total_synthetic_anomalies / len(df_result) * 100)
            logger.info(f"({percentage:.2f}% synthetic)")
        
        # Log anomaly type mapping
        if hasattr(self, 'anomaly_type_mapping'):
            logger.info("Anomaly type mapping:")
            for anomaly_type, type_number in (
                    self.anomaly_type_mapping.items()):
                count = np.sum(self.anomaly_labels[anomaly_type])
                logger.info(f"  - {type_number}: {anomaly_type} "
                            f"({count} anomalies)")
        
        return df_result
    
    def _create_replacement_dataset(self, df_original: pd.DataFrame,
                                   all_anomalous_data: List[Dict],
                                   anomaly_types: List[str]) -> pd.DataFrame:
        """Create dataset by replacing original rows with anomalous ones."""
        
        # Start with original data
        df_result = df_original.copy()
        
        # Create anomaly type mapping
        anomaly_type_mapping = {anomaly_type: i + 1
                               for i, anomaly_type in enumerate(anomaly_types)}
        self.anomaly_type_mapping = anomaly_type_mapping
        
        # Initialize anomaly indicator
        df_result['is_synthetic_anomaly'] = 0
        
        # Initialize anomaly labels
        self.anomaly_labels = {}
        for anomaly_type in anomaly_types:
            self.anomaly_labels[anomaly_type] = np.zeros(len(df_result),
                                                        dtype=bool)
        
        # Replace rows with anomalous versions using vectorized operations
        if all_anomalous_data:
            # Extract data for batch processing
            indices_to_replace = []
            anomaly_rows_data = []
            anomaly_type_per_index = {}
            
            for anomaly_data in all_anomalous_data:
                row_data = anomaly_data['row_data']
                template_idx = anomaly_data['template_index']
                anomaly_type = anomaly_data['anomaly_type']
                
                indices_to_replace.append(template_idx)
                anomaly_rows_data.append(row_data)
                anomaly_type_per_index[template_idx] = anomaly_type
            
            # Create DataFrame from anomalous data
            if anomaly_rows_data:
                anomalous_df = pd.DataFrame(anomaly_rows_data, 
                                            index=indices_to_replace)
                
                # Update original dataframe with anomalous data
                # Only update columns that exist in both dataframes
                common_cols = [col for col in anomalous_df.columns 
                               if col in df_result.columns]
                df_result.loc[indices_to_replace, common_cols] = (
                    anomalous_df[common_cols])
                
                # Set anomaly indicators using vectorized operations
                for template_idx, anomaly_type in (
                        anomaly_type_per_index.items()):
                    if anomaly_type in anomaly_type_mapping:
                        type_number = anomaly_type_mapping[anomaly_type]
                        df_result.loc[template_idx, 'is_synthetic_anomaly'] = (
                            type_number)
                        self.anomaly_labels[anomaly_type][template_idx] = True
        
        return df_result
    
    def _create_additive_dataset(self, df_original: pd.DataFrame,
                                all_anomalous_data: List[Dict],
                                anomaly_types: List[str]) -> pd.DataFrame:
        """Create dataset by adding new anomalous rows."""
        
        # Extract just the row data for adding
        new_rows = [item['row_data'] for item in all_anomalous_data]
        
        # Combine original data with synthetic anomalies
        if new_rows:
            anomalous_df = pd.DataFrame(new_rows)
            df_combined = pd.concat([df_original, anomalous_df],
                                  ignore_index=True)
        else:
            df_combined = df_original.copy()
        
        # Create anomaly labels for the combined dataset
        n_original = len(df_original)
        n_total = len(df_combined)
        
        # Create anomaly type mapping
        anomaly_type_mapping = {anomaly_type: i + 1
                               for i, anomaly_type in enumerate(anomaly_types)}
        self.anomaly_type_mapping = anomaly_type_mapping
        
        # Initialize labels - original rows are False, synthetic rows will be
        # True
        self.anomaly_labels = {}
        for anomaly_type in anomaly_types:
            self.anomaly_labels[anomaly_type] = np.zeros(n_total, dtype=bool)
        
        # Add general anomaly indicator column
        df_combined['is_synthetic_anomaly'] = np.zeros(len(df_combined),
                                                      dtype=int)
        
        # Mark synthetic anomalies
        current_idx = n_original
        for anomaly_data in all_anomalous_data:
            row_data = anomaly_data['row_data']
            anomaly_type = anomaly_data['anomaly_type']
            
            if 'anomaly_type' in row_data:
                if anomaly_type in self.anomaly_labels:
                    self.anomaly_labels[anomaly_type][current_idx] = True
                    # Set the anomaly type number in the indicator column
                    if anomaly_type in anomaly_type_mapping:
                        type_number = anomaly_type_mapping[anomaly_type]
                        df_combined.loc[current_idx, 
                                       'is_synthetic_anomaly'] = type_number
            current_idx += 1
        
        return df_combined
    
    def _generate_single_bid_anomalies(self, df: pd.DataFrame, 
                                     n_anomalies: int, 
                                     anomaly_type: str) -> Tuple[List[Dict], 
                                                                List[int]]:
        """Generate new rows with single bid competitive anomalies.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find competitive procedures with more than 1 bid as templates
        competitive_procedures = ["Appel d'offres ouvert", 
                                "Appel d'offres restreint"]
        
        mask = (df['procedure'].isin(competitive_procedures) & 
                (df['offresRecues'] > 1) & 
                df['offresRecues'].notna())
        
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for "
                          "single_bid_competitive anomalies")
            return [], []
        
        # Randomly select template rows
        selected_rows = eligible_rows.sample(
            n=min(n_anomalies, len(eligible_rows)), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for idx, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            
            # Make it anomalous: set to exactly 1 bid
            new_row['offresRecues'] = 1
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} single bid competitive "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_price_inflation_anomalies(self, df: pd.DataFrame, 
                                          n_anomalies: int, 
                                          anomaly_type: str) -> Tuple[
                                              List[Dict], List[int]]:
        """Generate new rows with artificially inflated prices.
        
        Uses the same logic as OriginalAnomalyAnalyzer: inflates prices to be
        >3 standard deviations above the mean within CPV categories.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Check required columns
        required_cols = ['montant', 'codeCPV_3']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for price inflation: {missing_cols}")
            return [], []
        
        if df['montant'].isna().all():
            logger.warning("No amount data for price inflation anomalies")
            return [], []
        
        # Filter to valid data
        valid_data = df[df['montant'].notna() & df['codeCPV_3'].notna() & 
                       (df['montant'] > 0)].copy()
        if len(valid_data) == 0:
            logger.warning("No valid amount/CPV data for price inflation")
            return [], []
        
        # Calculate statistics per CPV category
        cpv_stats = valid_data.groupby('codeCPV_3')['montant'].agg(
            ['mean', 'std', 'count']).reset_index()
        cpv_stats = cpv_stats[cpv_stats['count'] >= 200]  # Need at least 200 contracts for reliable stats
        
        if len(cpv_stats) == 0:
            logger.warning("No CPV categories with sufficient data for price inflation")
            return [], []
        
        # Select contracts from categories with valid statistics
        eligible_contracts = []
        for _, cpv_row in cpv_stats.iterrows():
            cpv = cpv_row['codeCPV_3']
            mean_amount = cpv_row['mean']
            std_amount = cpv_row['std']
            
            if pd.notna(std_amount) and std_amount > 0:
                # Find contracts in this CPV category that are below the inflation threshold
                cpv_contracts = valid_data[valid_data['codeCPV_3'] == cpv]
                inflation_threshold = mean_amount + (3 * std_amount)
                
                # Select contracts that are currently below the threshold
                below_threshold = cpv_contracts[cpv_contracts['montant'] < inflation_threshold]
                for idx, contract in below_threshold.iterrows():
                    eligible_contracts.append((idx, contract, mean_amount, std_amount))
        
        if len(eligible_contracts) == 0:
            logger.warning("No eligible contracts found for price inflation anomalies")
            return [], []
        
        # Randomly select contracts to inflate
        selected_contracts = random.sample(
            eligible_contracts, 
            min(n_anomalies, len(eligible_contracts)))
        
        new_rows = []
        template_indices = []
        
        for idx, contract, mean_amount, std_amount in selected_contracts:
            new_row = contract.copy()
            
            # Inflate to be >3 std dev above mean (add some randomness)
            inflation_threshold = mean_amount + (3 * std_amount)
            multiplier = random.uniform(1.1, 1.5)  # 10-50% above threshold
            new_row['montant'] = inflation_threshold * multiplier
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} price inflation anomaly rows")
        return new_rows, template_indices
    
    def _generate_price_deflation_anomalies(self, df: pd.DataFrame, 
                                          n_anomalies: int, 
                                          anomaly_type: str) -> Tuple[
                                              List[Dict], List[int]]:
        """Generate new rows with artificially deflated prices.
        
        Uses the same logic as OriginalAnomalyAnalyzer: deflates prices to be
        <3 standard deviations below the mean within CPV categories.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Check required columns
        required_cols = ['montant', 'codeCPV_3']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for price deflation: {missing_cols}")
            return [], []
        
        if df['montant'].isna().all():
            logger.warning("No amount data for price deflation anomalies")
            return [], []
        
        # Filter to valid data
        valid_data = df[df['montant'].notna() & df['codeCPV_3'].notna() & 
                       (df['montant'] > 0)].copy()
        if len(valid_data) == 0:
            logger.warning("No valid amount/CPV data for price deflation")
            return [], []
        
        # Calculate statistics per CPV category
        cpv_stats = valid_data.groupby('codeCPV_3')['montant'].agg(
            ['mean', 'std', 'count']).reset_index()
        cpv_stats = cpv_stats[cpv_stats['count'] >= 200]  # Need at least 200 contracts for reliable stats
        
        if len(cpv_stats) == 0:
            logger.warning("No CPV categories with sufficient data for price deflation")
            return [], []
        
        # Select contracts from categories with valid statistics
        eligible_contracts = []
        for _, cpv_row in cpv_stats.iterrows():
            cpv = cpv_row['codeCPV_3']
            mean_amount = cpv_row['mean']
            std_amount = cpv_row['std']
            
            if pd.notna(std_amount) and std_amount > 0:
                # Find contracts in this CPV category that are above the deflation threshold
                cpv_contracts = valid_data[valid_data['codeCPV_3'] == cpv]
                deflation_threshold = max(mean_amount - (3 * std_amount), 0)
                
                # Select contracts that are currently above the threshold
                above_threshold = cpv_contracts[cpv_contracts['montant'] > deflation_threshold]
                for idx, contract in above_threshold.iterrows():
                    eligible_contracts.append((idx, contract, mean_amount, std_amount))
        
        if len(eligible_contracts) == 0:
            logger.warning("No eligible contracts found for price deflation anomalies")
            return [], []
        
        # Randomly select contracts to deflate
        selected_contracts = random.sample(
            eligible_contracts, 
            min(n_anomalies, len(eligible_contracts)))
        
        new_rows = []
        template_indices = []
        
        for idx, contract, mean_amount, std_amount in selected_contracts:
            new_row = contract.copy()
            
            # Deflate to be <3 std dev below mean (add some randomness)
            deflation_threshold = max(mean_amount - (3 * std_amount), 0)
            # Ensure we don't go negative and add some randomness below threshold
            if deflation_threshold > 0:
                multiplier = random.uniform(0.5, 0.9)  # 50-90% of threshold
                new_row['montant'] = deflation_threshold * multiplier
            else:
                # If threshold is 0, use a small positive value
                new_row['montant'] = mean_amount * random.uniform(0.01, 0.05)
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} price deflation anomaly rows")
        return new_rows, template_indices
    
    def _generate_procedure_manipulation_anomalies(self, df: pd.DataFrame, 
                                                  n_anomalies: int, 
                                                  anomaly_type: str) -> Tuple[
                                                      List[Dict], List[int]]:
        """Generate new rows with suspicious procedure manipulation.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find competitive procedures as templates
        competitive_procedures = ["Appel d'offres ouvert", 
                                "Appel d'offres restreint"]
        mask = df['procedure'].isin(competitive_procedures)
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for procedure "
                          "manipulation anomalies")
            return [], []
        
        # Non-competitive procedures to switch to
        non_competitive = ['Procédure adaptée', 
                          'Marché négocié sans publicité']
        
        # Select random template rows
        selected_rows = eligible_rows.sample(
            n=min(n_anomalies, len(eligible_rows)), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for idx, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            
            # Switch to non-competitive procedure
            new_row['procedure'] = random.choice(non_competitive)
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} procedure manipulation "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_suspicious_modification_anomalies(self, df: pd.DataFrame, 
                                                   n_anomalies: int, 
                                                   anomaly_type: str) -> Tuple[
                                                       List[Dict], List[int]]:
        """Generate new rows suggesting suspicious contract modifications.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find contracts with reasonable duration as templates
        mask = (df['dureeMois'].notna() & (df['dureeMois'] > 0) & 
                (df['dureeMois'] < 36))
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for suspicious "
                          "modification anomalies")
            return [], []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(
            n=min(n_anomalies, len(eligible_rows)), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for idx, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            original_duration = new_row['dureeMois']
            
            # Dramatically increase duration (simulate contract modification)
            new_duration = original_duration * random.uniform(2.5, 5.0)
            new_row['dureeMois'] = new_duration
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} suspicious modification "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_high_market_concentration_anomalies(self, df: pd.DataFrame, 
                                                      n_anomalies: int, 
                                                      anomaly_type: str) -> Tuple[
                                                          List[Dict], List[int]]:
        """Generate new rows with high market concentration anomalies.
        
        Creates anomalies where a single supplier dominates a buyer's contracts,
        indicating potential market concentration issues.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find buyer-CPV combinations with multiple suppliers as templates
        buyer_cpv_groups = df.groupby(['acheteur_id', 'codeCPV_3'])
        
        eligible_groups = []
        for (buyer_id, cpv), group in buyer_cpv_groups:
            if len(group) >= 3 and group['titulaire_id'].nunique() > 1:
                eligible_groups.append((buyer_id, cpv, group))
        
        if len(eligible_groups) == 0:
            logger.warning("No eligible buyer-CPV groups found for market "
                          "concentration anomalies")
            return [], []
        
        # Select random groups to create anomalies from
        selected_groups = random.sample(eligible_groups, 
                                       min(n_anomalies // 3, 
                                           len(eligible_groups)))
        
        new_rows = []
        template_indices = []
        
        for buyer_id, cpv, group in selected_groups:
            # Pick one supplier to dominate
            suppliers = group['titulaire_id'].unique()
            dominant_supplier = random.choice(suppliers)
            
            # Create new contracts for this dominant supplier
            template_rows = group.sample(n=min(3, len(group)), 
                                       random_state=self.random_seed)
            
            for idx, row in template_rows.iterrows():
                new_row = row.copy()
                new_row['titulaire_id'] = dominant_supplier
                
                # Add anomaly metadata
                new_row['anomaly_type'] = anomaly_type
                new_row['source_type'] = 'synthetic'
                
                new_rows.append(new_row.to_dict())
                template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} high market concentration "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_temporal_clustering_anomalies(self, df: pd.DataFrame, 
                                                n_anomalies: int, 
                                                anomaly_type: str) -> Tuple[
                                                    List[Dict], List[int]]:
        """Generate new rows with suspicious temporal clustering patterns.
        
        Uses the same logic as OriginalAnomalyAnalyzer: creates clusters of 4+
        contracts within 30 days for the same buyer-supplier-CPV combination.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Check for required columns
        if 'dateNotification' not in df.columns:
            logger.warning("No date data for temporal clustering anomalies")
            return [], []
        
        # Convert dates and filter to valid data
        try:
            df_temp = df.copy()
            df_temp['date_parsed'] = pd.to_datetime(
                df_temp['dateNotification'], errors='coerce')
            df_temp = df_temp.dropna(subset=['date_parsed', 'acheteur_id', 
                                           'titulaire_id', 'codeCPV_3'])
            
            if len(df_temp) == 0:
                logger.warning("No valid dates for temporal clustering anomalies")
                return [], []
            
            # Find buyer-supplier-CPV combinations with multiple contracts
            buyer_supplier_cpv_groups = df_temp.groupby(['acheteur_id', 
                                                        'titulaire_id', 
                                                        'codeCPV_3'])
            
            eligible_groups = []
            for (buyer_id, supplier_id, cpv), group in buyer_supplier_cpv_groups:
                if len(group) >= 2:  # Need at least 2 to create cluster of 4
                    eligible_groups.append((buyer_id, supplier_id, cpv, group))
            
            if len(eligible_groups) == 0:
                logger.warning("No eligible buyer-supplier-CPV groups found for "
                              "temporal clustering anomalies")
                return [], []
            
            # Select groups to create clustered contracts from
            selected_groups = random.sample(
                eligible_groups,
                min(len(eligible_groups), max(1, n_anomalies // 4)))
            
            new_rows = []
            template_indices = []
            
            for buyer_id, supplier_id, cpv, group in selected_groups:
                # Pick template contracts (at least 4 for proper clustering)
                n_contracts_to_create = 4
                if len(group) < n_contracts_to_create:
                    # If not enough templates, duplicate some
                    template_contracts = group.sample(
                        n=len(group), random_state=self.random_seed)
                    # Repeat templates to get 4 contracts
                    template_list = list(template_contracts.iterrows())
                    while len(template_list) < n_contracts_to_create:
                        template_list.extend(list(template_contracts.iterrows()))
                    template_contracts = template_list[:n_contracts_to_create]
                else:
                    template_contracts = list(group.sample(
                        n=n_contracts_to_create, 
                        random_state=self.random_seed).iterrows())
                
                # Pick a random base date and cluster contracts within 30 days
                base_date = datetime(2023, random.randint(1, 12), 
                                   random.randint(1, 28))
                
                for i, (idx, row) in enumerate(template_contracts):
                    new_row = row.copy()
                    
                    # Create clustered date (within 30 days, like analyzer)
                    # Ensure all 4+ contracts are within 30 days of first one
                    clustered_date = base_date + timedelta(
                        days=random.randint(0, 29))  # 0-29 days = within 30 days
                    new_row['dateNotification'] = clustered_date.strftime(
                        '%Y-%m-%d')
                    
                    # Add anomaly metadata
                    new_row['anomaly_type'] = anomaly_type
                    new_row['source_type'] = 'synthetic'
                    
                    new_rows.append(new_row.to_dict())
                    template_indices.append(idx)
            
            logger.info(f"Generated {len(new_rows)} temporal clustering "
                       "anomaly rows")
            return new_rows, template_indices
            
        except Exception as e:
            logger.error(f"Error generating temporal clustering anomalies: {str(e)}")
            return [], []
    
    def _generate_excessive_subcontracting_anomalies(self, df: pd.DataFrame, 
                                                      n_anomalies: int, 
                                                      anomaly_type: str) -> Tuple[
                                                          List[Dict], List[int]]:
        """Generate new rows with excessive subcontracting patterns.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find contracts that currently don't declare subcontracting as templates
        mask = (df['sousTraitanceDeclaree'].notna() & 
                (df['sousTraitanceDeclaree'] == 0))
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for excessive "
                          "subcontracting anomalies")
            return [], []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(
            n=min(n_anomalies, len(eligible_rows)), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for idx, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            
            # Make it anomalous: mark as having declared subcontracting
            new_row['sousTraitanceDeclaree'] = 1
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} excessive subcontracting "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_short_duration_anomalies(self, df: pd.DataFrame, 
                                           n_anomalies: int, 
                                           anomaly_type: str) -> Tuple[
                                               List[Dict], List[int]]:
        """Generate new rows with unusually short contract durations.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find contracts with valid duration (> 1 month) as templates
        mask = df['dureeMois'].notna() & (df['dureeMois'] > 1)
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for unusual duration "
                          "anomalies")
            return [], []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(
            n=min(n_anomalies, len(eligible_rows)), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for idx, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            
            # Make it anomalous: set to very short duration (< 1 month)
            new_row['dureeMois'] = random.uniform(0.1, 0.9)
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} unusual short duration "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_suspicious_pairs_anomalies(self, df: pd.DataFrame, 
                                             n_anomalies: int, 
                                             anomaly_type: str) -> Tuple[
                                                 List[Dict], List[int]]:
        """Generate new rows with suspicious buyer-supplier relationship patterns.
        
        Creates anomalies by inflating contract amounts for buyer-supplier pairs
        to exceed mean + 2 standard deviations of total pair amounts.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Check required columns
        required_cols = ['acheteur_id', 'titulaire_id', 'montant']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for suspicious pairs: {missing_cols}")
            return [], []
        
        if df['montant'].isna().all():
            logger.warning("No amount data for suspicious pairs anomalies")
            return [], []
        
        # Calculate current pair totals to determine target threshold
        pair_totals = df.groupby(['acheteur_id', 'titulaire_id'])['montant'].agg(['sum', 'count'])
        pair_sums = pair_totals['sum'].dropna()
        
        if len(pair_sums) == 0:
            logger.warning("No valid pair amounts for suspicious pairs anomalies")
            return [], []
        
        mean_amount = pair_sums.mean()
        std_amount = pair_sums.std()
        
        if pd.isna(std_amount) or std_amount == 0:
            logger.warning("Cannot calculate standard deviation for suspicious pairs")
            return [], []
        
        # Target threshold for suspicious pairs
        suspicious_threshold = mean_amount + (2 * std_amount)
        
        # Find pairs that are currently below the threshold
        below_threshold_pairs = pair_totals[
            (pair_totals['sum'] < suspicious_threshold) & 
            (pair_totals['count'] >= 2) &
            (pair_totals['sum'].notna()) &
            (pair_totals['sum'] > 0)
        ]
        
        if len(below_threshold_pairs) == 0:
            logger.warning("No eligible pairs below threshold for suspicious pairs anomalies")
            return [], []
        
        # Select pairs to make suspicious
        selected_pairs = below_threshold_pairs.sample(
            min(len(below_threshold_pairs), n_anomalies // 2), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for (buyer_id, supplier_id), pair_info in selected_pairs.iterrows():
            current_total = pair_info['sum']
            
            # Find contracts for this pair as templates
            pair_contracts = df[(df['acheteur_id'] == buyer_id) & 
                              (df['titulaire_id'] == supplier_id)]
            
            # Calculate how much we need to inflate to exceed threshold
            target_total = suspicious_threshold * random.uniform(1.1, 1.4)  # 10-40% above threshold
            inflation_factor = target_total / current_total
            
            # Create new contracts with inflated amounts for this pair
            template_contracts = pair_contracts.sample(
                n=min(2, len(pair_contracts)), 
                random_state=self.random_seed)
            
            for idx, template_row in template_contracts.iterrows():
                new_row = template_row.copy()
                
                # Make it anomalous: inflate to reach target threshold
                original_amount = new_row['montant']
                if pd.notna(original_amount) and original_amount > 0:
                    new_row['montant'] = original_amount * inflation_factor
                
                # Add anomaly metadata
                new_row['anomaly_type'] = anomaly_type
                new_row['source_type'] = 'synthetic'
                
                new_rows.append(new_row.to_dict())
                template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} suspicious buyer-supplier "
                   f"pair anomaly rows (target threshold: {suspicious_threshold:,.0f})")
        return new_rows, template_indices
    
    def get_anomaly_summary(self) -> pd.DataFrame:
        """Get a summary of generated anomalies."""
        
        summary_data = []
        for anomaly_type, labels in self.anomaly_labels.items():
            summary_data.append({
                'anomaly_type': anomaly_type,
                'count': np.sum(labels),
                'percentage': (np.sum(labels) / len(labels) * 100 
                             if len(labels) > 0 else 0)
            })
        
        return pd.DataFrame(summary_data).sort_values('count', 
                                                     ascending=False)
    
    def get_replaced_indices(self) -> set:
        """Get the indices of rows that were replaced (if replace mode was 
        used)."""
        return self.replaced_indices.copy()
    
    def save_anomaly_labels(self, filepath: str):
        """Save anomaly labels to a file."""
        
        # Convert boolean arrays to a DataFrame
        labels_df = pd.DataFrame(self.anomaly_labels)
        labels_df.to_csv(filepath, index=True)
        logger.info(f"Anomaly labels saved to {filepath}")
    
    def load_anomaly_labels(self, filepath: str):
        """Load anomaly labels from a file."""
        
        labels_df = pd.read_csv(filepath, index_col=0)
        self.anomaly_labels = {}
        for col in labels_df.columns:
            self.anomaly_labels[col] = labels_df[col].values.astype(bool)
        logger.info(f"Anomaly labels loaded from {filepath}")
    
    def get_anomaly_type_mapping(self) -> Dict[str, int]:
        """Get the mapping between anomaly types and their numeric codes.
        
        Returns:
            Dictionary mapping anomaly type names to numeric codes (1-N)
        """
        return getattr(self, 'anomaly_type_mapping', {})
    
    def get_reverse_anomaly_mapping(self) -> Dict[int, str]:
        """Get the reverse mapping from numeric codes to anomaly type names.
        
        Returns:
            Dictionary mapping numeric codes to anomaly type names
        """
        mapping = self.get_anomaly_type_mapping()
        return {v: k for k, v in mapping.items()}


def demonstrate_anomaly_generation(df: pd.DataFrame, 
                                   sample_size: int = 1000) -> None:
    """Demonstrate the anomaly generation functionality with replacement 
    option."""
    
    print("=== Synthetic Anomaly Generation Demonstration ===\n")
    
    # Take a sample for demonstration
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Using a sample of {sample_size} contracts for demonstration")
    else:
        df_sample = df.copy()
        print(f"Using full dataset of {len(df)} contracts")
    
    # Initialize generator
    generator = SyntheticAnomalyGenerator(random_seed=42)
    
    # Test both modes
    print("\n--- Mode 1: Adding new rows ---")
    df_with_added = generator.generate_anomalies(
        df_sample,
        anomaly_percentage=0.05,
        replace_rows=False,
        anomaly_types=['single_bid_competitive', 'price_inflation']
    )
    
    print(f"Original: {len(df_sample)} rows")
    print(f"With additions: {len(df_with_added)} rows")
    print(f"Added: {len(df_with_added) - len(df_sample)} rows")
    
    print("\n--- Mode 2: Replacing existing rows ---")
    generator2 = SyntheticAnomalyGenerator(random_seed=42)
    df_with_replaced = generator2.generate_anomalies(
        df_sample,
        anomaly_percentage=0.05,
        replace_rows=True,
        anomaly_types=['price_deflation', 'procedure_manipulation']
    )
    
    print(f"Original: {len(df_sample)} rows")
    print(f"With replacements: {len(df_with_replaced)} rows")
    print(f"Replaced indices: {len(generator2.get_replaced_indices())}")
    
    # Show summaries
    print("\nSummary for addition mode:")
    summary1 = generator.get_anomaly_summary()
    print(summary1.to_string(index=False))
    
    print("\nSummary for replacement mode:")
    summary2 = generator2.get_anomaly_summary()
    print(summary2.to_string(index=False))
    
    # Show example of replaced indices
    replaced_indices = generator2.get_replaced_indices()
    print(f"\nFirst 5 replaced indices: {list(replaced_indices)[:5]}")


if __name__ == "__main__":
    # Example usage
    print("Synthetic Anomaly Generator - Example Usage")
    print("This module creates synthetic anomalies by either adding new rows")
    print("or replacing existing rows in the dataset.")
    print("Each anomaly generation method returns both the new rows and the")
    print("indices of the template rows that were used.")
    print("See the demonstrate_anomaly_generation() function for usage "
          "examples.") 