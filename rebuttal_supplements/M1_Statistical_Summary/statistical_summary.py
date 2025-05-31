import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

class StartupDataStatisticalSummary:
    """
    Comprehensive statistical analysis of startup success/failure dataset
    Based on L1-L5 founder segmentation levels
    """
    
    def __init__(self):
        # Set up project-relative paths
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.project_root, 'data')
        
        # Load datasets
        self.successful_df = None
        self.unsuccessful_df = None
        self.combined_df = None
        
        # Results storage
        self.summary_stats = {}
        
    def load_data(self):
        """Load the finalized segmented datasets"""
        try:
            # Load successful startups data
            successful_path = os.path.join(self.data_path, 'successful', 'finalised_segmented_profiles_successful.csv')
            self.successful_df = pd.read_csv(successful_path)
            self.successful_df['outcome'] = 'Successful'
            
            # Load unsuccessful startups data  
            unsuccessful_path = os.path.join(self.data_path, 'Unsuccessful', 'finalised_segmented_profiles_unsuccessful.csv')
            self.unsuccessful_df = pd.read_csv(unsuccessful_path)
            self.unsuccessful_df['outcome'] = 'Unsuccessful'
            
            # Combine datasets
            self.combined_df = pd.concat([self.successful_df, self.unsuccessful_df], ignore_index=True)
            
            print(f"✅ Data loaded successfully!")
            print(f"📊 Successful startups: {len(self.successful_df):,} records")
            print(f"📉 Unsuccessful startups: {len(self.unsuccessful_df):,} records")
            print(f"📈 Total dataset: {len(self.combined_df):,} records")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        return True
    
    def data_quality_assessment(self):
        """Comprehensive data quality and structure analysis"""
        print("\n" + "="*80)
        print("🔍 DATA QUALITY & STRUCTURE ASSESSMENT")
        print("="*80)
        
        # Dataset dimensions
        print("\n📏 DATASET DIMENSIONS:")
        print(f"Successful startups shape: {self.successful_df.shape}")
        print(f"Unsuccessful startups shape: {self.unsuccessful_df.shape}")
        print(f"Combined dataset shape: {self.combined_df.shape}")
        
        # Column information
        print(f"\n📋 COLUMNS: {list(self.combined_df.columns)}")
        
        # Data types
        print(f"\n🏷️ DATA TYPES:")
        for col in self.combined_df.columns:
            print(f"  {col}: {self.combined_df[col].dtype}")
        
        # Missing values analysis
        print(f"\n❓ MISSING VALUES ANALYSIS:")
        missing_stats = []
        for dataset_name, df in [("Successful", self.successful_df), 
                                ("Unsuccessful", self.unsuccessful_df),
                                ("Combined", self.combined_df)]:
            missing_counts = df.isnull().sum()
            missing_pct = (missing_counts / len(df)) * 100
            
            print(f"\n  {dataset_name} Dataset:")
            for col in df.columns:
                if missing_counts[col] > 0:
                    print(f"    {col}: {missing_counts[col]:,} ({missing_pct[col]:.2f}%)")
                    
        # Segment data quality
        print(f"\n🏷️ SEGMENT DATA QUALITY:")
        for dataset_name, df in [("Successful", self.successful_df), ("Unsuccessful", self.unsuccessful_df)]:
            print(f"\n  {dataset_name} - Segment Distribution:")
            if 'segment' in df.columns:
                segment_counts = df['segment'].value_counts()
                total = len(df)
                for segment, count in segment_counts.items():
                    percentage = (count / total) * 100
                    print(f"    {segment}: {count:,} ({percentage:.2f}%)")
                    
                # Check for invalid segments
                valid_segments = ['L1', 'L2', 'L3', 'L4', 'L5']
                invalid_segments = df[~df['segment'].isin(valid_segments)]['segment'].unique()
                if len(invalid_segments) > 0:
                    print(f"    ⚠️ Invalid segments found: {invalid_segments}")
    
    def segment_distribution_analysis(self):
        """Detailed analysis of L1-L5 segment distributions"""
        print("\n" + "="*80)
        print("📊 FOUNDER SEGMENTATION ANALYSIS (L1-L5)")
        print("="*80)
        
        # Overall segment distribution
        print("\n🎯 OVERALL SEGMENT DISTRIBUTION:")
        combined_segments = self.combined_df['segment'].value_counts().sort_index()
        total_records = len(self.combined_df)
        
        for segment in ['L1', 'L2', 'L3', 'L4', 'L5']:
            if segment in combined_segments.index:
                count = combined_segments[segment]
                percentage = (count / total_records) * 100
                print(f"  {segment}: {count:,} ({percentage:.2f}%)")
        
        # Segment distribution by outcome
        print(f"\n📈 SEGMENT DISTRIBUTION BY OUTCOME:")
        segment_outcome_crosstab = pd.crosstab(self.combined_df['segment'], 
                                             self.combined_df['outcome'], 
                                             margins=True)
        print(segment_outcome_crosstab)
        
        # Success rates by segment
        print(f"\n🎖️ SUCCESS RATES BY FOUNDER SEGMENT:")
        success_rates = {}
        
        for segment in ['L1', 'L2', 'L3', 'L4', 'L5']:
            segment_data = self.combined_df[self.combined_df['segment'] == segment]
            if len(segment_data) > 0:
                successful_count = len(segment_data[segment_data['outcome'] == 'Successful'])
                total_count = len(segment_data)
                success_rate = (successful_count / total_count) * 100
                success_rates[segment] = success_rate
                
                print(f"  {segment}: {successful_count}/{total_count} = {success_rate:.2f}% success rate")
        
        # Relative success multipliers (compared to L1)
        print(f"\n🔢 SUCCESS RATE MULTIPLIERS (vs L1):")
        if 'L1' in success_rates and success_rates['L1'] > 0:
            l1_rate = success_rates['L1']
            for segment, rate in success_rates.items():
                multiplier = rate / l1_rate
                print(f"  {segment}: {multiplier:.2f}x more likely to succeed than L1")
        
        self.summary_stats['segment_distribution'] = segment_outcome_crosstab
        self.summary_stats['success_rates'] = success_rates
        
    def statistical_significance_tests(self):
        """Statistical tests for segment-outcome relationships"""
        print("\n" + "="*80)
        print("🧪 STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)
        
        # Chi-square test for independence
        contingency_table = pd.crosstab(self.combined_df['segment'], self.combined_df['outcome'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        print(f"\n🔬 CHI-SQUARE TEST OF INDEPENDENCE:")
        print(f"  H0: Founder segment and startup outcome are independent")
        print(f"  H1: Founder segment and startup outcome are dependent")
        print(f"  Chi-square statistic: {chi2:.4f}")
        print(f"  p-value: {p_value:.2e}")
        print(f"  Degrees of freedom: {dof}")
        print(f"  Result: {'REJECT H0' if p_value < 0.05 else 'FAIL TO REJECT H0'} (α = 0.05)")
        
        if p_value < 0.05:
            print(f"  ✅ Strong evidence that founder segment significantly affects startup outcome")
        else:
            print(f"  ❌ Insufficient evidence that founder segment affects startup outcome")
        
        # Cramér's V for effect size
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        print(f"  Cramér's V (effect size): {cramers_v:.4f}")
        
        effect_interpretation = "Small" if cramers_v < 0.3 else "Medium" if cramers_v < 0.5 else "Large"
        print(f"  Effect size interpretation: {effect_interpretation}")
        
        # Pairwise comparisons between segments
        print(f"\n🔍 PAIRWISE SEGMENT COMPARISONS:")
        segments = ['L1', 'L2', 'L3', 'L4', 'L5']
        
        for i, seg1 in enumerate(segments):
            for seg2 in segments[i+1:]:
                seg1_data = self.combined_df[self.combined_df['segment'] == seg1]
                seg2_data = self.combined_df[self.combined_df['segment'] == seg2]
                
                if len(seg1_data) > 0 and len(seg2_data) > 0:
                    # Fisher's exact test for small samples
                    seg1_success = len(seg1_data[seg1_data['outcome'] == 'Successful'])
                    seg1_total = len(seg1_data)
                    seg2_success = len(seg2_data[seg2_data['outcome'] == 'Successful'])
                    seg2_total = len(seg2_data)
                    
                    # Create 2x2 contingency table
                    table_2x2 = [[seg1_success, seg1_total - seg1_success],
                                [seg2_success, seg2_total - seg2_success]]
                    
                    odds_ratio, p_val = stats.fisher_exact(table_2x2)
                    
                    print(f"  {seg1} vs {seg2}:")
                    print(f"    Odds ratio: {odds_ratio:.3f}")
                    print(f"    p-value: {p_val:.4f}")
                    print(f"    Significant: {'Yes' if p_val < 0.05 else 'No'}")
        
        self.summary_stats['chi_square_test'] = {
            'chi2': chi2, 'p_value': p_value, 'cramers_v': cramers_v
        }
    
    def descriptive_statistics(self):
        """Comprehensive descriptive statistics"""
        print("\n" + "="*80)
        print("📈 DESCRIPTIVE STATISTICS")
        print("="*80)
        
        # Basic dataset statistics
        print(f"\n📊 BASIC DATASET STATISTICS:")
        print(f"  Total records: {len(self.combined_df):,}")
        print(f"  Successful startups: {len(self.successful_df):,} ({len(self.successful_df)/len(self.combined_df)*100:.1f}%)")
        print(f"  Unsuccessful startups: {len(self.unsuccessful_df):,} ({len(self.unsuccessful_df)/len(self.combined_df)*100:.1f}%)")
        
        # Segment concentration metrics
        print(f"\n🎯 SEGMENT CONCENTRATION METRICS:")
        segment_counts = self.combined_df['segment'].value_counts()
        
        # Gini coefficient for segment distribution
        def gini_coefficient(x):
            sorted_x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        gini = gini_coefficient(segment_counts.values)
        print(f"  Gini coefficient (segment inequality): {gini:.4f}")
        print(f"  Interpretation: {'Highly unequal' if gini > 0.4 else 'Moderately unequal' if gini > 0.25 else 'Relatively equal'} distribution")
        
        # Entropy for segment diversity
        segment_probs = segment_counts / segment_counts.sum()
        entropy = -np.sum(segment_probs * np.log2(segment_probs))
        max_entropy = np.log2(len(segment_counts))
        normalized_entropy = entropy / max_entropy
        
        print(f"  Shannon entropy: {entropy:.4f}")
        print(f"  Normalized entropy: {normalized_entropy:.4f}")
        print(f"  Diversity interpretation: {'High' if normalized_entropy > 0.8 else 'Medium' if normalized_entropy > 0.6 else 'Low'} segment diversity")
        
        # Success rate statistics
        print(f"\n🏆 SUCCESS RATE STATISTICS:")
        if 'success_rates' in self.summary_stats:
            rates = list(self.summary_stats['success_rates'].values())
            print(f"  Mean success rate: {np.mean(rates):.2f}%")
            print(f"  Median success rate: {np.median(rates):.2f}%")
            print(f"  Standard deviation: {np.std(rates):.2f}%")
            print(f"  Range: {np.min(rates):.2f}% - {np.max(rates):.2f}%")
            print(f"  Coefficient of variation: {np.std(rates)/np.mean(rates):.4f}")
    
    def advanced_segment_analysis(self):
        """Advanced analysis of segment characteristics"""
        print("\n" + "="*80)
        print("🔬 ADVANCED SEGMENT ANALYSIS")
        print("="*80)
        
        # Detailed segment profiles
        print(f"\n👥 DETAILED SEGMENT PROFILES:")
        
        for segment in ['L1', 'L2', 'L3', 'L4', 'L5']:
            segment_data = self.combined_df[self.combined_df['segment'] == segment]
            if len(segment_data) == 0:
                continue
                
            successful_count = len(segment_data[segment_data['outcome'] == 'Successful'])
            unsuccessful_count = len(segment_data[segment_data['outcome'] == 'Unsuccessful'])
            total_count = len(segment_data)
            success_rate = (successful_count / total_count) * 100
            
            print(f"\n  {segment} FOUNDER PROFILE:")
            print(f"    Total founders: {total_count:,}")
            print(f"    Successful: {successful_count:,}")
            print(f"    Unsuccessful: {unsuccessful_count:,}")
            print(f"    Success rate: {success_rate:.2f}%")
            
            # Calculate confidence interval for success rate
            if total_count > 0:
                p = success_rate / 100
                margin_error = 1.96 * np.sqrt(p * (1 - p) / total_count)
                ci_lower = max(0, (p - margin_error) * 100)
                ci_upper = min(100, (p + margin_error) * 100)
                print(f"    95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
            
            # Market share within successful/unsuccessful
            if successful_count > 0:
                success_market_share = (successful_count / len(self.successful_df)) * 100
                print(f"    Share of successful startups: {success_market_share:.2f}%")
            
            if unsuccessful_count > 0:
                failure_market_share = (unsuccessful_count / len(self.unsuccessful_df)) * 100
                print(f"    Share of unsuccessful startups: {failure_market_share:.2f}%")
        
        # Risk analysis
        print(f"\n⚠️ RISK ANALYSIS:")
        total_unsuccessful = len(self.unsuccessful_df)
        total_successful = len(self.successful_df)
        
        for segment in ['L1', 'L2', 'L3', 'L4', 'L5']:
            segment_data = self.combined_df[self.combined_df['segment'] == segment]
            if len(segment_data) == 0:
                continue
                
            unsuccessful_in_segment = len(segment_data[segment_data['outcome'] == 'Unsuccessful'])
            risk_ratio = (unsuccessful_in_segment / total_unsuccessful) / (len(segment_data) / len(self.combined_df))
            
            print(f"  {segment}: Risk ratio = {risk_ratio:.2f}")
            print(f"    Interpretation: {'Higher' if risk_ratio > 1 else 'Lower'} failure risk than average")
    
    def export_summary_report(self):
        """Export comprehensive summary to file"""
        print("\n" + "="*80)
        print("📄 EXPORTING SUMMARY REPORT")
        print("="*80)
        
        # Create summary DataFrame
        summary_data = []
        
        for segment in ['L1', 'L2', 'L3', 'L4', 'L5']:
            segment_data = self.combined_df[self.combined_df['segment'] == segment]
            if len(segment_data) == 0:
                continue
                
            successful_count = len(segment_data[segment_data['outcome'] == 'Successful'])
            unsuccessful_count = len(segment_data[segment_data['outcome'] == 'Unsuccessful'])
            total_count = len(segment_data)
            success_rate = (successful_count / total_count) * 100 if total_count > 0 else 0
            
            summary_data.append({
                'Segment': segment,
                'Total_Founders': total_count,
                'Successful': successful_count,
                'Unsuccessful': unsuccessful_count,
                'Success_Rate_Percent': success_rate,
                'Share_of_Dataset_Percent': (total_count / len(self.combined_df)) * 100,
                'Share_of_Successful_Percent': (successful_count / len(self.successful_df)) * 100 if successful_count > 0 else 0,
                'Share_of_Unsuccessful_Percent': (unsuccessful_count / len(self.unsuccessful_df)) * 100 if unsuccessful_count > 0 else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Export to CSV
        output_path = os.path.join(self.project_root, 'statistical_summary_report.csv')
        summary_df.to_csv(output_path, index=False)
        print(f"📊 Summary report exported to: {output_path}")
        
        # Display summary table
        print(f"\n📋 SUMMARY TABLE:")
        print(summary_df.to_string(index=False, float_format='%.2f'))
        
        return summary_df
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*80)
        print("📊 CREATING VISUALIZATIONS")
        print("="*80)
        
        # Set up matplotlib style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Startup Success Forecasting Framework - Statistical Analysis', fontsize=16, fontweight='bold')
        
        # 1. Segment distribution pie chart
        segment_counts = self.combined_df['segment'].value_counts().sort_index()
        axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Overall Segment Distribution')
        
        # 2. Success rates by segment
        if 'success_rates' in self.summary_stats:
            segments = list(self.summary_stats['success_rates'].keys())
            rates = list(self.summary_stats['success_rates'].values())
            bars = axes[0, 1].bar(segments, rates, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
            axes[0, 1].set_title('Success Rates by Founder Segment')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].set_xlabel('Founder Segment')
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. Stacked bar chart of outcomes by segment
        segment_outcome = pd.crosstab(self.combined_df['segment'], self.combined_df['outcome'])
        segment_outcome.plot(kind='bar', stacked=True, ax=axes[0, 2], color=['#ff6b6b', '#4ecdc4'])
        axes[0, 2].set_title('Startup Outcomes by Founder Segment')
        axes[0, 2].set_xlabel('Founder Segment')
        axes[0, 2].set_ylabel('Number of Startups')
        axes[0, 2].legend(title='Outcome')
        axes[0, 2].tick_params(axis='x', rotation=0)
        
        # 4. Success rate comparison with confidence intervals
        segments_data = []
        for segment in ['L1', 'L2', 'L3', 'L4', 'L5']:
            segment_data = self.combined_df[self.combined_df['segment'] == segment]
            if len(segment_data) > 0:
                successful_count = len(segment_data[segment_data['outcome'] == 'Successful'])
                total_count = len(segment_data)
                success_rate = (successful_count / total_count) * 100
                
                # Calculate confidence interval
                p = success_rate / 100
                margin_error = 1.96 * np.sqrt(p * (1 - p) / total_count) * 100 if total_count > 0 else 0
                
                segments_data.append({
                    'segment': segment,
                    'success_rate': success_rate,
                    'margin_error': margin_error,
                    'total_count': total_count
                })
        
        if segments_data:
            segments_df = pd.DataFrame(segments_data)
            axes[1, 0].errorbar(segments_df['segment'], segments_df['success_rate'], 
                              yerr=segments_df['margin_error'], fmt='o-', capsize=5)
            axes[1, 0].set_title('Success Rates with 95% Confidence Intervals')
            axes[1, 0].set_xlabel('Founder Segment')
            axes[1, 0].set_ylabel('Success Rate (%)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Sample size by segment
        sample_sizes = self.combined_df['segment'].value_counts().sort_index()
        bars = axes[1, 1].bar(sample_sizes.index, sample_sizes.values, color='lightblue', edgecolor='navy')
        axes[1, 1].set_title('Sample Size by Founder Segment')
        axes[1, 1].set_xlabel('Founder Segment')
        axes[1, 1].set_ylabel('Number of Founders')
        
        # Add value labels on bars
        for bar, count in zip(bars, sample_sizes.values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(sample_sizes)*0.01,
                           f'{count:,}', ha='center', va='bottom')
        
        # 6. Success multiplier vs L1
        if 'success_rates' in self.summary_stats:
            rates = self.summary_stats['success_rates']
            if 'L1' in rates and rates['L1'] > 0:
                l1_rate = rates['L1']
                multipliers = {seg: rate/l1_rate for seg, rate in rates.items()}
                
                segments = list(multipliers.keys())
                mult_values = list(multipliers.values())
                
                bars = axes[1, 2].bar(segments, mult_values, color='lightcoral', edgecolor='darkred')
                axes[1, 2].axhline(y=1, color='black', linestyle='--', alpha=0.7)
                axes[1, 2].set_title('Success Rate Multiplier vs L1 Founders')
                axes[1, 2].set_xlabel('Founder Segment')
                axes[1, 2].set_ylabel('Multiplier (vs L1)')
                
                # Add value labels
                for bar, mult in zip(bars, mult_values):
                    height = bar.get_height()
                    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                                   f'{mult:.1f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.project_root, 'statistical_analysis_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualizations saved to: {output_path}")
        
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete statistical analysis"""
        print("🚀 STARTING COMPREHENSIVE STATISTICAL ANALYSIS")
        print("=" * 80)
        
        # Load data
        if not self.load_data():
            return
        
        # Run all analyses
        self.data_quality_assessment()
        self.segment_distribution_analysis()
        self.statistical_significance_tests()
        self.descriptive_statistics()
        self.advanced_segment_analysis()
        
        # Export results
        summary_df = self.export_summary_report()
        
        # Create visualizations
        self.create_visualizations()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"📊 Total records analyzed: {len(self.combined_df):,}")
        print(f"📈 Key finding: L5 founders are {self.summary_stats['success_rates']['L5']/self.summary_stats['success_rates']['L1']:.1f}x more likely to succeed than L1 founders")
        print(f"🔬 Statistical significance: p-value = {self.summary_stats['chi_square_test']['p_value']:.2e}")
        print(f"📄 Reports saved to project directory")
        
        return summary_df

def main():
    """Main execution function"""
    analyzer = StartupDataStatisticalSummary()
    summary_df = analyzer.run_complete_analysis()
    return analyzer, summary_df

if __name__ == "__main__":
    analyzer, summary = main() 