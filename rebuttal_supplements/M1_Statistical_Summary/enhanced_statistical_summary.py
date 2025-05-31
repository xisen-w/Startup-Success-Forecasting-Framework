import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
import json
from collections import Counter
import re
warnings.filterwarnings('ignore')

class EnhancedStartupDataInvestigation:
    """
    Deep investigation of startup founders and companies beyond L1-L5 segmentation
    WHO are the founders? WHAT are the companies? WHERE is the investigation?
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
        self.investigation_results = {}
        
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
    
    def parse_founder_profiles(self):
        """Extract detailed founder information from structured_info column"""
        print("\n" + "="*80)
        print("👥 FOUNDER PROFILE INVESTIGATION")
        print("="*80)
        
        founder_profiles = []
        
        for idx, row in self.combined_df.iterrows():
            try:
                # Parse structured_info if it exists
                if pd.notna(row.get('structured_info')):
                    if isinstance(row['structured_info'], str):
                        founder_info = eval(row['structured_info'])  # Convert string dict to dict
                    else:
                        founder_info = row['structured_info']
                    
                    profile = {
                        'org_name': row.get('org_name', 'Unknown'),
                        'outcome': row.get('outcome', 'Unknown'),
                        'segment': row.get('segment', 'Unknown'),
                        'founder_name': founder_info.get('name', 'Unknown'),
                        'gender': founder_info.get('gender', 'Unknown'),
                        'birth_date': founder_info.get('birthDate', 'Unknown'),
                        'country_code': row.get('country_code', 'Unknown'),
                        'city': row.get('city', 'Unknown'),
                        'category_list': row.get('category_list', 'Unknown'),
                        'founded_on': row.get('founded_on', 'Unknown'),
                        'status': row.get('status', 'Unknown')
                    }
                    founder_profiles.append(profile)
                    
            except Exception as e:
                continue
        
        self.founder_profiles_df = pd.DataFrame(founder_profiles)
        
        print(f"📋 Extracted {len(self.founder_profiles_df)} founder profiles")
        print(f"🔍 Sample founder names: {list(self.founder_profiles_df['founder_name'].head(10))}")
        
        return self.founder_profiles_df
    
    def investigate_founder_demographics(self):
        """Deep dive into founder demographics and characteristics"""
        print("\n" + "="*80)
        print("🔍 FOUNDER DEMOGRAPHICS INVESTIGATION")
        print("="*80)
        
        # Gender analysis
        print("\n👥 GENDER DISTRIBUTION:")
        gender_analysis = self.founder_profiles_df.groupby(['outcome', 'gender']).size().unstack(fill_value=0)
        print(gender_analysis)
        
        gender_success_rates = {}
        for gender in self.founder_profiles_df['gender'].unique():
            if gender != 'Unknown' and gender != '':
                gender_data = self.founder_profiles_df[self.founder_profiles_df['gender'] == gender]
                if len(gender_data) > 0:
                    success_rate = len(gender_data[gender_data['outcome'] == 'Successful']) / len(gender_data) * 100
                    gender_success_rates[gender] = success_rate
                    print(f"  {gender}: {success_rate:.1f}% success rate")
        
        # Geographic analysis
        print("\n🌍 GEOGRAPHIC DISTRIBUTION:")
        geo_analysis = self.founder_profiles_df.groupby(['outcome', 'country_code']).size().unstack(fill_value=0)
        top_countries = self.founder_profiles_df['country_code'].value_counts().head(10)
        print("Top 10 countries by founder count:")
        for country, count in top_countries.items():
            if country != 'Unknown':
                country_data = self.founder_profiles_df[self.founder_profiles_df['country_code'] == country]
                success_rate = len(country_data[country_data['outcome'] == 'Successful']) / len(country_data) * 100
                print(f"  {country}: {count} founders ({success_rate:.1f}% success rate)")
        
        # City analysis
        print("\n🏙️ TOP STARTUP CITIES:")
        top_cities = self.founder_profiles_df.groupby(['city', 'country_code']).size().sort_values(ascending=False).head(15)
        for (city, country), count in top_cities.items():
            if city != 'Unknown' and country != 'Unknown':
                city_data = self.founder_profiles_df[
                    (self.founder_profiles_df['city'] == city) & 
                    (self.founder_profiles_df['country_code'] == country)
                ]
                success_rate = len(city_data[city_data['outcome'] == 'Successful']) / len(city_data) * 100
                print(f"  {city}, {country}: {count} founders ({success_rate:.1f}% success rate)")
        
        self.investigation_results['demographics'] = {
            'gender_success_rates': gender_success_rates,
            'top_countries': top_countries.to_dict(),
            'top_cities': top_cities.to_dict()
        }
    
    def investigate_company_characteristics(self):
        """Deep dive into company types, industries, and characteristics"""
        print("\n" + "="*80)
        print("🏢 COMPANY CHARACTERISTICS INVESTIGATION")
        print("="*80)
        
        # Industry analysis
        print("\n🏭 INDUSTRY ANALYSIS:")
        all_categories = []
        for categories in self.founder_profiles_df['category_list'].dropna():
            if categories != 'Unknown':
                # Split categories and clean them
                cats = [cat.strip() for cat in str(categories).split(',')]
                all_categories.extend(cats)
        
        category_counter = Counter(all_categories)
        top_categories = category_counter.most_common(20)
        
        print("Top 20 industries by startup count:")
        for category, count in top_categories:
            # Calculate success rate for this category
            category_startups = self.founder_profiles_df[
                self.founder_profiles_df['category_list'].str.contains(category, na=False)
            ]
            if len(category_startups) > 0:
                success_rate = len(category_startups[category_startups['outcome'] == 'Successful']) / len(category_startups) * 100
                print(f"  {category}: {count} startups ({success_rate:.1f}% success rate)")
        
        # Company status analysis
        print("\n📊 COMPANY STATUS DISTRIBUTION:")
        status_analysis = self.founder_profiles_df.groupby(['outcome', 'status']).size().unstack(fill_value=0)
        print(status_analysis)
        
        # Founding year analysis
        print("\n📅 FOUNDING YEAR TRENDS:")
        # Extract year from founded_on
        self.founder_profiles_df['founding_year'] = self.founder_profiles_df['founded_on'].apply(
            lambda x: self._extract_year(x) if pd.notna(x) else None
        )
        
        year_analysis = self.founder_profiles_df.groupby(['founding_year', 'outcome']).size().unstack(fill_value=0)
        recent_years = year_analysis.tail(10)
        print("Recent founding years (last 10 years with data):")
        print(recent_years)
        
        self.investigation_results['companies'] = {
            'top_categories': top_categories,
            'status_distribution': status_analysis.to_dict(),
            'founding_trends': recent_years.to_dict()
        }
    
    def investigate_success_patterns(self):
        """Investigate patterns that distinguish successful from unsuccessful startups"""
        print("\n" + "="*80)
        print("🎯 SUCCESS PATTERN INVESTIGATION")
        print("="*80)
        
        # Segment-specific success patterns
        print("\n📈 SUCCESS PATTERNS BY SEGMENT:")
        for segment in ['L1', 'L2', 'L3', 'L4', 'L5']:
            segment_data = self.founder_profiles_df[self.founder_profiles_df['segment'] == segment]
            if len(segment_data) > 0:
                print(f"\n  {segment} FOUNDERS:")
                
                # Top successful companies in this segment
                successful_segment = segment_data[segment_data['outcome'] == 'Successful']
                if len(successful_segment) > 0:
                    print(f"    Sample successful companies:")
                    sample_companies = successful_segment[['org_name', 'founder_name', 'category_list', 'country_code']].head(5)
                    for _, company in sample_companies.iterrows():
                        print(f"      • {company['org_name']} (Founder: {company['founder_name']}, Industry: {str(company['category_list'])[:50]}...)")
                
                # Geographic concentration
                top_countries_segment = segment_data['country_code'].value_counts().head(3)
                print(f"    Top countries: {dict(top_countries_segment)}")
                
                # Industry concentration
                segment_categories = []
                for categories in segment_data['category_list'].dropna():
                    if categories != 'Unknown':
                        cats = [cat.strip() for cat in str(categories).split(',')]
                        segment_categories.extend(cats)
                
                if segment_categories:
                    top_segment_categories = Counter(segment_categories).most_common(3)
                    print(f"    Top industries: {[cat for cat, count in top_segment_categories]}")
    
    def investigate_notable_founders(self):
        """Identify and investigate notable founders and companies"""
        print("\n" + "="*80)
        print("⭐ NOTABLE FOUNDERS & COMPANIES INVESTIGATION")
        print("="*80)
        
        # L5 founders (most elite)
        print("\n🏆 L5 FOUNDERS (ELITE LEVEL):")
        l5_founders = self.founder_profiles_df[self.founder_profiles_df['segment'] == 'L5']
        if len(l5_founders) > 0:
            print(f"Total L5 founders: {len(l5_founders)}")
            successful_l5 = l5_founders[l5_founders['outcome'] == 'Successful']
            print(f"Successful L5 founders: {len(successful_l5)}")
            
            print("\nNotable L5 successful founders:")
            for _, founder in successful_l5.head(10).iterrows():
                print(f"  • {founder['founder_name']} - {founder['org_name']} ({founder['category_list'][:50]}...)")
        
        # High-growth industries
        print("\n🚀 HIGH-GROWTH INDUSTRIES:")
        industry_success = {}
        for categories in self.founder_profiles_df['category_list'].dropna():
            if categories != 'Unknown':
                cats = [cat.strip() for cat in str(categories).split(',')]
                for cat in cats:
                    if cat not in industry_success:
                        industry_success[cat] = {'total': 0, 'successful': 0}
                    
                    cat_data = self.founder_profiles_df[
                        self.founder_profiles_df['category_list'].str.contains(cat, na=False)
                    ]
                    industry_success[cat]['total'] = len(cat_data)
                    industry_success[cat]['successful'] = len(cat_data[cat_data['outcome'] == 'Successful'])
        
        # Calculate success rates and filter for significant industries
        high_success_industries = []
        for industry, stats in industry_success.items():
            if stats['total'] >= 10:  # At least 10 companies
                success_rate = stats['successful'] / stats['total'] * 100
                if success_rate >= 60:  # At least 60% success rate
                    high_success_industries.append((industry, success_rate, stats['total']))
        
        high_success_industries.sort(key=lambda x: x[1], reverse=True)
        print("Industries with >60% success rate (min 10 companies):")
        for industry, rate, total in high_success_industries[:10]:
            print(f"  • {industry}: {rate:.1f}% success rate ({total} companies)")
    
    def _extract_year(self, date_str):
        """Extract year from various date formats"""
        if pd.isna(date_str) or date_str == 'Unknown':
            return None
        
        # Try to extract 4-digit year
        year_match = re.search(r'(\d{4})', str(date_str))
        if year_match:
            year = int(year_match.group(1))
            if 1990 <= year <= 2024:  # Reasonable range
                return year
        return None
    
    def create_investigation_visualizations(self):
        """Create visualizations for the investigation"""
        print("\n" + "="*80)
        print("📊 CREATING INVESTIGATION VISUALIZATIONS")
        print("="*80)
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Startup Success Investigation - Beyond L1-L5 Segmentation', fontsize=16, fontweight='bold')
        
        # 1. Geographic distribution
        top_countries = self.founder_profiles_df['country_code'].value_counts().head(10)
        axes[0, 0].bar(range(len(top_countries)), top_countries.values)
        axes[0, 0].set_xticks(range(len(top_countries)))
        axes[0, 0].set_xticklabels(top_countries.index, rotation=45)
        axes[0, 0].set_title('Top 10 Countries by Founder Count')
        axes[0, 0].set_ylabel('Number of Founders')
        
        # 2. Gender distribution by outcome
        gender_data = self.founder_profiles_df[self.founder_profiles_df['gender'].isin(['Male', 'Female'])]
        gender_outcome = pd.crosstab(gender_data['gender'], gender_data['outcome'])
        gender_outcome.plot(kind='bar', ax=axes[0, 1], color=['#ff6b6b', '#4ecdc4'])
        axes[0, 1].set_title('Gender Distribution by Outcome')
        axes[0, 1].set_xlabel('Gender')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend(title='Outcome')
        axes[0, 1].tick_params(axis='x', rotation=0)
        
        # 3. Founding year trends
        if 'founding_year' in self.founder_profiles_df.columns:
            year_data = self.founder_profiles_df.dropna(subset=['founding_year'])
            year_outcome = pd.crosstab(year_data['founding_year'], year_data['outcome'])
            recent_years = year_outcome.tail(15)  # Last 15 years
            recent_years.plot(kind='line', ax=axes[1, 0], marker='o')
            axes[1, 0].set_title('Founding Trends (Recent Years)')
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('Number of Startups')
            axes[1, 0].legend(title='Outcome')
        
        # 4. Success rate by segment (detailed)
        segment_success = []
        for segment in ['L1', 'L2', 'L3', 'L4', 'L5']:
            segment_data = self.founder_profiles_df[self.founder_profiles_df['segment'] == segment]
            if len(segment_data) > 0:
                success_rate = len(segment_data[segment_data['outcome'] == 'Successful']) / len(segment_data) * 100
                segment_success.append((segment, success_rate, len(segment_data)))
        
        if segment_success:
            segments, rates, counts = zip(*segment_success)
            bars = axes[1, 1].bar(segments, rates, color='lightblue', edgecolor='navy')
            axes[1, 1].set_title('Success Rate by Founder Segment')
            axes[1, 1].set_ylabel('Success Rate (%)')
            axes[1, 1].set_xlabel('Founder Segment')
            
            # Add count labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # 5. Top cities
        top_cities_data = self.founder_profiles_df.groupby(['city', 'country_code']).size().sort_values(ascending=False).head(10)
        city_labels = [f"{city}, {country}" for (city, country), count in top_cities_data.items()]
        axes[2, 0].barh(range(len(top_cities_data)), top_cities_data.values)
        axes[2, 0].set_yticks(range(len(top_cities_data)))
        axes[2, 0].set_yticklabels(city_labels, fontsize=8)
        axes[2, 0].set_title('Top 10 Startup Cities')
        axes[2, 0].set_xlabel('Number of Founders')
        
        # 6. Company status distribution
        status_counts = self.founder_profiles_df['status'].value_counts()
        axes[2, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
        axes[2, 1].set_title('Company Status Distribution')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.project_root, 'enhanced_investigation_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Investigation visualizations saved to: {output_path}")
        
        plt.show()
    
    def export_investigation_report(self):
        """Export detailed investigation findings"""
        print("\n" + "="*80)
        print("📄 EXPORTING INVESTIGATION REPORT")
        print("="*80)
        
        # Create detailed founder profiles export
        detailed_profiles = self.founder_profiles_df.copy()
        
        # Add success rate by various dimensions
        detailed_profiles['country_success_rate'] = detailed_profiles.groupby('country_code')['outcome'].transform(
            lambda x: (x == 'Successful').mean() * 100
        )
        
        detailed_profiles['segment_success_rate'] = detailed_profiles.groupby('segment')['outcome'].transform(
            lambda x: (x == 'Successful').mean() * 100
        )
        
        # Export to CSV
        output_path = os.path.join(self.project_root, 'detailed_founder_investigation.csv')
        detailed_profiles.to_csv(output_path, index=False)
        print(f"📊 Detailed investigation exported to: {output_path}")
        
        # Create summary statistics
        summary_stats = {
            'total_founders': len(self.founder_profiles_df),
            'unique_companies': self.founder_profiles_df['org_name'].nunique(),
            'countries_represented': self.founder_profiles_df['country_code'].nunique(),
            'cities_represented': self.founder_profiles_df['city'].nunique(),
            'overall_success_rate': len(self.founder_profiles_df[self.founder_profiles_df['outcome'] == 'Successful']) / len(self.founder_profiles_df) * 100
        }
        
        print(f"\n📋 INVESTIGATION SUMMARY:")
        for key, value in summary_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        return detailed_profiles
    
    def run_complete_investigation(self):
        """Run the complete investigation"""
        print("🔍 STARTING COMPREHENSIVE STARTUP INVESTIGATION")
        print("=" * 80)
        
        # Load data
        if not self.load_data():
            return
        
        # Parse founder profiles
        self.parse_founder_profiles()
        
        # Run all investigations
        self.investigate_founder_demographics()
        self.investigate_company_characteristics()
        self.investigate_success_patterns()
        self.investigate_notable_founders()
        
        # Create visualizations
        self.create_investigation_visualizations()
        
        # Export results
        detailed_report = self.export_investigation_report()
        
        print("\n" + "="*80)
        print("✅ INVESTIGATION COMPLETE!")
        print("="*80)
        print(f"🔍 Deep investigation reveals:")
        print(f"  • {len(self.founder_profiles_df)} founder profiles analyzed")
        print(f"  • {self.founder_profiles_df['org_name'].nunique()} unique companies")
        print(f"  • {self.founder_profiles_df['country_code'].nunique()} countries represented")
        print(f"  • Detailed patterns beyond L1-L5 segmentation discovered")
        print(f"📄 Comprehensive reports saved to project directory")
        
        return detailed_report

def main():
    """Main execution function"""
    investigator = EnhancedStartupDataInvestigation()
    detailed_report = investigator.run_complete_investigation()
    return investigator, detailed_report

if __name__ == "__main__":
    investigator, report = main() 