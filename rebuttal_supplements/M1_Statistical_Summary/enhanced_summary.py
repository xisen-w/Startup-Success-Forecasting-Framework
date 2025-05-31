import pandas as pd
import numpy as np
from collections import Counter

def run_real_investigation():
    print("🔍 REAL STARTUP DATA INVESTIGATION")
    print("="*50)
    
    try:
        # Load the company data files we know exist
        print("Loading company data...")
        successful_companies = pd.read_csv('data/successful/Moneyball 1.1_ success - Company.csv')
        unsuccessful_companies = pd.read_csv('data/Unsuccessful/Moneyball 1.1_ fail - Company.csv')
        
        successful_companies['outcome'] = 'Successful'
        unsuccessful_companies['outcome'] = 'Unsuccessful'
        
        # Combine the datasets
        all_companies = pd.concat([successful_companies, unsuccessful_companies], ignore_index=True)
        
        print(f"✅ Data loaded successfully!")
        print(f"📊 Successful companies: {len(successful_companies):,}")
        print(f"📉 Unsuccessful companies: {len(unsuccessful_companies):,}")
        print(f"📈 Total companies: {len(all_companies):,}")
        
        # Show actual column structure
        print(f"\n📋 Available columns:")
        for i, col in enumerate(all_companies.columns):
            print(f"  {i+1:2d}. {col}")
        
        # Geographic analysis
        print(f"\n🌍 TOP 15 COUNTRIES:")
        country_analysis = all_companies.groupby(['country_code', 'outcome']).size().unstack(fill_value=0)
        country_totals = country_analysis.sum(axis=1).sort_values(ascending=False).head(15)
        
        for country in country_totals.index:
            if pd.notna(country):
                successful = country_analysis.loc[country, 'Successful'] if 'Successful' in country_analysis.columns else 0
                unsuccessful = country_analysis.loc[country, 'Unsuccessful'] if 'Unsuccessful' in country_analysis.columns else 0
                total = successful + unsuccessful
                success_rate = (successful / total * 100) if total > 0 else 0
                print(f"  {country}: {total:,} companies ({success_rate:.1f}% success rate)")
        
        # City analysis
        print(f"\n🏙️ TOP 15 CITIES:")
        city_analysis = all_companies.groupby(['city', 'country_code']).size().sort_values(ascending=False).head(15)
        
        for (city, country), count in city_analysis.items():
            if pd.notna(city) and pd.notna(country):
                city_data = all_companies[(all_companies['city'] == city) & (all_companies['country_code'] == country)]
                successful_count = len(city_data[city_data['outcome'] == 'Successful'])
                success_rate = (successful_count / count * 100) if count > 0 else 0
                print(f"  {city}, {country}: {count:,} companies ({success_rate:.1f}% success rate)")
        
        # Industry analysis
        print(f"\n🏭 TOP 20 INDUSTRIES:")
        all_categories = []
        for categories in all_companies['category_list'].dropna():
            if pd.notna(categories) and categories != '':
                cats = [cat.strip() for cat in str(categories).split(',')]
                all_categories.extend(cats)
        
        category_counter = Counter(all_categories)
        top_categories = category_counter.most_common(20)
        
        for category, count in top_categories:
            if category and category.strip():
                category_companies = all_companies[all_companies['category_list'].str.contains(category, na=False)]
                successful_count = len(category_companies[category_companies['outcome'] == 'Successful'])
                success_rate = (successful_count / len(category_companies) * 100) if len(category_companies) > 0 else 0
                print(f"  {category}: {count:,} companies ({success_rate:.1f}% success rate)")
        
        # Status analysis
        print(f"\n📊 COMPANY STATUS DISTRIBUTION:")
        status_analysis = all_companies.groupby(['status', 'outcome']).size().unstack(fill_value=0)
        print(status_analysis)
        
        # Sample successful companies
        print(f"\n⭐ SAMPLE SUCCESSFUL COMPANIES:")
        successful_sample = successful_companies.sample(min(10, len(successful_companies)))
        for _, company in successful_sample.iterrows():
            categories = str(company['category_list'])[:50] + "..." if len(str(company['category_list'])) > 50 else str(company['category_list'])
            print(f"  • {company['org_name']} ({categories}) - {company['city']}, {company['country_code']}")
        
        # Overall statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        total_companies = len(all_companies)
        successful_count = len(successful_companies)
        success_rate = (successful_count / total_companies * 100) if total_companies > 0 else 0
        
        print(f"  Total companies analyzed: {total_companies:,}")
        print(f"  Successful companies: {successful_count:,}")
        print(f"  Unsuccessful companies: {len(unsuccessful_companies):,}")
        print(f"  Overall success rate: {success_rate:.1f}%")
        print(f"  Countries represented: {all_companies['country_code'].nunique():,}")
        print(f"  Cities represented: {all_companies['city'].nunique():,}")
        print(f"  Unique industries: {len(set(all_categories)):,}")
        
        return all_companies
        
    except Exception as e:
        print(f"❌ Error during investigation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = run_real_investigation()