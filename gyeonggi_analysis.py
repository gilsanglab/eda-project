import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Korean font (MacOS)
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'eda_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    print("Loading data...")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    
    # Preprocessing
    price_cols = ['결제금액', '주문취소 금액', '실결제 금액', '판매단가', '공급단가']
    for col in price_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)
            
    return df

def analyze_gyeonggi_revenue(df):
    print("Analyzing Gyeonggi-do Revenue Logic...")
    
    # Check necessary columns
    required_cols = ['광역지역', '셀러명', '실결제 금액']
    for col in required_cols:
        if col not in df.columns:
            print(f"Missing column: {col}")
            return

    # 1. Basic Stats by Region
    region_stats = df.groupby('광역지역').agg(
        TotalRevenue=('실결제 금액', 'sum'),
        SellerCount=('셀러명', 'nunique'),
        OrderCount=('주문번호', 'count')
    ).reset_index()
    
    region_stats['AvgRevenuePerSeller'] = region_stats['TotalRevenue'] / region_stats['SellerCount']
    region_stats = region_stats.sort_values(by='TotalRevenue', ascending=False)
    
    print("\n[Region Revenue Stats]")
    print(region_stats[['광역지역', 'TotalRevenue', 'SellerCount', 'AvgRevenuePerSeller']].head(10))

    # 2. Seller Revenue Distribution
    seller_revenues = df.groupby(['광역지역', '셀러명'])['실결제 금액'].sum().reset_index()
    
    # Identify "High Revenue Sellers" (Top 20% overall)
    revenue_threshold = seller_revenues['실결제 금액'].quantile(0.80)
    print(f"\nTop 20% Seller Revenue Threshold: {revenue_threshold:,.0f} KRW")
    
    seller_revenues['IsHighRevenue'] = seller_revenues['실결제 금액'] > revenue_threshold
    
    # Count High Revenue Sellers per Region
    high_revenue_stats = seller_revenues.groupby('광역지역').agg(
        TotalSellers=('셀러명', 'count'),
        HighRevenueSellers=('IsHighRevenue', 'sum')
    ).reset_index()
    
    high_revenue_stats['HighRevenuePercent'] = (high_revenue_stats['HighRevenueSellers'] / high_revenue_stats['TotalSellers']) * 100
    high_revenue_stats = high_revenue_stats.sort_values(by='HighRevenueSellers', ascending=False)
    
    print("\n[High Revenue Seller Stats by Region]")
    print(high_revenue_stats.head(10))
    
    # 3. Visualization
    plt.figure(figsize=(12, 6))
    
    # 3.1 Total Revenue by Region
    plt.subplot(1, 2, 1)
    sns.barplot(x='TotalRevenue', y='광역지역', data=region_stats.head(10), palette='Blues_r')
    plt.title('상위 10개 지역 총 매출')
    plt.xlabel('총 매출 (원)')
    
    # 3.2 High Revenue Seller Count by Region
    plt.subplot(1, 2, 2)
    sns.barplot(x='HighRevenueSellers', y='광역지역', data=high_revenue_stats.head(10), palette='Reds_r')
    plt.title('상위 10개 지역 고매출 셀러 수 (Top 20%)')
    plt.xlabel('고매출 셀러 수')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gyeonggi_revenue_analysis.png')
    print(f"\nSaved visualization to {OUTPUT_DIR}/gyeonggi_revenue_analysis.png")
    
    # 3.3 Boxplot of Seller Revenues (Log scale for visibility)
    plt.figure(figsize=(12, 6))
    # Filter top regions for cleaner plot
    top_regions = region_stats.head(10)['광역지역'].tolist()
    filtered_sellers = seller_revenues[seller_revenues['광역지역'].isin(top_regions)]
    
    sns.boxplot(x='광역지역', y='실결제 금액', data=filtered_sellers, order=top_regions, palette='Set2')
    plt.yscale('log')
    plt.title('지역별 셀러 매출 분포 (Log Scale)')
    plt.ylabel('셀러별 총 매출 (Log)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gyeonggi_seller_revenue_dist.png')
    
    # Specific Answer Construction
    gyeonggi_row = high_revenue_stats[high_revenue_stats['광역지역'].isin(['경기', '경기도'])]
    if not gyeonggi_row.empty:
        gg_total = gyeonggi_row['TotalSellers'].sum()
        gg_high = gyeonggi_row['HighRevenueSellers'].sum()
        gg_pct = (gg_high / gg_total * 100) if gg_total > 0 else 0
        print(f"\n[Conclusion Hint]")
        print(f"Gyeonggi-do has {gg_high} high-revenue sellers out of {gg_total} total sellers ({gg_pct:.1f}%).")

def main():
    filepath = 'data/project1 - preprocessed_data.csv'
    df = load_data(filepath)
    if df is not None:
        analyze_gyeonggi_revenue(df)

if __name__ == "__main__":
    main()
