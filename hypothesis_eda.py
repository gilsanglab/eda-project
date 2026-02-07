import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm

# Set Korean font
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'hypothesis_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    print("Loading data...")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    
    # Preprocessing
    if '주문일' in df.columns:
        df['주문일'] = pd.to_datetime(df['주문일'])
        df['Month'] = df['주문일'].dt.to_period('M')
        
    price_cols = ['결제금액', '주문취소 금액', '실결제 금액', '판매단가', '공급가 총합']
    for col in price_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)
            
    # Calculate Margin if possible
    if '실결제 금액' in df.columns and '공급단가' in df.columns and '주문수량' in df.columns:
        # Ensure numerical types
        if df['공급단가'].dtype == 'object':
             df['공급단가'] = df['공급단가'].str.replace(',', '').astype(float)
             
        # Avoid division by zero
        # Margin = (Revenue - Cost) / Revenue
        # Cost = Supply Unit Price * Quantity
        df['TotalSupplyCost'] = df['공급단가'] * df['주문수량']
        df['Profit'] = df['실결제 금액'] - df['TotalSupplyCost']
        
        # Filter out cases where Revenue is <= 0 to avoid -inf/nan
        df['MarginRate'] = df.apply(lambda x: (x['Profit'] / x['실결제 금액']) if x['실결제 금액'] > 0 else 0, axis=1)

    print(f"Data Loaded. Shape: {df.shape}")
    return df

def hypothesis_1_gyeonggi_sellers(df):
    print("Analyzing Hypothesis 1: Gyeonggi-do & Sellers")
    if '광역지역' not in df.columns or '셀러명' not in df.columns: return

    gyeonggi_df = df[df['광역지역'].isin(['경기', '경기도'])]
    
    # Top Sellers in Gyeonggi (by Revenue)
    top_sellers_gg = gyeonggi_df.groupby('셀러명')['실결제 금액'].sum().sort_values(ascending=False).head(10)
    
    # Compare with Total Revenue share
    total_revenue_per_seller = df.groupby('셀러명')['실결제 금액'].sum()
    gg_revenue_per_seller = gyeonggi_df.groupby('셀러명')['실결제 금액'].sum()
    
    share_df = pd.DataFrame({'Total': total_revenue_per_seller, 'Gyeonggi': gg_revenue_per_seller})
    share_df = share_df.fillna(0)
    share_df['Gyeonggi_Share'] = share_df['Gyeonggi'] / share_df['Total']
    
    # Filter for significant sellers (e.g., Total Revenue > 1,000,000 KRW)
    share_df = share_df[share_df['Total'] > 1000000].sort_values(by='Gyeonggi', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=share_df.index, y=share_df['Gyeonggi_Share'] * 100, palette='Blues_r')
    plt.title('상위 셀러의 경기도 매출 비중 (%)')
    plt.ylabel('경기도 매출 비중 (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/h1_gyeonggi_seller_share.png')
    plt.close()

def hypothesis_2_3_event_products(df):
    print("Analyzing Hypothesis 2 & 3: Event Products (Sales & Margin)")
    if '이벤트 여부' not in df.columns: return

    # Sales Quantity
    avg_qty = df.groupby('이벤트 여부')['주문수량'].mean()
    
    plt.figure(figsize=(6, 5))
    sns.barplot(x=avg_qty.index, y=avg_qty.values, palette='Set2')
    plt.title('이벤트 여부에 따른 평균 주문 수량')
    plt.ylabel('평균 주문 수량')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/h2_event_qty.png')
    plt.close()

    # Margin Rate
    if 'MarginRate' in df.columns:
        avg_margin = df.groupby('이벤트 여부')['MarginRate'].mean()
        
        plt.figure(figsize=(6, 5))
        sns.barplot(x=avg_margin.index, y=avg_margin.values * 100, palette='RdYlGn')
        plt.title('이벤트 여부에 따른 평균 순이익률')
        plt.ylabel('순이익률 (%)')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/h3_event_margin.png')
        plt.close()

def hypothesis_4_gift_options(df):
    print("Analyzing Hypothesis 4: Gift Purpose & Options")
    if '목적' not in df.columns: return

    gift_df = df[df['목적'] == '선물']
    personal_df = df[df['목적'] == '개인소비']

    # 1. Average Price Comparison
    avg_price = df.groupby('목적')['실결제 금액'].mean()
    
    plt.figure(figsize=(6, 5))
    sns.barplot(x=avg_price.index, y=avg_price.values, palette='Purples')
    plt.title('구매 목적별 평균 객단가 비교')
    plt.ylabel('평균 객단가 (원)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/h4_gift_price_comparison.png')
    plt.close()
    
    # 2. Fruit Size Preference (if exists)
    if '과수 크기' in df.columns:
        plt.figure(figsize=(10, 6))
        # Normalize to 100% for comparison
        props = df.groupby(['목적', '과수 크기'])['주문번호'].count().unstack(fill_value=0)
        props = props.div(props.sum(axis=1), axis=0) * 100
        
        props.T.plot(kind='bar', stacked=False, figsize=(10, 6), colormap='coolwarm')
        plt.title('구매 목적별 과수 크기 선호도 (%)')
        plt.ylabel('비율 (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/h4_gift_fruit_size.png')
        plt.close()

def hypothesis_5_seller_repurchase(df):
    print("Analyzing Hypothesis 5: Seller Repurchase")
    if '셀러명' not in df.columns or '주문자연락처' not in df.columns: return

    # For each seller, finding their own loyal customers is complex if global repurchase is already calc'd.
    # Let's simple calc: For each seller, what % of their Unique Customers have count > 1 IN THIS DATASET for THAT SELLER?
    
    seller_stats = []
    
    sellers = df['셀러명'].unique()
    for seller in sellers:
        s_df = df[df['셀러명'] == seller]
        user_counts = s_df['주문자연락처'].value_counts()
        total_users = len(user_counts)
        repurchase_users = len(user_counts[user_counts > 1])
        
        # Only consider sellers with decent volume (> 50 orders)
        if len(s_df) > 50:
            seller_stats.append({
                'Seller': seller,
                'RepurchaseRate': repurchase_users / total_users if total_users > 0 else 0,
                'TotalOrders': len(s_df)
            })
            
    stats_df = pd.DataFrame(seller_stats).sort_values(by='RepurchaseRate', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(y='Seller', x='RepurchaseRate', data=stats_df, palette='Blues_r')
    plt.title('셀러별 재구매 고객 비율 (Top 10)')
    plt.xlabel('재구매율 (해당 셀러 내)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/h5_seller_repurchase.png')
    plt.close()

def hypothesis_6_seller_strategy(df):
    print("Analyzing Hypothesis 6: Seller Product Strategy")
    # Practical vs Premium? Let's use avg unit price per seller
    if '판매단가' not in df.columns: return
    
    seller_avg_price = df.groupby('셀러명')['판매단가'].mean().reset_index()
    seller_sales = df.groupby('셀러명')['실결제 금액'].sum().reset_index()
    
    merged = pd.merge(seller_avg_price, seller_sales, on='셀러명')
    merged.columns = ['Seller', 'AvgPrice', 'TotalRevenue']
    
    # Filter small sellers
    merged = merged[merged['TotalRevenue'] > 500000]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='AvgPrice', y='TotalRevenue', data=merged, hue='Seller', legend=False)
    plt.title('셀러별 전략: 평균 단가 vs 총 매출')
    plt.xlabel('평균 판매 단가')
    plt.ylabel('총 매출')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/h6_seller_strategy.png')
    plt.close()

def hypothesis_7_seller_trends(df):
    print("Analyzing Hypothesis 7: Seller Sales Trends")
    if 'Month' not in df.columns: return
    
    # Top 5 Sellers
    top_sellers = df.groupby('셀러명')['실결제 금액'].sum().sort_values(ascending=False).head(5).index
    
    trend_data = df[df['셀러명'].isin(top_sellers)]
    monthly_sales = trend_data.groupby(['Month', '셀러명'])['실결제 금액'].sum().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='line', marker='o')
    plt.title('상위 5개 셀러 월별 매출 추이')
    plt.xlabel('월')
    plt.ylabel('매출액')
    plt.legend(title='셀러명')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/h7_seller_trends.png')
    plt.close()

def hypothesis_8_seller_retention(df):
    print("Analyzing Hypothesis 8: Seller Retention")
    if 'Month' not in df.columns: return

    # Active sellers per month
    active_sellers = df.groupby('Month')['셀러명'].nunique()
    
    plt.figure(figsize=(10, 5))
    active_sellers.plot(kind='bar', color='skyblue')
    plt.title('월별 활동 셀러 수')
    plt.xlabel('월')
    plt.ylabel('셀러 수')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/h8_active_sellers.png')
    plt.close()

def hypothesis_9_seller_channel(df):
    print("Analyzing Hypothesis 9: Seller Preferred Channel")
    if '주문경로' not in df.columns: return
    
    top_sellers = df.groupby('셀러명')['실결제 금액'].sum().sort_values(ascending=False).head(5).index
    filtered_df = df[df['셀러명'].isin(top_sellers)]
    
    ct = pd.crosstab(filtered_df['셀러명'], filtered_df['주문경로'])
    # Normalize
    ct = ct.div(ct.sum(axis=1), axis=0) * 100
    
    ct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab10')
    plt.title('상위 셀러별 판매 채널 선호도 (%)')
    plt.ylabel('비율 (%)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/h9_seller_channel.png')
    plt.close()

def main():
    filepath = 'data/project1 - preprocessed_data.csv'
    df = load_data(filepath)
    if df is None: return

    hypothesis_1_gyeonggi_sellers(df)
    hypothesis_2_3_event_products(df)
    hypothesis_4_gift_options(df)
    hypothesis_5_seller_repurchase(df)
    hypothesis_6_seller_strategy(df)
    hypothesis_7_seller_trends(df)
    hypothesis_8_seller_retention(df)
    hypothesis_9_seller_channel(df)
    
    print("Hypothesis Analysis Completed.")

if __name__ == "__main__":
    main()
