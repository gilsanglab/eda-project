import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set font for Korean support based on OS
import platform
system_name = platform.system()
if system_name == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
elif system_name == 'Windows': # Windows
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

def analyze_seller_repurchase(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # 1. Date Conversion
    if '주문일' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['주문일'])
        df['Date'] = df['OrderDate'].dt.date
    else:
        print("Error: '주문일' column not found.")
        return

    # Check required columns
    required_cols = ['셀러명', '주문자연락처', 'Date', '실결제 금액']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: '{col}' column not found.")
            return

    # 2. Logic: Group by Seller, then Customer
    print("Calculating repurchase rates...")
    
    # Filter for valid sellers (e.g., at least N orders to be significant)
    # But first let's calculate for everyone
    
    results = []
    
    # Get list of unique sellers
    sellers = df['셀러명'].unique()
    
    for seller in sellers:
        seller_df = df[df['셀러명'] == seller]
        
        # Unique customers count
        total_customers = seller_df['주문자연락처'].nunique()
        
        # For each customer, count unique Dates
        # Group by Customer, count unique Date
        customer_date_counts = seller_df.groupby('주문자연락처')['Date'].nunique()
        
        # Repurchasing customers are those with Date count > 1
        repurchasing_customers = customer_date_counts[customer_date_counts > 1].count()
        
        repurchase_rate = (repurchasing_customers / total_customers * 100) if total_customers > 0 else 0
        
        total_revenue = seller_df['실결제 금액'].astype(str).str.replace(',', '').astype(float).sum()
        
        results.append({
            'Seller': seller,
            'TotalCustomers': total_customers,
            'RepurchasingCustomers': repurchasing_customers,
            'RepurchaseRate': repurchase_rate,
            'TotalRevenue': total_revenue
        })
        
    results_df = pd.DataFrame(results)
    
    # Filter for significant sellers (e.g., > 10 customers)
    significant_sellers = results_df[results_df['TotalCustomers'] >= 10].sort_values(by='RepurchaseRate', ascending=False)
    
    print("\n=== Top 20 Sellers by Repurchase Rate (Min 10 Customers) ===")
    print(significant_sellers[['Seller', 'RepurchaseRate', 'TotalCustomers', 'RepurchasingCustomers', 'TotalRevenue']].head(20).to_string(index=False))

    # Also sort by Total Revenue to see if big sellers have high retention
    print("\n=== Top 10 Revenue Sellers & Their Repurchase Rate ===")
    top_revenue = results_df.sort_values(by='TotalRevenue', ascending=False).head(10)
    print(top_revenue[['Seller', 'RepurchaseRate', 'TotalCustomers', 'RepurchasingCustomers', 'TotalRevenue']].to_string(index=False))

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(x='RepurchaseRate', y='Seller', data=significant_sellers.head(15), palette='Purples_r')
    plt.title('Top 15 Sellers by Repurchase Rate (Min 10 Customers)')
    plt.xlabel('Repurchase Rate (%)')
    plt.tight_layout()
    plt.savefig('seller_repurchase_top15.png')
    print("\nSaved chart to seller_repurchase_top15.png")

if __name__ == "__main__":
    analyze_seller_repurchase('data/project1 - preprocessed_data.csv')
