import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm
import numpy as np

# Set Korean font
plt.rc('font', family='AppleGothic') # For Mac
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'eda_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_region_from_zipcode(zipcode):
    """
    Map first 2 digits of 5-digit postal code to 17 regions.
    Based on Korea Post standards.
    """
    try:
        zipcode = str(int(zipcode)).zfill(5) # Ensure string and 5 digits
        prefix = int(zipcode[:2])
        
        if 1 <= prefix <= 9: return "서울"
        elif 10 <= prefix <= 20: return "경기" # 10-18 is Gyeonggi, 19-20 reserved/unused but safe to map if any
        elif 21 <= prefix <= 23: return "인천"
        elif 24 <= prefix <= 26: return "강원"
        elif 27 <= prefix <= 29: return "충북"
        elif prefix == 30: return "세종"
        elif 31 <= prefix <= 33: return "충남"
        elif 34 <= prefix <= 35: return "대전"
        elif 36 <= prefix <= 39: return "경북"
        elif 40 <= prefix <= 43: return "대구"
        elif 44 <= prefix <= 45: return "울산"
        elif 46 <= prefix <= 49: return "부산"
        elif 50 <= prefix <= 53: return "경남"
        elif 54 <= prefix <= 56: return "전북"
        elif 57 <= prefix <= 60: return "전남"
        elif 61 <= prefix <= 62: return "광주"
        elif prefix == 63: return "제주"
        else: return "기타"
    except:
        return "Unknown"

def calculate_repurchase(df):
    """
    Calculate repurchase count based on '주문자연락처' and '주문일' (date only).
    """
    print("Calculating repurchase counts...")
    if '주문자연락처' not in df.columns or '주문일' not in df.columns:
        print("Columns for repurchase calculation missing.")
        return df

    # Create a Date column (without time)
    df['OrderDate'] = df['주문일'].dt.date
    
    # Sort by User and Date
    df = df.sort_values(by=['주문자연락처', 'OrderDate'])
    
    # Group by User and count unique dates
    # We want to assign the TOTAL historical order count to the user, OR 
    # the 'N-th purchase' status to each order? 
    # The requirement is: "재구매 횟수는 '동일한 주문자 연락처를 사용하되, 고유한 주문 날짜(시간 제외)가 다른 건들을 카운팅'해서 만들어주고."
    # Usually this means "How many times has this user purchased BEFORE this order?" or "Total purchases for this user?".
    # Let's assume "Total purchase count for this user" (as a customer attribute) for the 'Repurchase Count' distribution.
    
    # Count unique order dates per user
    user_order_counts = df.groupby('주문자연락처')['OrderDate'].nunique()
    
    # Map back to df
    # '재구매 횟수' usually means (Total Orders - 1), if 1 order -> 0 repurchase.
    # But let's verify if the user means "Order Count" or "Repurchase Count".
    # "재구매 횟수" literally means "Repurchase Count". So 1 order = 0 repurchases.
    
    df['UserTotalOrders'] = df['주문자연락처'].map(user_order_counts)
    df['재구매 횟수'] = df['UserTotalOrders'] - 1
    
    # Handle cases where contact might be missing
    df['재구매 횟수'] = df['재구매 횟수'].fillna(0).astype(int)
    
    return df

def load_and_preprocess(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    print("Columns:", df.columns)
    
    # 1. Convert '주문일' to datetime
    if '주문일' in df.columns:
        df['주문일'] = pd.to_datetime(df['주문일'])
    
    # 2. Price columns
    price_cols = ['결제금액', '주문취소 금액', '실결제 금액', '판매단가', '공급가 총합']
    for col in price_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').astype(float)
    
    # 3. Calculated Repurchase Count
    df = calculate_repurchase(df)
    
    # 4. Region Mapping
    print("Mapping regions...")
    if '우편번호' in df.columns:
        # Fill missing '광역지역' or overwrite to ensure consistency? 
        # Requirement: "지역 정보는 텍스트로 판별이 불가능한 경우... 활용하여... 자동 매핑해줘"
        # Since '광역지역' exists but might be empty or messy, let's create 'DerivedRegion' or fill '광역지역'
        
        # Determine strict mapping for all rows to ensure accuracy
        df['MappedRegion'] = df['우편번호'].apply(get_region_from_zipcode)
        
        # Use MappedRegion if valid, otherwise keep original if Mapped is Unknown? 
        # Or just trust Postal Code primarily. Let's trust Postal Code.
        df['광역지역'] = df['MappedRegion']
        
    # Fill remaining NaNs for safety
    cat_cols = ['주문경로', '감귤 세부', '품종']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    print("Data loaded and preprocessed. Shape:", df.shape)
    return df

def analyze_time_and_day(df):
    print("시간대 및 요일별 분석 중...")
    if '주문일' not in df.columns: return

    # Hour analysis
    df['Hour'] = df['주문일'].dt.hour
    hourly_counts = df.groupby('Hour')['주문번호'].count()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette='viridis', hue=hourly_counts.index, legend=False)
    plt.title('시간대별 주문 건수')
    plt.xlabel('시간 (0-23)')
    plt.ylabel('주문 건수')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/orders_by_hour.png')
    plt.close()

    # Day of Week analysis
    # dt.day_name() returns English names by default. Let's map to Korean.
    day_map = {
        'Monday': '월', 'Tuesday': '화', 'Wednesday': '수', 'Thursday': '목',
        'Friday': '금', 'Saturday': '토', 'Sunday': '일'
    }
    df['DayOfWeek'] = df['주문일'].dt.day_name().map(day_map)
    # Order for plotting
    day_order = ['월', '화', '수', '목', '금', '토', '일']
    
    daily_counts = df['DayOfWeek'].value_counts().reindex(day_order)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=daily_counts.index, y=daily_counts.values, palette='pastel', hue=daily_counts.index, legend=False)
    plt.title('요일별 주문 건수')
    plt.xlabel('요일')
    plt.ylabel('주문 건수')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/orders_by_day_of_week.png')
    plt.close()

def analyze_channel_sales(df):
    print("채널별 매출 분석 중...")
    if '주문경로' not in df.columns or '실결제 금액' not in df.columns: return

    channel_sales = df.groupby('주문경로')['실결제 금액'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=channel_sales.index, y=channel_sales.values, palette='coolwarm', hue=channel_sales.index, legend=False)
    plt.title('유입 채널별 매출액')
    plt.xlabel('주문 경로')
    plt.ylabel('매출액')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sales_by_channel.png')
    plt.close()

def analyze_sellers(df):
    print("셀러 분석 중...")
    if '셀러명' not in df.columns or '실결제 금액' not in df.columns: return

    # Top Sellers by Revenue
    top_sellers = df.groupby('셀러명')['실결제 금액'].sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(y=top_sellers.index, x=top_sellers.values, palette='magma', hue=top_sellers.index, legend=False)
    plt.title('상위 10개 셀러 (매출액 기준)')
    plt.xlabel('매출액')
    plt.ylabel('셀러명')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top_10_sellers.png')
    plt.close()

    # High Unit Price Sellers
    # Average Unit Price = Total Revenue / Total Quantity (or just average '판매단가' if unique items)
    # Let's use average '판매단가' per seller
    if '판매단가' in df.columns:
        avg_unit_price = df.groupby('셀러명')['판매단가'].mean().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(y=avg_unit_price.index, x=avg_unit_price.values, palette='Blues_r', hue=avg_unit_price.index, legend=False)
        plt.title('객단가가 높은 상위 10개 셀러 (평균 판매단가)')
        plt.xlabel('평균 판매단가')
        plt.ylabel('셀러명')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/high_unit_price_sellers.png')
        plt.close()

def analyze_sales_trend(df):
    print("매출 추이 분석 중...")
    if '주문일' not in df.columns or '실결제 금액' not in df.columns: return

    daily_sales = df.groupby(df['주문일'].dt.date)['실결제 금액'].sum()
    
    plt.figure(figsize=(12, 6))
    daily_sales.plot(kind='line', marker='o', color='skyblue')
    plt.title('일별 매출 추이')
    plt.xlabel('날짜')
    plt.ylabel('매출액')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/daily_sales_trend.png')
    plt.close()

def analyze_products(df):
    print("상품 분석 중...")
    if '상품명' in df.columns and '실결제 금액' in df.columns:
        top_products = df.groupby('상품명')['실결제 금액'].sum().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(y=top_products.index, x=top_products.values, palette='viridis', hue=top_products.index, legend=False)
        plt.title('상위 10개 상품 (매출액 기준)')
        plt.xlabel('매출액')
        plt.ylabel('상품명')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/top_10_products_revenue.png')
        plt.close()

    if '감귤 세부' in df.columns:
        plt.figure(figsize=(10, 6))
        # Filter out 'Unknown' if desired, or keep to show missing data
        data_to_plot = df[df['감귤 세부'] != 'Unknown']
        if not data_to_plot.empty:
            order = data_to_plot['감귤 세부'].value_counts().index
            sns.countplot(y='감귤 세부', data=data_to_plot, order=order, palette='pastel', hue='감귤 세부', legend=False)
            plt.title('감귤 세부 카테고리별 주문 건수')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/citrus_category_counts.png')
            plt.close()

def analyze_customers(df):
    print("고객 분석 중...")
    
    # Repurchase Count Distribution
    if '재구매 횟수' in df.columns:
        # We want to see how many customers have 0 repurchases (1-time buyers), 1 repurchase (2 orders), etc.
        # Unique customers only
        unique_customers = df.drop_duplicates(subset=['주문자연락처'])
        
        plt.figure(figsize=(8, 6))
        sns.countplot(x='재구매 횟수', data=unique_customers, palette='magma', hue='재구매 횟수', legend=False)
        plt.title('재구매 횟수별 고객 분포 (고객 기준)')
        plt.xlabel('재구매 횟수 (0 = 1회 구매)')
        plt.ylabel('고객 수')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/repurchase_distribution.png')
        plt.close()

    # Order Purpose
    if '목적' in df.columns:
        plt.figure(figsize=(8, 6))
        counts = df['목적'].value_counts()
        if not counts.empty:
            counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
            plt.title('주문 목적 비율')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/order_purpose_pie.png')
            plt.close()

def analyze_geography(df):
    print("지역 분석 중...")
    if '광역지역' in df.columns:
        plt.figure(figsize=(12, 6))
        order = df['광역지역'].value_counts().index
        sns.countplot(x='광역지역', data=df, order=order, palette='coolwarm', hue='광역지역', legend=False)
        plt.title('지역별 주문 건수 (우편번호 기반)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/region_order_counts.png')
        plt.close()

def analyze_product_attributes(df):
    print("상품 속성 분석 중...")
    
    # 1. 감귤 세부 & 품종
    if '감귤 세부' in df.columns and '실결제 금액' in df.columns:
        plt.figure(figsize=(10, 6))
        sales_by_citrus = df.groupby('감귤 세부')['실결제 금액'].sum().sort_values(ascending=False)
        sns.barplot(x=sales_by_citrus.index, y=sales_by_citrus.values, palette='Oranges_r', hue=sales_by_citrus.index, legend=False)
        plt.title('감귤 세부 종류별 매출액')
        plt.ylabel('매출액')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/sales_by_citrus_detail.png')
        plt.close()

    # 2. 과수 크기
    if '과수 크기' in df.columns:
        plt.figure(figsize=(12, 6))
        # Remove outliers or sort? 
        # Checking unique values might be needed, but assuming categorical text
        order = df['과수 크기'].value_counts().index
        sns.countplot(x='과수 크기', data=df, order=order, palette='Set2', hue='과수 크기', legend=False)
        plt.title('과수 크기별 선호도 (주문 건수)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/preference_by_fruit_size.png')
        plt.close()

def analyze_packaging(df):
    print("패키징 및 단량 분석 중...")
    
    # 1. 무게별 판매 (매출 기준)
    if '무게(kg)' in df.columns and '실결제 금액' in df.columns:
        plt.figure(figsize=(10, 6))
        # Ensure numerical safety? Assuming it is float from preprocessing or clean enough
        # Convert to string for categorical plotting or binning?
        # Let's try grouping by exact values first
        sales_by_weight = df.groupby('무게(kg)')['실결제 금액'].sum().sort_values(ascending=False)
        sns.barplot(x=sales_by_weight.index.astype(str), y=sales_by_weight.values, palette='Greens_r', hue=sales_by_weight.index, legend=False)
        plt.title('무게(kg)별 매출액')
        plt.xlabel('무게 (kg)')
        plt.ylabel('매출액')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/sales_by_weight.png')
        plt.close()

    # 2. 선물세트 여부
    if '선물세트_여부' in df.columns:
        plt.figure(figsize=(6, 6))
        df['선물세트_여부'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90)
        plt.title('선물세트 vs 일반상품 주문 비율')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/gift_vs_normal_pie.png')
        plt.close()

def analyze_price_purpose(df):
    print("가격 및 구매 목적 분석 중...")
    
    # 1. 목적별 평균 객단가
    if '목적' in df.columns and '실결제 금액' in df.columns:
        plt.figure(figsize=(8, 6))
        avg_price = df.groupby('목적')['실결제 금액'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_price.index, y=avg_price.values, palette='Purples_r', hue=avg_price.index, legend=False)
        plt.title('구매 목적별 평균 객단가')
        plt.ylabel('평균 결제 금액 (원)')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/avg_price_by_purpose.png')
        plt.close()

    # 2. 가격대별 구매 목적 (Heatmap or Stacked Bar)
    if '가격대' in df.columns and '목적' in df.columns:
        # Cross tabulation
        ct = pd.crosstab(df['가격대'], df['목적'])
        # Sort index if possible (needs custom sort order usually, but let's try auto)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('가격대와 구매 목적의 관계 (주문 건수)')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/price_purpose_heatmap.png')
        plt.close()

def analyze_cancellations(df):
    print("취소율 분석 중...")
    
    if '취소여부' in df.columns:
        # Calculate cancellation rate by Product
        # 취소여부 is 'Y' or 'N'
        df['IsCancelled'] = df['취소여부'].apply(lambda x: 1 if x == 'Y' else 0)
        
        # Filter products with meaningful order counts (e.g., > 10)
        prod_counts = df['상품명'].value_counts()
        valid_prods = prod_counts[prod_counts > 10].index
        
        df_valid = df[df['상품명'].isin(valid_prods)]
        
        cancel_rates = df_valid.groupby('상품명')['IsCancelled'].mean().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(y=cancel_rates.index, x=cancel_rates.values * 100, palette='RdBu', hue=cancel_rates.index, legend=False)
        plt.title('취소율이 높은 상위 10개 상품 (주문 10건 이상)')
        plt.xlabel('취소율 (%)')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/top_cancellation_rates.png')
        plt.close()

def main():
    filepath = 'data/project1 - preprocessed_data.csv'
    if not os.path.exists(filepath):
        print(f"파일을 찾을 수 없습니다: {filepath}")
        return

    df = load_and_preprocess(filepath)
    
    analyze_sales_trend(df)
    analyze_products(df)
    analyze_customers(df)
    analyze_geography(df)
    
    # New analysis functions
    analyze_time_and_day(df)
    analyze_channel_sales(df)
    analyze_sellers(df)
    
    # New Product Planning Analysis
    analyze_product_attributes(df)
    analyze_packaging(df)
    analyze_price_purpose(df)
    analyze_cancellations(df)
    
    # Additional Insight Print
    print("\n--- 추가 인사이트 ---")
    print(f"총 고유 고객 수: {df['주문자연락처'].nunique()}")
    if '재구매 횟수' in df.columns:
        re_customers = df[df['재구매 횟수'] > 0]['주문자연락처'].nunique()
        total_customers = df['주문자연락처'].nunique()
        print(f"재구매 고객 수: {re_customers} ({re_customers/total_customers*100:.1f}%)")
    
    print("분석 완료. 'eda_output' 디렉토리를 확인하세요.")

if __name__ == "__main__":
    main()
