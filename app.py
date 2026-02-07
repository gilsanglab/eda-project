import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm

# --- Configuration ---
st.set_page_config(page_title="프로젝트 1 판매 대시보드", layout="wide")

# Set font for Korean support based on OS
import platform

system_name = platform.system()

if system_name == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
elif system_name == 'Windows': # Windows
    plt.rc('font', family='Malgun Gothic')
else: # Linux (Streamlit Cloud)
    # Try to find Nanum font
    # Usually installed at /usr/share/fonts/truetype/nanum/NanumGothic.ttf
    # But matplotlib needs the font family name
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False

# --- Data Loading & Preprocessing ---
@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        return None
    
    df = pd.read_csv(filepath)
    
    # 1. Date Conversion
    if '주문일' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['주문일'])
        df['Date'] = df['OrderDate'].dt.date
        
    # 2. Price Columns (Remove comma and cast to float)
    price_cols = ['결제금액', '주문취소 금액', '실결제 금액', '판매단가', '공급가 총합']
    for col in price_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)
            
    # 3. Create English mapping/columns for key metrics where possible
    # We will use the original columns for calculation but label them in English
    
    # Repurchase Calculation
    if '주문자연락처' in df.columns and 'OrderDate' in df.columns:
        df = df.sort_values(by=['주문자연락처', 'Date'])
        user_order_counts = df.groupby('주문자연락처')['Date'].nunique()
        df['UserTotalOrders'] = df['주문자연락처'].map(user_order_counts)
        df['RepurchaseCount'] = df['UserTotalOrders'] - 1
        df['RepurchaseCount'] = df['RepurchaseCount'].fillna(0).astype(int)
        
    # Region Mapping (Simple Zipcode to Region Name - reusing logic if needed, 
    # but '광역지역' usually exists. Let's use '광역지역' directly)
    
    return df

FILEPATH = 'data/project1 - preprocessed_data.csv'
df = load_data(FILEPATH)

if df is None:
    st.error(f"데이터 파일을 찾을 수 없습니다: {FILEPATH}")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.title("필터 (Filters)")
if 'OrderDate' in df.columns:
    min_date = df['OrderDate'].min().date()
    max_date = df['OrderDate'].max().date()
    start_date, end_date = st.sidebar.date_input("날짜 범위 선택", [min_date, max_date])
    
    # Filter Data
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df.loc[mask]
    df_filtered = df

# 2.2. Seller Metrics Calculation Helper
@st.cache_data
def calculate_seller_metrics(df):
    # Data Cleaning (User Request)
    # 1) 실결제 금액 숫자화
    if '실결제 금액' in df.columns:
        # 이미 load_data에서 처리했지만, 안전장치로 한번 더
        if df['실결제 금액'].dtype == 'object':
             df['실결제 금액'] = df['실결제 금액'].astype(str).str.replace(r'[^\d\.-]', '', regex=True)
        df['실결제 금액'] = pd.to_numeric(df['실결제 금액'], errors='coerce').fillna(0)

    # 2) 공급단가 Cleaning & SupplyCost Calculation
    if '공급단가' in df.columns:
        # 공급단가: 쉼표/원/공백 등 제거 후 숫자화
        supply_price_clean = df['공급단가'].astype(str).str.replace(r'[^\d\.-]', '', regex=True)
        supply_price_clean = pd.to_numeric(supply_price_clean, errors='coerce').fillna(0)
        
        df['SupplyCost'] = supply_price_clean * df['주문수량']
        df['Margin'] = df['실결제 금액'] - df['SupplyCost']
    else:
        df['Margin'] = 0
        
    df['IsCancelled'] = df['취소여부'].apply(lambda x: 1 if x == 'Y' else 0) if '취소여부' in df.columns else 0
    
    # Aggregation
    seller_stats = df.groupby('셀러명').agg(
        TotalRevenue=('실결제 금액', 'sum'),
        TotalMargin=('Margin', 'sum'),
        OrderCount=('주문번호', 'nunique'),
        TotalQty=('주문수량', 'sum'),
        CancelCount=('IsCancelled', 'sum')
    ).reset_index()
    
    # Derived Metrics
    seller_stats['MarginRate'] = (seller_stats['TotalMargin'] / seller_stats['TotalRevenue']) * 100
    seller_stats['MarginRate'] = seller_stats['MarginRate'].fillna(0)
    seller_stats['CancelRate'] = (seller_stats['CancelCount'] / seller_stats['OrderCount']) * 100
    seller_stats['AOV'] = seller_stats['TotalRevenue'] / seller_stats['OrderCount']
    
    # Repurchase Rate (Complex calculation)
    # For each seller, find % of customers who ordered > 1 time
    if '주문자연락처' in df.columns:
        repurchase_data = []
        for seller in seller_stats['셀러명']:
            seller_df = df[df['셀러명'] == seller]
            user_counts = seller_df['주문자연락처'].value_counts()
            total_users = len(user_counts)
            re_users = len(user_counts[user_counts > 1])
            rate = (re_users / total_users * 100) if total_users > 0 else 0
            repurchase_data.append(rate)
        seller_stats['RepurchaseRate'] = repurchase_data
    else:
        seller_stats['RepurchaseRate'] = 0
        
    # Lifecycle Metrics
    if 'OrderDate' in df.columns:
        lifecycle = df.groupby('셀러명')['OrderDate'].agg(['min', 'max']).reset_index()
        lifecycle.columns = ['셀러명', 'FirstOrderDate', 'LastOrderDate']
        
        # Merge back
        seller_stats = pd.merge(seller_stats, lifecycle, on='셀러명', how='left')
        
        # Calculate Tenure & Recency
        # Tenure: Days between first and last order (+1)
        seller_stats['TenureDays'] = (seller_stats['LastOrderDate'] - seller_stats['FirstOrderDate']).dt.days + 1
        
        # Recency: Days since last order (using max date in dataset as reference)
        max_date = df['OrderDate'].max()
        seller_stats['RecencyDays'] = (max_date - seller_stats['LastOrderDate']).dt.days
    else:
        seller_stats['TenureDays'] = 0
        seller_stats['RecencyDays'] = 0
        seller_stats['FirstOrderDate'] = None

    return seller_stats.sort_values(by='TotalRevenue', ascending=False)


# --- Main Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "대시보드 개요", 
    "매출 분석", 
    "상품 및 셀러", 
    "고객 및 지역", 
    "신상품 기획",
    "지역 심층 분석", 
    "셀러 심층 분석"
])

# --- Tab 1: Overview ---
with tab1:
    st.header("대시보드 개요")
    
    col1, col2, col3, col4 = st.columns(4)
    
    model_revenue = df_filtered['실결제 금액'].sum()
    total_orders = df_filtered['주문번호'].nunique()
    total_users = df_filtered['주문자연락처'].nunique() if '주문자연락처' in df_filtered.columns else 0
    
    repurchase_rate = 0
    if 'RepurchaseCount' in df_filtered.columns and total_users > 0:
        re_users = df_filtered[df_filtered['RepurchaseCount'] > 0]['주문자연락처'].nunique()
        repurchase_rate = (re_users / total_users) * 100

    col1.metric("총 매출", f"₩{model_revenue:,.0f}")
    col2.metric("총 주문수", f"{total_orders:,}건")
    col3.metric("총 고객수", f"{total_users:,}명")
    col4.metric("재구매율", f"{repurchase_rate:.1f}%")
    
    st.markdown("---")
    st.subheader("최근 주문 내역")
    st.dataframe(df_filtered[['주문일', '주문번호', '상품명', '실결제 금액', '주문자명']].sort_values(by='주문일', ascending=False).head(10))

# --- Tab 2: Sales Analysis ---
with tab2:
    st.header("매출 분석")
    
    col1, col2 = st.columns(2)
    
    # 1. Daily Sales Trend
    with col1:
        st.subheader("일별 매출 추이")
        daily_sales = df_filtered.groupby('Date')['실결제 금액'].sum()
        fig, ax = plt.subplots(figsize=(10, 5))
        daily_sales.plot(kind='line', marker='o', ax=ax, color='skyblue')
        ax.set_title("일별 매출 추이")
        ax.set_ylabel("매출액 (원)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    # 2. Sales by Channel
    with col2:
        st.subheader("채널별 매출")
        if '주문경로' in df_filtered.columns:
            channel_sales = df_filtered.groupby('주문경로')['실결제 금액'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=channel_sales.index, y=channel_sales.values, palette='coolwarm', ax=ax)
            ax.set_title("유입 채널별 매출액")
            ax.set_ylabel("매출액")
            ax.set_xlabel("주문 경로")
            st.pyplot(fig)

    # 3. Orders by Hour
    st.subheader("시간대별 주문 건수")
    if 'OrderDate' in df_filtered.columns:
        df_filtered['Hour'] = df_filtered['OrderDate'].dt.hour
        hourly_counts = df_filtered.groupby('Hour')['주문번호'].count()
        
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette='viridis', ax=ax)
        ax.set_title("시간대별 주문 건수")
        ax.set_xlabel("시간 (0-23시)")
        ax.set_ylabel("주문 건수")
        st.pyplot(fig)

# --- Tab 3: Product & Seller ---
with tab3:
    st.header("상품 및 셀러 분석")
    
    col1, col2 = st.columns(2)
    
    # 1. Top 10 Products
    with col1:
        st.subheader("상위 10개 상품 (매출 기준)")
        top_products = df_filtered.groupby('상품명')['실결제 금액'].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(y=top_products.index, x=top_products.values, palette='magma', ax=ax)
        ax.set_title("매출 상위 10개 상품")
        ax.set_xlabel("매출액")
        st.pyplot(fig)
        
    # 2. Top 10 Sellers
    with col2:
        st.subheader("상위 10개 셀러 (매출 기준)")
        if '셀러명' in df_filtered.columns:
            top_sellers = df_filtered.groupby('셀러명')['실결제 금액'].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(y=top_sellers.index, x=top_sellers.values, palette='viridis', ax=ax)
            ax.set_title("매출 상위 10개 셀러")
            ax.set_xlabel("매출액")
            st.pyplot(fig)

    # 3. Citrus Categories
    st.subheader("감귤 품종별 매출")
    if '감귤 세부' in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        citrus_sales = df_filtered.groupby('감귤 세부')['실결제 금액'].sum().sort_values(ascending=False)
        sns.barplot(x=citrus_sales.index, y=citrus_sales.values, palette='Oranges_r', ax=ax)
        ax.set_title("감귤 세부 품종별 매출액")
        st.pyplot(fig)

# --- Tab 4: Customer & Geography ---
with tab4:
    st.header("고객 및 지역 분석")
    
    col1, col2 = st.columns(2)
    
    # 1. Repurchase Distribution
    with col1:
        st.subheader("재구매 횟수 분포")
        if 'RepurchaseCount' in df_filtered.columns:
            # Drop duplicates per user for this plot
            unique_users = df_filtered.drop_duplicates(subset=['주문자연락처'])
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='RepurchaseCount', data=unique_users, palette='pastel', ax=ax)
            ax.set_title("고객별 재구매 횟수 분포")
            ax.set_xlabel("재구매 횟수 (0 = 1회 구매)")
            ax.set_ylabel("고객 수")
            st.pyplot(fig)
            
    # 2. Order Purpose
    with col2:
        st.subheader("주문 목적")
        if '목적' in df_filtered.columns:
            fig, ax = plt.subplots(figsize=(6, 6))
            df_filtered['목적'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=sns.color_palette('pastel'))
            ax.set_ylabel('')
            ax.set_title("주문 목적 비율 (선물 vs 개인소비)")
            st.pyplot(fig)
            
    # 3. Geography
    st.subheader("지역별 주문 건수")
    if '광역지역' in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        order = df_filtered['광역지역'].value_counts().index
        sns.countplot(x='광역지역', data=df_filtered, order=order, palette='coolwarm', ax=ax)
        ax.set_title("지역별 주문 건수")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

# --- Tab 5: New Product Planning ---
with tab5:
    st.header("신상품 기획 분석")
    st.markdown("신상품 개발에 필요한 속성, 패키징, 가격, 취소율 분석입니다.")
    
    col1, col2 = st.columns(2)
    
    # 1. Preferred Fruit Size
    with col1:
        st.subheader("선호 과수 크기")
        if '과수 크기' in df_filtered.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            order = df_filtered['과수 크기'].value_counts().index
            sns.countplot(x='과수 크기', data=df_filtered, order=order, palette='Set2', ax=ax)
            ax.set_title("과수 크기별 주문 건수")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
            
    # 2. Preferred Weight
    with col2:
        st.subheader("선호 중량 (kg)")
        if '무게(kg)' in df_filtered.columns:
            weight_sales = df_filtered.groupby('무게(kg)')['실결제 금액'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=weight_sales.index.astype(str), y=weight_sales.values, palette='Greens_r', ax=ax)
            ax.set_title("무게(kg)별 매출액")
            ax.set_ylabel("매출액")
            st.pyplot(fig)
            
    col3, col4 = st.columns(2)
    
    # 3. Price by Purpose
    with col3:
        st.subheader("목적별 평균 객단가")
        if '목적' in df_filtered.columns:
            avg_price = df_filtered.groupby('목적')['실결제 금액'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=avg_price.index, y=avg_price.values, palette='Purples_r', ax=ax)
            ax.set_title("구매 목적별 평균 객단가")
            ax.set_ylabel("평균 금액 (원)")
            st.pyplot(fig)
            
    # 4. Cancellation Rates
    with col4:
        st.subheader("상위 취소율 상품")
        if '취소여부' in df_filtered.columns:
            df_filtered['IsCancelled'] = df_filtered['취소여부'].apply(lambda x: 1 if x == 'Y' else 0)
            prod_counts = df_filtered['상품명'].value_counts()
            valid_prods = prod_counts[prod_counts > 10].index
            df_valid = df_filtered[df_filtered['상품명'].isin(valid_prods)]
            
            cancel_rates = df_valid.groupby('상품명')['IsCancelled'].mean().sort_values(ascending=False).head(5)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(y=cancel_rates.index, x=cancel_rates.values * 100, palette='RdBu', ax=ax)
            ax.set_title("취소율 상위 5개 상품 (주문 10건 이상)")

# --- Tab 6: Regional Analysis (Gyeonggi Focus) ---
with tab6:
    st.header("지역 심층 분석 (경기도)")
    st.markdown("사용자 가설 검증: **'경기도의 높은 매출은 고매출 셀러가 많기 때문인가?'**")
    
    if '광역지역' in df_filtered.columns and '셀러명' in df_filtered.columns:
        # 1. Basic Stats by Region
        region_stats = df_filtered.groupby('광역지역').agg(
            TotalRevenue=('실결제 금액', 'sum'),
            SellerCount=('셀러명', 'nunique')
        ).reset_index()
        
        region_stats['AvgRevenuePerSeller'] = region_stats['TotalRevenue'] / region_stats['SellerCount']
        region_stats = region_stats.sort_values(by='TotalRevenue', ascending=False)
        
        # 2. High Revenue Seller Analysis
        st.subheader("1. 지역별 고매출 셀러 비율 분석")
        
        # Calculate global threshold for 'High Revenue' (Top 20%)
        # Note: We calculate this based on the FILTERED data to respect date range, 
        # OR we could calculate based on full data if standard. Let's use filtered for consistency.
        seller_revenues = df_filtered.groupby(['광역지역', '셀러명'])['실결제 금액'].sum().reset_index()
        revenue_threshold = seller_revenues['실결제 금액'].quantile(0.80)
        
        st.info(f"**고매출 셀러 기준**: 상위 20% (매출 {revenue_threshold:,.0f}원 이상)")
        
        seller_revenues['IsHighRevenue'] = seller_revenues['실결제 금액'] > revenue_threshold
        
        high_revenue_stats = seller_revenues.groupby('광역지역').agg(
            TotalSellers=('셀러명', 'count'),
            HighRevenueSellers=('IsHighRevenue', 'sum')
        ).reset_index()
        
        high_revenue_stats['HighRevenuePercent'] = (high_revenue_stats['HighRevenueSellers'] / high_revenue_stats['TotalSellers']) * 100
        high_revenue_stats = high_revenue_stats.sort_values(by='HighRevenueSellers', ascending=False)
        
        # Display Key Metric for Gyeonggi
        gyeonggi_row = high_revenue_stats[high_revenue_stats['광역지역'].isin(['경기', '경기도'])]
        if not gyeonggi_row.empty:
            gg_pct = gyeonggi_row['HighRevenuePercent'].values[0]
            st.success(f"**결론**: 경기도의 고매출 셀러 비율은 **{gg_pct:.1f}%**로, 전체 지역 중 가장 높습니다.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**지역별 고매출 셀러 수 (명)**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='HighRevenueSellers', y='광역지역', data=high_revenue_stats.head(10), palette='Reds_r', ax=ax)
            ax.set_xlabel("고매출 셀러 수")
            st.pyplot(fig)
            
        with col2:
            st.write("**지역별 고매출 셀러 비율 (%)**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='HighRevenuePercent', y='광역지역', data=high_revenue_stats.head(10), palette='Oranges_r', ax=ax)
            ax.set_xlabel("고매출 셀러 비율 (%)")
            st.pyplot(fig)

        # 3. Distribution Plot
        st.subheader("2. 지역별 셀러 매출 분포 (Box Plot)")
        st.markdown("매출이 0원 초과인 셀러만 대상으로, 로그 스케일로 분포를 확인합니다.")
        
        top_regions = region_stats.head(10)['광역지역'].tolist()
        filtered_sellers_plot = seller_revenues[(seller_revenues['광역지역'].isin(top_regions)) & (seller_revenues['실결제 금액'] > 0)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='광역지역', y='실결제 금액', data=filtered_sellers_plot, order=top_regions, palette='Set2', ax=ax)
        ax.set_yscale('log')
        ax.set_ylabel("셀러별 총 매출 (Log Scale)")
        st.pyplot(fig)
        

    else:
        st.warning("데이터에 '광역지역' 또는 '셀러명' 컬럼이 없어 분석할 수 없습니다.")

# --- Tab 7: Seller Deep Dive ---
with tab7:
    st.header("셀러 심층 분석 (Scorecard)")
    st.markdown("""
    각 셀러의 **수익성(마진)**, **운영 효율(취소율)**, **고객 충성도(재구매율)**를 종합적으로 분석합니다.
    """)
    
    if '셀러명' in df_filtered.columns:
        seller_metrics = calculate_seller_metrics(df_filtered)
        
        # 1. Top Filters
        min_revenue = st.slider("최소 매출 필터 (원)", 0, int(seller_metrics['TotalRevenue'].max()), 500000, 100000)
        filtered_metrics = seller_metrics[seller_metrics['TotalRevenue'] >= min_revenue]
        
        # 2. Scorecard Table
        st.subheader("📊 셀러 종합 스코어카드")
        st.markdown(f"매출 {min_revenue:,}원 이상 셀러: **{len(filtered_metrics)}명**")
        
        display_cols = ['셀러명', 'TotalRevenue', 'TotalMargin', 'MarginRate', 'CancelRate', 'RepurchaseRate', 'AOV', 'OrderCount']
        format_dict = {
            'TotalRevenue': '₩{0:,.0f}',
            'TotalMargin': '₩{0:,.0f}', 
            'MarginRate': '{0:.1f}%',
            'CancelRate': '{0:.1f}%',
            'RepurchaseRate': '{0:.1f}%',
            'AOV': '₩{0:,.0f}',
            'OrderCount': '{0:,}건'
        }
        
        # Renaming for display
        rename_dict = {
            'TotalRevenue': '총 매출',
            'TotalMargin': '총 이익',
            'MarginRate': '이익률',
            'CancelRate': '취소율',
            'RepurchaseRate': '재구매율',
            'AOV': '객단가',
            'OrderCount': '주문건수'
        }
        
        st.dataframe(
            filtered_metrics[display_cols].rename(columns=rename_dict).style.format(format_dict).background_gradient(cmap='Blues', subset=['총 매출', '총 이익']),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        # 3. Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 수익성 분석: 마진율 vs 매출")
            # Scatter Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=filtered_metrics, x='TotalRevenue', y='MarginRate', 
                size='OrderCount', hue='RepurchaseRate', sizes=(50, 500), alpha=0.7, palette='viridis', ax=ax
            )
            ax.set_xscale('log')
            ax.set_xlabel("총 매출 (Log Scale)")
            ax.set_ylabel("마진율 (%)")
            ax.set_title("매출 vs 마진율 (점 크기: 주문건수, 색상: 재구매율)")
            st.pyplot(fig)
            
        with col2:
            st.subheader("❤️ 충성도 Top 10 셀러")
            top_retention = filtered_metrics.sort_values(by='RepurchaseRate', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='RepurchaseRate', y='셀러명', data=top_retention, palette='Purples_r', ax=ax)
            ax.set_xlabel("재구매율 (%)")
            ax.set_title("재구매율 상위 10개 셀러")
            st.pyplot(fig)
            
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("⚠️ 운영 리스크: 취소율 Top 10")
            # Filter distinct cancel rates to avoid all 0s showing weirdly
            top_cancel = filtered_metrics.sort_values(by='CancelRate', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='CancelRate', y='셀러명', data=top_cancel, palette='Reds_r', ax=ax)
            ax.set_xlabel("취소율 (%)")
            ax.set_title("취소율 상위 10개 셀러")
            st.pyplot(fig)
            
        with col4:
            st.subheader("📦 객단가(AOV) 분석")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(filtered_metrics['AOV'], bins=20, kde=True, color='green', ax=ax)
            ax.set_xlabel("평균 객단가 (원)")
            ax.set_title("셀러별 객단가 분포")
            st.pyplot(fig)

    else:
        st.warning("데이터에 '셀러명' 컬럼이 없어 분석할 수 없습니다.")
        
    st.markdown("---")
    
    # 4. Seller Lifecycle Analysis
    st.subheader("🔄 셀러 생애 주기 분석 (Lifecycle)")
    
    if 'TenureDays' in seller_metrics.columns:
        # A. Summary Key Metrics
        avg_tenure = seller_metrics['TenureDays'].mean()
        
        # Active Definition: Sold within last 30 days
        active_sellers = seller_metrics[seller_metrics['RecencyDays'] <= 30]
        active_count = len(active_sellers)
        churn_risk_count = len(seller_metrics) - active_count
        
        col1, col2, col3 = st.columns(3)
        col1.metric("평균 활동 기간", f"{avg_tenure:.1f}일")
        col2.metric("활성 셀러 (최근 30일 이내 판매)", f"{active_count}명")
        col3.metric("이탈 위험/비활성 셀러", f"{churn_risk_count}명")
        
        # B. New Seller Entrants Trend
        st.write("**월별 신규 진입 셀러 수**")
        if 'FirstOrderDate' in seller_metrics.columns and not seller_metrics['FirstOrderDate'].isnull().all():
            seller_metrics['FirstOrderMonth'] = seller_metrics['FirstOrderDate'].dt.to_period('M').astype(str)
            new_entrants = seller_metrics.groupby('FirstOrderMonth')['셀러명'].count()
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(x=new_entrants.index, y=new_entrants.values, marker='o', color='orange', ax=ax)
            ax.set_title("월별 신규 셀러 진입 추이")
            ax.set_ylabel("신규 셀러 수")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**셀러 활동 기간(Tenure) 분포**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(seller_metrics['TenureDays'], bins=20, kde=True, color='teal', ax=ax)
            ax.set_xlabel("활동 기간 (일)")
            ax.set_title("셀러 생존 기간 분포")
            st.pyplot(fig)
            
        with col2:
            st.write("**🚨 이탈 위험 고매출 셀러 (Top 10)**")
            st.caption("최근 30일간 판매 없음 & 누적 매출 상위")
            
            risk_sellers = seller_metrics[seller_metrics['RecencyDays'] > 30].sort_values(by='TotalRevenue', ascending=False).head(10)
            
            if not risk_sellers.empty:
                display_cols_risk = ['셀러명', 'RecencyDays', 'TotalRevenue', 'TenureDays']
                risk_rename = {
                    'RecencyDays': '미판매 경과일',
                    'TotalRevenue': '누적 매출',
                    'TenureDays': '과거 활동 기간(일)'
                }
                st.dataframe(
                    risk_sellers[display_cols_risk].rename(columns=risk_rename).style.format({'누적 매출': '₩{0:,.0f}'}),
                    use_container_width=True
                )
            else:
                st.success("이탈 위험이 있는 고매출 셀러가 없습니다.")

