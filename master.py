import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

st.set_page_config(
    page_title="Peer Valuation Tool",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# 1. 기본 설정
# -------------------------------------------------

st.title("📊 Peer Valuation Tool")
st.caption("시장 데이터는 반실시간 업데이트, 재무 데이터는 기준일별 CSV 관리 방식")

MARKET_DATA_TTL = 3600  # 1시간 캐시


# -------------------------------------------------
# 2. 데이터 로드 함수
# -------------------------------------------------

@st.cache_data
def load_peer_master(file):
    df = pd.read_csv(file)
    return df


@st.cache_data
def load_financials(file):
    df = pd.read_csv(file)
    return df


@st.cache_data(ttl=MARKET_DATA_TTL)
def get_market_data(tickers):
    """
    주가 / 시가총액 등 시장 데이터만 반실시간으로 가져옴
    ttl=3600 → 1시간마다 자동 갱신
    """
    rows = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")

            if hist.empty:
                price = np.nan
            else:
                price = hist["Close"].iloc[-1]

            market_cap = info.get("marketCap", np.nan)
            currency = info.get("currency", "")
            company_name = info.get("shortName", ticker)

            rows.append({
                "Ticker": ticker,
                "Company Name": company_name,
                "Price": price,
                "Market Cap": market_cap,
                "Currency": currency,
                "Market Data Updated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        except Exception as e:
            rows.append({
                "Ticker": ticker,
                "Company Name": ticker,
                "Price": np.nan,
                "Market Cap": np.nan,
                "Currency": "",
                "Market Data Updated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Error": str(e)
            })

    return pd.DataFrame(rows)


# -------------------------------------------------
# 3. 계산 함수
# -------------------------------------------------

def calculate_peer_valuation(peer_df, financial_df, market_df, selected_period):
    """
    Peer Master + Financials + Market Data 결합
    EV, EV/EBITDA, P/E, EV/Sales 계산
    """

    financial_period = financial_df[financial_df["Period"] == selected_period].copy()

    df = peer_df.merge(financial_period, on="Ticker", how="left")
    df = df.merge(market_df, on="Ticker", how="left")

    # 단위 정리
    # Market Cap은 yfinance 기준 원 단위/달러 단위 그대로 들어올 수 있음
    # 보기 편하게 백만 단위로 변환
    df["Market Cap_M"] = df["Market Cap"] / 1_000_000

    # EV = Market Cap + Net Debt
    # Net Debt_M은 CSV에서 백만 단위로 관리한다고 가정
    df["EV_M"] = df["Market Cap_M"] + df["Net Debt_M"]

    # Multiple 계산
    df["EV/Revenue"] = df["EV_M"] / df["Revenue_M"]
    df["EV/EBITDA"] = df["EV_M"] / df["EBITDA_M"]
    df["P/E"] = df["Market Cap_M"] / df["Net Income_M"]

    # 비정상값 처리
    multiple_cols = ["EV/Revenue", "EV/EBITDA", "P/E"]
    for col in multiple_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def remove_outliers_iqr(df, col):
    """
    IQR 방식 Outlier 제거
    """
    clean_df = df.dropna(subset=[col]).copy()

    q1 = clean_df[col].quantile(0.25)
    q3 = clean_df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]


def calculate_target_value(target_metric, selected_multiple):
    return target_metric * selected_multiple


# -------------------------------------------------
# 4. Sidebar
# -------------------------------------------------

st.sidebar.header("⚙️ 설정")

peer_file = st.sidebar.file_uploader(
    "Peer Master CSV 업로드",
    type=["csv"],
    help="Ticker, Company, Peer Group, Country 등이 포함된 파일"
)

financial_file = st.sidebar.file_uploader(
    "Financials CSV 업로드",
    type=["csv"],
    help="Ticker, Period, Revenue_M, EBITDA_M, Net Debt_M 등이 포함된 파일"
)

st.sidebar.divider()

if st.sidebar.button("🔄 시장 데이터 강제 새로고침"):
    get_market_data.clear()
    st.sidebar.success("시장 데이터 캐시를 초기화했습니다. 다시 계산하면 최신 데이터를 불러옵니다.")

st.sidebar.caption(f"시장 데이터 자동 갱신 주기: {MARKET_DATA_TTL // 60}분")


# -------------------------------------------------
# 5. Sample CSV 안내
# -------------------------------------------------

with st.expander("📌 CSV 양식 예시 보기"):
    st.write("peer_master.csv")
    st.code(
        """Ticker,Company,Peer Group,Country
IMAX,IMAX Corporation,PLF,US
CNK,Cinemark Holdings,Exhibitor,US
AMC,AMC Entertainment,Exhibitor,US
DBO.TO,D-BOX Technologies,Motion Seat,Canada""",
        language="csv"
    )

    st.write("financials.csv")
    st.code(
        """Ticker,Period,Revenue_M,EBITDA_M,Net Income_M,Net Debt_M,Shares_M
IMAX,FY2025,352,120,52,120,55
CNK,FY2025,3100,650,210,1800,122
AMC,FY2025,4600,530,-220,4100,360
DBO.TO,FY2025,38,7,2,5,80""",
        language="csv"
    )


# -------------------------------------------------
# 6. 메인 로직
# -------------------------------------------------

if peer_file is None or financial_file is None:
    st.info("왼쪽 사이드바에서 Peer Master CSV와 Financials CSV를 업로드해줘.")
    st.stop()

peer_df = load_peer_master(peer_file)
financial_df = load_financials(financial_file)

required_peer_cols = {"Ticker", "Company", "Peer Group", "Country"}
required_financial_cols = {
    "Ticker", "Period", "Revenue_M", "EBITDA_M",
    "Net Income_M", "Net Debt_M", "Shares_M"
}

if not required_peer_cols.issubset(peer_df.columns):
    st.error(f"Peer Master CSV에 필요한 컬럼이 부족합니다: {required_peer_cols}")
    st.stop()

if not required_financial_cols.issubset(financial_df.columns):
    st.error(f"Financials CSV에 필요한 컬럼이 부족합니다: {required_financial_cols}")
    st.stop()

available_periods = sorted(financial_df["Period"].dropna().unique())
available_peer_groups = sorted(peer_df["Peer Group"].dropna().unique())

col1, col2, col3 = st.columns(3)

with col1:
    selected_period = st.selectbox("기준 실적 기간", available_periods, index=0)

with col2:
    selected_peer_groups = st.multiselect(
        "Peer Group 선택",
        available_peer_groups,
        default=available_peer_groups
    )

with col3:
    selected_multiple_type = st.selectbox(
        "Multiple 기준",
        ["EV/EBITDA", "EV/Revenue", "P/E"]
    )

filtered_peer_df = peer_df[peer_df["Peer Group"].isin(selected_peer_groups)].copy()
tickers = filtered_peer_df["Ticker"].dropna().unique().tolist()

market_df = get_market_data(tickers)

valuation_df = calculate_peer_valuation(
    filtered_peer_df,
    financial_df,
    market_df,
    selected_period
)

st.divider()

# -------------------------------------------------
# 7. Peer Table
# -------------------------------------------------

st.subheader("1. Peer Valuation Table")

display_cols = [
    "Ticker", "Company", "Peer Group", "Country",
    "Price", "Currency", "Market Cap_M", "Net Debt_M", "EV_M",
    "Revenue_M", "EBITDA_M", "Net Income_M",
    "EV/Revenue", "EV/EBITDA", "P/E",
    "Market Data Updated At"
]

existing_display_cols = [col for col in display_cols if col in valuation_df.columns]

st.dataframe(
    valuation_df[existing_display_cols].style.format({
        "Price": "{:,.2f}",
        "Market Cap_M": "{:,.1f}",
        "Net Debt_M": "{:,.1f}",
        "EV_M": "{:,.1f}",
        "Revenue_M": "{:,.1f}",
        "EBITDA_M": "{:,.1f}",
        "Net Income_M": "{:,.1f}",
        "EV/Revenue": "{:,.1f}x",
        "EV/EBITDA": "{:,.1f}x",
        "P/E": "{:,.1f}x"
    }),
    use_container_width=True,
    height=420
)


# -------------------------------------------------
# 8. Multiple Summary
# -------------------------------------------------

st.subheader("2. Peer Multiple Summary")

use_outlier_filter = st.checkbox("Outlier 제거 적용", value=True)

multiple_df = valuation_df.dropna(subset=[selected_multiple_type]).copy()

# 음수 multiple 제거 옵션
exclude_negative = st.checkbox("음수 Multiple 제외", value=True)

if exclude_negative:
    multiple_df = multiple_df[multiple_df[selected_multiple_type] > 0]

if use_outlier_filter:
    multiple_df_for_summary = remove_outliers_iqr(multiple_df, selected_multiple_type)
else:
    multiple_df_for_summary = multiple_df.copy()

avg_multiple = multiple_df_for_summary[selected_multiple_type].mean()
median_multiple = multiple_df_for_summary[selected_multiple_type].median()
min_multiple = multiple_df_for_summary[selected_multiple_type].min()
max_multiple = multiple_df_for_summary[selected_multiple_type].max()

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

summary_col1.metric("Average", f"{avg_multiple:,.1f}x")
summary_col2.metric("Median", f"{median_multiple:,.1f}x")
summary_col3.metric("Min", f"{min_multiple:,.1f}x")
summary_col4.metric("Max", f"{max_multiple:,.1f}x")

with st.expander("Multiple 산정 대상 Peer 보기"):
    st.dataframe(
        multiple_df_for_summary[
            ["Ticker", "Company", "Peer Group", selected_multiple_type]
        ].style.format({
            selected_multiple_type: "{:,.1f}x"
        }),
        use_container_width=True
    )


# -------------------------------------------------
# 9. Target Company Valuation
# -------------------------------------------------

st.subheader("3. Target Company Valuation")

target_col1, target_col2, target_col3 = st.columns(3)

with target_col1:
    target_company_name = st.text_input("Target Company", value="Target Company")

with target_col2:
    if selected_multiple_type == "EV/EBITDA":
        target_metric_name = "Target EBITDA_M"
        default_metric = 100.0
    elif selected_multiple_type == "EV/Revenue":
        target_metric_name = "Target Revenue_M"
        default_metric = 300.0
    else:
        target_metric_name = "Target Net Income_M"
        default_metric = 50.0

    target_metric = st.number_input(
        target_metric_name,
        value=default_metric,
        step=10.0
    )

with target_col3:
    selected_basis = st.radio(
        "적용 Multiple",
        ["Average", "Median", "Manual"],
        horizontal=True
    )

if selected_basis == "Average":
    applied_multiple = avg_multiple
elif selected_basis == "Median":
    applied_multiple = median_multiple
else:
    applied_multiple = st.number_input(
        "Manual Multiple",
        value=float(round(median_multiple, 1)) if not np.isnan(median_multiple) else 10.0,
        step=0.5
    )

target_value = calculate_target_value(target_metric, applied_multiple)

result_col1, result_col2, result_col3 = st.columns(3)

result_col1.metric("적용 Multiple", f"{applied_multiple:,.1f}x")
result_col2.metric("Target Metric", f"{target_metric:,.1f}M")
result_col3.metric("Implied Value", f"{target_value:,.1f}M")


# -------------------------------------------------
# 10. Sensitivity Table
# -------------------------------------------------

st.subheader("4. Sensitivity Table")

sensitivity_range = st.slider(
    "Multiple 민감도 범위",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="적용 Multiple ± 범위"
)

step = st.selectbox("간격", [0.5, 1.0], index=0)

multiple_values = np.arange(
    max(applied_multiple - sensitivity_range, 0),
    applied_multiple + sensitivity_range + step,
    step
)

sensitivity_df = pd.DataFrame({
    "Multiple": multiple_values,
    "Implied Value_M": multiple_values * target_metric
})

st.dataframe(
    sensitivity_df.style.format({
        "Multiple": "{:,.1f}x",
        "Implied Value_M": "{:,.1f}"
    }),
    use_container_width=True
)


# -------------------------------------------------
# 11. 다운로드
# -------------------------------------------------

st.subheader("5. Export")

export_df = valuation_df.copy()

csv = export_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="📥 Valuation Table CSV 다운로드",
    data=csv,
    file_name=f"peer_valuation_{selected_period}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)
