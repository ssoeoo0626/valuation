import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

# -------------------------------------------------
# 0. Page Setting
# -------------------------------------------------

st.set_page_config(
    page_title="Peer Valuation Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Peer Valuation Tool")
st.caption("Peer Group별 시장가치 및 Multiple 자동 산출 Tool")


# -------------------------------------------------
# 1. 기본 설정
# -------------------------------------------------

MARKET_DATA_TTL = 60 * 60 * 24  # 24시간 캐시


# -------------------------------------------------
# 2. 시장 데이터 호출
# -------------------------------------------------

@st.cache_data(ttl=MARKET_DATA_TTL)
def get_market_data(tickers):
    """
    yfinance에서 주가, 시가총액, 통화 데이터를 불러옴
    24시간 캐시 적용
    """
    rows = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")

            price = hist["Close"].iloc[-1] if not hist.empty else np.nan
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
    EV, EV/Revenue, EV/EBITDA, P/E 계산
    """

    financial_period = financial_df[financial_df["Period"] == selected_period].copy()

    df = peer_df.merge(financial_period, on="Ticker", how="left")
    df = df.merge(market_df, on="Ticker", how="left")

    # Market Cap은 yfinance에서 실제 단위로 들어옴 → 백만 단위로 변환
    df["Market Cap_M"] = df["Market Cap"] / 1_000_000

    # EV = Market Cap + Net Debt
    # Net Debt_M은 사용자가 백만 단위로 입력한다고 가정
    df["EV_M"] = df["Market Cap_M"] + df["Net Debt_M"]

    # Multiple 계산
    df["EV/Revenue"] = df["EV_M"] / df["Revenue_M"]
    df["EV/EBITDA"] = df["EV_M"] / df["EBITDA_M"]
    df["P/E"] = df["Market Cap_M"] / df["Net Income_M"]

    # 무한대 값 제거
    multiple_cols = ["EV/Revenue", "EV/EBITDA", "P/E"]
    for col in multiple_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def remove_outliers_iqr(df, col):
    """
    IQR 방식으로 Outlier 제거
    """
    clean_df = df.dropna(subset=[col]).copy()

    if clean_df.empty:
        return clean_df

    q1 = clean_df[col].quantile(0.25)
    q3 = clean_df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]


# -------------------------------------------------
# 4. Sidebar
# -------------------------------------------------

st.sidebar.header("⚙️ 설정")

if st.sidebar.button("🔄 시장 데이터 강제 새로고침"):
    get_market_data.clear()
    st.sidebar.success("시장 데이터 캐시를 초기화했습니다.")

st.sidebar.caption("시장 데이터 자동 갱신 주기: 24시간")
st.sidebar.caption("주가/시가총액은 yfinance 기준으로 불러옵니다.")


# -------------------------------------------------
# 5. 입력 데이터
# -------------------------------------------------

st.subheader("0. 입력 데이터")

st.info(
    "Peer 회사와 재무 데이터를 직접 입력한 뒤, 아래의 'Valuation 계산하기' 버튼을 눌러줘. "
    "Revenue, EBITDA, Net Income, Net Debt는 모두 백만 단위로 입력하면 됨."
)

default_peer_data = pd.DataFrame({
    "Ticker": ["IMAX", "CNK", "AMC", "CGX.TO", "DBO.TO", "DLB"],
    "Company": [
        "IMAX Corporation",
        "Cinemark Holdings",
        "AMC Entertainment",
        "Cineplex Inc.",
        "D-BOX Technologies",
        "Dolby Laboratories"
    ],
    "Peer Group": [
        "PLF",
        "Exhibitor",
        "Exhibitor",
        "Exhibitor",
        "Motion Seat",
        "Audio/PLF"
    ],
    "Country": [
        "US",
        "US",
        "US",
        "Canada",
        "Canada",
        "US"
    ]
})

default_financial_data = pd.DataFrame({
    "Ticker": ["IMAX", "CNK", "AMC", "CGX.TO", "DBO.TO", "DLB"],
    "Period": ["FY2025", "FY2025", "FY2025", "FY2025", "FY2025", "FY2025"],
    "Revenue_M": [410, 3100, 4600, 1050, 38, 1300],
    "EBITDA_M": [140, 650, 530, 130, 7, 420],
    "Net Income_M": [80, 210, -220, 20, 2, 300],
    "Net Debt_M": [120, 1800, 4100, 850, 5, -500],
    "Shares_M": [55, 122, 360, 64, 80, 95]
})

st.write("Peer Master 입력")
peer_df = st.data_editor(
    default_peer_data,
    num_rows="dynamic",
    use_container_width=True,
    key="peer_editor"
)

st.write("Financials 입력")
financial_df = st.data_editor(
    default_financial_data,
    num_rows="dynamic",
    use_container_width=True,
    key="financial_editor"
)

run_calculation = st.button("📊 Valuation 계산하기", type="primary")

if not run_calculation:
    st.stop()


# -------------------------------------------------
# 6. 입력값 검증
# -------------------------------------------------

required_peer_cols = {"Ticker", "Company", "Peer Group", "Country"}
required_financial_cols = {
    "Ticker",
    "Period",
    "Revenue_M",
    "EBITDA_M",
    "Net Income_M",
    "Net Debt_M",
    "Shares_M"
}

if not required_peer_cols.issubset(peer_df.columns):
    st.error(f"Peer Master에 필요한 컬럼이 부족합니다: {required_peer_cols}")
    st.stop()

if not required_financial_cols.issubset(financial_df.columns):
    st.error(f"Financials에 필요한 컬럼이 부족합니다: {required_financial_cols}")
    st.stop()

# 빈 티커 제거
peer_df = peer_df.dropna(subset=["Ticker"]).copy()
financial_df = financial_df.dropna(subset=["Ticker", "Period"]).copy()

# 숫자 컬럼 변환
numeric_cols = ["Revenue_M", "EBITDA_M", "Net Income_M", "Net Debt_M", "Shares_M"]

for col in numeric_cols:
    financial_df[col] = pd.to_numeric(financial_df[col], errors="coerce")

if peer_df.empty:
    st.error("Peer Master에 입력된 회사가 없습니다.")
    st.stop()

if financial_df.empty:
    st.error("Financials에 입력된 재무 데이터가 없습니다.")
    st.stop()


# -------------------------------------------------
# 7. 필터 설정
# -------------------------------------------------

available_periods = sorted(financial_df["Period"].dropna().unique())
available_peer_groups = sorted(peer_df["Peer Group"].dropna().unique())

if len(available_periods) == 0:
    st.error("Financials의 Period 값이 없습니다.")
    st.stop()

if len(available_peer_groups) == 0:
    st.error("Peer Group 값이 없습니다.")
    st.stop()

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    selected_period = st.selectbox(
        "기준 실적 기간",
        available_periods
    )

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

if len(selected_peer_groups) == 0:
    st.warning("Peer Group을 최소 1개 이상 선택해줘.")
    st.stop()


# -------------------------------------------------
# 8. 시장 데이터 호출 및 Valuation 계산
# -------------------------------------------------

filtered_peer_df = peer_df[peer_df["Peer Group"].isin(selected_peer_groups)].copy()
tickers = filtered_peer_df["Ticker"].dropna().unique().tolist()

with st.spinner("시장 데이터를 불러오는 중입니다..."):
    market_df = get_market_data(tickers)

valuation_df = calculate_peer_valuation(
    filtered_peer_df,
    financial_df,
    market_df,
    selected_period
)


# -------------------------------------------------
# 9. Peer Valuation Table
# -------------------------------------------------

st.subheader("1. Peer Valuation Table")

display_cols = [
    "Ticker",
    "Company",
    "Peer Group",
    "Country",
    "Price",
    "Currency",
    "Market Cap_M",
    "Net Debt_M",
    "EV_M",
    "Revenue_M",
    "EBITDA_M",
    "Net Income_M",
    "EV/Revenue",
    "EV/EBITDA",
    "P/E",
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
# 10. Peer Multiple Summary
# -------------------------------------------------

st.subheader("2. Peer Multiple Summary")

summary_option_col1, summary_option_col2 = st.columns(2)

with summary_option_col1:
    use_outlier_filter = st.checkbox("Outlier 제거 적용", value=True)

with summary_option_col2:
    exclude_negative = st.checkbox("음수 Multiple 제외", value=True)

multiple_df = valuation_df.dropna(subset=[selected_multiple_type]).copy()

if exclude_negative:
    multiple_df = multiple_df[multiple_df[selected_multiple_type] > 0]

if use_outlier_filter:
    multiple_df_for_summary = remove_outliers_iqr(multiple_df, selected_multiple_type)
else:
    multiple_df_for_summary = multiple_df.copy()

if multiple_df_for_summary.empty:
    st.warning("Multiple 산정 가능한 Peer가 없습니다. EBITDA/Net Income/Revenue 입력값을 확인해줘.")
    st.stop()

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
# 11. Target Company Valuation
# -------------------------------------------------

st.subheader("3. Target Company Valuation")

target_col1, target_col2, target_col3 = st.columns(3)

with target_col1:
    target_company_name = st.text_input(
        "Target Company",
        value="Target Company"
    )

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
        value=float(round(median_multiple, 1)),
        step=0.5
    )

target_value = target_metric * applied_multiple

result_col1, result_col2, result_col3 = st.columns(3)

result_col1.metric("적용 Multiple", f"{applied_multiple:,.1f}x")
result_col2.metric("Target Metric", f"{target_metric:,.1f}M")
result_col3.metric("Implied Value", f"{target_value:,.1f}M")


# -------------------------------------------------
# 12. Sensitivity Table
# -------------------------------------------------

st.subheader("4. Sensitivity Table")

sensitivity_col1, sensitivity_col2 = st.columns(2)

with sensitivity_col1:
    sensitivity_range = st.slider(
        "Multiple 민감도 범위",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5
    )

with sensitivity_col2:
    step = st.selectbox(
        "간격",
        [0.5, 1.0],
        index=0
    )

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
# 13. Export
# -------------------------------------------------

st.subheader("5. Export")

csv = valuation_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="📥 Valuation Table CSV 다운로드",
    data=csv,
    file_name=f"peer_valuation_{selected_period}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)
