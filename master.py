import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime


# -------------------------------------------------
# 0. Page Setting
# -------------------------------------------------

st.set_page_config(
    page_title="Peer Valuation Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Peer Valuation Tool")
st.caption("DART 재무데이터 + yfinance 시장데이터 기반 Peer Valuation 자동 산출")


# -------------------------------------------------
# 1. 기본 설정
# -------------------------------------------------

MARKET_DATA_TTL = 60 * 60 * 24

PEER_MASTER_PATH = "data/peer_master.csv"
FINANCIALS_PATH = "data/financials.csv"

DOMESTIC_PEER_GROUP = "CG/VFX"
DOMESTIC_CATEGORY = "CG/VFX"
DOMESTIC_COUNTRY = "Korea"

REQUIRED_PEER_COLS = ["Ticker", "Company", "Peer Group", "Country", "Category"]

REQUIRED_FINANCIAL_COLS = [
    "Ticker",
    "Period",
    "Revenue_M",
    "EBITDA_M",
    "Net Income_M",
    "Net Debt_M",
    "Shares_M"
]

try:
    DART_API_KEY = st.secrets["DART_API_KEY"]
except Exception:
    DART_API_KEY = None


# -------------------------------------------------
# 2. Master / Financials 정리 함수
# -------------------------------------------------

def is_korea_ticker_series(ticker_series):
    ticker_series = ticker_series.fillna("").astype(str).str.strip()
    return ticker_series.str.endswith(".KQ") | ticker_series.str.endswith(".KS")


def normalize_peer_master(df):
    """
    peer_master.csv 정리 함수

    - 컬럼명이 Peer_Group / peer_group 등으로 들어와도 Peer Group으로 보정
    - 국내 상장사(.KS/.KQ)는 Peer Group, Country, Category를 자동으로 CG/VFX / Korea로 통일
    - 최종 컬럼은 Ticker, Company, Peer Group, Country, Category만 유지
    """
    df = df.copy()

    rename_map = {
        "ticker": "Ticker",
        "Ticker": "Ticker",
        "company": "Company",
        "Company": "Company",
        "Peer_Group": "Peer Group",
        "peer_group": "Peer Group",
        "peer group": "Peer Group",
        "Peer group": "Peer Group",
        "Peer Group": "Peer Group",
        "country": "Country",
        "Country": "Country",
        "category": "Category",
        "Category": "Category",
    }

    df = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})

    for col in REQUIRED_PEER_COLS:
        if col not in df.columns:
            df[col] = ""

    for col in REQUIRED_PEER_COLS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    is_korea = is_korea_ticker_series(df["Ticker"])

    df.loc[is_korea, "Peer Group"] = DOMESTIC_PEER_GROUP
    df.loc[is_korea, "Country"] = DOMESTIC_COUNTRY
    df.loc[is_korea, "Category"] = DOMESTIC_CATEGORY

    return df[REQUIRED_PEER_COLS]


def normalize_financials(df):
    df = df.copy()

    for col in REQUIRED_FINANCIAL_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[REQUIRED_FINANCIAL_COLS]

    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip()
    df["Period"] = df["Period"].fillna("").astype(str).str.strip()

    numeric_cols = ["Revenue_M", "EBITDA_M", "Net Income_M", "Net Debt_M", "Shares_M"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Ticker"] != ""]
    df = df[df["Period"] != ""]

    return df


@st.cache_data
def load_peer_master():
    try:
        peer_df = pd.read_csv(PEER_MASTER_PATH)
    except FileNotFoundError:
        peer_df = pd.DataFrame(columns=REQUIRED_PEER_COLS)

    return normalize_peer_master(peer_df)


@st.cache_data
def load_financials():
    try:
        financial_df = pd.read_csv(FINANCIALS_PATH)
    except FileNotFoundError:
        financial_df = pd.DataFrame(columns=REQUIRED_FINANCIAL_COLS)

    return normalize_financials(financial_df)


# -------------------------------------------------
# 3. yfinance 시장데이터
# -------------------------------------------------

@st.cache_data(ttl=MARKET_DATA_TTL)
def get_market_data(tickers):
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
# 4. DART 함수
# -------------------------------------------------

@st.cache_data(show_spinner=False)
def get_dart_corp_code(api_key):
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    params = {"crtfc_key": api_key}

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    temp_dir = Path("temp_dart")
    temp_dir.mkdir(exist_ok=True)

    zip_path = temp_dir / "corpCode.zip"
    zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    xml_path = temp_dir / "CORPCODE.xml"

    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []

    for item in root.findall("list"):
        rows.append({
            "corp_code": item.findtext("corp_code"),
            "corp_name": item.findtext("corp_name"),
            "stock_code": item.findtext("stock_code"),
            "modify_date": item.findtext("modify_date")
        })

    corp_df = pd.DataFrame(rows)
    corp_df = corp_df[corp_df["stock_code"].notna()]
    corp_df = corp_df[corp_df["stock_code"] != ""]

    return corp_df


def find_corp_code(corp_df, ticker):
    stock_code = str(ticker).split(".")[0].zfill(6)
    matched = corp_df[corp_df["stock_code"] == stock_code]

    if matched.empty:
        return None

    return matched.iloc[0]["corp_code"]


def get_dart_financial_statement(api_key, corp_code, bsns_year="2025", reprt_code="11011"):
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"

    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": bsns_year,
        "reprt_code": reprt_code,
        "fs_div": "CFS"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "000":
        return pd.DataFrame(), data.get("message", "DART API Error")

    return pd.DataFrame(data.get("list", [])), None


def clean_amount(value):
    if value is None:
        return np.nan

    value = str(value)
    value = value.replace(",", "")
    value = value.replace("(", "-")
    value = value.replace(")", "")
    value = value.strip()

    if value in ["", "-", "nan", "None"]:
        return np.nan

    try:
        return float(value)
    except Exception:
        return np.nan


def pick_amount(fs_df, keywords, sj_div=None):
    if fs_df.empty:
        return np.nan

    temp = fs_df.copy()

    if sj_div is not None and "sj_div" in temp.columns:
        if isinstance(sj_div, list):
            temp = temp[temp["sj_div"].isin(sj_div)]
        else:
            temp = temp[temp["sj_div"] == sj_div]

    if temp.empty:
        return np.nan

    for keyword in keywords:
        matched = temp[
            temp["account_nm"]
            .astype(str)
            .str.contains(keyword, na=False, regex=False)
        ]

        if not matched.empty:
            return clean_amount(matched.iloc[0].get("thstrm_amount"))

    return np.nan


def safe_zero(value):
    if pd.isna(value):
        return 0.0
    return value


def calculate_dart_metrics(fs_df):
    """
    DART 원 단위 → 백만원 단위 변환

    Revenue_M = 연결 매출액
    EBITDA_M = 영업이익 + 감가상각비 + 무형자산상각비
    Net Income_M = 연결 당기순이익
    Net Debt_M = 이자부부채 - 현금및현금성자산

    주의:
    - DART 계정명이 회사마다 다르므로 일부 항목은 수동 검증 필요
    - 리스부채는 현재 Net Debt에 포함
    """

    revenue = pick_amount(
        fs_df,
        [
            "매출액",
            "수익(매출액)",
            "영업수익",
            "수익",
            "매출"
        ],
        sj_div=["IS", "CIS"]
    )

    operating_income = pick_amount(
        fs_df,
        [
            "영업이익",
            "영업손실"
        ],
        sj_div=["IS", "CIS"]
    )

    net_income = pick_amount(
        fs_df,
        [
            "당기순이익",
            "당기순손실",
            "당기순손익",
            "연결당기순이익",
            "연결당기순손실"
        ],
        sj_div=["IS", "CIS"]
    )

    depreciation_amortization = pick_amount(
        fs_df,
        [
            "감가상각비와 무형자산상각비",
            "감가상각비 및 무형자산상각비",
            "감가상각비와무형자산상각비",
            "감가상각비및무형자산상각비"
        ],
        sj_div="CF"
    )

    if pd.isna(depreciation_amortization):
        depreciation = pick_amount(
            fs_df,
            ["감가상각비"],
            sj_div="CF"
        )

        amortization = pick_amount(
            fs_df,
            ["무형자산상각비"],
            sj_div="CF"
        )

        depreciation_amortization = safe_zero(depreciation) + safe_zero(amortization)

    cash = pick_amount(
        fs_df,
        ["현금및현금성자산"],
        sj_div="BS"
    )

    short_borrowings = pick_amount(
        fs_df,
        ["단기차입금"],
        sj_div="BS"
    )

    current_long_debt = pick_amount(
        fs_df,
        [
            "유동성장기부채",
            "유동성장기차입금",
            "유동성 장기차입금",
            "유동성사채"
        ],
        sj_div="BS"
    )

    long_borrowings = pick_amount(
        fs_df,
        ["장기차입금"],
        sj_div="BS"
    )

    bonds = pick_amount(
        fs_df,
        ["사채"],
        sj_div="BS"
    )

    current_lease = pick_amount(
        fs_df,
        [
            "유동리스부채",
            "유동성리스부채",
            "유동성 리스부채"
        ],
        sj_div="BS"
    )

    noncurrent_lease = pick_amount(
        fs_df,
        [
            "비유동리스부채",
            "비유동 리스부채"
        ],
        sj_div="BS"
    )

    total_debt = (
        safe_zero(short_borrowings)
        + safe_zero(current_long_debt)
        + safe_zero(long_borrowings)
        + safe_zero(bonds)
        + safe_zero(current_lease)
        + safe_zero(noncurrent_lease)
    )

    operating_income = safe_zero(operating_income)
    cash = safe_zero(cash)

    ebitda = operating_income + safe_zero(depreciation_amortization)
    net_debt = total_debt - cash

    return {
        "Revenue_M": round(revenue / 1_000_000, 1) if pd.notna(revenue) else np.nan,
        "EBITDA_M": round(ebitda / 1_000_000, 1),
        "Net Income_M": round(net_income / 1_000_000, 1) if pd.notna(net_income) else np.nan,
        "Net Debt_M": round(net_debt / 1_000_000, 1),
        "Shares_M": 0.0
    }


def get_korea_peer_df(peer_df):
    korea_df = peer_df[is_korea_ticker_series(peer_df["Ticker"])].copy()
    korea_df = normalize_peer_master(korea_df)
    return korea_df


def fetch_dart_financials_for_korea_peers(peer_df, api_key, bsns_year="2025"):
    corp_df = get_dart_corp_code(api_key)
    korea_df = get_korea_peer_df(peer_df)

    rows = []
    logs = []

    for _, row in korea_df.iterrows():
        ticker = row["Ticker"]
        company = row.get("Company", "")

        corp_code = find_corp_code(corp_df, ticker)

        if corp_code is None:
            logs.append({
                "Ticker": ticker,
                "Company": company,
                "Status": "FAIL",
                "Message": "corp_code 찾기 실패"
            })
            continue

        fs_df, error_msg = get_dart_financial_statement(
            api_key=api_key,
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code="11011"
        )

        if error_msg:
            logs.append({
                "Ticker": ticker,
                "Company": company,
                "Status": "FAIL",
                "Message": error_msg
            })
            continue

        metrics = calculate_dart_metrics(fs_df)

        rows.append({
            "Ticker": ticker,
            "Period": f"FY{bsns_year}",
            "Revenue_M": metrics["Revenue_M"],
            "EBITDA_M": metrics["EBITDA_M"],
            "Net Income_M": metrics["Net Income_M"],
            "Net Debt_M": metrics["Net Debt_M"],
            "Shares_M": metrics["Shares_M"]
        })

        logs.append({
            "Ticker": ticker,
            "Company": company,
            "Status": "SUCCESS",
            "Message": "수집 완료"
        })

    return pd.DataFrame(rows), pd.DataFrame(logs)


# -------------------------------------------------
# 5. Valuation 계산 함수
# -------------------------------------------------

def calculate_peer_valuation(peer_df, financial_df, market_df, selected_period):
    financial_period = financial_df[financial_df["Period"] == selected_period].copy()

    df = peer_df.merge(financial_period, on="Ticker", how="left")
    df = df.merge(market_df, on="Ticker", how="left")

    df["Market Cap_M"] = df["Market Cap"] / 1_000_000
    df["EV_M"] = df["Market Cap_M"] + df["Net Debt_M"]

    df["EV/Revenue"] = df["EV_M"] / df["Revenue_M"]
    df["EV/EBITDA"] = df["EV_M"] / df["EBITDA_M"]
    df["P/E"] = df["Market Cap_M"] / df["Net Income_M"]

    for col in ["EV/Revenue", "EV/EBITDA", "P/E"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def remove_outliers_iqr(df, col):
    clean_df = df.dropna(subset=[col]).copy()

    if clean_df.empty:
        return clean_df

    q1 = clean_df[col].quantile(0.25)
    q3 = clean_df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]


def format_valuation_table(df):
    format_dict = {
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
    }

    existing_format = {
        col: fmt for col, fmt in format_dict.items()
        if col in df.columns
    }

    return df.style.format(existing_format)


# -------------------------------------------------
# 6. Sidebar
# -------------------------------------------------

st.sidebar.header("⚙️ 설정")

if DART_API_KEY:
    st.sidebar.success("DART API Key 연결 완료")
else:
    st.sidebar.warning("DART API Key 미연결")

if st.sidebar.button("🔄 시장 데이터 강제 새로고침"):
    get_market_data.clear()
    st.sidebar.success("시장 데이터 캐시를 초기화했습니다.")

if st.sidebar.button("🧹 DART 수집값 초기화"):
    st.session_state.pop("dart_financials_df", None)
    st.session_state.pop("dart_logs_df", None)
    st.session_state.pop("dart_valuation_df", None)
    st.sidebar.success("DART 수집값을 초기화했습니다.")

st.sidebar.caption("시장 데이터 자동 갱신 주기: 24시간")


# -------------------------------------------------
# 7. 기본 DB 로드
# -------------------------------------------------

default_peer_df = load_peer_master()
default_financial_df = load_financials()

if default_peer_df.empty:
    st.error("data/peer_master.csv 파일이 없거나 비어 있습니다.")
    st.stop()

missing_peer_cols = set(REQUIRED_PEER_COLS) - set(default_peer_df.columns)

if missing_peer_cols:
    st.error(f"peer_master.csv에 필요한 컬럼이 없습니다: {missing_peer_cols}")
    st.stop()


# -------------------------------------------------
# 8. DART 자동 수집 + 즉시 Valuation 산출
# -------------------------------------------------

st.subheader("0. DART 국내 Peer 재무데이터 / Multiple 자동 산출")

with st.expander("DART 수집 설정 / 실행", expanded=True):
    dart_year = st.text_input("사업연도", value="2025")

    st.caption(
        "국내 상장 Peer(.KQ / .KS)는 자동으로 Peer Group / Category가 CG/VFX로 통일됩니다. "
        "DART 연결 재무제표와 yfinance 시총을 결합해 EV/Revenue, EV/EBITDA, P/E를 계산합니다."
    )

    korea_peer_preview = get_korea_peer_df(default_peer_df)

    st.write("DART 수집 대상 국내 Peer")
    st.dataframe(korea_peer_preview, use_container_width=True, height=180)

    fetch_dart = st.button("📥 DART 재무데이터 + Multiple 가져오기", type="primary")

    if fetch_dart:
        if DART_API_KEY is None:
            st.error("DART API Key가 없습니다. Streamlit Secrets에 DART_API_KEY를 먼저 등록해주세요.")
        elif korea_peer_preview.empty:
            st.error("DART 수집 대상 국내 Peer가 없습니다. peer_master.csv의 Ticker에 .KQ 또는 .KS를 붙여주세요.")
        else:
            with st.spinner("DART 재무데이터와 시장데이터를 수집하는 중입니다..."):
                dart_financials_df, dart_logs_df = fetch_dart_financials_for_korea_peers(
                    default_peer_df,
                    DART_API_KEY,
                    bsns_year=dart_year
                )

                korea_tickers = korea_peer_preview["Ticker"].dropna().unique().tolist()
                korea_market_df = get_market_data(korea_tickers)

                dart_valuation_df = calculate_peer_valuation(
                    korea_peer_preview,
                    dart_financials_df,
                    korea_market_df,
                    selected_period=f"FY{dart_year}"
                )

            st.session_state["dart_financials_df"] = dart_financials_df
            st.session_state["dart_logs_df"] = dart_logs_df
            st.session_state["dart_valuation_df"] = dart_valuation_df

            st.success("DART 재무데이터 및 Multiple 산출 완료. 아래 Valuation 계산에도 바로 반영됩니다.")

    if "dart_logs_df" in st.session_state:
        st.write("수집 로그")
        st.dataframe(st.session_state["dart_logs_df"], use_container_width=True)

    if "dart_valuation_df" in st.session_state:
        st.write("DART 기반 국내 Peer Valuation Table")

        dart_display_cols = [
            "Ticker",
            "Company",
            "Peer Group",
            "Country",
            "Category",
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

        existing_dart_cols = [
            col for col in dart_display_cols
            if col in st.session_state["dart_valuation_df"].columns
        ]

        st.dataframe(
            format_valuation_table(st.session_state["dart_valuation_df"][existing_dart_cols]),
            use_container_width=True,
            height=420
        )


# -------------------------------------------------
# 9. Peer / Financials 입력
# -------------------------------------------------

st.subheader("1. Peer / Financials 입력")

tab1, tab2, tab3 = st.tabs(["기본 Peer DB", "현재 반영 Financials", "추가 입력"])

with tab1:
    st.dataframe(default_peer_df, use_container_width=True)

with tab2:
    financial_preview_list = []

    if not default_financial_df.empty:
        financial_preview_list.append(default_financial_df)

    if "dart_financials_df" in st.session_state and not st.session_state["dart_financials_df"].empty:
        financial_preview_list.append(st.session_state["dart_financials_df"])

    if len(financial_preview_list) > 0:
        preview_financial_df = pd.concat(financial_preview_list, ignore_index=True)
        preview_financial_df = preview_financial_df.drop_duplicates(
            subset=["Ticker", "Period"],
            keep="last"
        )
        st.dataframe(preview_financial_df, use_container_width=True)
    else:
        st.warning("현재 반영된 Financials 데이터가 없습니다. DART 재무데이터를 먼저 가져오거나 financials.csv를 입력해주세요.")

with tab3:
    st.caption("이번 분석에만 추가할 Peer가 있으면 여기에 입력하면 됩니다. 국내 티커는 .KQ 또는 .KS를 붙이면 자동으로 CG/VFX 처리됩니다.")

    extra_peer_template = pd.DataFrame({
        "Ticker": [""],
        "Company": [""],
        "Peer Group": [DOMESTIC_PEER_GROUP],
        "Country": [DOMESTIC_COUNTRY],
        "Category": [DOMESTIC_CATEGORY]
    })

    extra_financial_template = pd.DataFrame({
        "Ticker": [""],
        "Period": ["FY2025"],
        "Revenue_M": [0.0],
        "EBITDA_M": [0.0],
        "Net Income_M": [0.0],
        "Net Debt_M": [0.0],
        "Shares_M": [0.0]
    })

    st.write("추가 Peer 입력")
    extra_peer_df = st.data_editor(
        extra_peer_template,
        num_rows="dynamic",
        use_container_width=True,
        key="extra_peer_editor"
    )

    st.write("추가 Financials 입력")
    extra_financial_df = st.data_editor(
        extra_financial_template,
        num_rows="dynamic",
        use_container_width=True,
        key="extra_financial_editor"
    )


# -------------------------------------------------
# 10. 데이터 통합
# -------------------------------------------------

extra_peer_df = extra_peer_df[
    extra_peer_df["Ticker"].astype(str).str.strip() != ""
].copy()

extra_financial_df = extra_financial_df[
    extra_financial_df["Ticker"].astype(str).str.strip() != ""
].copy()

peer_df = pd.concat([default_peer_df, extra_peer_df], ignore_index=True)
peer_df = normalize_peer_master(peer_df)
peer_df = peer_df.drop_duplicates(subset=["Ticker"], keep="last")

financial_sources = []

if not default_financial_df.empty:
    financial_sources.append(default_financial_df)

if "dart_financials_df" in st.session_state and not st.session_state["dart_financials_df"].empty:
    financial_sources.append(st.session_state["dart_financials_df"])

if not extra_financial_df.empty:
    financial_sources.append(normalize_financials(extra_financial_df))

if len(financial_sources) > 0:
    financial_df = pd.concat(financial_sources, ignore_index=True)
else:
    financial_df = pd.DataFrame(columns=REQUIRED_FINANCIAL_COLS)

financial_df = normalize_financials(financial_df)
financial_df = financial_df.drop_duplicates(subset=["Ticker", "Period"], keep="last")


# -------------------------------------------------
# 11. Valuation 설정
# -------------------------------------------------

st.subheader("2. Valuation 설정")

available_periods = sorted(financial_df["Period"].dropna().unique())
available_categories = sorted(peer_df["Category"].dropna().unique())
available_peer_groups = sorted(peer_df["Peer Group"].dropna().unique())

if len(available_periods) == 0:
    st.warning("아직 Financials 데이터가 없습니다. 위에서 DART 재무데이터를 먼저 가져오거나 financials.csv를 입력해주세요.")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    selected_period = st.selectbox("기준 실적 기간", available_periods)

with col2:
    default_categories = (
        [DOMESTIC_CATEGORY]
        if DOMESTIC_CATEGORY in available_categories
        else available_categories
    )

    selected_categories = st.multiselect(
        "Category 선택",
        available_categories,
        default=default_categories
    )

with col3:
    selected_multiple_type = st.selectbox(
        "Multiple 기준",
        ["EV/EBITDA", "EV/Revenue", "P/E"]
    )

default_peer_groups = (
    [DOMESTIC_PEER_GROUP]
    if DOMESTIC_PEER_GROUP in available_peer_groups
    else available_peer_groups
)

selected_peer_groups = st.multiselect(
    "Peer Group 선택",
    available_peer_groups,
    default=default_peer_groups
)

run_calculation = st.button("📊 Valuation 계산하기", type="primary")

if not run_calculation:
    st.stop()

filtered_peer_df = peer_df[
    peer_df["Category"].isin(selected_categories)
    & peer_df["Peer Group"].isin(selected_peer_groups)
].copy()

if filtered_peer_df.empty:
    st.warning("선택된 Peer가 없습니다.")
    st.stop()

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
# 12. Peer Valuation Table
# -------------------------------------------------

st.subheader("3. Peer Valuation Table")

display_cols = [
    "Ticker",
    "Company",
    "Peer Group",
    "Country",
    "Category",
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
    format_valuation_table(valuation_df[existing_display_cols]),
    use_container_width=True,
    height=420
)


# -------------------------------------------------
# 13. Peer Multiple Summary
# -------------------------------------------------

st.subheader("4. Peer Multiple Summary")

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
    st.warning("Multiple 산정 가능한 Peer가 없습니다. 입력값 또는 필터를 확인해주세요.")
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
            ["Ticker", "Company", "Category", "Peer Group", selected_multiple_type]
        ].style.format({
            selected_multiple_type: "{:,.1f}x"
        }),
        use_container_width=True
    )


# -------------------------------------------------
# 14. Target Company Valuation
# -------------------------------------------------

st.subheader("5. Target Company Valuation")

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
        value=float(round(median_multiple, 1)),
        step=0.5
    )

target_value = target_metric * applied_multiple

result_col1, result_col2, result_col3 = st.columns(3)

result_col1.metric("적용 Multiple", f"{applied_multiple:,.1f}x")
result_col2.metric("Target Metric", f"{target_metric:,.1f}M")
result_col3.metric("Implied Value", f"{target_value:,.1f}M")


# -------------------------------------------------
# 15. Sensitivity Table
# -------------------------------------------------

st.subheader("6. Sensitivity Table")

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
# 16. Export
# -------------------------------------------------

st.subheader("7. Export")

csv = valuation_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="📥 Valuation Table CSV 다운로드",
    data=csv,
    file_name=f"peer_valuation_{selected_period}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)
