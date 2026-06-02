
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Peer Valuation Tool", page_icon="📊", layout="wide")
st.title("📊 Peer Valuation Tool")
st.caption("DART 재무데이터 + yfinance 시장데이터 기반 Peer Valuation 자동 산출")

MARKET_DATA_TTL = 60 * 60 * 24
PEER_MASTER_PATH = "data/peer_master.csv"
FINANCIALS_PATH = "data/financials.csv"

REQUIRED_PEER_COLS = ["Ticker", "Company", "Peer Group", "Country", "Category"]
REQUIRED_FINANCIAL_COLS = ["Ticker", "Period", "Revenue_M", "EBITDA_M", "Net Income_M", "Net Debt_M", "Shares_M"]

try:
    DART_API_KEY = st.secrets["DART_API_KEY"]
except Exception:
    DART_API_KEY = None


# =================================================
# 1. 파일 / 데이터 정리
# =================================================

def get_file_mtime(path: str):
    p = Path(path)
    return p.stat().st_mtime if p.exists() else None


def read_csv_safely(path: str, required_cols: list[str], label: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.warning(f"{label} 파일을 찾을 수 없습니다: {p.resolve()}")
        return pd.DataFrame(columns=required_cols)
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except pd.errors.ParserError:
        st.error(f"{label} CSV 형식이 깨져 있습니다.")
        st.caption(f"현재 앱이 읽는 경로: {p.resolve()}")
        st.caption("대부분 특정 줄의 콤마 개수가 맞지 않거나, ```csv 같은 코드블록 문자가 같이 저장된 경우입니다.")
        return pd.DataFrame(columns=required_cols)
    except Exception as e:
        st.error(f"{label} 파일을 읽는 중 오류가 발생했습니다: {e}")
        st.caption(f"현재 앱이 읽는 경로: {p.resolve()}")
        return pd.DataFrame(columns=required_cols)


def is_korea_ticker_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()
    return s.str.endswith(".KQ") | s.str.endswith(".KS")


def normalize_peer_master(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "ticker": "Ticker",
        "company": "Company",
        "Peer_Group": "Peer Group",
        "peer_group": "Peer Group",
        "peer group": "Peer Group",
        "Peer group": "Peer Group",
        "country": "Country",
        "category": "Category",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    for col in REQUIRED_PEER_COLS:
        if col not in df.columns:
            df[col] = ""

    for col in REQUIRED_PEER_COLS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    is_korea = is_korea_ticker_series(df["Ticker"])
    df.loc[is_korea & (df["Country"] == ""), "Country"] = "Korea"
    df.loc[is_korea & (df["Peer Group"] == ""), "Peer Group"] = "CG/VFX"
    df.loc[is_korea & (df["Category"] == ""), "Category"] = "Domestic Peer"

    df = df[df["Ticker"] != ""].copy()
    return df[REQUIRED_PEER_COLS]


def normalize_financials(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "ticker": "Ticker",
        "period": "Period",
        "Revenue": "Revenue_M",
        "EBITDA": "EBITDA_M",
        "Net Income": "Net Income_M",
        "Net_Income_M": "Net Income_M",
        "Net Debt": "Net Debt_M",
        "Net_Debt_M": "Net Debt_M",
        "Shares": "Shares_M",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    for col in REQUIRED_FINANCIAL_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[REQUIRED_FINANCIAL_COLS]
    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip()
    df["Period"] = df["Period"].fillna("").astype(str).str.strip()

    for col in ["Revenue_M", "EBITDA_M", "Net Income_M", "Net Debt_M", "Shares_M"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["Ticker"] != "") & (df["Period"] != "")]
    return df


def get_korea_peer_df(peer_df: pd.DataFrame) -> pd.DataFrame:
    peer_df = normalize_peer_master(peer_df)
    return peer_df[is_korea_ticker_series(peer_df["Ticker"])].copy()


@st.cache_data
def load_peer_master(file_mtime=None) -> pd.DataFrame:
    return normalize_peer_master(read_csv_safely(PEER_MASTER_PATH, REQUIRED_PEER_COLS, "peer_master.csv"))


@st.cache_data
def load_financials(file_mtime=None) -> pd.DataFrame:
    return normalize_financials(read_csv_safely(FINANCIALS_PATH, REQUIRED_FINANCIAL_COLS, "financials.csv"))


# =================================================
# 2. 시장 데이터
# =================================================

@st.cache_data(ttl=MARKET_DATA_TTL)
def get_market_data(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        ticker = str(ticker).strip()
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")
            price = hist["Close"].iloc[-1] if not hist.empty else np.nan
            rows.append({
                "Ticker": ticker,
                "Company Name": info.get("shortName", ticker),
                "Price": price,
                "Market Cap": info.get("marketCap", np.nan),
                "Currency": info.get("currency", ""),
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


# =================================================
# 3. DART
# =================================================

@st.cache_data(show_spinner=False)
def get_dart_corp_code(api_key: str) -> pd.DataFrame:
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    response = requests.get(url, params={"crtfc_key": api_key}, timeout=30)
    response.raise_for_status()

    temp_dir = Path("temp_dart")
    temp_dir.mkdir(exist_ok=True)
    zip_path = temp_dir / "corpCode.zip"
    zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    tree = ET.parse(temp_dir / "CORPCODE.xml")
    root = tree.getroot()

    rows = []
    for item in root.findall("list"):
        rows.append({
            "corp_code": item.findtext("corp_code"),
            "corp_name": item.findtext("corp_name"),
            "stock_code": item.findtext("stock_code"),
            "modify_date": item.findtext("modify_date")
        })

    df = pd.DataFrame(rows)
    df = df[df["stock_code"].notna()]
    df = df[df["stock_code"] != ""]
    return df


def find_corp_code(corp_df: pd.DataFrame, ticker: str):
    stock_code = str(ticker).split(".")[0].zfill(6)
    matched = corp_df[corp_df["stock_code"] == stock_code]
    if matched.empty:
        return None
    return matched.iloc[0]["corp_code"]


def get_dart_financial_statement(api_key: str, corp_code: str, bsns_year="2025", reprt_code="11011"):
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": bsns_year,
        "reprt_code": reprt_code,
        "fs_div": "CFS",
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
    value = str(value).replace(",", "").replace("(", "-").replace(")", "").strip()
    if value in ["", "-", "nan", "None"]:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def safe_zero(value):
    return 0.0 if pd.isna(value) else value


def pick_amount(fs_df: pd.DataFrame, keywords: list[str], sj_div=None):
    if fs_df.empty:
        return np.nan

    temp = fs_df.copy()
    if sj_div is not None and "sj_div" in temp.columns:
        temp = temp[temp["sj_div"].isin(sj_div)] if isinstance(sj_div, list) else temp[temp["sj_div"] == sj_div]

    if temp.empty:
        return np.nan

    for keyword in keywords:
        matched = temp[temp["account_nm"].astype(str).str.contains(keyword, na=False, regex=False)]
        if not matched.empty:
            return clean_amount(matched.iloc[0].get("thstrm_amount"))
    return np.nan


def calculate_dart_metrics(fs_df: pd.DataFrame) -> dict:
    revenue = pick_amount(fs_df, ["매출액", "수익(매출액)", "영업수익", "수익", "매출"], sj_div=["IS", "CIS"])
    operating_income = pick_amount(fs_df, ["영업이익", "영업손실"], sj_div=["IS", "CIS"])
    net_income = pick_amount(fs_df, ["당기순이익", "당기순손실", "당기순손익", "연결당기순이익", "연결당기순손실"], sj_div=["IS", "CIS"])

    depreciation_amortization = pick_amount(
        fs_df,
        ["감가상각비와 무형자산상각비", "감가상각비 및 무형자산상각비", "감가상각비와무형자산상각비", "감가상각비및무형자산상각비"],
        sj_div="CF"
    )
    if pd.isna(depreciation_amortization):
        depreciation = pick_amount(fs_df, ["감가상각비"], sj_div="CF")
        amortization = pick_amount(fs_df, ["무형자산상각비"], sj_div="CF")
        depreciation_amortization = safe_zero(depreciation) + safe_zero(amortization)

    cash = pick_amount(fs_df, ["현금및현금성자산"], sj_div="BS")
    short_borrowings = pick_amount(fs_df, ["단기차입금"], sj_div="BS")
    current_long_debt = pick_amount(fs_df, ["유동성장기부채", "유동성장기차입금", "유동성 장기차입금", "유동성사채"], sj_div="BS")
    long_borrowings = pick_amount(fs_df, ["장기차입금"], sj_div="BS")
    bonds = pick_amount(fs_df, ["사채"], sj_div="BS")
    current_lease = pick_amount(fs_df, ["유동리스부채", "유동성리스부채", "유동성 리스부채"], sj_div="BS")
    noncurrent_lease = pick_amount(fs_df, ["비유동리스부채", "비유동 리스부채"], sj_div="BS")

    total_debt = (
        safe_zero(short_borrowings)
        + safe_zero(current_long_debt)
        + safe_zero(long_borrowings)
        + safe_zero(bonds)
        + safe_zero(current_lease)
        + safe_zero(noncurrent_lease)
    )

    ebitda = safe_zero(operating_income) + safe_zero(depreciation_amortization)
    net_debt = total_debt - safe_zero(cash)

    return {
        "Revenue_M": round(revenue / 1_000_000, 1) if pd.notna(revenue) else np.nan,
        "EBITDA_M": round(ebitda / 1_000_000, 1),
        "Net Income_M": round(net_income / 1_000_000, 1) if pd.notna(net_income) else np.nan,
        "Net Debt_M": round(net_debt / 1_000_000, 1),
        "Shares_M": 0.0,
    }


def fetch_dart_financials_for_korea_peers(peer_df: pd.DataFrame, api_key: str, bsns_year="2025"):
    corp_df = get_dart_corp_code(api_key)
    korea_df = get_korea_peer_df(peer_df)

    rows, logs = [], []
    for _, row in korea_df.iterrows():
        ticker = row["Ticker"]
        company = row.get("Company", "")
        corp_code = find_corp_code(corp_df, ticker)

        if corp_code is None:
            logs.append({"Ticker": ticker, "Company": company, "Status": "FAIL", "Message": "corp_code 찾기 실패"})
            continue

        fs_df, error_msg = get_dart_financial_statement(api_key, corp_code, bsns_year=bsns_year, reprt_code="11011")
        if error_msg:
            logs.append({"Ticker": ticker, "Company": company, "Status": "FAIL", "Message": error_msg})
            continue

        metrics = calculate_dart_metrics(fs_df)
        rows.append({
            "Ticker": ticker,
            "Period": f"FY{bsns_year}",
            "Revenue_M": metrics["Revenue_M"],
            "EBITDA_M": metrics["EBITDA_M"],
            "Net Income_M": metrics["Net Income_M"],
            "Net Debt_M": metrics["Net Debt_M"],
            "Shares_M": metrics["Shares_M"],
        })
        logs.append({"Ticker": ticker, "Company": company, "Status": "SUCCESS", "Message": "DART 수집 완료"})

    return pd.DataFrame(rows), pd.DataFrame(logs)


# =================================================
# 4. Target 자동 분석
# =================================================

def pick_from_yf_df(df: pd.DataFrame, names: list[str]):
    if df is None or df.empty:
        return np.nan
    for name in names:
        if name in df.index:
            value = df.loc[name].dropna()
            if not value.empty:
                return float(value.iloc[0])
    return np.nan


def get_latest_yfinance_financials(ticker: str) -> dict:
    ticker = str(ticker).strip()
    stock = yf.Ticker(ticker)

    info = stock.info
    income_stmt = stock.financials
    balance_sheet = stock.balance_sheet
    hist = stock.history(period="1d")

    price = hist["Close"].iloc[-1] if not hist.empty else np.nan
    market_cap = info.get("marketCap", np.nan)

    revenue = pick_from_yf_df(income_stmt, ["Total Revenue", "Operating Revenue", "Revenue"])
    ebitda = pick_from_yf_df(income_stmt, ["EBITDA", "Normalized EBITDA"])
    net_income = pick_from_yf_df(income_stmt, ["Net Income", "Net Income Common Stockholders"])

    cash = pick_from_yf_df(balance_sheet, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
    total_debt = pick_from_yf_df(balance_sheet, ["Total Debt"])
    if pd.isna(total_debt):
        short_debt = pick_from_yf_df(balance_sheet, ["Current Debt", "Current Debt And Capital Lease Obligation"])
        long_debt = pick_from_yf_df(balance_sheet, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
        total_debt = safe_zero(short_debt) + safe_zero(long_debt)

    net_debt = safe_zero(total_debt) - safe_zero(cash)

    return {
        "Ticker": ticker,
        "Company": info.get("shortName", ticker),
        "Currency": info.get("currency", ""),
        "Price": price,
        "Market Cap_M": market_cap / 1_000_000 if pd.notna(market_cap) else np.nan,
        "Revenue_M": revenue / 1_000_000 if pd.notna(revenue) else np.nan,
        "EBITDA_M": ebitda / 1_000_000 if pd.notna(ebitda) else np.nan,
        "Net Income_M": net_income / 1_000_000 if pd.notna(net_income) else np.nan,
        "Net Debt_M": net_debt / 1_000_000 if pd.notna(net_debt) else np.nan,
        "Source": "yfinance",
    }


def get_target_company_financials(ticker: str, api_key=None, bsns_year="2025") -> dict:
    ticker = str(ticker).strip()

    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        if api_key is None:
            raise ValueError("DART API Key가 없습니다.")

        corp_df = get_dart_corp_code(api_key)
        corp_code = find_corp_code(corp_df, ticker)
        if corp_code is None:
            raise ValueError("DART corp_code를 찾지 못했습니다.")

        fs_df, error_msg = get_dart_financial_statement(api_key, corp_code, bsns_year=bsns_year, reprt_code="11011")
        if error_msg:
            raise ValueError(error_msg)

        metrics = calculate_dart_metrics(fs_df)
        market_df = get_market_data([ticker])

        price, market_cap_m, currency, company_name = np.nan, np.nan, "KRW", ticker
        if not market_df.empty:
            price = market_df.iloc[0].get("Price", np.nan)
            market_cap = market_df.iloc[0].get("Market Cap", np.nan)
            market_cap_m = market_cap / 1_000_000 if pd.notna(market_cap) else np.nan
            currency = market_df.iloc[0].get("Currency", "KRW")
            company_name = market_df.iloc[0].get("Company Name", ticker)

        return {
            "Ticker": ticker,
            "Company": company_name,
            "Currency": currency,
            "Price": price,
            "Market Cap_M": market_cap_m,
            "Revenue_M": metrics["Revenue_M"],
            "EBITDA_M": metrics["EBITDA_M"],
            "Net Income_M": metrics["Net Income_M"],
            "Net Debt_M": metrics["Net Debt_M"],
            "Source": "DART",
        }

    return get_latest_yfinance_financials(ticker)


# =================================================
# 5. Valuation / 스타일
# =================================================

def calculate_peer_valuation(peer_df: pd.DataFrame, financial_df: pd.DataFrame, market_df: pd.DataFrame, selected_period: str) -> pd.DataFrame:
    peer_df = peer_df.copy()
    financial_df = financial_df.copy()
    market_df = market_df.copy()

    peer_df["Ticker"] = peer_df["Ticker"].astype(str).str.strip()
    financial_df["Ticker"] = financial_df["Ticker"].astype(str).str.strip()
    financial_df["Period"] = financial_df["Period"].astype(str).str.strip()
    market_df["Ticker"] = market_df["Ticker"].astype(str).str.strip()

    selected_financial = financial_df[financial_df["Period"] == selected_period].copy()
    selected_financial["Financial Match"] = "Selected Period"
    selected_financial["Financial Period Used"] = selected_financial["Period"]

    fallback_financial = (
        financial_df.sort_values(["Ticker", "Period"])
        .drop_duplicates(subset=["Ticker"], keep="last")
        .copy()
    )
    fallback_financial["Financial Match"] = "Fallback Latest"
    fallback_financial["Financial Period Used"] = fallback_financial["Period"]

    selected_tickers = set(selected_financial["Ticker"].dropna().unique())
    fallback_financial = fallback_financial[~fallback_financial["Ticker"].isin(selected_tickers)].copy()

    financial_period = pd.concat([selected_financial, fallback_financial], ignore_index=True)
    financial_period = financial_period.drop_duplicates(subset=["Ticker"], keep="first")

    df = peer_df.merge(financial_period, on="Ticker", how="left")
    df = df.merge(market_df, on="Ticker", how="left")

    df["Financial Match"] = df["Financial Match"].fillna("No Financials")
    df["Financial Period Used"] = df["Financial Period Used"].fillna("")

    for col in ["Market Cap", "Revenue_M", "EBITDA_M", "Net Income_M", "Net Debt_M"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Market Cap_M"] = df["Market Cap"] / 1_000_000
    df["EV_M"] = df["Market Cap_M"] + df["Net Debt_M"]

    df["EV/Revenue"] = df["EV_M"] / df["Revenue_M"]
    df["EV/EBITDA"] = df["EV_M"] / df["EBITDA_M"]
    df["P/E"] = df["Market Cap_M"] / df["Net Income_M"]

    for col in ["EV/Revenue", "EV/EBITDA", "P/E"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def remove_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    clean_df = df.dropna(subset=[col]).copy()
    if clean_df.empty:
        return clean_df

    q1 = clean_df[col].quantile(0.25)
    q3 = clean_df[col].quantile(0.75)
    iqr = q3 - q1
    return clean_df[(clean_df[col] >= q1 - 1.5 * iqr) & (clean_df[col] <= q3 + 1.5 * iqr)]


def style_valuation_table(df: pd.DataFrame):
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
        "P/E": "{:,.1f}x",
    }
    return df.style.format({c: f for c, f in format_dict.items() if c in df.columns})


# =================================================
# 6. Sidebar / 데이터 로드
# =================================================

st.sidebar.header("⚙️ 설정")

if DART_API_KEY:
    st.sidebar.success("DART API Key 연결 완료")
else:
    st.sidebar.warning("DART API Key 미연결")

if st.sidebar.button("🔄 시장 데이터 새로고침"):
    get_market_data.clear()
    st.rerun()

if st.sidebar.button("🔄 Master / Financials 새로고침"):
    load_peer_master.clear()
    load_financials.clear()
    for key in ["dart_financials_df", "dart_logs_df", "valuation_ready", "valuation_df", "target_data"]:
        st.session_state.pop(key, None)
    st.rerun()

peer_master_mtime = get_file_mtime(PEER_MASTER_PATH)
financials_mtime = get_file_mtime(FINANCIALS_PATH)

default_peer_df = load_peer_master(peer_master_mtime)
default_financial_df = load_financials(financials_mtime)

if default_peer_df.empty:
    st.error("data/peer_master.csv 파일이 없거나 비어 있습니다.")
    st.stop()

if default_financial_df.empty:
    st.warning("data/financials.csv 파일이 없거나 비어 있습니다. 해외 Peer의 Net Debt / EV 계산은 financials.csv 입력 후 가능합니다.")

missing_peer_cols = set(REQUIRED_PEER_COLS) - set(default_peer_df.columns)
if missing_peer_cols:
    st.error(f"peer_master.csv에 필요한 컬럼이 없습니다: {missing_peer_cols}")
    st.stop()


# =================================================
# 7. DART 수집
# =================================================

st.subheader("0. DART 재무데이터 수집")

dart_year = "2025"
fetch_dart = st.button("📥 DART 재무데이터 가져오기", type="primary")

if fetch_dart:
    if DART_API_KEY is None:
        st.error("DART API Key가 없습니다. Streamlit Secrets에 DART_API_KEY를 먼저 등록해주세요.")
    else:
        korea_peer_df = get_korea_peer_df(default_peer_df)
        if korea_peer_df.empty:
            st.error("DART 수집 대상 국내 Peer가 없습니다. peer_master.csv의 Ticker에 .KQ 또는 .KS를 붙여주세요.")
        else:
            with st.spinner("DART 재무데이터를 수집하는 중입니다..."):
                dart_financials_df, dart_logs_df = fetch_dart_financials_for_korea_peers(
                    default_peer_df,
                    DART_API_KEY,
                    bsns_year=dart_year,
                )
            st.session_state["dart_financials_df"] = dart_financials_df
            st.session_state["dart_logs_df"] = dart_logs_df
            st.session_state.pop("valuation_ready", None)
            st.session_state.pop("valuation_df", None)
            st.success("DART 재무데이터 수집 완료. 아래 Valuation 계산에 반영됩니다.")


# =================================================
# 8. 데이터 통합
# =================================================

peer_df = normalize_peer_master(default_peer_df).drop_duplicates(subset=["Ticker"], keep="last")

financial_sources = []
if not default_financial_df.empty:
    financial_sources.append(default_financial_df)
if "dart_financials_df" in st.session_state and not st.session_state["dart_financials_df"].empty:
    financial_sources.append(st.session_state["dart_financials_df"])

if financial_sources:
    financial_df = pd.concat(financial_sources, ignore_index=True)
else:
    financial_df = pd.DataFrame(columns=REQUIRED_FINANCIAL_COLS)

financial_df = normalize_financials(financial_df).drop_duplicates(subset=["Ticker", "Period"], keep="last")


# =================================================
# 9. Valuation 설정
# =================================================

st.subheader("1. Valuation 설정")

available_periods = sorted(financial_df["Period"].dropna().unique())
available_categories = sorted(peer_df["Category"].dropna().unique())
available_peer_groups = sorted(peer_df["Peer Group"].dropna().unique())

if not available_periods:
    st.warning("아직 Financials 데이터가 없습니다. DART 재무데이터를 가져오거나 financials.csv를 입력해주세요.")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    selected_period = st.selectbox("기준 실적 기간", available_periods, key="selected_period")
with col2:
    selected_categories = st.multiselect("Category 선택", available_categories, default=available_categories, key="selected_categories")
with col3:
    selected_multiple_type = st.selectbox("Multiple 기준", ["EV/EBITDA", "EV/Revenue", "P/E"], key="selected_multiple_type")

selected_peer_groups = st.multiselect("Peer Group 선택", available_peer_groups, default=available_peer_groups, key="selected_peer_groups")

if "valuation_ready" not in st.session_state:
    st.session_state["valuation_ready"] = False

run_calculation = st.button("📊 Valuation 계산하기", type="primary")

if run_calculation:
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

    valuation_df = calculate_peer_valuation(filtered_peer_df, financial_df, market_df, selected_period)

    st.session_state["valuation_ready"] = True
    st.session_state["valuation_df"] = valuation_df
    st.session_state["selected_period_saved"] = selected_period
    st.session_state["selected_multiple_type_saved"] = selected_multiple_type

if not st.session_state["valuation_ready"]:
    st.stop()

valuation_df = st.session_state.get("valuation_df", pd.DataFrame())
selected_period = st.session_state.get("selected_period_saved", selected_period)
selected_multiple_type = st.session_state.get("selected_multiple_type_saved", selected_multiple_type)

if valuation_df.empty:
    st.warning("Valuation 계산 결과가 없습니다. Valuation 계산하기를 눌러주세요.")
    st.stop()


# =================================================
# 10. Peer Valuation Table
# =================================================

st.subheader("2. Peer Valuation Table")

display_cols = [
    "Ticker", "Company", "Peer Group", "Country", "Category",
    "Financial Match", "Financial Period Used",
    "Price", "Currency", "Market Cap_M", "Net Debt_M", "EV_M",
    "Revenue_M", "EBITDA_M", "Net Income_M",
    "EV/Revenue", "EV/EBITDA", "P/E", "Market Data Updated At"
]
existing_display_cols = [c for c in display_cols if c in valuation_df.columns]
st.dataframe(style_valuation_table(valuation_df[existing_display_cols]), use_container_width=True, height=420)


# =================================================
# 11. Peer Multiple Summary
# =================================================

st.subheader("3. Peer Multiple Summary")

c1, c2 = st.columns(2)
with c1:
    use_outlier_filter = st.checkbox("Outlier 제거 적용", value=True)
with c2:
    exclude_negative = st.checkbox("음수 Multiple 제외", value=True)

multiple_df = valuation_df.dropna(subset=[selected_multiple_type]).copy()
if exclude_negative:
    multiple_df = multiple_df[multiple_df[selected_multiple_type] > 0]
multiple_df_for_summary = remove_outliers_iqr(multiple_df, selected_multiple_type) if use_outlier_filter else multiple_df.copy()

if multiple_df_for_summary.empty:
    st.warning("Multiple 산정 가능한 Peer가 없습니다. 입력값 또는 필터를 확인해주세요.")
    st.stop()

avg_multiple = multiple_df_for_summary[selected_multiple_type].mean()
median_multiple = multiple_df_for_summary[selected_multiple_type].median()
min_multiple = multiple_df_for_summary[selected_multiple_type].min()
max_multiple = multiple_df_for_summary[selected_multiple_type].max()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Average", f"{avg_multiple:,.1f}x")
m2.metric("Median", f"{median_multiple:,.1f}x")
m3.metric("Min", f"{min_multiple:,.1f}x")
m4.metric("Max", f"{max_multiple:,.1f}x")

with st.expander("Multiple 산정 대상 Peer 보기"):
    st.dataframe(
        multiple_df_for_summary[["Ticker", "Company", "Category", "Peer Group", selected_multiple_type]].style.format({selected_multiple_type: "{:,.1f}x"}),
        use_container_width=True
    )


# =================================================
# 12. Target Company Valuation
# =================================================

st.subheader("4. Target Company Valuation")

with st.form("target_company_form"):
    t1, t2, t3 = st.columns(3)
    with t1:
        target_ticker = st.text_input("Target Ticker", value=st.session_state.get("target_ticker", ""), placeholder="예: 299900.KQ / IMAX / DLB")
    with t2:
        target_year = st.text_input("Target 사업연도", value=st.session_state.get("target_year", "2025"))
    with t3:
        selected_basis = st.radio(
            "적용 Multiple",
            ["Average", "Median", "Manual"],
            index=["Average", "Median", "Manual"].index(st.session_state.get("selected_basis", "Median")),
            horizontal=True
        )

    manual_multiple = None
    if selected_basis == "Manual":
        manual_multiple = st.number_input("Manual Multiple", value=float(st.session_state.get("manual_multiple", round(median_multiple, 1))), step=0.5)

    fetch_target = st.form_submit_button("Target 자동 분석")

if fetch_target:
    if target_ticker.strip() == "":
        st.warning("Target Ticker를 입력해주세요.")
    else:
        try:
            with st.spinner("Target Company 재무/시장 데이터를 가져오는 중입니다..."):
                target_data = get_target_company_financials(target_ticker, api_key=DART_API_KEY, bsns_year=target_year)

            st.session_state["target_ticker"] = target_ticker
            st.session_state["target_year"] = target_year
            st.session_state["selected_basis"] = selected_basis
            st.session_state["target_data"] = target_data
            if manual_multiple is not None:
                st.session_state["manual_multiple"] = manual_multiple
            st.success("Target Company 분석 완료")
        except Exception as e:
            st.error(f"Target Company 데이터를 가져오지 못했습니다: {e}")

target_data = st.session_state.get("target_data", None)

target_metric = np.nan
target_metric_name = ""
applied_multiple = median_multiple
implied_ev = np.nan
implied_equity = np.nan

if target_data is not None:
    selected_basis = st.session_state.get("selected_basis", "Median")
    if selected_basis == "Average":
        applied_multiple = avg_multiple
    elif selected_basis == "Median":
        applied_multiple = median_multiple
    else:
        applied_multiple = st.session_state.get("manual_multiple", median_multiple)

    if selected_multiple_type == "EV/EBITDA":
        target_metric_name = "EBITDA_M"
        target_metric = target_data.get("EBITDA_M", np.nan)
        implied_ev = target_metric * applied_multiple if pd.notna(target_metric) else np.nan
        implied_equity = implied_ev - target_data.get("Net Debt_M", np.nan) if pd.notna(implied_ev) else np.nan
    elif selected_multiple_type == "EV/Revenue":
        target_metric_name = "Revenue_M"
        target_metric = target_data.get("Revenue_M", np.nan)
        implied_ev = target_metric * applied_multiple if pd.notna(target_metric) else np.nan
        implied_equity = implied_ev - target_data.get("Net Debt_M", np.nan) if pd.notna(implied_ev) else np.nan
    else:
        target_metric_name = "Net Income_M"
        target_metric = target_data.get("Net Income_M", np.nan)
        implied_ev = np.nan
        implied_equity = target_metric * applied_multiple if pd.notna(target_metric) else np.nan

    target_summary_df = pd.DataFrame([{
        "Ticker": target_data.get("Ticker"),
        "Company": target_data.get("Company"),
        "Source": target_data.get("Source"),
        "Currency": target_data.get("Currency"),
        "Price": target_data.get("Price"),
        "Market Cap_M": target_data.get("Market Cap_M"),
        "Revenue_M": target_data.get("Revenue_M"),
        "EBITDA_M": target_data.get("EBITDA_M"),
        "Net Income_M": target_data.get("Net Income_M"),
        "Net Debt_M": target_data.get("Net Debt_M"),
        "Selected Multiple": selected_multiple_type,
        "Applied Multiple": applied_multiple,
        "Target Metric": target_metric_name,
        "Target Metric Value_M": target_metric,
        "Implied EV_M": implied_ev,
        "Implied Equity Value_M": implied_equity,
    }])

    st.write("Target Company 요약")
    st.dataframe(
        target_summary_df.style.format({
            "Price": "{:,.2f}", "Market Cap_M": "{:,.1f}", "Revenue_M": "{:,.1f}",
            "EBITDA_M": "{:,.1f}", "Net Income_M": "{:,.1f}", "Net Debt_M": "{:,.1f}",
            "Applied Multiple": "{:,.1f}x", "Target Metric Value_M": "{:,.1f}",
            "Implied EV_M": "{:,.1f}", "Implied Equity Value_M": "{:,.1f}"
        }),
        use_container_width=True
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("적용 Multiple", f"{applied_multiple:,.1f}x")
    r2.metric(target_metric_name, f"{target_metric:,.1f}M" if pd.notna(target_metric) else "N/A")
    r3.metric("Implied EV", f"{implied_ev:,.1f}M" if pd.notna(implied_ev) else "N/A")
    r4.metric("Implied Equity", f"{implied_equity:,.1f}M" if pd.notna(implied_equity) else "N/A")


# =================================================
# 13. Sensitivity Table
# =================================================

st.subheader("5. Sensitivity Table")

if target_data is None:
    st.info("Sensitivity Table을 보려면 Target Company 자동 분석을 먼저 실행해주세요.")
else:
    s1, s2 = st.columns(2)
    with s1:
        sensitivity_range = st.slider("Multiple 민감도 범위", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    with s2:
        step = st.selectbox("간격", [0.5, 1.0], index=0)

    if pd.isna(target_metric):
        st.warning("Target metric이 비어 있어 Sensitivity Table을 계산할 수 없습니다.")
    else:
        multiple_values = np.arange(max(applied_multiple - sensitivity_range, 0), applied_multiple + sensitivity_range + step, step)
        sensitivity_df = pd.DataFrame({"Multiple": multiple_values, "Implied Value_M": multiple_values * target_metric})
        st.dataframe(sensitivity_df.style.format({"Multiple": "{:,.1f}x", "Implied Value_M": "{:,.1f}"}), use_container_width=True)


# =================================================
# 14. Export
# =================================================

st.subheader("6. Export")
csv = valuation_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="📥 Valuation Table CSV 다운로드",
    data=csv,
    file_name=f"peer_valuation_{selected_period}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
)
