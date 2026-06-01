import io
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="기업가치평가 자동화 Tool",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# Utility Functions
# =========================================================
def format_money(value: float, unit: str = "$M") -> str:
    """Format number for dashboard display."""
    if pd.isna(value):
        return "-"
    return f"{unit} {value:,.1f}"


def format_multiple(value: float) -> str:
    if pd.isna(value) or np.isinf(value):
        return "-"
    return f"{value:,.1f}x"


def calc_ev(market_cap: float, debt: float, cash: float, minority_interest: float = 0.0) -> float:
    """Enterprise Value = Market Cap + Debt - Cash + Minority Interest."""
    return market_cap + debt - cash + minority_interest


def calc_peer_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EV and valuation multiples for peer companies."""
    result = df.copy()

    required_cols = ["Market Cap", "Debt", "Cash", "Revenue", "EBITDA"]
    for col in required_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    if "Minority Interest" not in result.columns:
        result["Minority Interest"] = 0.0
    result["Minority Interest"] = pd.to_numeric(result["Minority Interest"], errors="coerce").fillna(0.0)

    result["Net Debt"] = result["Debt"] - result["Cash"]
    result["EV"] = result["Market Cap"] + result["Debt"] - result["Cash"] + result["Minority Interest"]

    result["EV/Sales"] = np.where(result["Revenue"] > 0, result["EV"] / result["Revenue"], np.nan)
    result["EV/EBITDA"] = np.where(result["EBITDA"] > 0, result["EV"] / result["EBITDA"], np.nan)

    return result


def get_peer_stats(df: pd.DataFrame, multiple_col: str, exclude_negative: bool = True) -> pd.Series:
    """Return valuation multiple statistics."""
    values = pd.to_numeric(df[multiple_col], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    if exclude_negative:
        values = values[values > 0]

    if len(values) == 0:
        return pd.Series({
            "Mean": np.nan,
            "Median": np.nan,
            "25th Percentile": np.nan,
            "75th Percentile": np.nan,
            "Min": np.nan,
            "Max": np.nan,
        })

    return pd.Series({
        "Mean": values.mean(),
        "Median": values.median(),
        "25th Percentile": values.quantile(0.25),
        "75th Percentile": values.quantile(0.75),
        "Min": values.min(),
        "Max": values.max(),
    })


def calc_target_valuation(
    target_metric: float,
    net_debt: float,
    low_multiple: float,
    base_multiple: float,
    high_multiple: float,
) -> pd.DataFrame:
    """Calculate valuation range based on selected multiple."""
    rows = []
    for case, multiple in [
        ("Low", low_multiple),
        ("Base", base_multiple),
        ("High", high_multiple),
    ]:
        ev = target_metric * multiple
        equity_value = ev - net_debt
        rows.append({
            "Case": case,
            "Applied Multiple": multiple,
            "Enterprise Value": ev,
            "Net Debt": net_debt,
            "Equity Value": equity_value,
        })
    return pd.DataFrame(rows)


def build_dcf_projection(
    base_revenue: float,
    growth_rates: list,
    ebitda_margins: list,
    da_pct: float,
    capex_pct: float,
    nwc_pct: float,
    tax_rate: float,
) -> pd.DataFrame:
    """Build 5-year simplified DCF projection."""
    years = [f"Y{i}" for i in range(1, 6)]
    rows = []
    revenue = base_revenue

    for i in range(5):
        revenue = revenue * (1 + growth_rates[i])
        ebitda = revenue * ebitda_margins[i]
        da = revenue * da_pct
        ebit = ebitda - da
        tax = max(ebit, 0) * tax_rate
        nopat = ebit - tax
        capex = revenue * capex_pct
        nwc_increase = revenue * nwc_pct
        fcf = nopat + da - capex - nwc_increase

        rows.append({
            "Year": years[i],
            "Revenue": revenue,
            "Growth": growth_rates[i],
            "EBITDA Margin": ebitda_margins[i],
            "EBITDA": ebitda,
            "D&A": da,
            "EBIT": ebit,
            "Tax": tax,
            "NOPAT": nopat,
            "CapEx": capex,
            "NWC Increase": nwc_increase,
            "FCF": fcf,
        })

    return pd.DataFrame(rows)


def calc_dcf_value(
    fcf_list: list,
    wacc: float,
    terminal_growth: float,
    net_debt: float,
) -> Tuple[float, float, float, float]:
    """Calculate DCF Enterprise Value and Equity Value."""
    if wacc <= terminal_growth:
        return np.nan, np.nan, np.nan, np.nan

    pv_fcf = sum(fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(fcf_list))
    terminal_value = fcf_list[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + wacc) ** len(fcf_list))
    enterprise_value = pv_fcf + pv_terminal_value
    equity_value = enterprise_value - net_debt

    return pv_fcf, terminal_value, enterprise_value, equity_value


def build_sensitivity_table(
    fcf_list: list,
    net_debt: float,
    wacc_range: list,
    tg_range: list,
    value_type: str = "Enterprise Value",
) -> pd.DataFrame:
    """Build WACC x Terminal Growth sensitivity table."""
    data = []
    for tg in tg_range:
        row = {"Terminal Growth": f"{tg:.1%}"}
        for wacc in wacc_range:
            _, _, ev, equity = calc_dcf_value(fcf_list, wacc, tg, net_debt)
            row[f"{wacc:.1%}"] = ev if value_type == "Enterprise Value" else equity
        data.append(row)
    return pd.DataFrame(data)


def to_excel(peer_df: pd.DataFrame, target_df: pd.DataFrame, dcf_df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> bytes:
    """Export key outputs to Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        peer_df.to_excel(writer, index=False, sheet_name="Peer Valuation")
        target_df.to_excel(writer, index=False, sheet_name="Target Valuation")
        dcf_df.to_excel(writer, index=False, sheet_name="DCF Projection")
        sensitivity_df.to_excel(writer, index=False, sheet_name="Sensitivity")
    return output.getvalue()


# =========================================================
# Default Data
# =========================================================
def get_default_peer_data() -> pd.DataFrame:
    return pd.DataFrame({
        "Company": ["IMAX", "D-BOX", "Dolby", "AMC"],
        "Ticker": ["IMAX", "DBO.TO", "DLB", "AMC"],
        "Market Cap": [1500.0, 80.0, 7500.0, 1200.0],
        "Debt": [450.0, 5.0, 0.0, 8500.0],
        "Cash": [120.0, 8.0, 900.0, 700.0],
        "Minority Interest": [0.0, 0.0, 0.0, 0.0],
        "Revenue": [410.0, 35.0, 1300.0, 4600.0],
        "EBITDA": [120.0, 4.0, 420.0, 900.0],
        "Comment": ["PLF / Cinema Tech", "Motion Seat", "Audio / Imaging", "Exhibitor"],
    })


# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("📊 Valuation Tool")
st.sidebar.caption("Peer Multiple + DCF + Sensitivity")

page = st.sidebar.radio(
    "메뉴 선택",
    [
        "1. Peer Valuation",
        "2. Target Valuation",
        "3. DCF Valuation",
        "4. Summary & Export",
    ],
)

currency_unit = st.sidebar.selectbox("단위", ["$M", "₩억", "CAD M"], index=0)
st.sidebar.info("현재 버전은 입력값 기반 계산 Tool입니다. yfinance 자동 수집은 2차 기능으로 분리하는 것을 추천합니다.")


# =========================================================
# Session State
# =========================================================
if "peer_input" not in st.session_state:
    st.session_state.peer_input = get_default_peer_data()

if "peer_result" not in st.session_state:
    st.session_state.peer_result = calc_peer_multiples(st.session_state.peer_input)

if "target_result" not in st.session_state:
    st.session_state.target_result = pd.DataFrame()

if "dcf_projection" not in st.session_state:
    st.session_state.dcf_projection = pd.DataFrame()

if "sensitivity" not in st.session_state:
    st.session_state.sensitivity = pd.DataFrame()


# =========================================================
# Header
# =========================================================
st.title("기업가치평가 자동화 Tool")
st.caption("Peer Group Multiple, Target Valuation, DCF, Sensitivity Table 자동 계산")


# =========================================================
# Page 1: Peer Valuation
# =========================================================
if page == "1. Peer Valuation":
    st.subheader("1. Peer Group Valuation")
    st.write("비교기업 데이터를 입력하면 EV, EV/Sales, EV/EBITDA를 자동 계산합니다.")

    uploaded_file = st.file_uploader("Peer 데이터 CSV 업로드", type=["csv"])

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.session_state.peer_input = uploaded_df

    with st.expander("CSV 양식 보기", expanded=False):
        st.dataframe(get_default_peer_data(), use_container_width=True)

    peer_input = st.data_editor(
        st.session_state.peer_input,
        num_rows="dynamic",
        use_container_width=True,
        key="peer_editor",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        exclude_negative = st.checkbox("음수/무효 Multiple 제외", value=True)
    with col2:
        calc_button = st.button("계산하기", type="primary")

    if calc_button:
        st.session_state.peer_input = peer_input
        st.session_state.peer_result = calc_peer_multiples(peer_input)

    peer_result = calc_peer_multiples(peer_input)
    st.session_state.peer_result = peer_result

    st.markdown("#### 계산 결과")
    display_cols = [
        "Company", "Ticker", "Market Cap", "Debt", "Cash", "Net Debt", "EV",
        "Revenue", "EBITDA", "EV/Sales", "EV/EBITDA", "Comment"
    ]
    display_cols = [col for col in display_cols if col in peer_result.columns]
    st.dataframe(peer_result[display_cols], use_container_width=True)

    sales_stats = get_peer_stats(peer_result, "EV/Sales", exclude_negative)
    ebitda_stats = get_peer_stats(peer_result, "EV/EBITDA", exclude_negative)

    st.markdown("#### Peer Multiple 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EV/Sales Median", format_multiple(sales_stats["Median"]))
    c2.metric("EV/Sales Mean", format_multiple(sales_stats["Mean"]))
    c3.metric("EV/EBITDA Median", format_multiple(ebitda_stats["Median"]))
    c4.metric("EV/EBITDA Mean", format_multiple(ebitda_stats["Mean"]))

    stats_df = pd.DataFrame({
        "EV/Sales": sales_stats,
        "EV/EBITDA": ebitda_stats,
    })
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("#### Multiple 비교 차트")
    chart_df = peer_result[["Company", "EV/Sales", "EV/EBITDA"]].copy()
    chart_df = chart_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Company"])

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.bar_chart(chart_df.set_index("Company")[["EV/Sales"]])
    with chart_col2:
        st.bar_chart(chart_df.set_index("Company")[["EV/EBITDA"]])


# =========================================================
# Page 2: Target Valuation
# =========================================================
elif page == "2. Target Valuation":
    st.subheader("2. Target Company Valuation")
    st.write("대상기업 실적과 적용 Multiple을 입력하면 EV 및 Equity Value Range를 자동 계산합니다.")

    peer_result = st.session_state.peer_result
    if peer_result.empty:
        st.warning("먼저 Peer Valuation을 계산해주세요.")
        st.stop()

    ebitda_stats = get_peer_stats(peer_result, "EV/EBITDA", True)
    sales_stats = get_peer_stats(peer_result, "EV/Sales", True)

    col1, col2 = st.columns(2)
    with col1:
        target_company = st.text_input("대상기업명", value="Target Company")
        valuation_method = st.selectbox("Valuation 기준", ["EV/EBITDA", "EV/Sales"], index=0)
        target_metric = st.number_input(
            "대상기업 EBITDA 또는 Revenue",
            min_value=0.0,
            value=100.0,
            step=10.0,
        )
        target_debt = st.number_input("대상기업 총차입금", value=200.0, step=10.0)
        target_cash = st.number_input("대상기업 현금성자산", value=50.0, step=10.0)

    with col2:
        if valuation_method == "EV/EBITDA":
            default_low = float(ebitda_stats["25th Percentile"]) if not pd.isna(ebitda_stats["25th Percentile"]) else 8.0
            default_base = float(ebitda_stats["Median"]) if not pd.isna(ebitda_stats["Median"]) else 10.0
            default_high = float(ebitda_stats["75th Percentile"]) if not pd.isna(ebitda_stats["75th Percentile"]) else 12.0
        else:
            default_low = float(sales_stats["25th Percentile"]) if not pd.isna(sales_stats["25th Percentile"]) else 1.0
            default_base = float(sales_stats["Median"]) if not pd.isna(sales_stats["Median"]) else 2.0
            default_high = float(sales_stats["75th Percentile"]) if not pd.isna(sales_stats["75th Percentile"]) else 3.0

        low_multiple = st.number_input("Low Multiple", value=round(default_low, 1), step=0.1)
        base_multiple = st.number_input("Base Multiple", value=round(default_base, 1), step=0.1)
        high_multiple = st.number_input("High Multiple", value=round(default_high, 1), step=0.1)
        net_debt = target_debt - target_cash
        st.metric("Net Debt", format_money(net_debt, currency_unit))

    target_result = calc_target_valuation(
        target_metric=target_metric,
        net_debt=net_debt,
        low_multiple=low_multiple,
        base_multiple=base_multiple,
        high_multiple=high_multiple,
    )
    st.session_state.target_result = target_result

    st.markdown("#### Valuation Range")
    st.dataframe(target_result, use_container_width=True)

    base_ev = target_result.loc[target_result["Case"] == "Base", "Enterprise Value"].iloc[0]
    base_equity = target_result.loc[target_result["Case"] == "Base", "Equity Value"].iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Base EV", format_money(base_ev, currency_unit))
    m2.metric("Base Equity Value", format_money(base_equity, currency_unit))
    m3.metric("Applied Multiple", format_multiple(base_multiple))

    st.markdown("#### 보고서용 문장")
    auto_comment = (
        f"{target_company}의 기준 {'EBITDA' if valuation_method == 'EV/EBITDA' else 'Revenue'} "
        f"{format_money(target_metric, currency_unit)}에 Peer Group {valuation_method} Base Multiple "
        f"{format_multiple(base_multiple)}를 적용할 경우, 추정 EV는 {format_money(base_ev, currency_unit)}이며 "
        f"순차입금 {format_money(net_debt, currency_unit)} 차감 후 Equity Value는 "
        f"{format_money(base_equity, currency_unit)} 수준으로 산정됨."
    )
    st.text_area("자동 생성 코멘트", value=auto_comment, height=120)


# =========================================================
# Page 3: DCF Valuation
# =========================================================
elif page == "3. DCF Valuation":
    st.subheader("3. DCF Valuation")
    st.write("5개년 FCF 추정 및 Terminal Value 기반 기업가치를 산정합니다.")

    col1, col2, col3 = st.columns(3)
    with col1:
        base_revenue = st.number_input("기준 Revenue", value=400.0, step=10.0)
        tax_rate = st.number_input("Tax Rate", value=25.0, step=1.0) / 100
        net_debt_dcf = st.number_input("Net Debt", value=150.0, step=10.0)
    with col2:
        da_pct = st.number_input("D&A / Revenue", value=5.0, step=0.5) / 100
        capex_pct = st.number_input("CapEx / Revenue", value=6.0, step=0.5) / 100
        nwc_pct = st.number_input("NWC 증가 / Revenue", value=1.0, step=0.5) / 100
    with col3:
        wacc = st.number_input("WACC", value=9.0, step=0.5) / 100
        terminal_growth = st.number_input("Terminal Growth", value=2.0, step=0.5) / 100

    st.markdown("#### 연도별 가정")
    assumption_df = pd.DataFrame({
        "Year": ["Y1", "Y2", "Y3", "Y4", "Y5"],
        "Revenue Growth": [8.0, 7.0, 6.0, 5.0, 4.0],
        "EBITDA Margin": [25.0, 26.0, 27.0, 28.0, 28.0],
    })

    edited_assumption = st.data_editor(
        assumption_df,
        use_container_width=True,
        hide_index=True,
        key="dcf_assumption_editor",
    )

    growth_rates = [x / 100 for x in edited_assumption["Revenue Growth"].tolist()]
    ebitda_margins = [x / 100 for x in edited_assumption["EBITDA Margin"].tolist()]

    dcf_projection = build_dcf_projection(
        base_revenue=base_revenue,
        growth_rates=growth_rates,
        ebitda_margins=ebitda_margins,
        da_pct=da_pct,
        capex_pct=capex_pct,
        nwc_pct=nwc_pct,
        tax_rate=tax_rate,
    )

    fcf_list = dcf_projection["FCF"].tolist()
    pv_fcf, terminal_value, dcf_ev, dcf_equity = calc_dcf_value(
        fcf_list=fcf_list,
        wacc=wacc,
        terminal_growth=terminal_growth,
        net_debt=net_debt_dcf,
    )

    st.session_state.dcf_projection = dcf_projection

    st.markdown("#### DCF Projection")
    st.dataframe(dcf_projection, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PV of FCF", format_money(pv_fcf, currency_unit))
    m2.metric("Terminal Value", format_money(terminal_value, currency_unit))
    m3.metric("DCF EV", format_money(dcf_ev, currency_unit))
    m4.metric("DCF Equity Value", format_money(dcf_equity, currency_unit))

    st.markdown("#### Sensitivity Table")
    value_type = st.radio("민감도 기준", ["Enterprise Value", "Equity Value"], horizontal=True)

    wacc_range = [wacc - 0.01, wacc - 0.005, wacc, wacc + 0.005, wacc + 0.01]
    tg_range = [terminal_growth - 0.01, terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005, terminal_growth + 0.01]
    tg_range = [max(0.0, x) for x in tg_range]

    sensitivity = build_sensitivity_table(
        fcf_list=fcf_list,
        net_debt=net_debt_dcf,
        wacc_range=wacc_range,
        tg_range=tg_range,
        value_type=value_type,
    )
    st.session_state.sensitivity = sensitivity

    st.dataframe(sensitivity, use_container_width=True)


# =========================================================
# Page 4: Summary & Export
# =========================================================
elif page == "4. Summary & Export":
    st.subheader("4. Summary & Export")
    st.write("Peer Valuation, Target Valuation, DCF 결과를 요약하고 Excel로 다운로드합니다.")

    peer_df = st.session_state.peer_result
    target_df = st.session_state.target_result
    dcf_df = st.session_state.dcf_projection
    sensitivity_df = st.session_state.sensitivity

    if peer_df.empty:
        st.warning("Peer Valuation 결과가 없습니다.")
    else:
        ebitda_stats = get_peer_stats(peer_df, "EV/EBITDA", True)
        sales_stats = get_peer_stats(peer_df, "EV/Sales", True)

        st.markdown("#### Peer Multiple Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EV/Sales Median", format_multiple(sales_stats["Median"]))
        c2.metric("EV/Sales Mean", format_multiple(sales_stats["Mean"]))
        c3.metric("EV/EBITDA Median", format_multiple(ebitda_stats["Median"]))
        c4.metric("EV/EBITDA Mean", format_multiple(ebitda_stats["Mean"]))

    st.markdown("#### Target Valuation Summary")
    if target_df.empty:
        st.info("Target Valuation 페이지에서 먼저 계산해주세요.")
    else:
        st.dataframe(target_df, use_container_width=True)

    st.markdown("#### DCF Summary")
    if dcf_df.empty:
        st.info("DCF Valuation 페이지에서 먼저 계산해주세요.")
    else:
        st.dataframe(dcf_df, use_container_width=True)

    st.markdown("#### Sensitivity Summary")
    if sensitivity_df.empty:
        st.info("DCF Valuation 페이지에서 민감도표를 먼저 생성해주세요.")
    else:
        st.dataframe(sensitivity_df, use_container_width=True)

    st.markdown("#### Excel 다운로드")
    if not peer_df.empty:
        excel_data = to_excel(
            peer_df=peer_df,
            target_df=target_df if not target_df.empty else pd.DataFrame(),
            dcf_df=dcf_df if not dcf_df.empty else pd.DataFrame(),
            sensitivity_df=sensitivity_df if not sensitivity_df.empty else pd.DataFrame(),
        )
        st.download_button(
            label="결과 Excel 다운로드",
            data=excel_data,
            file_name="valuation_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown("#### 보고서용 요약 문구")
    summary_comment = """
- Peer Group 기준 EV/EBITDA 및 EV/Sales Multiple 자동 산출
- 대상기업 주요 재무지표 입력 시 EV 및 Equity Value Range 자동 계산
- 5개년 FCF 추정 및 WACC/Terminal Growth 민감도 분석 기반 DCF Valuation 수행
- 반복 계산 업무 표준화 통해 경쟁사 분석 및 전략 검토 자료 작성 효율화 가능
""".strip()
    st.text_area("요약 문구", value=summary_comment, height=160)

