import warnings
warnings.filterwarnings("ignore", message="ntree_limit is deprecated.*", category=UserWarning)

import streamlit as st
import math
from model import (
    load_model, predict, generate_random_driver,
    get_insights, get_actionables, format_value,
    FEATURE_LABELS, FREIGHT_TYPES, FREIGHT_LABELS,
    REGIONS, REGION_LABELS, STATS,
)

# ── page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Billor · Default Probability Scoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── global CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] { background: #0f1117 !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="block-container"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stSidebar"] { display: none; }

/* inputs */
input[type="number"], .stTextInput input {
    background: #1a1d26 !important;
    border: 1px solid #2a2e3a !important;
    color: #e8e6e1 !important;
    border-radius: 4px !important;
}
input[type="number"]:focus, .stTextInput input:focus {
    border-color: #c9a572 !important;
    background: #1f2330 !important;
}
div[data-baseweb="select"] > div {
    background: #1a1d26 !important;
    border: 1px solid #2a2e3a !important;
    color: #e8e6e1 !important;
    border-radius: 4px !important;
}
div[data-baseweb="select"] svg { fill: #888 !important; }
div[data-baseweb="popover"] { background: #1a1d26 !important; }
li[role="option"] { color: #e8e6e1 !important; }
li[role="option"]:hover { background: #232733 !important; }

/* labels */
label[data-testid="stWidgetLabel"] p {
    font-size: 10px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6b6f7a !important;
    font-weight: 500 !important;
    font-family: inherit !important;
}

/* button */
.stButton > button {
    background: transparent !important;
    color: #c9a572 !important;
    border: 1px solid #c9a572 !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    border-radius: 4px !important;
    padding: 8px 18px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover { background: rgba(201,165,114,0.1) !important; }

/* hide streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* plotly */
.js-plotly-plot .plotly { background: #151821 !important; }

/* number input arrows */
input::-webkit-outer-spin-button, input::-webkit-inner-spin-button { opacity: 0.3; }

/* column gap */
[data-testid="column"] { padding: 0 6px !important; }
</style>
""", unsafe_allow_html=True)

# ── helpers ────────────────────────────────────────────────────
def panel(content: str, border_color: str = "#232733", left_accent: str = ""):
    left = f"border-left: 3px solid {left_accent};" if left_accent else ""
    return f"""
    <div style="background:#151821;border:1px solid {border_color};{left}
                border-radius:6px;padding:26px;width:100%;margin-bottom:4px">
      {content}
    </div>"""

def panel_title(label: str, color: str = "#c9a572") -> str:
    return f"""<div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
                color:{color};font-weight:500;margin-bottom:18px;padding-bottom:12px;
                border-bottom:1px solid #232733">{label}</div>"""

def mono(text: str, color: str = "#888c96", size: int = 11) -> str:
    return f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:{size}px;color:{color}">{text}</span>'

def serif(text: str, size: str = "inherit", color: str = "#f4f1eb", weight: int = 400) -> str:
    return (f'<span style="font-family:\'Fraunces\',Georgia,serif;font-size:{size};'
            f'color:{color};font-weight:{weight}">{text}</span>')

# ── model load (cached) ────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on first run…")
def get_bundle():
    return load_model()

bundle = get_bundle()

# ── session state ──────────────────────────────────────────────
if "driver" not in st.session_state:
    st.session_state.driver = generate_random_driver()

def do_randomize():
    new_driver = generate_random_driver()
    st.session_state.driver = new_driver
    for k, v in new_driver.items():
        st.session_state[f"field_{k}"] = v

driver = st.session_state.driver

# ── HEADER ─────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:1px solid #232733;padding:22px 32px;
            background:linear-gradient(180deg,#11141c 0%,#0f1117 100%)">
  <div style="max-width:1280px;margin:0 auto;display:flex;
              justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:12px">
    <div>
      <div style="font-size:11px;letter-spacing:0.18em;text-transform:uppercase;
                  color:#c9a572;margin-bottom:6px">Lease-to-Own Risk Engine</div>
      <h1 style="font-family:'Fraunces',Georgia,serif;font-size:clamp(24px,5.2vw,34px);
                 font-weight:500;margin:0;color:#f4f1eb;letter-spacing:-0.02em">
        Default Probability Scoring
      </h1>
    </div>
    <div style="text-align:right;font-size:11px;color:#6b6f7a;letter-spacing:0.05em">
      <div>XGBoost · SHAP · Proof of Concept</div>
      <div style="margin-top:4px">Lucas A. Souza · 2026</div>
    </div>
  </div>
</div>
<div style="height:28px"/>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("<div style='max-width:1280px;margin:0 auto;padding:0 24px'>", unsafe_allow_html=True)

    # ── 01 · DRIVER PROFILE ────────────────────────────────────
    st.markdown(panel_title("01 · Driver Profile"), unsafe_allow_html=True)

    col_title, col_btn = st.columns([6, 1])
    with col_btn:
        st.button("↺ Generate fictional driver", on_click=do_randomize, key="rand_btn")

    INT_COLS = {"age","years_cdl","years_owner_op","fico_score","violations_3y",
                "avg_weekly_miles","loads_per_month","unique_brokers_6m",
                "gross_weekly_revenue","net_weekly_income","avg_account_balance",
                "prior_late_payments","contract_start_month","is_foreign_born"}

    FIELD_ORDER = [
        "age","years_cdl","years_owner_op","fico_score","is_foreign_born",
        "violations_3y","avg_weekly_miles","loads_per_month","freight_type",
        "unique_brokers_6m","on_time_delivery_pct","avg_days_between_loads",
        "gross_weekly_revenue","net_weekly_income","payment_to_income_ratio",
        "avg_account_balance","prior_late_payments","region","contract_start_month",
    ]

    rows = [FIELD_ORDER[i:i+3] for i in range(0, len(FIELD_ORDER), 3)]
    new_driver = dict(driver)

    for row_keys in rows:
        cols = st.columns(3)
        for col, key in zip(cols, row_keys):
            with col:
                label = FEATURE_LABELS[key]
                if key == "freight_type":
                    idx = FREIGHT_TYPES.index(driver[key]) if driver[key] in FREIGHT_TYPES else 0
                    val = st.selectbox(label, options=FREIGHT_TYPES,
                                       format_func=lambda x: FREIGHT_LABELS[x],
                                       index=idx, key=f"field_{key}")
                    new_driver[key] = val
                elif key == "region":
                    idx = REGIONS.index(driver[key]) if driver[key] in REGIONS else 0
                    val = st.selectbox(label, options=REGIONS,
                                       format_func=lambda x: REGION_LABELS[x],
                                       index=idx, key=f"field_{key}")
                    new_driver[key] = val
                elif key == "is_foreign_born":
                    opts = [0, 1]
                    idx = 1 if driver[key] == 1 else 0
                    val = st.selectbox(label, options=opts,
                                       format_func=lambda x: "Yes" if x == 1 else "No",
                                       index=idx, key=f"field_{key}")
                    new_driver[key] = val
                else:
                    s = STATS.get(key, {})
                    is_int = key in INT_COLS
                    if is_int:
                        val = st.number_input(
                            label,
                            min_value=int(s["min"]) if s else None,
                            max_value=int(s["max"]) if s else None,
                            value=int(round(driver[key])),
                            step=1,
                            key=f"field_{key}",
                        )
                        new_driver[key] = int(val)
                    else:
                        fmt = "%.3f" if key == "payment_to_income_ratio" else "%.2f"
                        val = st.number_input(
                            label,
                            min_value=float(s["min"]) if s else None,
                            max_value=float(s["max"]) if s else None,
                            value=float(driver[key]),
                            step=0.01,
                            format=fmt,
                            key=f"field_{key}",
                        )
                        new_driver[key] = round(float(val), 3)

    st.session_state.driver = new_driver
    driver = new_driver

    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

    # ── compute ─────────────────────────────────────────────────
    result    = predict(driver, bundle)
    prob      = result["probability"]
    prob_pct  = prob * 100
    base_val  = result["base_value"]
    contribs  = result["contributions"]
    total_shap = result["total_shap"]
    final_logit = base_val + total_shap
    fleet_avg = 12.0
    ratio     = prob_pct / fleet_avg

    risk_color = "#84a98c" if prob < 0.42 else "#d4a574" if prob < 0.64 else "#c9483b"
    if prob < 0.42:
        decision = {"label": "APPROVE", "color": "#84a98c", "desc": "Auto-approval recommended"}
    elif prob < 0.64:
        decision = {"label": "REVIEW",  "color": "#d4a574", "desc": "Manual analyst review required"}
    else:
        decision = {"label": "DECLINE", "color": "#c9483b", "desc": "Auto-decline recommended"}

    insights   = get_insights(driver, contribs)
    actionables = get_actionables(driver, contribs, bundle)

    sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:8]

    # ── 02 · RISK ASSESSMENT ───────────────────────────────────
    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)

    verdict_html = f"""
    <div style="display:flex;align-items:center;gap:16px;padding:14px 18px;
                background:#11141c;border:1px solid {decision['color']}40;
                border-left:3px solid {decision['color']};border-radius:4px;
                margin-bottom:24px;flex-wrap:wrap">
      <div style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
                  color:#6b6f7a;font-weight:500">Recommendation</div>
      <div style="font-family:'Fraunces',Georgia,serif;font-size:22px;
                  letter-spacing:0.06em;color:{decision['color']};font-weight:500">
        {decision['label']}
      </div>
      <div style="font-size:12px;color:#a8acb6">{decision['desc']}</div>
    </div>"""

    sign = "+" if total_shap >= 0 else ""
    big_number_html = f"""
    <div>
      <div style="font-size:11px;color:#6b6f7a;letter-spacing:0.08em;
                  text-transform:uppercase;margin-bottom:8px">
        Default probability (12 months)
      </div>
      <div style="font-family:'Fraunces',Georgia,serif;font-size:clamp(52px,10vw,80px);
                  font-weight:300;line-height:1;color:{risk_color}">
        {prob_pct:.1f}<span style="font-size:0.43em;color:#6b6f7a;margin-left:4px">%</span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                  color:#888c96;margin-top:8px">
        f(x) = {final_logit:.3f} &nbsp;·&nbsp; σ(f(x)) = {prob:.4f}
      </div>
    </div>"""

    stats_html = f"""
    <div style="display:flex;flex-direction:column;gap:18px">
      <div>
        <div style="font-size:10px;color:#6b6f7a;letter-spacing:0.08em;
                    text-transform:uppercase;margin-bottom:6px">Fleet average</div>
        <div style="font-family:'Fraunces',Georgia,serif;font-size:26px;color:#e8e6e1">
          {fleet_avg:.1f}<span style="font-size:15px;color:#6b6f7a">%</span>
        </div>
      </div>
      <div>
        <div style="font-size:10px;color:#6b6f7a;letter-spacing:0.08em;
                    text-transform:uppercase;margin-bottom:6px">Individual / fleet ratio</div>
        <div style="font-family:'Fraunces',Georgia,serif;font-size:26px;color:{risk_color}">
          {ratio:.2f}×
        </div>
      </div>
    </div>"""

    bar_pct = min(prob_pct, 100)
    bar_html = f"""
    <div style="margin-top:28px">
      <div style="position:relative;height:8px;background:#1f2330;border-radius:4px;overflow:hidden">
        <div style="position:absolute;left:0;top:0;height:100%;width:{bar_pct:.1f}%;
                    background:linear-gradient(90deg,{risk_color}aa,{risk_color});
                    border-radius:4px"></div>
        <div style="position:absolute;left:{fleet_avg}%;top:-4px;height:16px;
                    width:1px;background:#6b6f7a"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:8px;
                  font-size:10px;color:#6b6f7a">
        <span style="font-family:'JetBrains Mono',monospace">0%</span>
        <span style="font-family:'JetBrains Mono',monospace;color:#888c96">
          ↑ fleet avg ({fleet_avg}%)
        </span>
        <span style="font-family:'JetBrains Mono',monospace">100%</span>
      </div>
    </div>"""

    how_score_html = f"""
    <div style="margin-top:26px;padding-top:18px;border-top:1px solid #232733">
      <div style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;
                  color:#6b6f7a;margin-bottom:12px">How the score is built</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div style="padding:12px 14px;background:#11141c;border:1px solid #1f2330;border-radius:4px">
          <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:6px;flex-wrap:wrap">
            <span style="font-family:'JetBrains Mono',monospace;font-size:13px;
                         color:#c9a572;font-weight:500">f(x)</span>
            <span style="font-size:11px;color:#6b6f7a;letter-spacing:0.04em">raw model output · log-odds</span>
          </div>
          <div style="font-size:11px;color:#a8acb6;line-height:1.6">
            Sum of all feature contributions starting from the fleet baseline.
            Lives on an unbounded scale where contributions are <em>additive</em> —
            this is why the waterfall bars can be summed honestly.
          </div>
        </div>
        <div style="padding:12px 14px;background:#11141c;border:1px solid #1f2330;border-radius:4px">
          <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:6px;flex-wrap:wrap">
            <span style="font-family:'JetBrains Mono',monospace;font-size:13px;
                         color:#c9a572;font-weight:500">σ(f(x))</span>
            <span style="font-size:11px;color:#6b6f7a;letter-spacing:0.04em">sigmoid · final probability</span>
          </div>
          <div style="font-size:11px;color:#a8acb6;line-height:1.6">
            Squashes <span style="font-family:'JetBrains Mono',monospace">f(x)</span> into the
            <span style="font-family:'JetBrains Mono',monospace">[0, 1]</span> range
            via σ(z) = 1 / (1 + e<sup>−z</sup>).
            The displayed percentage is σ(f(x)) × 100.
          </div>
        </div>
      </div>
      <div style="margin-top:12px;font-size:10px;color:#6b6f7a;font-style:italic;line-height:1.6">
        The model computes in log-odds because contributions add up cleanly there;
        the sigmoid step only converts to a probability for display.
      </div>
    </div>"""

    thresholds_html = """
    <div style="margin-top:24px;padding-top:18px;border-top:1px solid #232733">
      <div style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;
                  color:#6b6f7a;margin-bottom:12px">Decision thresholds · cost-calibrated</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:14px">
        <div style="padding:10px 14px;background:#11141c;border:1px solid #1f2330;
                    border-left:2px solid #84a98c;border-radius:4px;
                    display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px">
          <div>
            <span style="font-family:'Fraunces',Georgia,serif;font-size:14px;color:#84a98c;
                         font-weight:500;letter-spacing:0.05em">APPROVE</span>
            <span style="font-size:11px;color:#6b6f7a;margin-left:8px">auto-approval</span>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#a8acb6">p &lt; 42%</div>
        </div>
        <div style="padding:10px 14px;background:#11141c;border:1px solid #1f2330;
                    border-left:2px solid #d4a574;border-radius:4px;
                    display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px">
          <div>
            <span style="font-family:'Fraunces',Georgia,serif;font-size:14px;color:#d4a574;
                         font-weight:500;letter-spacing:0.05em">REVIEW</span>
            <span style="font-size:11px;color:#6b6f7a;margin-left:8px">manual analyst</span>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#a8acb6">42% ≤ p &lt; 64%</div>
        </div>
        <div style="padding:10px 14px;background:#11141c;border:1px solid #1f2330;
                    border-left:2px solid #c9483b;border-radius:4px;
                    display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px">
          <div>
            <span style="font-family:'Fraunces',Georgia,serif;font-size:14px;color:#c9483b;
                         font-weight:500;letter-spacing:0.05em">DECLINE</span>
            <span style="font-size:11px;color:#6b6f7a;margin-left:8px">auto-decline</span>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#a8acb6">p ≥ 64%</div>
        </div>
      </div>
      <div style="font-size:11px;color:#a8acb6;line-height:1.7;padding:12px 14px;
                  background:#11141c;border:1px solid #232733;border-radius:4px">
        <strong style="color:#c9a572">How thresholds were chosen.</strong>
        Approving a defaulter costs the company truck repossession, accelerated depreciation,
        and lost margin — estimated at
        <span style="font-family:'JetBrains Mono',monospace">~US$ 45k</span> per case.
        Declining a good payer costs only the foregone margin — estimated at
        <span style="font-family:'JetBrains Mono',monospace">~US$ 20k</span> per case.
        The resulting <span style="font-family:'JetBrains Mono',monospace">FN : FP ≈ 2.25</span>
        ratio favors approval while still protecting against the worst losses. The thresholds
        minimize total expected loss on the test population, reducing cost by
        <span style="font-family:'JetBrains Mono',monospace">~24%</span> versus a naive
        <span style="font-family:'JetBrains Mono',monospace">p &gt; 0.5</span> cutoff.
        In production, these would be re-calibrated against actual loss data quarterly.
      </div>
    </div>"""

    risk_content = (
        panel_title("02 · Risk Assessment")
        + verdict_html
        + f"""<div style="display:grid;grid-template-columns:1.3fr 1fr;gap:56px;align-items:center">
               {big_number_html}{stats_html}</div>"""
        + bar_html
        + how_score_html
        + thresholds_html
    )
    st.markdown(panel(risk_content, left_accent=risk_color), unsafe_allow_html=True)
    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)

    # ── 03 · SHAP WATERFALL (HTML, idêntico ao JSX) ───────────────
    def _wf_base_row(label, value, sublabel, is_final=False):
        sign   = "+" if value > 0 else ""
        color  = "#c9a572" if is_final else "#6b6f7a"
        bg     = "#1f2330" if is_final else "#171a23"
        mono   = "font-family:'JetBrains Mono',monospace;"
        return (
            f'<div style="display:flex;flex-direction:column;gap:4px;padding:12px 14px;'
            f'margin:6px 0;background:{bg};border-radius:4px;'
            f'border-left:2px solid {color};width:100%">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'gap:8px;flex-wrap:wrap">'
            f'<div style="font-size:13px;color:#e8e6e1;font-weight:500">{label}</div>'
            f'<div style="{mono}font-size:13px;color:{color};white-space:nowrap">'
            f'{sign}{value:.3f}</div></div>'
            f'<div style="{mono}font-size:10px;color:#6b6f7a">{sublabel}</div>'
            f'</div>'
        )

    def _wf_contrib_row(label, value, sublabel, max_abs):
        is_pos   = value > 0
        fill_pct = min(50, (abs(value) / max_abs) * 50)
        color    = "#c9483b" if is_pos else "#84a98c"
        sign     = "+" if is_pos else ""
        mono     = "font-family:'JetBrains Mono',monospace;"
        if is_pos:
            bar = (f'position:absolute;left:50%;width:{fill_pct:.2f}%;top:2px;bottom:2px;'
                   f'background:linear-gradient(90deg,{color}88,{color});border-radius:2px')
        else:
            bar = (f'position:absolute;left:{50-fill_pct:.2f}%;width:{fill_pct:.2f}%;top:2px;bottom:2px;'
                   f'background:linear-gradient(90deg,{color},{color}88);border-radius:2px')
        return (
            f'<div style="display:flex;flex-direction:column;gap:8px;padding:12px 4px;'
            f'border-bottom:1px solid #1f2330;width:100%">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'gap:12px;flex-wrap:wrap">'
            f'<div style="min-width:0;flex:1 1 auto">'
            f'<div style="font-size:13px;color:#e8e6e1;line-height:1.3">{label}</div>'
            f'<div style="{mono}font-size:10px;color:#6b6f7a;margin-top:2px;word-break:break-word">'
            f'{sublabel}</div></div>'
            f'<div style="{mono}font-size:13px;color:{color};font-weight:500;white-space:nowrap">'
            f'{sign}{value:.3f}</div></div>'
            f'<div style="position:relative;height:14px;width:100%;background:#13161e;'
            f'border-radius:2px;overflow:hidden">'
            f'<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#2a2e3a"></div>'
            f'<div style="{bar}"></div>'
            f'</div></div>'
        )

    max_abs     = max((abs(v) for _, v in sorted_contribs), default=0.5)
    max_abs     = max(max_abs, 0.5)
    final_logit = base_val + sum(v for _, v in sorted_contribs)
    sigma_base  = 1 / (1 + math.exp(-base_val))
    sigma_final = 1 / (1 + math.exp(-final_logit))

    wf_rows = _wf_base_row(
        "E[f(x)] · base", base_val,
        f"model intercept · σ = {sigma_base:.3f}"
    )
    for key, value in sorted_contribs:
        wf_rows += _wf_contrib_row(
            FEATURE_LABELS[key], value,
            format_value(key, driver[key]), max_abs
        )
    wf_rows += _wf_base_row(
        "f(x) · final score", final_logit,
        f"σ(f(x)) = {sigma_final:.3f}", is_final=True
    )

    waterfall_desc = (
        f"Each feature contributes by shifting the prediction away from "
        f"<span style=\"font-family:'JetBrains Mono',monospace\">E[f(x)] = {base_val:.3f}</span> "
        f"(model intercept in log-odds space; σ({base_val:.3f}) = {sigma_base:.1%}). "
        f"Bars to the right increase risk; bars to the left decrease it."
    )
    wf_content = (
        panel_title("03 · SHAP Decomposition · Waterfall")
        + f'<div style="font-size:12px;color:#888c96;margin-bottom:18px;line-height:1.6">{waterfall_desc}</div>'
        + f'<div style="display:flex;flex-direction:column;width:100%;max-width:100%">{wf_rows}</div>'
    )
    st.markdown(panel(wf_content), unsafe_allow_html=True)
    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)

    # ── 04 · INSIGHTS ──────────────────────────────────────────
    cards = ""
    for i, ins in enumerate(insights):
        bc = "#c9483b" if ins["shap"] > 0 else "#84a98c"
        cards += f"""
        <div style="display:grid;grid-template-columns:auto 1fr auto;gap:14px;
                    align-items:center;padding:14px;background:#1a1d26;border-radius:4px;
                    border-left:2px solid {bc};margin-bottom:10px">
          <div style="font-family:'Fraunces',Georgia,serif;font-size:26px;
                      color:#6b6f7a;font-weight:300">0{i+1}</div>
          <div>
            <div style="font-size:13px;color:#f4f1eb;margin-bottom:4px;line-height:1.4">
              {ins['sentence']}
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#6b6f7a">
              {ins['label']}: {ins['driver_value']}
            </div>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                      color:{bc};font-weight:500;text-align:right;white-space:nowrap">
            {ins['shap_display']}
          </div>
        </div>"""

    insights_content = panel_title("04 · What's Driving This Score") + cards
    st.markdown(panel(insights_content), unsafe_allow_html=True)
    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)

    # ── 05 · SENSITIVITY / ACTION RULES ────────────────────────
    if not actionables:
        no_action_content = (
            panel_title("05 · Sensitivity Analysis · What Could Change the Score", color="#84a98c")
            + """<div style="font-size:12px;color:#888c96;margin-bottom:14px;line-height:1.6">
            A risk score alone tells you <em>who</em> is likely to default — but not <em>what to do about it</em>.
            This section identifies which operational behaviors this driver could realistically change,
            and estimates how much each improvement would reduce their predicted default probability.
            The output is a set of <strong style="color:#e8e6e1">Action Rules</strong> — concrete,
            model-backed recommendations for Billor's risk team.</div>"""
            + """<div style="padding:16px 18px;background:#11141c;border:1px solid #232733;
                            border-left:2px solid #6b6f7a;border-radius:4px;
                            font-size:12px;color:#a8acb6;line-height:1.7">
            <strong style="color:#888c96">No actionable improvement found.</strong>
            For this driver, the four operational features considered —
            payment-to-income ratio, on-time delivery rate, broker diversification,
            and days between loads — are already performing at or above the recommended
            thresholds, or improving them would not meaningfully reduce the predicted
            default probability (&lt; 0.5 pp change).
            The main risk drivers for this profile are structural factors such as
            credit score and financial reserves, which cannot be changed through
            day-to-day operational decisions.
            </div>"""
        )
        st.markdown(panel(no_action_content), unsafe_allow_html=True)
        st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)

    if actionables:
        # ── combined scenario: all targets applied simultaneously ──
        driver_all = dict(driver)
        for a in actionables:
            driver_all[a["key"]] = a["target"]
        combined_prob  = predict(driver_all, bundle)["probability"]
        combined_delta = (combined_prob - prob) * 100
        sum_individual = sum(float(a["delta"]) for a in actionables)
        interaction_note = (
            "smaller than" if combined_delta > sum_individual
            else "larger than" if combined_delta < sum_individual
            else "equal to"
        )

        if len(actionables) >= 2:
            combined_block = (
                '<div><strong style="color:#c9a572">Combined scenario</strong> — the last row simulates'
                ' all recommended changes applied simultaneously. Individual estimates assume only one'
                ' variable changes at a time; the combined simulation removes that constraint.'
                ' Because XGBoost captures non-linear relationships and feature interactions, the joint'
                ' effect (' + f'{combined_delta:.1f}' + ' pp) may differ from the simple sum of individual'
                ' estimates (' + f'{sum_individual:.1f}' + ' pp) — and is the more realistic figure.'
                '</div>'
            )
        else:
            combined_block = ""

        scope_note = (
            '<div style="font-size:11px;color:#a8acb6;margin-bottom:18px;padding:12px 14px;'
            'background:#11141c;border:1px solid #232733;border-radius:4px;line-height:1.7">'
            '<div style="margin-bottom:10px">'
            '<strong style="color:#c9a572">What is analysed</strong> — only behaviors the driver'
            ' controls directly: payment-to-income ratio, on-time delivery rate, broker diversification,'
            ' and days between loads. Each row below tests one change at a time, keeping all other'
            ' variables fixed. Structural attributes — credit score, age, origin, prior payment'
            ' history — are excluded because they cannot be changed through operational decisions.'
            '</div>'
            '<div style="margin-bottom:10px">'
            '<strong style="color:#c9a572">Δ pp (individual)</strong> — estimated drop in predicted'
            ' default probability, in percentage points, if that specific behavior reaches the target'
            ' while everything else stays the same. Example: Δ −5 pp means the model predicts a'
            ' 5-point reduction — e.g. from 38% to 33%.'
            '</div>'
            + combined_block
            + '</div>'
        )

        action_cards = ""
        for a in actionables:
            action_cards += f"""
            <div style="display:grid;grid-template-columns:1fr auto;gap:14px;align-items:center;
                        padding:14px;background:#1a1d26;border-radius:4px;margin-bottom:8px">
              <div>
                <div style="font-size:13px;color:#e8e6e1;line-height:1.5">
                  If {a['direction']} <strong>{a['label']}</strong> from
                  <span style="font-family:'JetBrains Mono',monospace;color:#c9a572">{a['current_fmt']}</span> to
                  <span style="font-family:'JetBrains Mono',monospace;color:#84a98c">{a['target']}</span>
                </div>
              </div>
              <div style="font-family:'JetBrains Mono',monospace;color:#84a98c;
                          font-size:13px;font-weight:500;white-space:nowrap">
                Δ {a['delta']} pp
              </div>
            </div>"""

        combined_card = ""
        if len(actionables) >= 2:
            combined_card = f"""
        <div style="display:grid;grid-template-columns:1fr auto;gap:14px;align-items:center;
                    padding:14px;background:#1a1d26;border-radius:4px;margin-top:4px;
                    border:1px solid #84a98c40;border-left:2px solid #84a98c">
          <div>
            <div style="font-size:13px;color:#e8e6e1;font-weight:500;line-height:1.5">
              All changes above applied simultaneously
            </div>
            <div style="font-size:11px;color:#6b6f7a;margin-top:3px">
              Combined scenario · predicted probability: {combined_prob*100:.1f}%
            </div>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;color:#84a98c;
                      font-size:15px;font-weight:500;white-space:nowrap">
            Δ {combined_delta:.1f} pp
          </div>
        </div>"""

        preamble = (
            '<div style="font-size:12px;color:#888c96;margin-bottom:14px;line-height:1.6">'
            'A risk score alone tells you <em>who</em> is likely to default — but not <em>what to do about it</em>. '
            'This section goes one step further: for each behavior the driver can realistically change, '
            'it estimates how much the predicted default probability would drop if that behavior reached '
            'the recommended level. The result is a set of <strong style="color:#e8e6e1">Action Rules</strong> — '
            'concrete, model-backed recommendations that Billor\'s risk team can use to guide conversations '
            'with drivers, set contract conditions, or define monitoring checkpoints during the lease.'
            '</div>'
        )
        action_content = (
            panel_title("05 · Sensitivity Analysis · What Could Change the Score", color="#84a98c")
            + preamble
            + scope_note
            + action_cards
            + combined_card
        )
        st.markdown(panel(action_content, left_accent="#84a98c"), unsafe_allow_html=True)
        st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)

    # ── TECH NOTE ───────────────────────────────────────────────
    st.markdown("""
    <div style="padding:16px 18px;background:#11141c;border:1px dashed #2a2e3a;
                border-radius:4px;font-size:11px;color:#6b6f7a;line-height:1.7;margin-bottom:40px">
      <strong style="color:#888c96">Technical note.</strong>
      The underlying model is an XGBoost classifier (AUC ≈ 0.92) trained on 5,000 synthetic drivers;
      SHAP values are computed via TreeExplainer on each prediction. Decision thresholds are
      cost-calibrated using a FN:FP cost ratio of 2.25 (repossession vs. foregone margin).
      In real deployment, the serialized model would serve predictions via REST API and SHAP
      would be computed server-side. The conceptual architecture is identical.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
