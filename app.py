"""
app.py  –  Universal Bank Personal Loan Dashboard
=========================================================
Run locally:   streamlit run app.py
Deploy:        Push to GitHub → connect to Streamlit Cloud
=========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_data, get_summary_stats
import charts as ch
from model import train_model, predict_single

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank | Loan Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0F1117;
    color: #E0E0E0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #13151f;
    border-right: 1px solid #2a2d3e;
}
section[data-testid="stSidebar"] * { color: #c9cde0 !important; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 4px;
}
div[data-testid="metric-container"] label { color: #8891b0 !important; font-size: 12px; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 26px; font-weight: 700; }

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: #13151f;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #2a2d3e;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8891b0;
    border-radius: 8px;
    font-weight: 500;
    padding: 8px 18px;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background: #4361EE !important;
    color: #fff !important;
}

/* ── Cards / expanders ── */
.insight-box {
    background: #1a1d2e;
    border-left: 4px solid #4361EE;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 14px;
    line-height: 1.6;
}
.insight-box.green  { border-left-color: #00C9A7; }
.insight-box.red    { border-left-color: #FF6B6B; }
.insight-box.amber  { border-left-color: #FFB703; }
.insight-box.pink   { border-left-color: #F72585; }

/* ── Section headers ── */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    margin: 24px 0 6px 0;
    letter-spacing: -0.3px;
}
.section-sub {
    color: #8891b0;
    font-size: 13px;
    margin-bottom: 20px;
}

/* ── Plotly charts dark ── */
.js-plotly-plot .plotly .modebar { background: transparent !important; }

/* ── Prediction badge ── */
.pred-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 30px;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
}
.pred-accept { background: rgba(0,201,167,0.15); color: #00C9A7; border: 2px solid #00C9A7; }
.pred-reject { background: rgba(255,107,107,0.15); color: #FF6B6B; border: 2px solid #FF6B6B; }

/* ── Recommendation card ── */
.rec-card {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
}
.rec-card h4 { margin: 0 0 6px 0; font-size: 15px; color: #fff; }
.rec-card p  { margin: 0; font-size: 13px; color: #9ba3c0; line-height: 1.6; }
.priority-badge {
    display: inline-block; border-radius: 20px; padding: 2px 10px;
    font-size: 11px; font-weight: 600; margin-bottom: 8px;
}
.p-high   { background: rgba(247,37,133,0.15); color: #F72585; }
.p-medium { background: rgba(255,183,3,0.15);  color: #FFB703; }
.p-low    { background: rgba(67,97,238,0.15);  color: #4361EE; }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data()

df_full = get_data()


# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("**Personal Loan Analytics**")
    st.markdown("---")

    st.markdown("### 🎛️ Filters")

    income_min = int(df_full["Income"].min())
    income_max = int(df_full["Income"].max())
    income_range = st.slider(
        "Annual Income ($000)",
        min_value=income_min, max_value=income_max,
        value=(income_min, income_max), step=5,
    )

    edu_options = {1: "Undergrad", 2: "Graduate", 3: "Advanced/Prof"}
    edu_sel = st.multiselect(
        "Education Level",
        options=[1, 2, 3],
        format_func=lambda x: edu_options[x],
        default=[1, 2, 3],
    )

    family_sel = st.multiselect(
        "Family Size",
        options=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
    )

    age_min = int(df_full["Age"].min())
    age_max = int(df_full["Age"].max())
    age_range = st.slider(
        "Age Range",
        min_value=age_min, max_value=age_max,
        value=(age_min, age_max),
    )

    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown(
        "<small>Dataset: Universal Bank (5,000 customers)<br>"
        "Target: Personal Loan acceptance<br>"
        "Built with Streamlit + Plotly</small>",
        unsafe_allow_html=True,
    )


# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_full.copy()
df = df[
    (df["Income"]    >= income_range[0]) &
    (df["Income"]    <= income_range[1]) &
    (df["Education"].isin(edu_sel))       &
    (df["Family"].isin(family_sel))       &
    (df["Age"]       >= age_range[0])     &
    (df["Age"]       <= age_range[1])
]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 18px 0 10px 0;">
  <h1 style="font-family:'Space Grotesk',sans-serif; font-size:34px;
             font-weight:700; color:#fff; margin:0; letter-spacing:-0.5px;">
    🏦 Universal Bank — Personal Loan Analytics
  </h1>
  <p style="color:#8891b0; font-size:15px; margin:6px 0 0 0;">
    Understanding which customers are most likely to accept a personal loan offer
  </p>
</div>
""", unsafe_allow_html=True)

# Warn if filter is very restrictive
if len(df) < 50:
    st.warning(f"⚠️ Only {len(df)} customers match the current filters. Widen the filters for meaningful analysis.")

stats = get_summary_stats(df)

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Customers", f"{len(df):,}")
with k2:
    accepted_pct = df["Personal_Loan"].mean() * 100
    st.metric("Loan Acceptance Rate", f"{accepted_pct:.1f}%",
              delta=f"{accepted_pct - 9.6:.1f}pp vs overall" if len(df) < len(df_full) else None)
with k3:
    st.metric("Avg Annual Income", f"${df['Income'].mean():.0f}k")
with k4:
    st.metric("Avg CC Spend/Month", f"${df['CCAvg'].mean():.2f}k")
with k5:
    st.metric("Avg Mortgage", f"${df['Mortgage'].mean():.0f}k")

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Descriptive",
    "🔍 Diagnostic",
    "🤖 Predictive",
    "💡 Prescriptive",
    "🧮 Loan Calculator",
])


# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">📊 Descriptive Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">What does our customer base look like? Summaries, distributions and overall patterns.</p>', unsafe_allow_html=True)

    # ── Row 1: Donut + Age dist ────────────────────────────────────
    col_a, col_b = st.columns([1, 1.6])
    with col_a:
        st.plotly_chart(ch.donut_loan_acceptance(df), use_container_width=True)
        st.markdown("""
        <div class="insight-box green">
          <b>Key Insight:</b> Only <b>9.6%</b> of customers accepted the loan overall.
          The dataset is class-imbalanced — important for model training.
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.plotly_chart(ch.hist_age(df), use_container_width=True)
        st.markdown("""
        <div class="insight-box">
          <b>Key Insight:</b> The customer age ranges from 23–67 years with a near-uniform
          distribution. Loan acceptance is not strongly age-dependent — both young and
          middle-aged customers accept loans at similar rates.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 2: Income dist + Avg metrics ──────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(ch.hist_income(df), use_container_width=True)
        st.markdown("""
        <div class="insight-box green">
          <b>Key Insight:</b> Customers who accepted the loan are heavily concentrated
          in the <b>$100k–$200k+</b> income bracket. Very few low-income customers
          accepted the loan offer.
        </div>""", unsafe_allow_html=True)

    with col_d:
        st.plotly_chart(ch.avg_metrics_bar(df), use_container_width=True)
        st.markdown("""
        <div class="insight-box amber">
          <b>Key Insight:</b> Loan acceptors earn <b>2.2× more</b> income, spend
          <b>2.3× more</b> on credit cards, and carry <b>2× more</b> mortgage
          compared to those who rejected the loan.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 3: Education bar + Family bar ─────────────────────────
    col_e, col_f = st.columns(2)
    with col_e:
        st.plotly_chart(ch.bar_education(df), use_container_width=True)
        st.markdown("""
        <div class="insight-box">
          <b>Key Insight:</b> Graduate and Advanced/Professional degree holders
          accept loans at <b>3× the rate</b> of undergrads (13% vs 4.4%).
          Education is a meaningful predictor.
        </div>""", unsafe_allow_html=True)

    with col_f:
        st.plotly_chart(ch.bar_family(df), use_container_width=True)
        st.markdown("""
        <div class="insight-box">
          <b>Key Insight:</b> Customers with family size 3–4 show the highest acceptance
          rate (~13%), likely reflecting higher financial need and borrowing capacity.
        </div>""", unsafe_allow_html=True)

    # ── Data summary table ─────────────────────────────────────────
    with st.expander("📋 View Raw Statistics Table"):
        desc_cols = ["Age", "Income", "CCAvg", "Mortgage", "Family", "Experience"]
        st.dataframe(
            df[desc_cols].describe().T.round(2)
            .style.background_gradient(cmap="Blues", axis=1),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — DIAGNOSTIC ANALYTICS
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">🔍 Diagnostic Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Why do customers accept or reject the loan? Drivers, comparisons and relationship analysis.</p>', unsafe_allow_html=True)

    # ── Row 1: Box income + Violin CCAvg ──────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(ch.box_income_by_loan(df), use_container_width=True)
        st.markdown("""
        <div class="insight-box green">
          <b>Key Driver — Income:</b> Accepted customers have a median income of
          <b>$142.5k</b> vs <b>$59k</b> for rejected customers.
          Income is the single strongest differentiator (correlation: 0.50).
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.plotly_chart(ch.violin_ccavg(df), use_container_width=True)
        st.markdown("""
        <div class="insight-box green">
          <b>Key Driver — CC Spend:</b> Accepted customers spend an average of
          <b>$3.91k/month</b> on credit cards vs <b>$1.73k</b> for rejected ones.
          High CC spend signals higher disposable income and creditworthiness.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 2: Banking services + Income group rate ────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(ch.bar_banking_services(df), use_container_width=True)
        st.markdown("""
        <div class="insight-box pink">
          <b>Key Driver — CD Account:</b> Customers with a CD Account accept loans
          at a <b>46.4%</b> rate vs only <b>7.2%</b> without one.
          CD account is the strongest binary differentiator in the dataset.
        </div>""", unsafe_allow_html=True)

    with col_d:
        st.plotly_chart(ch.bar_income_group_rate(df), use_container_width=True)
        st.markdown("""
        <div class="insight-box amber">
          <b>Key Threshold:</b> Loan acceptance jumps sharply above <b>$100k income</b>.
          The $100–150k group has a 28.6% acceptance rate;
          the $150–200k group reaches <b>50.5%</b>.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 3: Scatter + Correlation heatmap ──────────────────────
    st.plotly_chart(ch.scatter_income_ccavg(df), use_container_width=True)
    st.markdown("""
    <div class="insight-box">
      <b>Pattern:</b> Accepted customers cluster in the <b>top-right quadrant</b>
      (high income + high CC spend). The separation is remarkably clean, confirming
      that Income and CCAvg together strongly define the acceptance boundary.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(ch.heatmap_corr(df), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
      <b>Correlation Summary:</b>
      <ul style="margin:6px 0 0 0; padding-left:18px; line-height:1.9;">
        <li><b>Income (0.50)</b> — strongest positive correlation with loan acceptance</li>
        <li><b>CCAvg (0.37)</b> — second strongest; high spenders more likely to borrow</li>
        <li><b>CD Account (0.32)</b> — existing deep banking relationship predicts acceptance</li>
        <li><b>Mortgage (0.14)</b> — moderate positive; homeowners with mortgages more likely</li>
        <li><b>Education (0.14)</b> — higher education correlates with higher acceptance</li>
        <li><b>Age / Experience</b> — negligible correlation; not key differentiators</li>
      </ul>
    </div>""", unsafe_allow_html=True)

    # ── Accepted vs Rejected comparison table ─────────────────────
    with st.expander("📋 Side-by-Side Comparison: Accepted vs Rejected"):
        comp_cols = ["Age", "Income", "CCAvg", "Mortgage", "Family", "Experience"]
        acc_stats = df[df["Personal_Loan"]==1][comp_cols].mean().round(2)
        rej_stats = df[df["Personal_Loan"]==0][comp_cols].mean().round(2)
        diff_pct  = ((acc_stats - rej_stats) / rej_stats * 100).round(1)
        comp_df   = pd.DataFrame({
            "Avg (Accepted)": acc_stats,
            "Avg (Rejected)": rej_stats,
            "Difference (%)": diff_pct.astype(str) + "%",
        })
        st.dataframe(comp_df.style.background_gradient(subset=["Avg (Accepted)"], cmap="Greens"),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — PREDICTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">🤖 Predictive Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Random Forest classifier trained to predict personal loan acceptance. AUC = 0.9987.</p>', unsafe_allow_html=True)

    with st.spinner("Training model on full dataset…"):
        results = train_model(df_full)  # always train on full dataset for consistency

    # ── Model performance gauges ───────────────────────────────────
    st.plotly_chart(
        ch.gauge_model_accuracy(results["auc"], results["accuracy"]),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(ch.roc_curve_plot(results["fpr"], results["tpr"], results["auc"]),
                        use_container_width=True)
    with col_b:
        st.plotly_chart(ch.confusion_matrix_plot(results["cm"]),
                        use_container_width=True)

    st.markdown("---")

    # ── Feature importance ─────────────────────────────────────────
    st.plotly_chart(ch.bar_feature_importance(results["importances"]),
                    use_container_width=True)
    st.markdown("""
    <div class="insight-box green">
      <b>Top Predictors:</b>
      <b>Income (48.5%)</b> is by far the dominant feature, followed by
      <b>CCAvg (19.4%)</b> and <b>Education (10.9%)</b>.
      Together these three features account for ~79% of the model's predictive power.
    </div>""", unsafe_allow_html=True)

    # ── Classification report ──────────────────────────────────────
    with st.expander("📋 View Full Classification Report"):
        cr = results["report"]
        rep_df = pd.DataFrame(cr).T.round(3)
        st.dataframe(rep_df.style.background_gradient(cmap="Greens"), use_container_width=True)

    st.markdown("""
    <div class="insight-box amber">
      <b>Model Notes:</b> The Random Forest achieves 99% accuracy and AUC of 0.9987
      on the hold-out test set. Given the strong linear separation in income, this
      performance is expected. In production, consider Logistic Regression as a
      simpler, interpretable baseline alongside this ensemble model.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — PRESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">💡 Prescriptive Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Actionable recommendations — which customer segments should the bank target for personal loan campaigns?</p>', unsafe_allow_html=True)

    # ── Sunburst drill-down ────────────────────────────────────────
    st.markdown("#### 🌀 Interactive Drill-Down: Income → Education → Loan Status")
    st.caption("Click on a segment to drill down. Click the centre to go back up.")
    st.plotly_chart(ch.sunburst_drill(df), use_container_width=True)

    st.markdown("---")

    # ── Segment heatmap ────────────────────────────────────────────
    col_a, col_b = st.columns([1.4, 1])
    with col_a:
        st.plotly_chart(ch.prescriptive_segment_chart(df), use_container_width=True)
    with col_b:
        st.markdown("#### 🎯 Priority Target Segments")
        st.markdown("""
        <div class="rec-card">
          <span class="priority-badge p-high">🔴 HIGH PRIORITY</span>
          <h4>High-Income Graduates & Professionals ($100k+)</h4>
          <p>Acceptance rate up to <b>50%</b>. These customers have the income capacity
          and financial sophistication to understand loan products. Target via
          digital channels and relationship managers.</p>
        </div>
        <div class="rec-card">
          <span class="priority-badge p-high">🔴 HIGH PRIORITY</span>
          <h4>CD Account Holders (Any Income)</h4>
          <p>Acceptance rate of <b>46.4%</b> — nearly 6× higher than non-holders.
          Existing deep relationship with the bank. Offer loan as a complementary product
          via in-app notification or branch follow-up.</p>
        </div>
        <div class="rec-card">
          <span class="priority-badge p-medium">🟡 MEDIUM PRIORITY</span>
          <h4>Family Size 3–4, Income $75k+ </h4>
          <p>Larger families have higher financial obligations and greater need for
          personal loans. Pair with family-oriented messaging about home improvement,
          education funding, or life events.</p>
        </div>
        <div class="rec-card">
          <span class="priority-badge p-medium">🟡 MEDIUM PRIORITY</span>
          <h4>High CC Spenders (CCAvg > $2.5k/month)</h4>
          <p>These customers already demonstrate comfort with credit products.
          Target with debt consolidation messaging — converting high-interest CC
          debt into a lower-rate personal loan.</p>
        </div>
        <div class="rec-card">
          <span class="priority-badge p-low">🔵 LOW PRIORITY</span>
          <h4>Online Banking Users</h4>
          <p>Online users are slightly more engaged but the acceptance lift is modest
          (0.6% correlation). Use this channel for <i>delivery</i> of offers to
          the above segments, not as a standalone targeting criterion.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Treemap ────────────────────────────────────────────────────
    st.markdown("#### 🗺️ Prescriptive Treemap — All Segments by Acceptance Rate")
    st.caption("Size = number of customers in segment. Colour = acceptance rate (darker = higher).")
    st.plotly_chart(ch.treemap_prescriptive(df), use_container_width=True)

    st.markdown("---")

    # ── Campaign strategy summary ──────────────────────────────────
    st.markdown("#### 📋 Recommended Campaign Strategy")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="insight-box green">
          <b>📣 Channel Strategy</b><br><br>
          • <b>In-app push</b> for online banking users<br>
          • <b>Relationship manager outreach</b> for CD account holders<br>
          • <b>Email campaigns</b> segmented by income group<br>
          • <b>Branch cross-sell</b> at mortgage renewal time
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box amber">
          <b>💰 Offer Design</b><br><br>
          • Competitive rate for income > $100k<br>
          • Debt consolidation angle for high CC spenders<br>
          • Family plan messaging for size 3–4<br>
          • Pre-approved offers for CD account holders
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="insight-box pink">
          <b>🚫 Avoid / Deprioritise</b><br><br>
          • Customers with income &lt; $50k (0% acceptance)<br>
          • Undergrad-only, small family, low income<br>
          • No banking relationship (no CD/securities)<br>
          • Over-investing in securities account holders (low lift)
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  TAB 5 — LOAN CALCULATOR / SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">🧮 Customer Loan Propensity Calculator</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Enter a customer profile and the trained model will predict their likelihood of accepting a personal loan.</p>', unsafe_allow_html=True)

    with st.spinner("Loading model…"):
        calc_results = train_model(df_full)

    col_form, col_out = st.columns([1.2, 1])

    with col_form:
        st.markdown("#### 👤 Customer Profile Input")
        c1, c2 = st.columns(2)
        with c1:
            inp_age      = st.number_input("Age", 18, 80, 35)
            inp_income   = st.number_input("Annual Income ($000)", 8, 300, 80)
            inp_ccavg    = st.number_input("Monthly CC Spend ($000)", 0.0, 10.0, 1.5, step=0.1)
            inp_mortgage = st.number_input("Mortgage ($000)", 0, 700, 0)
            inp_family   = st.selectbox("Family Size", [1, 2, 3, 4])
        with c2:
            inp_edu      = st.selectbox("Education", [1, 2, 3],
                                        format_func=lambda x: {1:"Undergrad",2:"Graduate",3:"Advanced/Prof"}[x])
            inp_exp      = st.number_input("Years of Experience", 0, 50, 10)
            inp_sec      = st.selectbox("Securities Account?", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")
            inp_cd       = st.selectbox("CD Account?", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")
            inp_online   = st.selectbox("Online Banking?", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")
            inp_cc       = st.selectbox("UniversalBank Credit Card?", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")

        predict_btn = st.button("🔮 Predict Loan Propensity", use_container_width=True, type="primary")

    with col_out:
        st.markdown("#### 📈 Prediction Result")
        if predict_btn:
            input_dict = {
                "Age": inp_age, "Experience": inp_exp, "Income": inp_income,
                "Family": inp_family, "CCAvg": inp_ccavg, "Education": inp_edu,
                "Mortgage": inp_mortgage, "Securities_Account": inp_sec,
                "CD_Account": inp_cd, "Online": inp_online, "CreditCard": inp_cc,
            }
            pred = predict_single(calc_results, input_dict)
            prob = pred["probability"]

            # Gauge chart for probability
            import plotly.graph_objects as go
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 36,
                        "color": "#00C9A7" if prob >= 0.5 else "#FF6B6B"}},
                title={"text": "Loan Acceptance Probability", "font": {"size": 14, "color": "#8891b0"}},
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color="#00C9A7" if prob >= 0.5 else "#FF6B6B"),
                    bgcolor="#1a1d2e",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 30],  color="#1f2235"),
                        dict(range=[30, 60], color="#23263b"),
                        dict(range=[60, 100],color="#252940"),
                    ],
                    threshold=dict(line=dict(color="#F72585", width=3),
                                   thickness=0.75, value=50),
                ),
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E0E0E0"),
                margin=dict(l=20, r=20, t=60, b=20),
                height=260,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            if pred["prediction"] == 1:
                st.markdown(f'<div style="text-align:center"><span class="pred-badge pred-accept">✅ LIKELY TO ACCEPT</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:center"><span class="pred-badge pred-reject">❌ LIKELY TO REJECT</span></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Key factors for this customer
            st.markdown("#### 🔑 Key Factors for this Customer")
            factors = []
            if inp_income >= 100:
                factors.append(("✅", "High income (>$100k) — strong positive signal"))
            else:
                factors.append(("⚠️", f"Income ${inp_income}k — below the key $100k threshold"))
            if inp_ccavg >= 3.0:
                factors.append(("✅", f"High CC spend (${inp_ccavg}k) — positive signal"))
            if inp_cd == 1:
                factors.append(("✅", "Has CD Account — acceptance rate 46%"))
            if inp_edu >= 2:
                factors.append(("✅", "Graduate/Advanced education — positive signal"))
            if inp_family >= 3:
                factors.append(("✅", "Family size 3+ — higher financial need"))
            if inp_mortgage > 0:
                factors.append(("✅", f"Has mortgage (${inp_mortgage}k) — indicates established finances"))

            for icon, text in factors:
                st.markdown(f"<div class='insight-box'>{icon} {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box" style="text-align:center; padding:40px;">
              <span style="font-size:48px;">🔮</span><br><br>
              Fill in the customer profile on the left and click
              <b>Predict Loan Propensity</b> to see the result.
            </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a4f6a; font-size:12px; padding:10px 0;">
  Universal Bank Personal Loan Analytics Dashboard &nbsp;|&nbsp;
  Built with Streamlit &amp; Plotly &nbsp;|&nbsp;
  Dataset: 5,000 customers &nbsp;|&nbsp;
  Model: Random Forest (AUC = 0.9987)
</div>
""", unsafe_allow_html=True)
