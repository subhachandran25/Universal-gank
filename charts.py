"""
charts.py  –  All Plotly chart builders for the Universal Bank Dashboard.
Each function returns a plotly.graph_objects.Figure ready for st.plotly_chart().
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
C_ACCEPT  = "#00C9A7"   # teal-green  → accepted
C_REJECT  = "#FF6B6B"   # coral-red   → rejected
C_PRIMARY = "#4361EE"   # indigo blue → general
C_ACCENT  = "#F72585"   # magenta
C_WARN    = "#FFB703"   # amber
C_BG      = "#0F1117"   # dark canvas
C_CARD    = "#1A1D2E"   # card bg
C_TEXT    = "#E0E0E0"

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C_TEXT, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
)

EDU_ORDER = ["Undergrad", "Graduate", "Advanced/Prof"]


# ═══════════════════════════════════════════════════════════════════════
#  DESCRIPTIVE
# ═══════════════════════════════════════════════════════════════════════

def donut_loan_acceptance(df: pd.DataFrame) -> go.Figure:
    counts = df["Personal_Loan"].value_counts().sort_index()
    labels = ["Rejected", "Accepted"]
    colors = [C_REJECT, C_ACCEPT]
    fig = go.Figure(go.Pie(
        labels=labels, values=counts.values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color=C_BG, width=3)),
        textinfo="label+percent",
        textfont=dict(size=13),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{counts[1]:,}</b><br>Accepted",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=C_ACCEPT),
        align="center",
    )
    fig.update_layout(
        title="Loan Acceptance Overview",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        **LAYOUT_DEFAULTS,
    )
    return fig


def hist_age(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for val, label, color in [(0, "Rejected", C_REJECT), (1, "Accepted", C_ACCEPT)]:
        sub = df[df["Personal_Loan"] == val]["Age"]
        fig.add_trace(go.Histogram(
            x=sub, name=label, nbinsx=20,
            marker_color=color, opacity=0.75,
            hovertemplate=f"Age: %{{x}}<br>Count: %{{y}}<extra>{label}</extra>",
        ))
    fig.update_layout(
        barmode="overlay",
        title="Age Distribution by Loan Status",
        xaxis_title="Age (years)", yaxis_title="Count",
        legend=dict(orientation="h", y=1.1),
        **LAYOUT_DEFAULTS,
    )
    return fig


def hist_income(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for val, label, color in [(0, "Rejected", C_REJECT), (1, "Accepted", C_ACCEPT)]:
        sub = df[df["Personal_Loan"] == val]["Income"]
        fig.add_trace(go.Histogram(
            x=sub, name=label, nbinsx=30,
            marker_color=color, opacity=0.75,
            hovertemplate=f"Income: $%{{x}}k<br>Count: %{{y}}<extra>{label}</extra>",
        ))
    fig.update_layout(
        barmode="overlay",
        title="Income Distribution by Loan Status",
        xaxis_title="Annual Income ($000)", yaxis_title="Count",
        legend=dict(orientation="h", y=1.1),
        **LAYOUT_DEFAULTS,
    )
    return fig


def bar_family(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("Family")["Personal_Loan"].agg(["sum", "count"]).reset_index()
    grp["rate"] = grp["sum"] / grp["count"] * 100
    fig = go.Figure(go.Bar(
        x=grp["Family"].astype(str), y=grp["rate"],
        marker=dict(color=grp["rate"], colorscale="teal", showscale=False),
        text=grp["rate"].round(1).astype(str) + "%",
        textposition="outside",
        hovertemplate="Family size: %{x}<br>Acceptance rate: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Loan Acceptance Rate by Family Size",
        xaxis_title="Family Size", yaxis_title="Acceptance Rate (%)",
        yaxis=dict(range=[0, grp["rate"].max() * 1.25]),
        **LAYOUT_DEFAULTS,
    )
    return fig


def bar_education(df: pd.DataFrame) -> go.Figure:
    grp = (
        df.groupby("Education_Label")["Personal_Loan"]
        .agg(["sum", "count"])
        .reindex(["Undergrad", "Graduate", "Advanced/Prof"])
        .reset_index()
    )
    grp["rate"] = grp["sum"] / grp["count"] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Total Customers",
        x=grp["Education_Label"], y=grp["count"],
        marker_color=C_PRIMARY, opacity=0.45,
        yaxis="y1",
        hovertemplate="%{x}<br>Total: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Acceptance Rate",
        x=grp["Education_Label"], y=grp["rate"],
        mode="lines+markers+text",
        line=dict(color=C_ACCEPT, width=3),
        marker=dict(size=10, color=C_ACCEPT),
        text=grp["rate"].round(1).astype(str) + "%",
        textposition="top center",
        yaxis="y2",
        hovertemplate="%{x}<br>Rate: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Loan Acceptance by Education Level",
        xaxis_title="Education Level",
        yaxis=dict(title="Total Customers", showgrid=False),
        yaxis2=dict(title="Acceptance Rate (%)", overlaying="y", side="right",
                    range=[0, grp["rate"].max() * 1.4]),
        legend=dict(orientation="h", y=1.1),
        **LAYOUT_DEFAULTS,
    )
    return fig


def avg_metrics_bar(df: pd.DataFrame) -> go.Figure:
    metrics = ["Income", "CCAvg", "Mortgage"]
    labels  = ["Avg Income ($k)", "Avg CC Spend ($k)", "Avg Mortgage ($k)"]
    data_acc = [df[df["Personal_Loan"]==1][m].mean() for m in metrics]
    data_rej = [df[df["Personal_Loan"]==0][m].mean() for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accepted", x=labels, y=data_acc,
                         marker_color=C_ACCEPT,
                         text=[f"{v:.1f}" for v in data_acc], textposition="outside"))
    fig.add_trace(go.Bar(name="Rejected", x=labels, y=data_rej,
                         marker_color=C_REJECT,
                         text=[f"{v:.1f}" for v in data_rej], textposition="outside"))
    fig.update_layout(
        barmode="group",
        title="Average Financial Metrics: Accepted vs Rejected",
        yaxis_title="Value ($000)",
        legend=dict(orientation="h", y=1.1),
        **LAYOUT_DEFAULTS,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════

def box_income_by_loan(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for val, label, color in [(0, "Rejected", C_REJECT), (1, "Accepted", C_ACCEPT)]:
        fig.add_trace(go.Box(
            y=df[df["Personal_Loan"]==val]["Income"],
            name=label, marker_color=color,
            boxmean="sd",
            hovertemplate=f"{label}<br>Income: $%{{y}}k<extra></extra>",
        ))
    fig.update_layout(
        title="Income Distribution: Accepted vs Rejected",
        yaxis_title="Annual Income ($000)",
        **LAYOUT_DEFAULTS,
    )
    return fig


def violin_ccavg(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for val, label, color in [(0, "Rejected", C_REJECT), (1, "Accepted", C_ACCEPT)]:
        fig.add_trace(go.Violin(
            y=df[df["Personal_Loan"]==val]["CCAvg"],
            name=label, line_color=color, fillcolor=color,
            opacity=0.6, box_visible=True, meanline_visible=True,
        ))
    fig.update_layout(
        title="Credit Card Spending vs Loan Status",
        yaxis_title="Monthly CC Spend ($000)",
        **LAYOUT_DEFAULTS,
    )
    return fig


def heatmap_corr(df: pd.DataFrame) -> go.Figure:
    cols = ["Age", "Experience", "Income", "Family", "CCAvg",
            "Education", "Mortgage", "Securities_Account",
            "CD_Account", "Online", "CreditCard", "Personal_Loan"]
    corr = df[cols].corr()
    labels = [c.replace("_", " ") for c in cols]

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Correlation Heatmap",
        xaxis=dict(tickangle=-45),
        height=520,
        **LAYOUT_DEFAULTS,
    )
    return fig


def bar_banking_services(df: pd.DataFrame) -> go.Figure:
    services = {
        "Securities Acct": "Securities_Account",
        "CD Account":      "CD_Account",
        "Online Banking":  "Online",
        "Credit Card":     "CreditCard",
    }
    rows = []
    for label, col in services.items():
        for val, status in [(0, "Without"), (1, "With")]:
            rate = df[df[col]==val]["Personal_Loan"].mean() * 100
            rows.append({"Service": label, "Status": status, "Rate": rate})
    grp = pd.DataFrame(rows)

    fig = px.bar(
        grp, x="Service", y="Rate", color="Status",
        barmode="group",
        color_discrete_map={"Without": C_REJECT, "With": C_ACCEPT},
        text=grp["Rate"].round(1).astype(str) + "%",
        labels={"Rate": "Loan Acceptance Rate (%)"},
        title="Loan Acceptance Rate by Banking Service Usage",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def scatter_income_ccavg(df: pd.DataFrame) -> go.Figure:
    # Sample for performance
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Income", y="CCAvg",
        color="Loan_Status",
        color_discrete_map={"Accepted": C_ACCEPT, "Rejected": C_REJECT},
        opacity=0.6,
        hover_data=["Age", "Education_Label", "Family"],
        labels={"Income": "Annual Income ($000)", "CCAvg": "Monthly CC Spend ($000)"},
        title="Income vs Credit Card Spending (coloured by Loan Status)",
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def bar_income_group_rate(df: pd.DataFrame) -> go.Figure:
    grp = (
        df.groupby("Income_Group", observed=True)["Personal_Loan"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grp["rate"] = grp["sum"] / grp["count"] * 100
    fig = go.Figure(go.Bar(
        x=grp["Income_Group"].astype(str),
        y=grp["rate"],
        marker=dict(color=grp["rate"], colorscale="Teal", showscale=True,
                    colorbar=dict(title="Rate %")),
        text=grp["rate"].round(1).astype(str) + "%",
        textposition="outside",
        hovertemplate="Income: %{x}<br>Rate: %{y:.1f}%<br>Count: %{customdata:,}<extra></extra>",
        customdata=grp["count"],
    ))
    fig.update_layout(
        title="Loan Acceptance Rate by Income Group",
        xaxis_title="Income Group", yaxis_title="Acceptance Rate (%)",
        yaxis=dict(range=[0, grp["rate"].max() * 1.3]),
        **LAYOUT_DEFAULTS,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  PREDICTIVE
# ═══════════════════════════════════════════════════════════════════════

def bar_feature_importance(importances: pd.Series) -> go.Figure:
    imp = importances.sort_values()
    colors = [C_ACCENT if v > imp.median() else C_PRIMARY for v in imp.values]
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index.str.replace("_", " "),
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in imp.values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title="Random Forest – Feature Importance",
        xaxis_title="Importance Score", yaxis_title="",
        xaxis=dict(tickformat=".0%"),
        **LAYOUT_DEFAULTS,
    )
    return fig


def gauge_model_accuracy(auc: float, accuracy: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "indicator"}, {"type": "indicator"}]])
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=auc * 100,
        title={"text": "ROC-AUC Score", "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 22, "color": C_ACCEPT}},
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=C_ACCEPT),
            steps=[
                dict(range=[0, 60], color="#2d2d2d"),
                dict(range=[60, 80], color="#3a3a3a"),
                dict(range=[80, 100], color="#444"),
            ],
            threshold=dict(line=dict(color=C_ACCENT, width=4), thickness=0.75, value=90),
        ),
    ), row=1, col=1)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=accuracy * 100,
        title={"text": "Model Accuracy", "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 22, "color": C_PRIMARY}},
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=C_PRIMARY),
            steps=[
                dict(range=[0, 60], color="#2d2d2d"),
                dict(range=[60, 80], color="#3a3a3a"),
                dict(range=[80, 100], color="#444"),
            ],
            threshold=dict(line=dict(color=C_ACCENT, width=4), thickness=0.75, value=90),
        ),
    ), row=1, col=2)
    fig.update_layout(
        title="Predictive Model Performance",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_TEXT),
        margin=dict(l=20, r=20, t=60, b=20),
        height=260,
    )
    return fig


def roc_curve_plot(fpr, tpr, auc) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color=C_ACCEPT, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,201,167,0.15)",
        name=f"ROC (AUC={auc:.4f})",
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#555", dash="dash", width=1.5),
        name="Random Classifier",
    ))
    fig.update_layout(
        title="ROC Curve – Random Forest",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        **LAYOUT_DEFAULTS,
    )
    return fig


def confusion_matrix_plot(cm) -> go.Figure:
    labels = ["Rejected", "Accepted"]
    text   = [[f"{cm[i][j]:,}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"Pred: {l}" for l in labels],
        y=[f"Actual: {l}" for l in labels],
        colorscale=[[0, "#1A1D2E"], [1, C_ACCEPT]],
        text=text, texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(side="top"),
        height=340,
        **LAYOUT_DEFAULTS,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  PRESCRIPTIVE
# ═══════════════════════════════════════════════════════════════════════

def prescriptive_segment_chart(df: pd.DataFrame) -> go.Figure:
    """Heatmap: Income group × Education → acceptance rate."""
    pivot = (
        df.groupby(["Income_Group", "Education_Label"], observed=True)["Personal_Loan"]
        .mean()
        .mul(100)
        .reset_index()
        .pivot(index="Education_Label", columns="Income_Group", values="Personal_Loan")
        .reindex(["Undergrad", "Graduate", "Advanced/Prof"])
    )
    text = pivot.applymap(lambda v: f"{v:.1f}%" if not pd.isna(v) else "")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str).tolist(),
        y=pivot.index.tolist(),
        colorscale="Teal",
        text=text.values,
        texttemplate="%{text}",
        textfont=dict(size=12),
        hovertemplate="Education: %{y}<br>Income: %{x}<br>Rate: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Accept %"),
    ))
    fig.update_layout(
        title="Target Segment Map: Income × Education → Acceptance Rate",
        xaxis_title="Income Group",
        yaxis_title="Education Level",
        **LAYOUT_DEFAULTS,
    )
    return fig


def sunburst_drill(df: pd.DataFrame) -> go.Figure:
    """Interactive sunburst: Income Group → Education → Loan Status."""
    grp = (
        df.groupby(["Income_Group", "Education_Label", "Loan_Status"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    grp["Income_Group"] = grp["Income_Group"].astype(str)
    fig = px.sunburst(
        grp,
        path=["Income_Group", "Education_Label", "Loan_Status"],
        values="Count",
        color="Loan_Status",
        color_discrete_map={"Accepted": C_ACCEPT, "Rejected": C_REJECT, "(?)": "#888"},
        title="Drill-Down: Income Group → Education → Loan Status",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_TEXT),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.update_traces(textinfo="label+percent entry")
    return fig


def treemap_prescriptive(df: pd.DataFrame) -> go.Figure:
    """Treemap of high-propensity segments."""
    grp = (
        df.groupby(["Income_Group", "Education_Label", "Family"], observed=True)
        .agg(total=("Personal_Loan","count"), accepted=("Personal_Loan","sum"))
        .reset_index()
    )
    grp["rate"] = grp["accepted"] / grp["total"] * 100
    grp = grp[grp["total"] >= 20]  # filter small cells
    grp["Income_Group"] = grp["Income_Group"].astype(str)
    grp["Family"]       = grp["Family"].astype(str)

    fig = px.treemap(
        grp,
        path=["Income_Group", "Education_Label", "Family"],
        values="total",
        color="rate",
        color_continuous_scale="Teal",
        title="Prescriptive Treemap: Target Segments by Acceptance Rate",
        hover_data={"rate": ":.1f", "accepted": True, "total": True},
        labels={"rate": "Acceptance Rate (%)"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_TEXT),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig
