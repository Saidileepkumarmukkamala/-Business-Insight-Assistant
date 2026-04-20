"""
Streamlit Cloud entrypoint.

Streamlit Cloud runs `streamlit run …` and probes http://localhost:8501/healthz.
Do not replace this with Uvicorn on 8501 — use this file as the app Main file.

Secrets (Cloud): Settings → Secrets → TOML with:
  OPENAI_API_KEY = "sk-..."
Optional: OPENAI_MODEL = "gpt-4o-mini"
"""
from __future__ import annotations

import os

import streamlit as st

# `main` builds AIService at import time using os.environ — map Cloud secrets first.
try:
    for key in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

import pandas as pd
import plotly.express as px

from main import ai_service, get_metrics_dict

st.set_page_config(page_title="Business Insight Assistant", layout="wide")
st.title("AI-Powered Business Insight Assistant")

metrics = get_metrics_dict()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total revenue", f"{metrics['total_revenue']:.2f}")
c2.metric("Units sold", f"{metrics['total_units_sold']}")
c3.metric("Avg satisfaction", f"{metrics['avg_customer_satisfaction']:.2f}/5")
c4.metric("Top category", metrics["top_category"])

st.subheader("AI recommendation engine (on load)")
if not ai_service.enabled:
    st.warning(
        "OpenAI is not configured. Add `OPENAI_API_KEY` in Streamlit Cloud **Secrets** "
        "(not only a local `.env` file)."
    )
else:
    with st.spinner("Generating proactive insights…"):
        try:
            ins = ai_service.generate_insights(metrics)
            st.markdown(f"**Summary**  \n{ins.get('summary', '')}")
            st.markdown(f"**Trend analysis**  \n{ins.get('trends', '')}")
            st.markdown(f"**Recommendations**  \n{ins.get('recommendations', '')}")
        except Exception as exc:
            st.error(f"Insights failed: {exc}")

left, right = st.columns(2)
with left:
    st.subheader("Monthly revenue")
    monthly_df = pd.DataFrame(metrics["monthly_revenue"])
    fig_m = px.line(monthly_df, x="month", y="sales_amount", markers=True, title="Revenue by month")
    st.plotly_chart(fig_m, width="stretch")
with right:
    st.subheader("Revenue by category")
    cat_df = pd.DataFrame(metrics["category_revenue"])
    fig_c = px.bar(
        cat_df,
        x="product_category",
        y="sales_amount",
        title="Category performance",
        labels={"product_category": "Category", "sales_amount": "Revenue"},
    )
    st.plotly_chart(fig_c, width="stretch")

st.subheader("AI chatbot")
if not ai_service.enabled:
    st.info("Configure `OPENAI_API_KEY` in Cloud secrets to enable the chatbot.")
else:
    presets = [
        "",
        "What are the sales trends this month?",
        "Which product category performed the best?",
        "What recommendations can improve revenue?",
    ]
    preset = st.selectbox("Sample question", presets, format_func=lambda x: x or "— Choose or type below —")
    q = st.text_input("Your question", value=preset or "")
    if st.button("Ask AI", type="primary"):
        q = (q or "").strip()
        if len(q) < 3:
            st.warning("Enter at least 3 characters.")
        else:
            with st.spinner("Generating…"):
                try:
                    ans = ai_service.generate_response(q, metrics)
                    st.success("Response")
                    st.write(ans)
                except Exception as exc:
                    st.error(str(exc))
