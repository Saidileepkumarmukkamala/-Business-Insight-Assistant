"""
Single-file app: FastAPI backend + embedded web dashboard (Plotly via CDN).
Run: python main.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


@dataclass
class BusinessMetrics:
    total_revenue: float
    total_units_sold: int
    avg_customer_satisfaction: float
    top_category: str
    top_category_revenue: float
    monthly_revenue: list[dict[str, Any]]
    category_revenue: list[dict[str, Any]]


class DataService:
    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def compute_metrics(self, df: pd.DataFrame) -> BusinessMetrics:
        total_revenue = float(df["sales_amount"].sum())
        total_units_sold = int(df["units_sold"].sum())
        avg_customer_satisfaction = float(df["customer_satisfaction"].mean())

        category_totals = (
            df.groupby("product_category", as_index=False)["sales_amount"]
            .sum()
            .sort_values("sales_amount", ascending=False)
        )
        top_row = category_totals.iloc[0]
        top_category = str(top_row["product_category"])
        top_category_revenue = float(top_row["sales_amount"])

        monthly = df.copy()
        monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
        monthly_revenue = (
            monthly.groupby("month", as_index=False)["sales_amount"]
            .sum()
            .sort_values("month")
            .to_dict(orient="records")
        )

        category_revenue = category_totals.to_dict(orient="records")

        return BusinessMetrics(
            total_revenue=total_revenue,
            total_units_sold=total_units_sold,
            avg_customer_satisfaction=avg_customer_satisfaction,
            top_category=top_category,
            top_category_revenue=top_category_revenue,
            monthly_revenue=monthly_revenue,
            category_revenue=category_revenue,
        )


class AIService:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def generate_response(self, question: str, metrics: dict[str, Any]) -> str:
        if not self.client:
            raise RuntimeError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file."
            )
        prompt = (
            "You are a business intelligence assistant. Use ONLY the provided metrics to answer.\n"
            "If data is insufficient, say what is missing.\n\n"
            f"Metrics:\n{metrics}\n\n"
            f"User Question:\n{question}\n\n"
            "Respond with: (1) direct answer, (2) short insight, (3) recommended action."
        )
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        return response.output_text.strip()


data_path = Path(__file__).resolve().parent / "data" / "sample_business_data.csv"
data_service = DataService(data_path=data_path)
ai_service = AIService()

app = FastAPI(title="AI-Powered Business Insight Assistant")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    df = data_service.load_data()
    stats = data_service.compute_metrics(df)
    return stats.__dict__


@app.post("/ask")
def ask(request: QueryRequest) -> dict[str, str]:
    try:
        cleaned_question = request.question.strip()
        if len(cleaned_question) < 3:
            raise HTTPException(
                status_code=400,
                detail="Please enter a question with at least 3 characters.",
            )
        if not ai_service.client:
            raise HTTPException(
                status_code=503,
                detail="OpenAI API key is missing. Set OPENAI_API_KEY in .env and restart the server.",
            )

        df = data_service.load_data()
        stats = data_service.compute_metrics(df)
        answer = ai_service.generate_response(cleaned_question, stats.__dict__)
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Business Insight Assistant</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; padding: 1.25rem; background: #0f1419; color: #e6edf3; }
    h1 { margin: 0 0 0.25rem; font-size: 1.5rem; }
    .sub { color: #8b949e; margin-bottom: 1.25rem; }
    .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }
    .kpi { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 0.75rem 1rem; }
    .kpi .label { font-size: 0.75rem; color: #8b949e; }
    .kpi .value { font-size: 1.15rem; font-weight: 600; margin-top: 0.25rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; margin-bottom: 1.25rem; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; }
    .card h2 { margin: 0 0 0.75rem; font-size: 1rem; }
    #monthly, #category { min-height: 320px; }
    label { display: block; font-size: 0.85rem; color: #8b949e; margin-bottom: 0.35rem; }
    select, textarea, button { width: 100%; box-sizing: border-box; border-radius: 6px; border: 1px solid #30363d; background: #0d1117; color: #e6edf3; padding: 0.5rem 0.65rem; font-size: 0.95rem; }
    button { margin-top: 0.5rem; cursor: pointer; background: #238636; border-color: #238636; font-weight: 600; }
    button:hover { filter: brightness(1.08); }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .info { background: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.9rem; color: #8b949e; margin-bottom: 0.75rem; }
    #answer { white-space: pre-wrap; margin-top: 0.75rem; padding: 0.75rem; background: #0d1117; border-radius: 6px; border: 1px solid #30363d; min-height: 3rem; }
    .err { color: #f85149; }
    .risks { font-size: 0.9rem; color: #8b949e; line-height: 1.5; }
  </style>
</head>
<body>
  <h1>AI-Powered Business Insight Assistant</h1>
  <p class="sub">Dashboard and chatbot (same server). OpenAI API key required for chat.</p>

  <div class="kpis">
    <div class="kpi"><div class="label">Total revenue</div><div class="value" id="k1">—</div></div>
    <div class="kpi"><div class="label">Units sold</div><div class="value" id="k2">—</div></div>
    <div class="kpi"><div class="label">Avg satisfaction</div><div class="value" id="k3">—</div></div>
    <div class="kpi"><div class="label">Top category</div><div class="value" id="k4">—</div></div>
  </div>

  <div class="grid">
    <div class="card"><h2>Monthly revenue</h2><div id="monthly"></div></div>
    <div class="card"><h2>Revenue by category</h2><div id="category"></div></div>
  </div>

  <div class="card">
    <h2>AI chatbot</h2>
    <div class="info">OpenAI is required for answers. Without <code>OPENAI_API_KEY</code>, <code>POST /ask</code> returns 503.</div>
    <label for="preset">Sample question</label>
    <select id="preset">
      <option value="">— Choose or type below —</option>
      <option>What are the sales trends this month?</option>
      <option>Which product category performed the best?</option>
      <option>What recommendations can improve revenue?</option>
    </select>
    <label for="q" style="margin-top:0.75rem">Your question</label>
    <textarea id="q" rows="3" placeholder="Ask about the dataset…"></textarea>
    <button type="button" id="btn">Ask AI</button>
    <div id="answer"></div>
  </div>

  <div class="card" style="margin-top:1rem">
    <h2>Ethical and risk considerations</h2>
    <p class="risks">Bias from incomplete data; protect business data before external APIs; models can hallucinate; keep API keys and endpoints secured.</p>
  </div>

  <script>
    const plotLayout = { paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117', font: { color: '#e6edf3' }, margin: { t: 36, r: 16, b: 48, l: 48 } };
    async function loadMetrics() {
      const r = await fetch('/metrics');
      if (!r.ok) throw new Error(await r.text());
      const m = await r.json();
      document.getElementById('k1').textContent = Number(m.total_revenue).toFixed(2);
      document.getElementById('k2').textContent = String(m.total_units_sold);
      document.getElementById('k3').textContent = Number(m.avg_customer_satisfaction).toFixed(2) + '/5';
      document.getElementById('k4').textContent = m.top_category;

      const months = m.monthly_revenue.map((x) => x.month);
      const rev = m.monthly_revenue.map((x) => x.sales_amount);
      Plotly.newPlot('monthly', [{ x: months, y: rev, type: 'scatter', mode: 'lines+markers', line: { color: '#58a6ff' } }], { ...plotLayout, title: 'Revenue by month', xaxis: { title: 'Month' }, yaxis: { title: 'Revenue' } }, { responsive: true, displayModeBar: false });

      const cats = m.category_revenue.map((x) => x.product_category);
      const catRev = m.category_revenue.map((x) => x.sales_amount);
      Plotly.newPlot('category', [{ x: cats, y: catRev, type: 'bar', marker: { color: '#3fb950' } }], { ...plotLayout, title: 'Category performance', xaxis: { title: 'Category' }, yaxis: { title: 'Revenue' } }, { responsive: true, displayModeBar: false });
    }

    document.getElementById('preset').addEventListener('change', (e) => {
      const v = e.target.value;
      if (v) document.getElementById('q').value = v;
    });

    document.getElementById('btn').addEventListener('click', async () => {
      const q = document.getElementById('q').value.trim();
      const out = document.getElementById('answer');
      const btn = document.getElementById('btn');
      out.textContent = '';
      out.className = '';
      if (q.length < 3) { out.className = 'err'; out.textContent = 'Enter at least 3 characters.'; return; }
      btn.disabled = true;
      try {
        const r = await fetch('/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q }) });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          out.className = 'err';
          out.textContent = (data && data.detail) ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : ('Error ' + r.status);
          return;
        }
        out.textContent = data.answer || '';
      } catch (e) {
        out.className = 'err';
        out.textContent = String(e);
      } finally {
        btn.disabled = false;
      }
    });

    loadMetrics().catch((e) => {
      document.getElementById('answer').className = 'err';
      document.getElementById('answer').textContent = 'Failed to load metrics: ' + e;
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return _DASHBOARD_HTML


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
