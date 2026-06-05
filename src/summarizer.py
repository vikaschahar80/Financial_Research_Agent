"""
Financial Summarizer — Groq (Llama 3)
"""

import json
from typing import Dict
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


INVESTMENT_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context", "company"],
    template="""You are preparing an investment memo for {company}.
Using ONLY the information in the following financial documents, produce a structured summary.

DOCUMENTS:
{context}

Return a JSON object with this exact structure (no markdown, no backticks, pure JSON only):
{{
  "company": "{company}",
  "one_liner": "One sentence describing what this company does",
  "financial_snapshot": {{
    "revenue_trend": "...",
    "margin_profile": "...",
    "key_metrics": ["metric 1", "metric 2"]
  }},
  "bull_case": ["reason 1", "reason 2", "reason 3"],
  "bear_case": ["risk 1", "risk 2", "risk 3"],
  "management_tone": "positive or neutral or cautious or negative",
  "key_quote": "Most important quote from the documents",
  "verdict": "One sentence investment view based purely on the documents"
}}"""
)


class FinancialSummarizer:

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(model=model, temperature=0.1)

    def generate_investment_memo(self, company: str, context: str) -> Dict:
        prompt = INVESTMENT_SUMMARY_PROMPT.format(company=company, context=context[:8000])
        response = self.llm.invoke(prompt)
        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return {"raw_response": response.content, "parse_error": True}

    def format_report_markdown(self, memo: Dict) -> str:
        if memo.get("parse_error"):
            return memo.get("raw_response", "Error generating report.")

        snap = memo.get("financial_snapshot", {})
        bull = "\n".join(f"- {r}" for r in memo.get("bull_case", []))
        bear = "\n".join(f"- {r}" for r in memo.get("bear_case", []))
        metrics = ", ".join(snap.get("key_metrics", []))

        return f"""# 📊 Investment Memo: {memo.get('company', '')}

## Overview
{memo.get('one_liner', 'N/A')}

## Financial Snapshot
- **Revenue Trend:** {snap.get('revenue_trend', 'N/A')}
- **Margin Profile:** {snap.get('margin_profile', 'N/A')}
- **Key Metrics:** {metrics}

## Bull Case 🟢
{bull}

## Bear Case 🔴
{bear}

## Management Tone
**{memo.get('management_tone', 'N/A').upper()}**

## Key Quote
> {memo.get('key_quote', 'N/A')}

## Verdict
**{memo.get('verdict', 'N/A')}**
"""
