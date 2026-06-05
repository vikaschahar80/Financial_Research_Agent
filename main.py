"""
CLI Entrypoint — run the agent without the Streamlit UI.
Usage:
    python main.py --file path/to/filing.pdf --company Apple --type 10-K --year 2024
    python main.py --url https://... --company Apple --type earnings_call --year 2024
"""

import argparse
import os
import json
from src.rag_pipeline import FinancialRAGPipeline
from src.agent import FinancialResearchAgent
from src.summarizer import FinancialSummarizer


def main():
    parser = argparse.ArgumentParser(description="Financial Research Agent CLI")
    parser.add_argument("--file", help="Path to PDF or TXT document")
    parser.add_argument("--url", help="URL to load document from")
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument("--type", default="10-K", help="Document type (10-K, earnings_call, etc.)")
    parser.add_argument("--year", default="2024", help="Document year")
    parser.add_argument("--query", help="Single question to ask (optional; skips full report)")
    parser.add_argument("--report", action="store_true", help="Generate a full investment report")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY environment variable first.")

    # ── Load and ingest document ──
    rag = FinancialRAGPipeline()

    if args.file:
        ext = args.file.split(".")[-1].lower()
        docs = rag.load_pdf(args.file) if ext == "pdf" else rag.load_text(args.file)
    elif args.url:
        docs = rag.load_from_url(args.url)
    else:
        raise ValueError("Provide either --file or --url")

    docs = rag.add_metadata(docs, doc_type=args.type, company=args.company, year=args.year)
    rag.ingest(docs)

    agent = FinancialResearchAgent(rag=rag)
    summarizer = FinancialSummarizer()

    # ── Run query or report ──
    if args.query:
        result = agent.run(args.query)
        print("\n" + "="*60)
        print(f"QUERY: {result['query']}")
        print("="*60)
        print(result["answer"])

    elif args.report:
        print(f"\n📊 Generating investment report for {args.company}...\n")
        docs_ctx = rag.retrieve(f"{args.company} financials overview", k=10)
        context = "\n\n".join([d.page_content for d in docs_ctx])
        memo = summarizer.generate_investment_memo(company=args.company, context=context)
        markdown = summarizer.format_report_markdown(memo)
        print(markdown)

        output_path = f"{args.company.lower().replace(' ', '_')}_report.md"
        with open(output_path, "w") as f:
            f.write(markdown)
        print(f"\n✅ Report saved to {output_path}")

    else:
        # Interactive mode
        print(f"\n✅ Documents loaded. Entering interactive mode for {args.company}.")
        print("Type 'quit' to exit.\n")
        while True:
            query = input("Your question: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            result = agent.run(query)
            print(f"\n{result['answer']}\n")


if __name__ == "__main__":
    main()
