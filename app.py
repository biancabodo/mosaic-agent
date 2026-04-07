"""Gradio web UI — interactive front-end for the AlphaSignal research pipeline."""

import gradio as gr

from agents.orchestrator import extract_result, run_pipeline
from storage import signals as signal_store


async def analyze(ticker_input: str) -> tuple[str, str, str, str, str]:
    """Run the pipeline for a single ticker and return formatted results."""
    ticker = ticker_input.strip().upper()
    if not ticker:
        return "Enter a ticker symbol", "", "", "", ""

    state = await run_pipeline(ticker, stream_research=False)

    if state.get("error"):
        return f"Error: {state['error']}", "", "", "", ""

    signal, backtest = extract_result(state)

    if not signal:
        return "No signal generated — confidence may be below threshold", "", "", "", ""

    direction = f"{signal.direction.upper()} — confidence {signal.confidence:.0%}"
    period = signal.filing_period
    rationale = signal.rationale

    if backtest:
        bt_text = (
            f"Period:  {backtest.start_date} → {backtest.end_date}\n"
            f"Sharpe:  {backtest.sharpe_ratio:.2f}\n"
            f"Max DD:  {backtest.max_drawdown:.1%}\n"
            f"CAGR:    {backtest.cagr:.1%}\n"
            f"Return:  {backtest.total_return:.1%}\n"
            f"vs SPY — Sharpe: {backtest.benchmark_sharpe:.2f}  "
            f"CAGR: {backtest.benchmark_cagr:.1%}"
        )
    else:
        bt_text = "Confidence below threshold — backtest skipped"

    signal_store.save(signal, backtest)

    research = state.get("research_context", "")
    return direction, period, rationale, bt_text, research


with gr.Blocks(title="AlphaSignal Research Agent") as demo:
    gr.Markdown("# AlphaSignal Research Agent")
    gr.Markdown(
        "Analyses SEC filings (10-K / 10-Q / 8-K) and generates a directional "
        "alpha signal with historical backtest performance."
    )

    with gr.Row():
        ticker_box = gr.Textbox(
            label="Ticker",
            placeholder="e.g. NVDA",
            scale=3,
        )
        run_btn = gr.Button("Run pipeline", variant="primary", scale=1)

    with gr.Row():
        signal_out = gr.Textbox(label="Signal", interactive=False)
        period_out = gr.Textbox(label="Filing period", interactive=False)

    rationale_out = gr.Textbox(label="Rationale", interactive=False, lines=3)
    backtest_out = gr.Textbox(label="Backtest", interactive=False, lines=7)
    research_out = gr.Textbox(
        label="Research synthesis", interactive=False, lines=12
    )

    run_btn.click(
        fn=analyze,
        inputs=ticker_box,
        outputs=[signal_out, period_out, rationale_out, backtest_out, research_out],
    )
    ticker_box.submit(
        fn=analyze,
        inputs=ticker_box,
        outputs=[signal_out, period_out, rationale_out, backtest_out, research_out],
    )

if __name__ == "__main__":
    demo.launch()
