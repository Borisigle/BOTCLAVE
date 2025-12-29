#!/usr/bin/env python3
"""
Generate Report Script

Generates comprehensive performance reports from backtest results.
"""

import argparse
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from botclave.dashboard.metrics import MetricsCalculator


def load_backtest_results(filepath: str) -> dict:
    """Load backtest results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def generate_html_report(results: dict, output_file: str):
    """Generate HTML report."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 8px;
            color: white;
        }}
        .metric-card.positive {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .metric-card.negative {{
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin-top: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Backtest Performance Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Key Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{results.get('total_trades', 0)}</div>
            </div>
            <div class="metric-card {'positive' if results.get('win_rate', 0) > 50 else 'negative'}">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{results.get('win_rate', 0):.1f}%</div>
            </div>
            <div class="metric-card {'positive' if results.get('total_pnl', 0) > 0 else 'negative'}">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value">${results.get('total_pnl', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{results.get('sharpe_ratio', 0):.2f}</div>
            </div>
        </div>

        <h2>Detailed Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Winning Trades</td>
                <td>{results.get('winning_trades', 0)}</td>
            </tr>
            <tr>
                <td>Losing Trades</td>
                <td>{results.get('losing_trades', 0)}</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td>{results.get('profit_factor', 0):.2f}</td>
            </tr>
            <tr>
                <td>Max Drawdown</td>
                <td>{results.get('max_drawdown', 0):.2f}%</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td>${results.get('avg_win', 0):.2f}</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td>${results.get('avg_loss', 0):.2f}</td>
            </tr>
            <tr>
                <td>Largest Win</td>
                <td>${results.get('largest_win', 0):.2f}</td>
            </tr>
            <tr>
                <td>Largest Loss</td>
                <td>${results.get('largest_loss', 0):.2f}</td>
            </tr>
            <tr>
                <td>Total Fees</td>
                <td>${results.get('total_fees', 0):.2f}</td>
            </tr>
            <tr>
                <td>Start Capital</td>
                <td>${results.get('start_capital', 0):.2f}</td>
            </tr>
            <tr>
                <td>End Capital</td>
                <td>${results.get('end_capital', 0):.2f}</td>
            </tr>
            <tr>
                <td>ROI</td>
                <td>{results.get('total_pnl_percent', 0):.2f}%</td>
            </tr>
        </table>
    </div>
</body>
</html>
    """

    with open(output_file, "w") as f:
        f.write(html)

    print(f"✅ HTML report generated: {output_file}")


def generate_markdown_report(results: dict, output_file: str):
    """Generate Markdown report."""
    md = f"""# Backtest Performance Report

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| Total Trades | {results.get('total_trades', 0)} |
| Win Rate | {results.get('win_rate', 0):.1f}% |
| Total P&L | ${results.get('total_pnl', 0):.2f} |
| ROI | {results.get('total_pnl_percent', 0):.2f}% |
| Sharpe Ratio | {results.get('sharpe_ratio', 0):.2f} |
| Profit Factor | {results.get('profit_factor', 0):.2f} |
| Max Drawdown | {results.get('max_drawdown', 0):.2f}% |

## 📈 Trade Statistics

| Metric | Value |
|--------|-------|
| Winning Trades | {results.get('winning_trades', 0)} |
| Losing Trades | {results.get('losing_trades', 0)} |
| Average Win | ${results.get('avg_win', 0):.2f} |
| Average Loss | ${results.get('avg_loss', 0):.2f} |
| Largest Win | ${results.get('largest_win', 0):.2f} |
| Largest Loss | ${results.get('largest_loss', 0):.2f} |

## 💰 Capital

| Metric | Value |
|--------|-------|
| Start Capital | ${results.get('start_capital', 0):.2f} |
| End Capital | ${results.get('end_capital', 0):.2f} |
| Total Fees | ${results.get('total_fees', 0):.2f} |
"""

    with open(output_file, "w") as f:
        f.write(md)

    print(f"✅ Markdown report generated: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate performance reports")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to backtest results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["html", "markdown", "both"],
        default="both",
        help="Report format",
    )

    args = parser.parse_args()

    print(f"Loading results from {args.results}...")
    results = load_backtest_results(args.results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.format in ["html", "both"]:
        html_file = output_dir / f"report_{timestamp}.html"
        generate_html_report(results, str(html_file))

    if args.format in ["markdown", "both"]:
        md_file = output_dir / f"report_{timestamp}.md"
        generate_markdown_report(results, str(md_file))

    print(f"\n✅ Reports generated in {output_dir}")


if __name__ == "__main__":
    main()
