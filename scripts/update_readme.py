#!/usr/bin/env python3
"""
Update README Status Script

Updates the implementation status table in README.md based on actual file existence.
"""

import os
from pathlib import Path
from datetime import datetime


STATUS_EMOJI = {
    "complete": "✅",
    "partial": "🔄",
    "pending": "⏳",
    "not_started": "❌",
}


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    return Path(filepath).exists()


def get_module_status() -> dict:
    """Get status of all modules."""
    base_path = Path(__file__).parent.parent
    src_path = base_path / "src" / "botclave"

    modules = {
        "Engine": {
            "depth.py": src_path / "engine" / "depth.py",
            "footprint.py": src_path / "engine" / "footprint.py",
            "dom_builder.py": src_path / "engine" / "dom_builder.py",
            "indicators.py": src_path / "engine" / "indicators.py",
            "strategy.py": src_path / "engine" / "strategy.py",
        },
        "Exchange": {
            "binance_connector.py": src_path / "exchange" / "binance_connector.py",
            "order_manager.py": src_path / "exchange" / "order_manager.py",
        },
        "Backtest": {
            "backtester.py": src_path / "backtest" / "backtester.py",
            "validator.py": src_path / "backtest" / "validator.py",
        },
        "Dashboard": {
            "app.py": src_path / "dashboard" / "app.py",
            "charts.py": src_path / "dashboard" / "charts.py",
            "metrics.py": src_path / "dashboard" / "metrics.py",
        },
    }

    status = {}
    for module_name, files in modules.items():
        total = len(files)
        completed = sum(1 for filepath in files.values() if filepath.exists())

        if completed == total:
            status[module_name] = "complete"
        elif completed > 0:
            status[module_name] = "partial"
        else:
            status[module_name] = "not_started"

        status[f"{module_name}_progress"] = f"{completed}/{total}"

    return status


def generate_status_table(status: dict) -> str:
    """Generate markdown status table."""
    table = """
## 📊 Implementation Status

| Module | Status | Progress | Description |
|--------|--------|----------|-------------|
"""

    modules_info = {
        "Engine": "Core order flow analysis components",
        "Exchange": "Exchange connectivity and order management",
        "Backtest": "Backtesting and strategy validation",
        "Dashboard": "Streamlit visualization dashboard",
    }

    for module, description in modules_info.items():
        emoji = STATUS_EMOJI.get(status.get(module, "not_started"), "❌")
        progress = status.get(f"{module}_progress", "0/0")
        table += f"| {module} | {emoji} | {progress} | {description} |\n"

    table += f"\n*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

    return table


def main():
    """Main function."""
    print("Checking module implementation status...")

    status = get_module_status()

    print("\nModule Status:")
    for module in ["Engine", "Exchange", "Backtest", "Dashboard"]:
        emoji = STATUS_EMOJI.get(status.get(module, "not_started"), "❌")
        progress = status.get(f"{module}_progress", "0/0")
        print(f"  {emoji} {module}: {progress}")

    status_table = generate_status_table(status)

    readme_path = Path(__file__).parent.parent / "README.md"

    if readme_path.exists():
        with open(readme_path, "r") as f:
            content = f.read()

        if "## 📊 Implementation Status" in content:
            start = content.find("## 📊 Implementation Status")
            end = content.find("\n## ", start + 1)

            if end == -1:
                end = len(content)

            new_content = content[:start] + status_table + content[end:]

            with open(readme_path, "w") as f:
                f.write(new_content)

            print(f"\n✅ README.md updated successfully!")
        else:
            print("\n⚠️  Status section not found in README.md")
    else:
        print("\n❌ README.md not found!")


if __name__ == "__main__":
    main()
