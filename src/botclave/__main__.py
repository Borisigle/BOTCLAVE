"""Main entry point for Botclave package."""

import sys
from .scripts.fetch_data import main as fetch_data_main


def main() -> int:
    """Main entry point for the botclave package."""
    # For now, delegate to the fetch_data script
    # In the future, this could be expanded to provide a CLI menu
    return fetch_data_main()


if __name__ == "__main__":
    sys.exit(main())
