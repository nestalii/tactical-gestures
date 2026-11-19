"""GUI entry point.

Run from the project root::

    python -m gui.main                          # random-weight smoke run
    python -m gui.main --checkpoint <PATH>      # use a trained checkpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

import config
from gui.main_window import MainWindow
from src.utils import setup_logging


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Help text is in Ukrainian."""
    parser = argparse.ArgumentParser(
        description=(
            "Графічний інтерфейс системи розпізнавання тактичних жестів."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.CHECKPOINTS_DIR / "best_model.pt",
        help=(
            "Шлях до чекпойнта (за замовч. checkpoints/best_model.pt). "
            "Якщо файла немає — використовуються випадкові ваги."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Launch the Qt application."""
    setup_logging()
    args = _parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(model_checkpoint_path=args.checkpoint)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
