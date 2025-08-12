"""
Aplicativo de Calculadora em Python

Uso:
  - CLI interativa (padrão):
      python main.py

  - GUI (Tkinter):
      python main.py --gui

Requer apenas biblioteca padrão do Python.
"""

from __future__ import annotations

import argparse

from cli_calculator import run_interactive_cli, run_step_cli


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculadora simples com modos CLI e GUI"
    )
    parser.add_argument("--gui", action="store_true", help="Inicia a interface gráfica (Tkinter)")
    parser.add_argument(
        "--expressao",
        action="store_true",
        help="Modo expressão (digitar expressões completas). Por padrão usa passo-a-passo.",
    )

    args = parser.parse_args()

    if args.gui:
        # Importação tardia para evitar carregar Tkinter quando não necessário
        from gui_calculator import CalculatorApp

        app = CalculatorApp()
        app.run()
    else:
        if args.expressao:
            run_interactive_cli()
        else:
            run_step_cli()


if __name__ == "__main__":
    main()
