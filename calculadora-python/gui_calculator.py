"""
Interface gráfica da calculadora usando Tkinter.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from calculator_core import evaluate_expression


class CalculatorApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Calculadora")
        self.root.resizable(False, False)

        self._create_widgets()
        self._bind_keys()

    def _create_widgets(self) -> None:
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.display_var = tk.StringVar(value="")
        display = ttk.Entry(
            main_frame, textvariable=self.display_var, font=("Segoe UI", 16))
        display.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 8))
        display.focus_set()
        self.display = display

        buttons = [
            ("7", self._append("7")), ("8", self._append("8")
                                       ), ("9", self._append("9")), ("/", self._append("/")),
            ("4", self._append("4")), ("5", self._append("5")
                                       ), ("6", self._append("6")), ("*", self._append("*")),
            ("1", self._append("1")), ("2", self._append("2")
                                       ), ("3", self._append("3")), ("-", self._append("-")),
            ("0", self._append("0")), (".", self._append(".")
                                       ), ("(", self._append("(")), (")", self._append(")")),
            ("C", self._clear), ("⌫", self._backspace), ("^",
                                                         self._append("^")), ("√", self._append("sqrt(")),
            ("=", self._evaluate), ("+", self._append("+")),
        ]

        # Layout em grade 5x4 (última linha com '=' grande)
        grid_positions = [
            (1, 0), (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3),
            (3, 0), (3, 1), (3, 2), (3, 3),
            (4, 0), (4, 1), (4, 2), (4, 3),
            (5, 0), (5, 1), (5, 2), (5, 3),
        ]

        for (text, cmd), (r, c) in zip(buttons, grid_positions):
            style = "TButton"
            width = 5
            if text == "=":
                width = 12
            btn = ttk.Button(main_frame, text=text, width=width, command=cmd)
            btn.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")

        # Expansão horizontal da entrada
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.columnconfigure(3, weight=1)

    def _bind_keys(self) -> None:
        self.root.bind("<Return>", lambda _e: self._evaluate())
        self.root.bind("<KP_Enter>", lambda _e: self._evaluate())
        self.root.bind("<Escape>", lambda _e: self._clear())
        self.root.bind("<BackSpace>", lambda _e: self._backspace())

    def _append(self, s: str):
        def _cmd() -> None:
            self.display_var.set(self.display_var.get() + s)
            self.display.icursor(tk.END)
        return _cmd

    def _clear(self) -> None:
        self.display_var.set("")

    def _backspace(self) -> None:
        cur = self.display_var.get()
        if cur:
            self.display_var.set(cur[:-1])

    def _evaluate(self) -> None:
        expr = self.display_var.get()
        try:
            result = evaluate_expression(expr)
            if float(result).is_integer():
                self.display_var.set(str(int(result)))
            else:
                self.display_var.set(str(result))
        except ZeroDivisionError:
            self.display_var.set("Erro: div/0")
        except Exception:
            self.display_var.set("Erro")

    def run(self) -> None:
        self.root.mainloop()
