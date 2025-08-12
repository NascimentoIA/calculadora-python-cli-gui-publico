"""
CLI interativa para a calculadora.

Comandos:
  - Digite expressões como: 1+2*3, (2+3)^2, sqrt(9), 10/4
  - 'ajuda' para ver ajuda
  - 'sair', 'exit' ou 'q' para encerrar
"""

from __future__ import annotations

from calculator_core import (
    evaluate_expression,
    add,
    subtract,
    multiply,
    divide,
    power,
    sqrt,
)


HELP_TEXT = (
    "Digite expressões usando +, -, *, /, ^ (potência) ou **, parênteses e sqrt(x).\n"
    "Exemplos: 1+2*3, (2+3)^2, sqrt(9), -5/2\n"
    "Comandos: 'ajuda' para esta mensagem, 'sair'/'exit'/'q' para encerrar."
)


def run_interactive_cli() -> None:
    print("Calculadora (CLI). Digite 'ajuda' para ajuda, 'sair' para encerrar.")
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # linha em branco
            break

        if not raw:
            continue

        lowered = raw.lower()
        if lowered in {"sair", "exit", "q"}:
            break
        if lowered in {"ajuda", "help", "h"}:
            print(HELP_TEXT)
            continue

        try:
            result = evaluate_expression(raw)
            # Remover .0 quando o resultado for inteiro
            if float(result).is_integer():
                print(int(result))
            else:
                print(result)
        except ZeroDivisionError:
            print("Erro: divisão por zero")
        except Exception as exc:  # noqa: BLE001 - reportar erro ao usuário
            print(f"Erro: {exc}")


def run_step_cli() -> None:
    print("Calculadora (passo-a-passo). Digite 'sair' a qualquer momento para encerrar.")
    while True:
        try:
            a_raw = input("Primeiro número: ").strip()
            if a_raw.lower() in {"sair", "exit", "q"}:
                break
            a_val = float(a_raw)

            op = input("Operador (+, -, *, /, ^, sqrt): ").strip().lower()
            if op in {"sair", "exit", "q"}:
                break

            if op == "sqrt":
                result = sqrt(a_val)
            else:
                b_raw = input("Segundo número: ").strip()
                if b_raw.lower() in {"sair", "exit", "q"}:
                    break
                b_val = float(b_raw)

                if op == "+":
                    result = add(a_val, b_val)
                elif op == "-":
                    result = subtract(a_val, b_val)
                elif op == "*":
                    result = multiply(a_val, b_val)
                elif op == "/":
                    result = divide(a_val, b_val)
                elif op == "^" or op == "**":
                    result = power(a_val, b_val)
                else:
                    print("Operador inválido. Use +, -, *, /, ^, sqrt.")
                    continue

            if float(result).is_integer():
                print(f"Resultado: {int(result)}")
            else:
                print(f"Resultado: {result}")

        except ZeroDivisionError:
            print("Erro: divisão por zero")
        except ValueError:
            print("Entrada inválida. Digite números válidos.")
        except Exception as exc:  # noqa: BLE001
            print(f"Erro: {exc}")
