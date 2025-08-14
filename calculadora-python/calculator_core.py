"""
Núcleo da calculadora: avaliação segura de expressões aritméticas.

Suporta:
- Números inteiros e decimais
- Operadores: +, -, *, /, potência (** ou ^), parênteses
- Função: sqrt(x)

Observações:
- Implementa avaliador via AST para evitar riscos de eval()
"""

from __future__ import annotations

import ast
import math
from typing import Any


class _SafeEval(ast.NodeVisitor):
    """Avaliador de expressões aritméticas com AST.

    Permite apenas nós e operadores seguros.
    """

    allowed_binary_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
    )
    allowed_unary_ops = (
        ast.UAdd,
        ast.USub,
    )

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, self.allowed_binary_ops):
            raise ValueError("Operador binário não suportado")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self._apply_binop(node.op, left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, self.allowed_unary_ops):
            raise ValueError("Operador unário não suportado")
        operand = self.visit(node.operand)
        return self._apply_unaryop(node.op, operand)

    def visit_Call(self, node: ast.Call) -> Any:
        # Permitir apenas chamadas a sqrt(x)
        if isinstance(node.func, ast.Name) and node.func.id == "sqrt":
            if len(node.args) != 1 or node.keywords:
                raise ValueError("Uso inválido de sqrt")
            arg_val = self.visit(node.args[0])
            if arg_val < 0:
                raise ValueError("sqrt não aceita número negativo")
            return math.sqrt(arg_val)
        raise ValueError("Função não suportada")

    def visit_Name(self, node: ast.Name) -> Any:
        # Impedir uso de nomes/variáveis
        raise ValueError("Identificadores não são permitidos")

    def visit_Constant(self, node: ast.Constant) -> Any:  # py>=3.8
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Constante inválida na expressão")

    # Compatibilidade com versões mais antigas (opcional)
    def visit_Num(self, node: ast.Num) -> Any:  # type: ignore[override]
        return float(node.n)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError("Expressão contém construção não permitida")

    @staticmethod
    def _apply_binop(op: ast.operator, left: float, right: float) -> float:
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("Divisão por zero")
            return left / right
        if isinstance(op, ast.Pow):
            return left ** right
        raise ValueError("Operador binário não suportado")

    @staticmethod
    def _apply_unaryop(op: ast.unaryop, operand: float) -> float:
        if isinstance(op, ast.UAdd):
            return +operand
        if isinstance(op, ast.USub):
            return -operand
        raise ValueError("Operador unário não suportado")


def _preprocess_expression(expression: str) -> str:
    # Permitir '^' como potência substituindo por '**'
    return expression.replace("^", "**").strip()


def evaluate_expression(expression: str) -> float:
    """Avalia uma expressão aritmética de forma segura.

    Levanta ValueError para entradas inválidas e ZeroDivisionError para divisão por zero.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expressão vazia")
    processed = _preprocess_expression(expression)
    try:
        tree = ast.parse(processed, mode="eval")
        evaluator = _SafeEval()
        result = evaluator.visit(tree)
        return float(result)
    except ZeroDivisionError:
        raise
    except Exception as exc:  # noqa: BLE001 - reportar erro claramente ao usuário
        raise ValueError(f"Expressão inválida: {exc}") from exc


# Operações diretas (úteis para testes e reuso)
def add(a: float, b: float) -> float:
    return float(a) + float(b)


def subtract(a: float, b: float) -> float:
    return float(a) - float(b)


def multiply(a: float, b: float) -> float:
    return float(a) * float(b)


def divide(a: float, b: float) -> float:
    b = float(b)
    if b == 0:
        raise ZeroDivisionError("Divisão por zero")
    return float(a) / b


def power(a: float, b: float) -> float:
    return float(a) ** float(b)


def sqrt(a: float) -> float:
    a = float(a)
    if a < 0:
        raise ValueError("sqrt não aceita número negativo")
    return math.sqrt(a)
