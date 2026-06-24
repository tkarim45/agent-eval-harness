"""Deterministic tools + a tiny in-memory world the agent operates over.

Tools are pure and reproducible so the benchmark has a single correct answer and a
reference tool-trace per task. Each tool exposes an Anthropic-style JSON schema (for the
Claude agent) and a plain Python `fn` (executed by both agents).
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Callable

# --- the world ---------------------------------------------------------------
CATALOG = {
    "widget": {"price": 12.5, "stock": 40},
    "gadget": {"price": 19.0, "stock": 12},
    "sprocket": {"price": 7.25, "stock": 100},
    "cog": {"price": 3.5, "stock": 8},
    "flange": {"price": 22.0, "stock": 25},
}
EMPLOYEES = {
    "alice": {"dept": "Engineering", "salary": 145000, "manager": "carol"},
    "bob": {"dept": "Sales", "salary": 98000, "manager": "dan"},
    "carol": {"dept": "Engineering", "salary": 210000, "manager": "erin"},
    "dan": {"dept": "Sales", "salary": 175000, "manager": "erin"},
    "erin": {"dept": "Executive", "salary": 320000, "manager": None},
}
FACTS = [
    "Acme Corp was founded in 1993 in Toledo.",
    "The Zephyr product line launched in 2018.",
    "Acme's headquarters moved to Denver in 2007.",
    "The Sprocket division was spun off in 2015.",
]

# --- safe calculator ---------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str):
    """Evaluate an arithmetic expression (+ - * / ** and parentheses)."""
    return _safe_eval(ast.parse(expression, mode="eval").body)


def catalog_lookup(item: str):
    """Return {price, stock} for a catalog item, or an error string."""
    rec = CATALOG.get(item.strip().lower())
    return rec if rec else f"unknown item: {item!r}"


def employee_lookup(name: str):
    """Return {dept, salary, manager} for an employee, or an error string."""
    rec = EMPLOYEES.get(name.strip().lower())
    return rec if rec else f"unknown employee: {name!r}"


def search(query: str):
    """Keyword search over the company fact corpus; returns matching sentences."""
    terms = [t for t in query.lower().split() if len(t) > 2]
    hits = [f for f in FACTS if any(t in f.lower() for t in terms)]
    return hits or ["no results"]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable


REGISTRY: dict[str, Tool] = {
    "calculator": Tool(
        "calculator", "Evaluate an arithmetic expression and return the number.",
        {"type": "object", "properties": {"expression": {"type": "string"}},
         "required": ["expression"]}, lambda expression: calculator(expression)),
    "catalog_lookup": Tool(
        "catalog_lookup", "Look up an item's price and stock in the product catalog.",
        {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]},
        lambda item: catalog_lookup(item)),
    "employee_lookup": Tool(
        "employee_lookup", "Look up an employee's department, salary, and manager.",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        lambda name: employee_lookup(name)),
    "search": Tool(
        "search", "Search the company fact corpus for a sentence answering a question.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        lambda query: search(query)),
}


def anthropic_tools() -> list[dict]:
    """Tool definitions in Anthropic Messages-API shape."""
    return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in REGISTRY.values()]


def run_tool(name: str, args: dict):
    if name not in REGISTRY:
        return f"unknown tool: {name}"
    try:
        return REGISTRY[name].fn(**args)
    except Exception as e:  # noqa: BLE001 — surface tool errors to the agent
        return f"tool error: {e}"
