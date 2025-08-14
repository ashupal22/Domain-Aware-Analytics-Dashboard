import pandas as pd
import numpy as np
import re

# -------------------------------
# Formula evaluation for KPIs
# -------------------------------
def eval_formula(df: pd.DataFrame, formula: str, as_scalar: bool = True):
    """
    Evaluate KPI formulas like:
    - SUM(quantity * unitprice)
    - COUNT(DISTINCT customerid)
    - SUM(quantity * unitprice) / COUNT(DISTINCT invoiceno)

    If as_scalar=True -> returns scalar (for KPIs).
    If as_scalar=False -> returns vector/series (for charts).
    """

    formula = formula.strip()

    # Handle SUM(...)
    if formula.startswith("SUM(") and formula.endswith(")"):
        inner = formula[4:-1]
        values = eval_expression(inner, df)
        return values.sum() if as_scalar else values

    # Handle COUNT(DISTINCT col)
    if formula.startswith("COUNT(DISTINCT "):
        col = formula[15:-1].strip().lower()
        return df[col].nunique() if as_scalar else df[col]

    # Handle COUNT(col)
    if formula.startswith("COUNT(") and formula.endswith(")"):
        col = formula[6:-1].strip().lower()
        return df[col].count() if as_scalar else df[col]

    # Handle arithmetic expressions (e.g., SUM(...) / COUNT(...))
    try:
        expr = formula

        # Replace SUM(...)
        for match in re.findall(r"SUM\((.*?)\)", formula):
            val = eval_formula(f"SUM({match})", df, as_scalar=True)
            expr = expr.replace(f"SUM({match})", str(val))

        # Replace COUNT(DISTINCT ...)
        for match in re.findall(r"COUNT\(DISTINCT (.*?)\)", formula):
            val = eval_formula(f"COUNT(DISTINCT {match})", df, as_scalar=True)
            expr = expr.replace(f"COUNT(DISTINCT {match})", str(val))

        # Replace COUNT(...)
        for match in re.findall(r"COUNT\((.*?)\)", formula):
            val = eval_formula(f"COUNT({match})", df, as_scalar=True)
            expr = expr.replace(f"COUNT({match})", str(val))

        return eval(expr) if as_scalar else eval_expression(expr, df)
    except Exception as e:
        return f"Error: {e}"


# -------------------------------
# Expression evaluator for vectors
# -------------------------------
def eval_expression(expr: str, df: pd.DataFrame):
    """
    Evaluate vector expressions like:
    - quantity * unitprice
    - unitprice
    Returns a pandas Series.
    """
    expr = expr.lower().strip()
    try:
        return df.eval(expr)
    except Exception:
        # If column not found, return series of 1s (fallback for histograms)
        return pd.Series(np.ones(len(df)))


# -------------------------------
# Chart data preparation
# -------------------------------
def prepare_chart_data(chart_cfg: dict, df: pd.DataFrame):
    """
    Prepare data for chart plotting.
    Handles x and y formulas.
    """

    x_col = chart_cfg.get("x")
    y_col = chart_cfg.get("y")

    data = pd.DataFrame()

    # Process X
    if x_col:
        if any(op in x_col for op in ["*", "+", "-", "/"]):
            data["x"] = eval_expression(x_col, df)
        else:
            data["x"] = df[x_col.lower()]
    else:
        data["x"] = pd.Series(np.arange(len(df)))

    # Process Y
    if y_col:
        if any(op in y_col for op in ["*", "+", "-", "/"]):
            data["y"] = eval_expression(y_col, df)
        else:
            data["y"] = df[y_col.lower()]
    else:
        data["y"] = pd.Series(np.ones(len(df)))

    return data
