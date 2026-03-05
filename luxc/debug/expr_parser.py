"""Parse debug expressions by reusing the full Lux parser.

This avoids maintaining a separate expression grammar — conditions,
watch expressions, and REPL eval all go through the same parser that
compiles .lux sources.
"""

from __future__ import annotations

from luxc.parser.tree_builder import parse_lux


def parse_debug_expr(expr_str: str):
    """Parse a debug expression string into an AST node.

    Wraps the expression in a minimal stage + function so the full
    parser can handle it, then extracts the return expression AST.

    Supports: arithmetic, comparisons, function calls, field access,
    swizzle, index access, literals, constructors — everything the
    Lux parser supports.

    Returns the AST expression node, or raises ValueError on parse failure.
    """
    wrapper = (
        f"fragment {{\n"
        f"  fn __dbg__() -> scalar {{\n"
        f"    return {expr_str};\n"
        f"  }}\n"
        f"}}\n"
    )
    try:
        module = parse_lux(wrapper)
        fn = module.stages[0].functions[0]
        # The body should contain a single ReturnStmt
        ret_stmt = fn.body[0]
        return ret_stmt.value
    except Exception as e:
        raise ValueError(f"Failed to parse expression: {expr_str!r} — {e}") from e
