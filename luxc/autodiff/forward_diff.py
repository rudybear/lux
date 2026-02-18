"""Forward-mode automatic differentiation via symbolic AST transformation.

Given a function annotated with @differentiable, generates gradient functions
for each scalar parameter using the chain rule.

Example:
    @differentiable
    fn energy(x: scalar) -> scalar { return x * x + sin(x); }

Generates:
    fn energy_d_x(x: scalar) -> scalar { return x * 1.0 + 1.0 * x + cos(x) * 1.0; }
"""

from __future__ import annotations
import copy
from luxc.parser.ast_nodes import (
    Module, FunctionDef, Param,
    LetStmt, ReturnStmt, IfStmt,
    NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp, CallExpr,
    ConstructorExpr, SwizzleAccess, TernaryExpr,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def autodiff_expand(module: Module) -> None:
    """Find all @differentiable functions and generate gradient functions."""
    new_functions: list[FunctionDef] = []

    for func in list(module.functions):
        if "differentiable" not in func.attributes:
            continue
        # Generate a gradient function for each scalar parameter
        scalar_params = [p for p in func.params if p.type_name == "scalar"]
        for param in scalar_params:
            grad_fn = _differentiate_function(func, param.name)
            new_functions.append(grad_fn)

    module.functions.extend(new_functions)

    # Also check stage-level functions
    for stage in module.stages:
        new_stage_fns: list[FunctionDef] = []
        for func in list(stage.functions):
            if "differentiable" not in func.attributes:
                continue
            scalar_params = [p for p in func.params if p.type_name == "scalar"]
            for param in scalar_params:
                grad_fn = _differentiate_function(func, param.name)
                new_stage_fns.append(grad_fn)
        stage.functions.extend(new_stage_fns)


# ---------------------------------------------------------------------------
# Function differentiation
# ---------------------------------------------------------------------------

def _differentiate_function(func: FunctionDef, wrt: str) -> FunctionDef:
    """Generate a gradient function d(func)/d(wrt).

    The generated function contains both the original computations
    (needed for derivative expressions) and additional derivative variables.
    """
    grad_name = f"{func.name}_d_{wrt}"
    params = copy.deepcopy(func.params)

    # Environment: maps variable names to their derivative expressions
    env: dict[str, object] = {}

    # Initialize derivatives: wrt parameter gets 1.0, others get 0.0
    for p in func.params:
        if p.type_name == "scalar":
            env[p.name] = _one() if p.name == wrt else _zero()
        # Non-scalar params: derivative is zero (not tracked)

    grad_body: list = []

    for stmt in func.body:
        if isinstance(stmt, LetStmt):
            # Keep original statement (derivative exprs may reference the value)
            grad_body.append(copy.deepcopy(stmt))

            # Compute derivative of the value
            d_value = _diff_expr(stmt.value, wrt, env)
            d_value = _simplify(d_value)
            d_name = f"_d_{stmt.name}"
            env[stmt.name] = VarRef(d_name)
            grad_body.append(LetStmt(d_name, stmt.type_name, d_value))

        elif isinstance(stmt, ReturnStmt):
            d_value = _diff_expr(stmt.value, wrt, env)
            d_value = _simplify(d_value)
            grad_body.append(ReturnStmt(d_value))

        elif isinstance(stmt, IfStmt):
            # Differentiate both branches, condition unchanged
            then_stmts = _diff_stmts(stmt.then_body, wrt, env)
            else_stmts = _diff_stmts(stmt.else_body, wrt, env)
            grad_body.append(IfStmt(
                copy.deepcopy(stmt.condition),
                then_stmts, else_stmts
            ))
        else:
            # Other statements pass through
            grad_body.append(copy.deepcopy(stmt))

    return FunctionDef(grad_name, params, func.return_type, grad_body)


def _diff_stmts(stmts: list, wrt: str, env: dict) -> list:
    """Differentiate a list of statements (used for if branches)."""
    result = []
    local_env = dict(env)  # copy for local scope
    for stmt in stmts:
        if isinstance(stmt, LetStmt):
            result.append(copy.deepcopy(stmt))
            d_value = _diff_expr(stmt.value, wrt, local_env)
            d_value = _simplify(d_value)
            d_name = f"_d_{stmt.name}"
            local_env[stmt.name] = VarRef(d_name)
            result.append(LetStmt(d_name, stmt.type_name, d_value))
        elif isinstance(stmt, ReturnStmt):
            d_value = _diff_expr(stmt.value, wrt, local_env)
            d_value = _simplify(d_value)
            result.append(ReturnStmt(d_value))
        elif isinstance(stmt, IfStmt):
            then_stmts = _diff_stmts(stmt.then_body, wrt, local_env)
            else_stmts = _diff_stmts(stmt.else_body, wrt, local_env)
            result.append(IfStmt(copy.deepcopy(stmt.condition), then_stmts, else_stmts))
        else:
            result.append(copy.deepcopy(stmt))
    return result


# ---------------------------------------------------------------------------
# Expression differentiation  (chain rule throughout)
# ---------------------------------------------------------------------------

def _diff_expr(expr, wrt: str, env: dict):
    """Compute the symbolic derivative of an expression w.r.t. `wrt`."""

    if isinstance(expr, NumberLit):
        return _zero()

    if isinstance(expr, BoolLit):
        return _zero()

    if isinstance(expr, VarRef):
        if expr.name == wrt:
            return _one()
        if expr.name in env:
            d = env[expr.name]
            return copy.deepcopy(d) if isinstance(d, (VarRef, NumberLit)) else copy.deepcopy(d)
        return _zero()

    if isinstance(expr, BinaryOp):
        a, b = expr.left, expr.right
        da = _diff_expr(a, wrt, env)
        db = _diff_expr(b, wrt, env)

        if expr.op == "+":
            return _simplify_add(da, db)
        elif expr.op == "-":
            return _simplify_sub(da, db)
        elif expr.op == "*":
            # Product rule: a*db + da*b
            return _simplify_add(
                _simplify_mul(copy.deepcopy(a), db),
                _simplify_mul(da, copy.deepcopy(b))
            )
        elif expr.op == "/":
            # Quotient rule: (da*b - a*db) / (b*b)
            return BinaryOp("/",
                _simplify_sub(
                    _simplify_mul(da, copy.deepcopy(b)),
                    _simplify_mul(copy.deepcopy(a), db)
                ),
                BinaryOp("*", copy.deepcopy(b), copy.deepcopy(b))
            )
        else:
            # Comparison/logical operators have zero derivative
            return _zero()

    if isinstance(expr, UnaryOp):
        if expr.op == "-":
            da = _diff_expr(expr.operand, wrt, env)
            if _is_zero(da):
                return _zero()
            return UnaryOp("-", da)
        elif expr.op == "!":
            return _zero()
        return _zero()

    if isinstance(expr, CallExpr) and isinstance(expr.func, VarRef):
        return _diff_call(expr, wrt, env)

    if isinstance(expr, ConstructorExpr):
        # vec3(a, b, c) -> vec3(da, db, dc)
        d_args = [_diff_expr(a, wrt, env) for a in expr.args]
        return ConstructorExpr(expr.type_name, d_args)

    if isinstance(expr, TernaryExpr):
        # condition unchanged, differentiate both branches
        dt = _diff_expr(expr.then_expr, wrt, env)
        de = _diff_expr(expr.else_expr, wrt, env)
        return TernaryExpr(copy.deepcopy(expr.condition), dt, de)

    if isinstance(expr, SwizzleAccess):
        d_obj = _diff_expr(expr.object, wrt, env)
        return SwizzleAccess(d_obj, expr.components)

    return _zero()


# ---------------------------------------------------------------------------
# Builtin function differentiation
# ---------------------------------------------------------------------------

def _diff_call(expr: CallExpr, wrt: str, env: dict):
    """Differentiate a function call using known derivative rules."""
    fname = expr.func.name
    args = expr.args

    # --- 1-arg builtins ---
    if len(args) == 1:
        u = args[0]
        du = _diff_expr(u, wrt, env)

        if fname == "sin":
            # cos(u) * du
            return _simplify_mul(
                CallExpr(VarRef("cos"), [copy.deepcopy(u)]),
                du
            )
        elif fname == "cos":
            # -sin(u) * du
            return _simplify_mul(
                UnaryOp("-", CallExpr(VarRef("sin"), [copy.deepcopy(u)])),
                du
            )
        elif fname == "tan":
            # du / (cos(u) * cos(u))
            cos_u = CallExpr(VarRef("cos"), [copy.deepcopy(u)])
            return BinaryOp("/", du,
                BinaryOp("*", cos_u, CallExpr(VarRef("cos"), [copy.deepcopy(u)])))
        elif fname == "exp":
            # exp(u) * du
            return _simplify_mul(
                CallExpr(VarRef("exp"), [copy.deepcopy(u)]),
                du
            )
        elif fname == "exp2":
            # exp2(u) * log(2.0) * du
            return _simplify_mul(
                _simplify_mul(
                    CallExpr(VarRef("exp2"), [copy.deepcopy(u)]),
                    CallExpr(VarRef("log"), [NumberLit("2.0")])
                ),
                du
            )
        elif fname == "log":
            # du / u
            return BinaryOp("/", du, copy.deepcopy(u))
        elif fname == "log2":
            # du / (u * log(2.0))
            return BinaryOp("/", du,
                BinaryOp("*", copy.deepcopy(u),
                    CallExpr(VarRef("log"), [NumberLit("2.0")])))
        elif fname == "sqrt":
            # du / (2.0 * sqrt(u))
            return BinaryOp("/", du,
                BinaryOp("*", NumberLit("2.0"),
                    CallExpr(VarRef("sqrt"), [copy.deepcopy(u)])))
        elif fname == "abs":
            # sign(u) * du
            return _simplify_mul(
                CallExpr(VarRef("sign"), [copy.deepcopy(u)]),
                du
            )
        elif fname in ("sign", "floor", "ceil", "step"):
            # Piecewise constant -> derivative is 0
            return _zero()
        elif fname == "fract":
            # fract(u) = u - floor(u), d/du = 1 - 0 = du
            return du
        elif fname == "normalize":
            # (dv - n*dot(n, dv)) / length(v)
            v = copy.deepcopy(u)
            dv = du
            n = CallExpr(VarRef("normalize"), [copy.deepcopy(u)])
            return BinaryOp("/",
                BinaryOp("-",
                    dv,
                    BinaryOp("*",
                        copy.deepcopy(n),
                        CallExpr(VarRef("dot"), [n, copy.deepcopy(dv) if not _is_zero(dv) else _zero()])
                    )
                ),
                CallExpr(VarRef("length"), [v])
            )
        elif fname == "length":
            # dot(v, dv) / length(v)
            v = copy.deepcopy(u)
            return BinaryOp("/",
                CallExpr(VarRef("dot"), [copy.deepcopy(u), du]),
                CallExpr(VarRef("length"), [v])
            )

    # --- 2-arg builtins ---
    if len(args) == 2:
        u, v = args[0], args[1]
        du = _diff_expr(u, wrt, env)
        dv = _diff_expr(v, wrt, env)

        if fname == "pow":
            # If v is constant: v * pow(u, v-1) * du
            if _is_zero(dv):
                return _simplify_mul(
                    _simplify_mul(
                        copy.deepcopy(v),
                        CallExpr(VarRef("pow"), [
                            copy.deepcopy(u),
                            BinaryOp("-", copy.deepcopy(v), NumberLit("1.0"))
                        ])
                    ),
                    du
                )
            # General: pow(u,v) * (v*du/u + log(u)*dv)
            return _simplify_mul(
                CallExpr(VarRef("pow"), [copy.deepcopy(u), copy.deepcopy(v)]),
                BinaryOp("+",
                    BinaryOp("*", copy.deepcopy(v),
                        BinaryOp("/", du, copy.deepcopy(u))),
                    BinaryOp("*",
                        CallExpr(VarRef("log"), [copy.deepcopy(u)]),
                        dv)
                )
            )
        elif fname == "min":
            # u < v ? du : dv
            return TernaryExpr(
                BinaryOp("<", copy.deepcopy(u), copy.deepcopy(v)),
                du, dv
            )
        elif fname == "max":
            # u > v ? du : dv
            return TernaryExpr(
                BinaryOp(">", copy.deepcopy(u), copy.deepcopy(v)),
                du, dv
            )
        elif fname == "step":
            # Piecewise constant
            return _zero()
        elif fname == "dot":
            # dot(da, b) + dot(a, db)
            return _simplify_add(
                CallExpr(VarRef("dot"), [du, copy.deepcopy(v)]),
                CallExpr(VarRef("dot"), [copy.deepcopy(u), dv])
            )

    # --- 3-arg builtins ---
    if len(args) == 3:
        a, b, t = args[0], args[1], args[2]
        da = _diff_expr(a, wrt, env)
        db = _diff_expr(b, wrt, env)
        dt = _diff_expr(t, wrt, env)

        if fname == "mix":
            # da*(1-t) + db*t + (b-a)*dt
            return _simplify_add(
                _simplify_add(
                    _simplify_mul(da,
                        BinaryOp("-", NumberLit("1.0"), copy.deepcopy(t))),
                    _simplify_mul(db, copy.deepcopy(t))
                ),
                _simplify_mul(
                    BinaryOp("-", copy.deepcopy(b), copy.deepcopy(a)),
                    dt
                )
            )
        elif fname == "clamp":
            # (u > lo && u < hi) ? du : 0.0
            return TernaryExpr(
                BinaryOp("&&",
                    BinaryOp(">", copy.deepcopy(a), copy.deepcopy(b)),
                    BinaryOp("<", copy.deepcopy(a), copy.deepcopy(t))
                ),
                da, _zero()
            )
        elif fname == "smoothstep":
            # 6*t_val*(1-t_val) * dx / (e1-e0) where t_val = clamp((x-e0)/(e1-e0), 0, 1)
            e0, e1, x = a, b, t
            dx = dt
            diff = BinaryOp("-", copy.deepcopy(e1), copy.deepcopy(e0))
            t_val = CallExpr(VarRef("clamp"), [
                BinaryOp("/",
                    BinaryOp("-", copy.deepcopy(x), copy.deepcopy(e0)),
                    copy.deepcopy(diff)
                ),
                NumberLit("0.0"), NumberLit("1.0")
            ])
            return BinaryOp("/",
                BinaryOp("*",
                    BinaryOp("*",
                        NumberLit("6.0"),
                        BinaryOp("*",
                            copy.deepcopy(t_val),
                            BinaryOp("-", NumberLit("1.0"), t_val)
                        )
                    ),
                    dx
                ),
                diff
            )

    # User-defined function call: apply chain rule
    # d/dx f(a(x), b(x)) = sum_i f_d_pi(args) * d(arg_i)/dx
    d_args = [_diff_expr(a, wrt, env) for a in args]
    terms = []
    # Find scalar params to generate gradient calls
    for i, (arg, d_arg) in enumerate(zip(args, d_args)):
        if _is_zero(d_arg):
            continue
        # Generate call to f_d_paramname
        # We don't know param names from call site, so use positional: _d_p{i}
        # Actually, we use the convention f_d_{param_name} but we don't have
        # the callee's param names here. Use index-based fallback.
        grad_fn_name = f"{fname}_d_p{i}"
        grad_call = CallExpr(VarRef(grad_fn_name), [copy.deepcopy(a) for a in args])
        terms.append(_simplify_mul(grad_call, d_arg))

    if not terms:
        return _zero()
    result = terms[0]
    for t in terms[1:]:
        result = _simplify_add(result, t)
    return result


# ---------------------------------------------------------------------------
# Zero/one helpers and simplification
# ---------------------------------------------------------------------------

def _zero() -> NumberLit:
    return NumberLit("0.0")


def _one() -> NumberLit:
    return NumberLit("1.0")


def _is_zero(expr) -> bool:
    """Check if expression is a literal 0."""
    if isinstance(expr, NumberLit):
        try:
            return float(expr.value) == 0.0
        except ValueError:
            return False
    return False


def _is_one(expr) -> bool:
    """Check if expression is a literal 1."""
    if isinstance(expr, NumberLit):
        try:
            return float(expr.value) == 1.0
        except ValueError:
            return False
    return False


def _simplify_add(a, b):
    """a + b with zero elimination."""
    if _is_zero(a):
        return b
    if _is_zero(b):
        return a
    return BinaryOp("+", a, b)


def _simplify_sub(a, b):
    """a - b with zero elimination."""
    if _is_zero(b):
        return a
    if _is_zero(a):
        return UnaryOp("-", b)
    return BinaryOp("-", a, b)


def _simplify_mul(a, b):
    """a * b with zero/one elimination."""
    if _is_zero(a) or _is_zero(b):
        return _zero()
    if _is_one(a):
        return b
    if _is_one(b):
        return a
    return BinaryOp("*", a, b)


def _simplify(expr):
    """Single-pass simplification of an expression tree."""
    if isinstance(expr, BinaryOp):
        expr.left = _simplify(expr.left)
        expr.right = _simplify(expr.right)
        if expr.op == "+":
            return _simplify_add(expr.left, expr.right)
        elif expr.op == "-":
            return _simplify_sub(expr.left, expr.right)
        elif expr.op == "*":
            return _simplify_mul(expr.left, expr.right)
        return expr
    if isinstance(expr, UnaryOp):
        expr.operand = _simplify(expr.operand)
        if expr.op == "-" and _is_zero(expr.operand):
            return _zero()
        return expr
    if isinstance(expr, ConstructorExpr):
        expr.args = [_simplify(a) for a in expr.args]
        return expr
    if isinstance(expr, CallExpr):
        expr.args = [_simplify(a) for a in expr.args]
        return expr
    if isinstance(expr, TernaryExpr):
        expr.then_expr = _simplify(expr.then_expr)
        expr.else_expr = _simplify(expr.else_expr)
        return expr
    if isinstance(expr, SwizzleAccess):
        expr.object = _simplify(expr.object)
        return expr
    return expr
