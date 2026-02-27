"""Type checker for Lux programs."""

from __future__ import annotations
from luxc.parser.ast_nodes import (
    Module, StageBlock, FunctionDef, LetStmt, AssignStmt, ReturnStmt,
    IfStmt, ExprStmt, NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp,
    CallExpr, ConstructorExpr, FieldAccess, SwizzleAccess, IndexAccess,
    TernaryExpr, AssignTarget,
    TypeAlias, ImportDecl, SurfaceDecl, GeometryDecl, PipelineDecl,
    DebugPrintStmt, AssertStmt, DebugBlock,
)
from luxc.builtins.types import (
    LuxType, resolve_type, TYPE_MAP, SCALAR, BOOL, VOID,
    VEC2, VEC3, VEC4, MAT2, MAT3, MAT4,
    VectorType, MatrixType, ScalarType, BoolType, is_numeric,
    register_type_alias, clear_type_aliases,
    RuntimeArrayType,
)
from luxc.builtins.functions import lookup_builtin
from luxc.analysis.symbols import Scope, Symbol


class TypeCheckError(Exception):
    pass


# Field types for BindlessMaterialData struct (SSBO)
_BINDLESS_MATERIAL_FIELD_TYPES = {
    "baseColorFactor": "vec4",
    "emissiveFactor": "vec3",
    "metallicFactor": "scalar",
    "roughnessFactor": "scalar",
    "emissionStrength": "scalar",
    "ior": "scalar",
    "clearcoatFactor": "scalar",
    "clearcoatRoughness": "scalar",
    "transmissionFactor": "scalar",
    "sheenRoughness": "scalar",
    "sheenColorFactor": "vec3",
    "base_color_tex_index": "int",
    "normal_tex_index": "int",
    "metallic_roughness_tex_index": "int",
    "occlusion_tex_index": "int",
    "emissive_tex_index": "int",
    "clearcoat_tex_index": "int",
    "clearcoat_roughness_tex_index": "int",
    "sheen_color_tex_index": "int",
    "transmission_tex_index": "int",
    "material_flags": "uint",
}

# Field types for LightData struct (SSBO) — multi-light evaluation
_LIGHT_DATA_FIELD_TYPES = {
    "light_type": "scalar",
    "intensity": "scalar",
    "range": "scalar",
    "inner_cone": "scalar",
    "position": "vec3",
    "outer_cone": "scalar",
    "direction": "vec3",
    "shadow_index": "scalar",
    "color": "vec3",
}

# Field types for ShadowEntry struct (SSBO) — shadow map evaluation
_SHADOW_ENTRY_FIELD_TYPES = {
    "view_projection": "mat4",
    "bias": "scalar",
    "normal_bias": "scalar",
    "resolution": "scalar",
    "light_size": "scalar",
}


def type_check(module: Module) -> None:
    checker = TypeChecker(module)
    checker.check()


class TypeChecker:
    def __init__(self, module: Module):
        self.module = module
        self.global_scope = Scope()
        self.current_scope: Scope = self.global_scope
        self.current_stage: StageBlock | None = None
        self._register_type_aliases()
        self._register_constants()

    def _register_type_aliases(self):
        for ta in self.module.type_aliases:
            # Verify the target type resolves
            t = resolve_type(ta.target_type)
            if t is None:
                raise TypeCheckError(f"Unknown target type '{ta.target_type}' in type alias '{ta.name}'")
            if ta.strict:
                from luxc.builtins.types import register_strict_type_alias
                register_strict_type_alias(ta.name, t)
            else:
                register_type_alias(ta.name, ta.target_type)

    def _register_constants(self):
        for c in self.module.constants:
            t = resolve_type(c.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown type '{c.type_name}' for constant '{c.name}'")
            self.global_scope.define(Symbol(c.name, t, "constant"))

    def check(self):
        for fn in self.module.functions:
            self._check_function(fn, self.global_scope)
        for stage in self.module.stages:
            self._check_stage(stage)

    def _check_stage(self, stage: StageBlock):
        self.current_stage = stage
        scope = self.global_scope.child()

        # Register inputs
        for inp in stage.inputs:
            t = resolve_type(inp.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown type '{inp.type_name}'")
            scope.define(Symbol(inp.name, t, "input"))

        # Register outputs
        for out in stage.outputs:
            t = resolve_type(out.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown type '{out.type_name}'")
            scope.define(Symbol(out.name, t, "output"))

        # Register builtin_position for vertex shaders
        if stage.stage_type == "vertex":
            scope.define(Symbol("builtin_position", VEC4, "builtin_position"))

        # Register uniform block fields as directly accessible
        for ub in stage.uniforms:
            for field in ub.fields:
                t = resolve_type(field.type_name)
                if t is None:
                    raise TypeCheckError(f"Unknown type '{field.type_name}' in uniform block '{ub.name}'")
                scope.define(Symbol(field.name, t, "uniform_field"))
            # Also register the block name for qualified access (e.g. Material.field)
            from luxc.builtins.types import UniformBlockType
            block_type = UniformBlockType(ub.name, tuple((f.name, f.type_name) for f in ub.fields))
            scope.define(Symbol(ub.name, block_type, "uniform_block"))

        # Register push constant fields
        for pb in stage.push_constants:
            for field in pb.fields:
                t = resolve_type(field.type_name)
                if t is None:
                    raise TypeCheckError(f"Unknown type '{field.type_name}' in push block '{pb.name}'")
                scope.define(Symbol(field.name, t, "push_field"))

        # Register samplers
        for sam in stage.samplers:
            from luxc.builtins.types import SAMPLER2D, SAMPLER_CUBE, SAMPLER_2D_ARRAY, SAMPLER_CUBE_ARRAY
            sam_type_name = getattr(sam, 'type_name', 'sampler2d')
            _SAMPLER_TYPE_MAP = {
                "sampler2d": SAMPLER2D, "samplerCube": SAMPLER_CUBE,
                "sampler2DArray": SAMPLER_2D_ARRAY, "samplerCubeArray": SAMPLER_CUBE_ARRAY,
            }
            sam_type = _SAMPLER_TYPE_MAP.get(sam_type_name, SAMPLER2D)
            scope.define(Symbol(sam.name, sam_type, "sampler"))

        # Register RT-specific variables
        for rp in stage.ray_payloads:
            t = resolve_type(rp.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown type '{rp.type_name}' in ray_payload")
            scope.define(Symbol(rp.name, t, "ray_payload"))

        for ha in stage.hit_attributes:
            t = resolve_type(ha.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown type '{ha.type_name}' in hit_attribute")
            scope.define(Symbol(ha.name, t, "hit_attribute"))

        for cd in stage.callable_data:
            t = resolve_type(cd.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown type '{cd.type_name}' in callable_data")
            scope.define(Symbol(cd.name, t, "callable_data"))

        for accel in stage.accel_structs:
            from luxc.builtins.types import ACCELERATION_STRUCTURE
            scope.define(Symbol(accel.name, ACCELERATION_STRUCTURE, "accel_struct"))

        for si in getattr(stage, 'storage_images', []):
            from luxc.builtins.types import STORAGE_IMAGE
            scope.define(Symbol(si.name, STORAGE_IMAGE, "storage_image"))

        for bta in getattr(stage, 'bindless_texture_arrays', []):
            from luxc.builtins.types import BINDLESS_TEXTURE_ARRAY
            scope.define(Symbol(bta.name, BINDLESS_TEXTURE_ARRAY, "bindless_texture_array"))

        # Register storage buffer variables (runtime arrays)
        for sb in getattr(stage, 'storage_buffers', []):
            elem_t = resolve_type(sb.element_type)
            if elem_t is None:
                raise TypeCheckError(f"Unknown element type '{sb.element_type}' in storage_buffer '{sb.name}'")
            arr_type = RuntimeArrayType(f"_rtarr_{sb.element_type}", sb.element_type)
            scope.define(Symbol(sb.name, arr_type, "storage_buffer"))

        # Register RT built-in variables
        _RT_STAGE_TYPES = {"raygen", "closest_hit", "any_hit", "miss", "intersection", "callable"}
        if stage.stage_type in _RT_STAGE_TYPES:
            from luxc.codegen.spirv_builder import _RT_BUILTINS
            for bname, (btype, _, valid_stages) in _RT_BUILTINS.items():
                if stage.stage_type in valid_stages:
                    t = resolve_type(btype)
                    if t is not None:
                        scope.define(Symbol(bname, t, "rt_builtin"))

        # Register mesh/task built-in variables
        _MESH_STAGE_TYPES = {"mesh", "task"}
        if stage.stage_type in _MESH_STAGE_TYPES:
            from luxc.codegen.spirv_builder import _MESH_BUILTINS
            for bname, (btype, _, valid_stages) in _MESH_BUILTINS.items():
                if stage.stage_type in valid_stages:
                    t = resolve_type(btype)
                    if t is not None:
                        scope.define(Symbol(bname, t, "mesh_builtin"))

        # Register mesh output built-in variables as arrays
        # (IndexAccess on RuntimeArrayType returns the element type correctly)
        if stage.stage_type == "mesh":
            scope.define(Symbol("gl_MeshVerticesEXT",
                                RuntimeArrayType("gl_MeshVerticesEXT_arr", "vec4"),
                                "mesh_builtin"))
            scope.define(Symbol("gl_PrimitiveTriangleIndicesEXT",
                                RuntimeArrayType("gl_PrimitiveTriangleIndicesEXT_arr", "uvec3"),
                                "mesh_builtin"))

            # Also register per-vertex output arrays (stage.outputs in mesh stage)
            for out in stage.outputs:
                t = resolve_type(out.type_name)
                if t is not None:
                    scope.define(Symbol(out.name,
                                        RuntimeArrayType(f"{out.name}_arr", out.type_name),
                                        "mesh_output"))

        # Register task payload variables
        for tp in getattr(stage, 'task_payloads', []):
            t = resolve_type(tp.type_name)
            if t is not None:
                scope.define(Symbol(tp.name, t, "task_payload"))

        for fn in stage.functions:
            self._check_function(fn, scope)

        self.current_stage = None

    def _check_function(self, fn: FunctionDef, parent_scope: Scope):
        scope = parent_scope.child()
        for p in fn.params:
            t = resolve_type(p.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown param type '{p.type_name}'")
            scope.define(Symbol(p.name, t, "param"))

        ret_type = resolve_type(fn.return_type) if fn.return_type else VOID

        for stmt in fn.body:
            self._check_stmt(stmt, scope, ret_type)

    def _check_stmt(self, stmt, scope: Scope, ret_type: LuxType):
        if isinstance(stmt, LetStmt):
            t = resolve_type(stmt.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown type '{stmt.type_name}'")
            vt = self._check_expr(stmt.value, scope)
            if vt.name != t.name:
                # Allow scalar literal in vector context etc — but for now strict
                if not _implicit_convert(vt, t):
                    raise TypeCheckError(
                        f"Type mismatch in let '{stmt.name}': declared {t.name}, got {vt.name}"
                    )
            scope.define(Symbol(stmt.name, t, "variable"))

        elif isinstance(stmt, AssignStmt):
            target_type = self._check_assign_target(stmt.target, scope)
            val_type = self._check_expr(stmt.value, scope)
            if target_type.name != val_type.name:
                if not _implicit_convert(val_type, target_type):
                    raise TypeCheckError(
                        f"Type mismatch in assignment: target is {target_type.name}, value is {val_type.name}"
                    )

        elif isinstance(stmt, ReturnStmt):
            vt = self._check_expr(stmt.value, scope)
            if ret_type != VOID and vt.name != ret_type.name:
                raise TypeCheckError(f"Return type mismatch: expected {ret_type.name}, got {vt.name}")

        elif isinstance(stmt, IfStmt):
            ct = self._check_expr(stmt.condition, scope)
            if not isinstance(ct, type(BOOL)):
                pass  # Allow non-bool conditions for now
            for s in stmt.then_body:
                self._check_stmt(s, scope, ret_type)
            for s in stmt.else_body:
                self._check_stmt(s, scope, ret_type)

        elif isinstance(stmt, ExprStmt):
            self._check_expr(stmt.expr, scope)

        elif isinstance(stmt, DebugPrintStmt):
            # Type-check each arg — must be numeric/vector
            for arg in stmt.args:
                t = self._check_expr(arg, scope)
                if not is_numeric(t) and not isinstance(t, BoolType):
                    pass  # Allow any type for debug_print flexibility

        elif isinstance(stmt, AssertStmt):
            ct = self._check_expr(stmt.condition, scope)
            # Condition should be bool-like (bool or scalar comparison result)

        elif isinstance(stmt, DebugBlock):
            for s in stmt.body:
                self._check_stmt(s, scope, ret_type)

    def _check_assign_target(self, target, scope: Scope) -> LuxType:
        if isinstance(target, AssignTarget):
            return self._check_expr(target.expr, scope)
        return self._check_expr(target, scope)

    def _check_expr(self, expr, scope: Scope) -> LuxType:
        if isinstance(expr, NumberLit):
            # If it has a decimal point, it's a scalar; otherwise could be int
            # For simplicity, all numeric literals are scalar in v1
            expr.resolved_type = "scalar"
            return SCALAR

        elif isinstance(expr, BoolLit):
            expr.resolved_type = "bool"
            return BOOL

        elif isinstance(expr, VarRef):
            sym = scope.lookup(expr.name)
            if sym is None:
                raise TypeCheckError(f"Undefined variable '{expr.name}'")
            expr.resolved_type = sym.type.name
            return sym.type

        elif isinstance(expr, BinaryOp):
            lt = self._check_expr(expr.left, scope)
            rt = self._check_expr(expr.right, scope)
            result = _check_binary_op(expr.op, lt, rt)
            expr.resolved_type = result.name
            return result

        elif isinstance(expr, UnaryOp):
            ot = self._check_expr(expr.operand, scope)
            if expr.op == "-":
                if not is_numeric(ot):
                    raise TypeCheckError(f"Cannot negate type {ot.name}")
                expr.resolved_type = ot.name
                return ot
            elif expr.op == "!":
                expr.resolved_type = "bool"
                return BOOL
            return ot

        elif isinstance(expr, CallExpr):
            # func could be a VarRef to a built-in function name
            if isinstance(expr.func, VarRef):
                fname = expr.func.name
                arg_types = [self._check_expr(a, scope) for a in expr.args]
                sig = lookup_builtin(fname, arg_types)
                if sig is None:
                    # Check user-defined functions
                    fn = self._find_user_function(fname)
                    if fn is not None:
                        ret = resolve_type(fn.return_type) if fn.return_type else VOID
                        expr.resolved_type = ret.name
                        return ret
                    raise TypeCheckError(
                        f"No matching overload for '{fname}' with args ({', '.join(t.name for t in arg_types)})"
                    )
                expr.resolved_type = sig.return_type.name
                return sig.return_type
            # Method-style calls not supported in v1
            arg_types = [self._check_expr(a, scope) for a in expr.args]
            raise TypeCheckError(f"Unsupported call expression")

        elif isinstance(expr, ConstructorExpr):
            arg_types = [self._check_expr(a, scope) for a in expr.args]
            t = resolve_type(expr.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown constructor type '{expr.type_name}'")
            # Validate constructor args (permissive for v1)
            expr.resolved_type = t.name
            return t

        elif isinstance(expr, FieldAccess):
            obj_type = self._check_expr(expr.object, scope)
            # Qualified uniform/push block field access (e.g. Material.roughness_factor)
            from luxc.builtins.types import UniformBlockType
            if isinstance(obj_type, UniformBlockType):
                field_map = dict(obj_type.fields)
                if expr.field in field_map:
                    field_type = resolve_type(field_map[expr.field])
                    if field_type is None:
                        field_type = SCALAR
                    expr.resolved_type = field_type.name
                    return field_type
                raise TypeCheckError(f"Unknown field '{expr.field}' in block '{obj_type.name}'")
            # SSBO struct field access (e.g. materials[idx].baseColorFactor)
            if obj_type.name == "BindlessMaterialData":
                field_types = _BINDLESS_MATERIAL_FIELD_TYPES.get(expr.field)
                if field_types is not None:
                    result = resolve_type(field_types)
                    if result is None:
                        result = SCALAR
                    expr.resolved_type = result.name
                    return result
                # Unknown field — fallback to scalar
            if obj_type.name == "LightData":
                field_types = _LIGHT_DATA_FIELD_TYPES.get(expr.field)
                if field_types is not None:
                    result = resolve_type(field_types)
                    if result is None:
                        result = SCALAR
                    expr.resolved_type = result.name
                    return result
                # Unknown field — fallback to scalar
            if obj_type.name == "ShadowEntry":
                field_types = _SHADOW_ENTRY_FIELD_TYPES.get(expr.field)
                if field_types is not None:
                    result = resolve_type(field_types)
                    if result is None:
                        result = SCALAR
                    expr.resolved_type = result.name
                    return result
                # Unknown field — fallback to scalar
            # For other field access (struct fields, etc.) - simplified
            expr.resolved_type = SCALAR.name
            return SCALAR

        elif isinstance(expr, SwizzleAccess):
            obj_type = self._check_expr(expr.object, scope)
            n = len(expr.components)
            if n == 1:
                result = SCALAR
            else:
                result = {2: VEC2, 3: VEC3, 4: VEC4}.get(n)
                if result is None:
                    raise TypeCheckError(f"Invalid swizzle length {n}")
            expr.resolved_type = result.name
            return result

        elif isinstance(expr, IndexAccess):
            obj_type = self._check_expr(expr.object, scope)
            self._check_expr(expr.index, scope)
            # storage_buffer[i] -> element_type
            if isinstance(obj_type, RuntimeArrayType):
                result = resolve_type(obj_type.element_type_name)
                if result is None:
                    result = SCALAR
            # mat[i] -> vecN, vec[i] -> scalar
            elif isinstance(obj_type, MatrixType):
                result = {2: VEC2, 3: VEC3, 4: VEC4}[obj_type.size]
            elif isinstance(obj_type, VectorType):
                result = resolve_type(obj_type.component)
            else:
                result = SCALAR
            expr.resolved_type = result.name
            return result

        elif isinstance(expr, TernaryExpr):
            self._check_expr(expr.condition, scope)
            tt = self._check_expr(expr.then_expr, scope)
            et = self._check_expr(expr.else_expr, scope)
            if tt.name != et.name:
                raise TypeCheckError(f"Ternary branches have different types: {tt.name} vs {et.name}")
            expr.resolved_type = tt.name
            return tt

        raise TypeCheckError(f"Unknown expression type: {type(expr).__name__}")

    def _find_user_function(self, name: str) -> FunctionDef | None:
        for fn in self.module.functions:
            if fn.name == name:
                return fn
        if self.current_stage:
            for fn in self.current_stage.functions:
                if fn.name == name:
                    return fn
        return None


def _check_binary_op(op: str, left: LuxType, right: LuxType) -> LuxType:
    from luxc.builtins.types import SemanticType, unwrap_semantic_type

    # Handle SemanticType in binary operations
    left_is_semantic = isinstance(left, SemanticType)
    right_is_semantic = isinstance(right, SemanticType)

    if left_is_semantic or right_is_semantic:
        # Both semantic: must be the same semantic type (not just same underlying)
        if left_is_semantic and right_is_semantic:
            if left.name != right.name:
                raise TypeCheckError(
                    f"Cannot apply '{op}' to incompatible semantic types {left.name} and {right.name}"
                )
            # Delegate to base-type check, then wrap result back
            base_result = _check_binary_op(op, left.base_type, right.base_type)
            # Comparison/logical ops always return bool
            if isinstance(base_result, type(BOOL)) or base_result == BOOL:
                return BOOL
            return left

        # One semantic, one plain: allow scalar promotion (scalar op SemanticType -> SemanticType)
        if left_is_semantic and isinstance(right, ScalarType):
            base_result = _check_binary_op(op, left.base_type, right)
            if isinstance(base_result, type(BOOL)) or base_result == BOOL:
                return BOOL
            return left
        if right_is_semantic and isinstance(left, ScalarType):
            base_result = _check_binary_op(op, left, right.base_type)
            if isinstance(base_result, type(BOOL)) or base_result == BOOL:
                return BOOL
            return right

        # One semantic, one non-scalar plain: check if base types match
        if left_is_semantic:
            base_result = _check_binary_op(op, left.base_type, right)
            if isinstance(base_result, type(BOOL)) or base_result == BOOL:
                return BOOL
            return left
        if right_is_semantic:
            base_result = _check_binary_op(op, left, right.base_type)
            if isinstance(base_result, type(BOOL)) or base_result == BOOL:
                return BOOL
            return right

    if op in ("==", "!=", "<", ">", "<=", ">="):
        return BOOL
    if op in ("&&", "||"):
        return BOOL

    # Bitwise operators: &, |, ^
    if op in ("&", "|", "^"):
        if isinstance(left, ScalarType) and isinstance(right, ScalarType):
            # Prefer uint if either operand is uint
            if left.name == "uint" or right.name == "uint":
                return resolve_type("uint")
            return left
        raise TypeCheckError(f"Cannot apply '{op}' to {left.name} and {right.name}")

    # Arithmetic: +, -, *, /, %
    if op in ("+", "-", "/", "%"):
        if left.name == right.name and is_numeric(left):
            return left
        # Mixed scalar types (int/uint/scalar) -> scalar (codegen handles conversion)
        if isinstance(left, ScalarType) and isinstance(right, ScalarType):
            return SCALAR
        # scalar op vec -> vec, vec op scalar -> vec
        if isinstance(left, ScalarType) and isinstance(right, VectorType):
            return right
        if isinstance(left, VectorType) and isinstance(right, ScalarType):
            return left
        raise TypeCheckError(f"Cannot apply '{op}' to {left.name} and {right.name}")

    if op == "*":
        # Same types
        if left.name == right.name and is_numeric(left):
            return left
        # Mixed scalar types (int/uint/scalar) -> scalar (codegen handles conversion)
        if isinstance(left, ScalarType) and isinstance(right, ScalarType):
            return SCALAR
        # scalar * vec -> vec
        if isinstance(left, ScalarType) and isinstance(right, VectorType):
            return right
        if isinstance(left, VectorType) and isinstance(right, ScalarType):
            return left
        # mat * vec -> vec
        if isinstance(left, MatrixType) and isinstance(right, VectorType):
            return right
        # mat * mat -> mat
        if isinstance(left, MatrixType) and isinstance(right, MatrixType):
            return left
        # vec * mat -> vec
        if isinstance(left, VectorType) and isinstance(right, MatrixType):
            return left
        # scalar * mat -> mat
        if isinstance(left, ScalarType) and isinstance(right, MatrixType):
            return right
        if isinstance(left, MatrixType) and isinstance(right, ScalarType):
            return left
        raise TypeCheckError(f"Cannot apply '*' to {left.name} and {right.name}")

    raise TypeCheckError(f"Unknown operator '{op}'")


def _implicit_convert(from_type: LuxType, to_type: LuxType) -> bool:
    from luxc.builtins.types import SemanticType, unwrap_semantic_type
    # SemanticType("A") -> its base type: OK (implicit unwrap)
    if isinstance(from_type, SemanticType) and not isinstance(to_type, SemanticType):
        return from_type.base_type.name == to_type.name
    # base type -> SemanticType("A"): OK (implicit wrap / construction)
    if not isinstance(from_type, SemanticType) and isinstance(to_type, SemanticType):
        return from_type.name == to_type.base_type.name
    # SemanticType("A") -> SemanticType("B"): NO (even if same underlying)
    if isinstance(from_type, SemanticType) and isinstance(to_type, SemanticType):
        return from_type.name == to_type.name
    # No other implicit conversions
    return False
