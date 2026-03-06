"""SPIR-V assembly text generator."""

from __future__ import annotations
from luxc.parser.ast_nodes import (
    Module, StageBlock, FunctionDef, LetStmt, AssignStmt, ReturnStmt,
    IfStmt, ExprStmt, NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp,
    CallExpr, ConstructorExpr, FieldAccess, SwizzleAccess, IndexAccess,
    TernaryExpr, AssignTarget,
    RayPayloadDecl, HitAttributeDecl, CallableDataDecl, AccelDecl,
    StorageImageDecl, StorageBufferDecl, BindlessTextureArrayDecl,
    SharedDecl, SpecConstDecl,
    DebugPrintStmt, AssertStmt, DebugBlock,
    ForStmt, WhileStmt, BreakStmt, ContinueStmt,
    DiscardStmt,
)
from luxc.codegen.spirv_types import TypeRegistry
from luxc.codegen.glsl_ext import GLSL_STD_450, LUX_TO_GLSL
from luxc.analysis.layout_assigner import compute_std140_offsets
from luxc.builtins.types import (
    resolve_type, resolve_alias_chain, VectorType, MatrixType, ScalarType, TYPE_MAP,
)


# RT execution model mapping
_RT_EXEC_MODELS = {
    "raygen": "RayGenerationKHR",
    "closest_hit": "ClosestHitKHR",
    "any_hit": "AnyHitKHR",
    "miss": "MissKHR",
    "intersection": "IntersectionKHR",
    "callable": "CallableKHR",
}

# RT stage types
_RT_STAGES = frozenset(_RT_EXEC_MODELS.keys())

# Mesh shader execution model mapping
_MESH_EXEC_MODELS = {"mesh": "MeshEXT", "task": "TaskEXT"}
_MESH_STAGES = frozenset(_MESH_EXEC_MODELS.keys())

# Workgroup built-in variables (shared by mesh, task, and compute stages)
_WORKGROUP_BUILTINS = {
    "local_invocation_id":    ("uvec3", "LocalInvocationId",    {"mesh", "task", "compute"}),
    "local_invocation_index": ("uint",  "LocalInvocationIndex", {"mesh", "task", "compute"}),
    "workgroup_id":           ("uvec3", "WorkgroupId",          {"mesh", "task", "compute"}),
    "num_workgroups":         ("uvec3", "NumWorkgroups",        {"mesh", "task", "compute"}),
    "global_invocation_id":   ("uvec3", "GlobalInvocationId",   {"mesh", "task", "compute"}),
}

# Compute stage types
_COMPUTE_STAGES = frozenset({"compute"})
_WORKGROUP_STAGES = _MESH_STAGES | _COMPUTE_STAGES

# RT built-in variables: (name, type, SPIR-V BuiltIn, stages)
_RT_BUILTINS = {
    "launch_id": ("uvec3", "LaunchIdKHR", {"raygen"}),
    "launch_size": ("uvec3", "LaunchSizeKHR", {"raygen"}),
    "world_ray_origin": ("vec3", "WorldRayOriginKHR", {"closest_hit", "any_hit", "miss", "intersection"}),
    "world_ray_direction": ("vec3", "WorldRayDirectionKHR", {"closest_hit", "any_hit", "miss", "intersection"}),
    "ray_tmin": ("scalar", "RayTminKHR", {"closest_hit", "any_hit", "miss", "intersection"}),
    "ray_tmax": ("scalar", "RayTmaxKHR", {"closest_hit", "any_hit", "miss", "intersection"}),
    "hit_t": ("scalar", "RayTmaxKHR", {"closest_hit", "any_hit"}),
    "instance_id": ("int", "InstanceCustomIndexKHR", {"closest_hit", "any_hit", "intersection"}),
    "primitive_id": ("int", "PrimitiveId", {"closest_hit", "any_hit", "intersection"}),
    "hit_kind": ("uint", "HitKindKHR", {"closest_hit", "any_hit"}),
    "object_to_world": ("mat4x3", "ObjectToWorldKHR", {"closest_hit", "any_hit", "intersection"}),
    "world_to_object": ("mat4x3", "WorldToObjectKHR", {"closest_hit", "any_hit", "intersection"}),
    "incoming_ray_flags": ("uint", "IncomingRayFlagsKHR", {"closest_hit", "any_hit", "miss", "intersection"}),
    "geometry_index": ("int", "RayGeometryIndexKHR", {"closest_hit", "any_hit", "intersection"}),
}

# BindlessMaterialData struct fields: (name, lux_type)
# Must match the GPU-side struct layout (std430)
_BINDLESS_MATERIAL_FIELDS = [
    ("baseColorFactor", "vec4"),
    ("emissiveFactor", "vec3"),
    ("metallicFactor", "scalar"),
    ("roughnessFactor", "scalar"),
    ("emissionStrength", "scalar"),
    ("ior", "scalar"),
    ("clearcoatFactor", "scalar"),
    ("clearcoatRoughness", "scalar"),
    ("transmissionFactor", "scalar"),
    ("sheenRoughness", "scalar"),
    ("_pad0", "scalar"),
    ("sheenColorFactor", "vec3"),
    ("_pad1", "scalar"),
    ("base_color_tex_index", "int"),
    ("normal_tex_index", "int"),
    ("metallic_roughness_tex_index", "int"),
    ("occlusion_tex_index", "int"),
    ("emissive_tex_index", "int"),
    ("clearcoat_tex_index", "int"),
    ("clearcoat_roughness_tex_index", "int"),
    ("sheen_color_tex_index", "int"),
    ("transmission_tex_index", "int"),
    ("material_flags", "uint"),
    ("index_offset", "scalar"),
    ("_pad3", "uint"),
]

# LightData struct fields: (name, lux_type)
# Must match the GPU-side struct layout (std430)
# 64 bytes per light: 4 vec4s
_LIGHT_DATA_FIELDS = [
    ("light_type", "scalar"),   # 0=dir, 1=point, 2=spot
    ("intensity", "scalar"),
    ("range", "scalar"),
    ("inner_cone", "scalar"),
    ("position", "vec3"),
    ("outer_cone", "scalar"),
    ("direction", "vec3"),
    ("shadow_index", "scalar"), # -1.0 = no shadow
    ("color", "vec3"),
    ("_pad", "scalar"),
]

# ShadowEntry struct fields for shadow matrix data
# 80 bytes per entry (std430)
_SHADOW_ENTRY_FIELDS = [
    ("view_projection", "mat4"),   # 64 bytes
    ("bias", "scalar"),            # 4 bytes
    ("normal_bias", "scalar"),     # 4 bytes
    ("resolution", "scalar"),      # 4 bytes
    ("light_size", "scalar"),      # 4 bytes — PCSS light source size
]


def generate_spirv(module: Module, stage: StageBlock, debug: bool = False, source_name: str = "",
                    assert_kill: bool = False, source_text: str = "",
                    rich_debug: bool = False,
                    precision_map: dict[str, str] | None = None) -> str:
    gen = SpvGenerator(module, stage, debug=debug, source_name=source_name)
    gen.assert_kill = assert_kill
    gen._dbg_source_text = source_text
    gen._enable_nonsemantic_debug = rich_debug
    if precision_map:
        gen._precision_map = precision_map
    return gen.generate()


class SpvGenerator:
    def __init__(self, module: Module, stage: StageBlock, debug: bool = False, source_name: str = ""):
        self.module = module
        self.stage = stage
        self.debug = debug
        self.source_name = source_name
        self.reg = TypeRegistry()
        self.body_lines: list[str] = []
        self.decorations: list[str] = []
        self.annotations: list[str] = []  # OpName etc.
        self.global_vars: list[str] = []
        self.var_map: dict[str, str] = {}        # lux name -> %id of variable (pointer)
        self.var_types: dict[str, str] = {}       # lux name -> lux type name
        self.var_storage: dict[str, str] = {}     # lux name -> storage class
        self._storage_image_type_ids: dict[str, str] = {}  # image name -> SPIR-V image type id
        self.interface_ids: list[str] = []         # for OpEntryPoint
        self.interface_storage: dict[str, str] = {}  # var_id -> storage class (Input/Output/Uniform/etc)
        self.glsl_ext_id: str | None = None
        self._label_counter = 0

        # Uniform block info
        self.uniform_var_ids: dict[str, str] = {}     # block name -> %id
        self.uniform_struct_ids: dict[str, str] = {}   # block name -> struct type %id
        self.uniform_field_indices: dict[str, int] = {} # field name -> index
        self.uniform_block_for_field: dict[str, str] = {} # field name -> block name

        # Push constant info
        self.push_var_ids: dict[str, str] = {}
        self.push_struct_ids: dict[str, str] = {}
        self.push_field_indices: dict[str, int] = {}
        self.push_block_for_field: dict[str, str] = {}

        # Storage buffer info
        self.storage_buffer_var_ids: dict[str, str] = {}     # buffer name -> %id
        self.storage_buffer_element_types: dict[str, str] = {} # buffer name -> element type name

        # Shared memory (Workgroup storage class)
        self.shared_var_ids: dict[str, str] = {}           # shared var name -> %id
        self.shared_element_types: dict[str, str] = {}     # shared var name -> element type name
        self.shared_array_sizes: dict[str, int | None] = {} # shared var name -> array size (None for scalar)

        # Specialization constants
        self.spec_const_ids: dict[str, str] = {}  # spec const name -> %id (OpSpecConstant result)

        # gl_PerVertex output block for vertex shader
        self.per_vertex_var_id: str | None = None
        self.per_vertex_struct_id: str | None = None

        # Debug info tracking
        self._debug_source_id: str | None = None
        self._debug_current_line: int = -1  # track last emitted line to avoid redundancy

        # Loop label stack for break/continue
        self._loop_label_stack: list[tuple[str, str]] = []  # (merge_label, continue_label)

        # DebugPrintf extension (NonSemantic.DebugPrintf)
        self._debug_printf_ext_id: str | None = None
        self._has_debug_printf: bool = False  # set to True if any debug_print/assert is found
        self._needs_image_query: bool = False  # set to True if texture_levels/texture_size/image_size used
        self._debug_string_ids: dict[str, str] = {}  # format_string -> OpString %id

        # SSA value forwarding for mem2reg (populated per-function in _gen_function)
        self._ssa_values: dict[str, str] = {}      # name -> SSA ID (single-assignment lets)
        self._mutable_vars: set[str] = set()        # names that need OpVariable

        # Local variable names for OpName emission in debug mode
        self._all_local_names: list[tuple[str, str]] = []  # (name, var_id)

        # Assert kill mode: emit OpDemoteToHelperInvocation on assertion failure
        self.assert_kill: bool = False

        # Auto-type precision map: var_name -> "fp16" for RelaxedPrecision decoration
        self._precision_map: dict[str, str] = {}

        # NonSemantic.Shader.DebugInfo.100 emitter (initialized in generate() if debug)
        self._dbg_emitter: object | None = None  # DebugInfoEmitter
        self._dbg_source_text: str = ""  # full source for embedding
        self._enable_nonsemantic_debug: bool = False  # opt-in for NonSemantic debug info

    def _next_label(self) -> str:
        self._label_counter += 1
        return f"%label_{self._label_counter}"

    @staticmethod
    def _block_terminated(lines: list[str]) -> bool:
        """Check if a list of SPIR-V lines ends with a block terminator."""
        _TERMINATORS = ("OpReturn", "OpReturnValue", "OpKill",
                        "OpBranch", "OpBranchConditional", "OpUnreachable",
                        "OpDemoteToHelperInvocation")
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                continue
            for t in _TERMINATORS:
                if stripped.startswith(t):
                    return True
            # Skip debug/decoration lines
            if stripped.startswith("OpLine") or stripped.startswith("OpNoLine"):
                continue
            if "DebugScope" in stripped or "DebugNoScope" in stripped:
                continue
            return False
        return False

    def _scan_for_debug_stmts(self, stmts: list) -> bool:
        """Check if any debug_print or assert statements exist in the function bodies."""
        for stmt in stmts:
            if isinstance(stmt, (DebugPrintStmt, AssertStmt)):
                return True
            if isinstance(stmt, DebugBlock):
                if self._scan_for_debug_stmts(stmt.body):
                    return True
            if isinstance(stmt, IfStmt):
                if self._scan_for_debug_stmts(stmt.then_body):
                    return True
                if self._scan_for_debug_stmts(stmt.else_body):
                    return True
            if isinstance(stmt, ForStmt):
                if self._scan_for_debug_stmts(stmt.body):
                    return True
            if isinstance(stmt, WhileStmt):
                if self._scan_for_debug_stmts(stmt.body):
                    return True
        return False

    def _scan_mutable_vars(self, stmts: list) -> None:
        """Pre-scan statements to find variables that need OpVariable (reassigned or loop vars)."""
        for stmt in stmts:
            if isinstance(stmt, AssignStmt):
                name = self._extract_assign_root_name(stmt.target)
                if name is not None:
                    self._mutable_vars.add(name)
            elif isinstance(stmt, ForStmt):
                self._mutable_vars.add(stmt.loop_var)
                self._scan_mutable_vars(stmt.body)
            elif isinstance(stmt, IfStmt):
                self._scan_mutable_vars(stmt.then_body)
                self._scan_mutable_vars(stmt.else_body)
            elif isinstance(stmt, WhileStmt):
                self._scan_mutable_vars(stmt.body)
            elif isinstance(stmt, DebugBlock):
                self._scan_mutable_vars(stmt.body)

    def _scan_forward_refs(self, stmts: list) -> None:
        """Mark SSA-eligible variables that are referenced before their LetStmt definition.

        This handles CSE ordering quirks where one CSE variable's value references
        another CSE variable whose LetStmt appears later in the statement list.
        Such variables need OpVariable to avoid undefined SSA references.
        """
        let_pos: dict[str, int] = {}
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, LetStmt):
                let_pos[stmt.name] = i

        for i, stmt in enumerate(stmts):
            refs: set[str] = set()
            if isinstance(stmt, LetStmt):
                self._expr_var_refs(stmt.value, refs)
            elif isinstance(stmt, AssignStmt):
                self._expr_var_refs(stmt.value, refs)
            elif isinstance(stmt, ReturnStmt):
                self._expr_var_refs(stmt.value, refs)
            elif isinstance(stmt, ExprStmt):
                self._expr_var_refs(getattr(stmt, 'expr', None), refs)
            elif isinstance(stmt, IfStmt):
                self._expr_var_refs(stmt.condition, refs)
            elif isinstance(stmt, ForStmt):
                self._expr_var_refs(stmt.init_value, refs)
                self._expr_var_refs(stmt.condition, refs)
            for name in refs:
                if name in let_pos and let_pos[name] > i:
                    self._mutable_vars.add(name)

        for stmt in stmts:
            if isinstance(stmt, IfStmt):
                self._scan_forward_refs(stmt.then_body)
                self._scan_forward_refs(stmt.else_body)
            elif isinstance(stmt, ForStmt):
                self._scan_forward_refs(stmt.body)
            elif isinstance(stmt, WhileStmt):
                self._scan_forward_refs(stmt.body)
            elif isinstance(stmt, DebugBlock):
                self._scan_forward_refs(stmt.body)

    @staticmethod
    def _expr_var_refs(expr, refs: set) -> None:
        """Collect all VarRef names from an expression tree."""
        if expr is None:
            return
        if isinstance(expr, VarRef):
            refs.add(expr.name)
        elif isinstance(expr, BinaryOp):
            SpvGenerator._expr_var_refs(expr.left, refs)
            SpvGenerator._expr_var_refs(expr.right, refs)
        elif isinstance(expr, UnaryOp):
            SpvGenerator._expr_var_refs(expr.operand, refs)
        elif isinstance(expr, CallExpr):
            SpvGenerator._expr_var_refs(expr.func, refs)
            for a in expr.args:
                SpvGenerator._expr_var_refs(a, refs)
        elif isinstance(expr, ConstructorExpr):
            for a in expr.args:
                SpvGenerator._expr_var_refs(a, refs)
        elif isinstance(expr, FieldAccess):
            SpvGenerator._expr_var_refs(expr.object, refs)
        elif isinstance(expr, SwizzleAccess):
            SpvGenerator._expr_var_refs(expr.object, refs)
        elif isinstance(expr, IndexAccess):
            SpvGenerator._expr_var_refs(expr.object, refs)
            SpvGenerator._expr_var_refs(expr.index, refs)
        elif isinstance(expr, TernaryExpr):
            SpvGenerator._expr_var_refs(expr.condition, refs)
            SpvGenerator._expr_var_refs(expr.then_expr, refs)
            SpvGenerator._expr_var_refs(expr.else_expr, refs)

    @staticmethod
    def _extract_assign_root_name(target) -> str | None:
        """Extract the root variable name from an assignment target."""
        if isinstance(target, AssignTarget):
            return SpvGenerator._extract_assign_root_name(target.expr)
        if isinstance(target, VarRef):
            return target.name
        if isinstance(target, (SwizzleAccess, FieldAccess)):
            return SpvGenerator._extract_assign_root_name(target.object)
        if isinstance(target, IndexAccess):
            return SpvGenerator._extract_assign_root_name(target.object)
        return None

    def generate(self) -> str:
        self.glsl_ext_id = self.reg.next_id()

        # Pre-scan for debug_print/assert to know if we need DebugPrintf extension
        for fn in self.stage.functions:
            if self._scan_for_debug_stmts(fn.body):
                self._has_debug_printf = True
                break

        # Reserve ID for DebugPrintf extension if needed
        if self._has_debug_printf:
            self._debug_printf_ext_id = self.reg.next_id()

        # Pre-allocate debug source ID so _emit_debug_line can reference it
        # during function body generation (before the header is assembled)
        if self.debug and self.source_name:
            self._debug_source_id = self.reg.next_id()

        # Initialize NonSemantic debug info emitter for RenderDoc support
        # Activated by passing source_text to generate_spirv (only when debug=True)
        if self.debug and self._dbg_source_text and self._enable_nonsemantic_debug:
            from luxc.codegen.debug_info import DebugInfoEmitter
            void_type = self.reg.void()
            self._dbg_emitter = DebugInfoEmitter(self.reg, self.source_name, self._dbg_source_text)
            self._dbg_emitter.init(void_type)

        # Pre-declare types we'll need
        self._declare_globals()

        # Initialize NonSemantic debug info declarations before function generation
        # (DebugFunction uses debug type IDs that must exist first)
        self._dbg_pre_fn_lines: list[str] = []
        if self._dbg_emitter:
            self._dbg_pre_fn_lines = self._dbg_emitter.emit_pre_function_decls()

        # Generate function bodies
        fn_code = []
        for fn in self.stage.functions:
            fn_code.extend(self._gen_function(fn))

        # Assemble sections
        lines = []

        # Header
        is_rt = self.stage.stage_type in _RT_STAGES
        is_mesh = self.stage.stage_type in _MESH_STAGES
        lines.append("; SPIR-V")
        lines.append("; Generated by luxc")
        lines.append("OpCapability Shader")
        if is_rt:
            lines.append("OpCapability RayTracingKHR")
        if is_mesh:
            lines.append("OpCapability MeshShadingEXT")
        # Check for half precision attribute on any function
        has_half_precision = any(
            "precision(half)" in fn.attributes
            for fn in self.stage.functions
        )
        if has_half_precision:
            lines.append("OpCapability Float16")
        has_storage_buffers = bool(getattr(self.stage, 'storage_buffers', []))
        has_bindless = bool(getattr(self.stage, 'bindless_texture_arrays', []))
        # All OpCapability instructions must come before any OpExtension
        if has_bindless:
            lines.append("OpCapability RuntimeDescriptorArray")
            lines.append("OpCapability ShaderNonUniform")
            lines.append("OpCapability SampledImageArrayNonUniformIndexing")
        if self._needs_image_query:
            lines.append("OpCapability ImageQuery")
        if self.assert_kill and self.stage.stage_type == "fragment" and self._has_debug_printf:
            lines.append("OpCapability DemoteToHelperInvocationEXT")
        # StorageImageExtendedFormats needed for Rg16f, R32f, etc.
        _EXTENDED_FMTS = {"Rg16f", "Rg32f", "R16f", "R32f", "R32i", "R32ui", "R11fG11fB10f"}
        needs_ext_fmt = any(
            getattr(si, 'format', 'rgba8').lower() not in ('rgba8', 'rgba16f', 'rgba32f')
            for si in getattr(self.stage, 'storage_images', [])
        )
        if needs_ext_fmt:
            lines.append("OpCapability StorageImageExtendedFormats")
        if has_storage_buffers:
            lines.append('OpExtension "SPV_KHR_storage_buffer_storage_class"')
        if is_rt:
            lines.append('OpExtension "SPV_KHR_ray_tracing"')
        if is_mesh:
            lines.append('OpExtension "SPV_EXT_mesh_shader"')
        if has_bindless:
            lines.append('OpExtension "SPV_EXT_descriptor_indexing"')
        if self.assert_kill and self.stage.stage_type == "fragment" and self._has_debug_printf:
            lines.append('OpExtension "SPV_EXT_demote_to_helper_invocation"')
        if self._has_debug_printf or self._dbg_emitter:
            lines.append('OpExtension "SPV_KHR_non_semantic_info"')
        lines.append(f"{self.glsl_ext_id} = OpExtInstImport \"GLSL.std.450\"")
        if self._has_debug_printf:
            lines.append(f'{self._debug_printf_ext_id} = OpExtInstImport "NonSemantic.DebugPrintf"')
        if self._dbg_emitter:
            lines.append(
                f'{self._dbg_emitter.ext_id} = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"'
            )
        lines.append("OpMemoryModel Logical GLSL450")

        # Entry point
        if is_rt:
            exec_model = _RT_EXEC_MODELS[self.stage.stage_type]
        elif is_mesh:
            exec_model = _MESH_EXEC_MODELS[self.stage.stage_type]
        elif self.stage.stage_type == "compute":
            exec_model = "GLCompute"
        elif self.stage.stage_type == "vertex":
            exec_model = "Vertex"
        else:
            exec_model = "Fragment"
        # Include ALL global variables in OpEntryPoint interface.
        # SPIR-V 1.4+ requires this, and spirv-opt/spirv-val enforce it
        # even for earlier versions under vulkan1.2+ target environments.
        iface = " ".join(self.interface_ids)
        lines.append(f"OpEntryPoint {exec_model} %main \"main\" {iface}")

        # Execution mode
        if self.stage.stage_type == "fragment":
            lines.append("OpExecutionMode %main OriginUpperLeft")

        if self.stage.stage_type == "mesh":
            defines = getattr(self.module, '_defines', {})
            wg_size = defines.get('workgroup_size', 32)
            max_verts = defines.get('max_vertices', 64)
            max_prims = defines.get('max_primitives', 124)
            lines.append(f"OpExecutionMode %main LocalSize {wg_size} 1 1")
            lines.append(f"OpExecutionMode %main OutputVertices {max_verts}")
            lines.append(f"OpExecutionMode %main OutputPrimitivesEXT {max_prims}")
            lines.append("OpExecutionMode %main OutputTrianglesEXT")
        elif self.stage.stage_type == "task":
            defines = getattr(self.module, '_defines', {})
            wg_size = defines.get('workgroup_size', 32)
            lines.append(f"OpExecutionMode %main LocalSize {wg_size} 1 1")
        elif self.stage.stage_type == "compute":
            defines = getattr(self.module, '_defines', {})
            wg_x = defines.get('workgroup_size_x', defines.get('workgroup_size', 64))
            wg_y = defines.get('workgroup_size_y', 1)
            wg_z = defines.get('workgroup_size_z', 1)
            lines.append(f"OpExecutionMode %main LocalSize {wg_x} {wg_y} {wg_z}")

        # Debug source info (OpString / OpSource) — must come after OpEntryPoint/OpExecutionMode
        if self._debug_source_id is not None:
            lines.append(f'{self._debug_source_id} = OpString "{self.source_name}"')
            lines.append(f"OpSource GLSL 450 {self._debug_source_id}")

        # DebugPrintf format strings (OpString)
        for fmt_str, str_id in self._debug_string_ids.items():
            lines.append(f'{str_id} = OpString "{fmt_str}"')

        # Debug names
        lines.append('OpName %main "main"')
        for name, vid in self.var_map.items():
            lines.append(f'OpName {vid} "{name}"')
        for bname, sid in self.uniform_struct_ids.items():
            lines.append(f'OpName {sid} "{bname}"')
        for bname, sid in self.push_struct_ids.items():
            lines.append(f'OpName {sid} "{bname}"')
        # Local variable OpNames (debug mode only)
        if self.debug:
            for name, var_id in self._all_local_names:
                lines.append(f'OpName {var_id} "{name}"')
        if self.per_vertex_struct_id:
            lines.append(f'OpName {self.per_vertex_struct_id} "gl_PerVertex"')
            lines.append(f'OpName {self.per_vertex_var_id} ""')

        # OpMemberName for struct fields (RenderDoc / spirv-cross readability)
        for ub in self.stage.uniforms:
            sid = self.uniform_struct_ids.get(ub.name)
            if sid:
                for i, field in enumerate(ub.fields):
                    lines.append(f'OpMemberName {sid} {i} "{field.name}"')
        for pb in self.stage.push_constants:
            sid = self.push_struct_ids.get(pb.name)
            if sid:
                for i, field in enumerate(pb.fields):
                    lines.append(f'OpMemberName {sid} {i} "{field.name}"')
        if self.per_vertex_struct_id:
            lines.append(f'OpMemberName {self.per_vertex_struct_id} 0 "gl_Position"')
            lines.append(f'OpMemberName {self.per_vertex_struct_id} 1 "gl_PointSize"')
            lines.append(f'OpMemberName {self.per_vertex_struct_id} 2 "gl_ClipDistance"')
            lines.append(f'OpMemberName {self.per_vertex_struct_id} 3 "gl_CullDistance"')
        # Bindless material struct member names
        if "struct_BindlessMaterialData" in self.reg._types:
            bms_id = self.reg._types["struct_BindlessMaterialData"]
            bms_fields = getattr(self, '_bindless_material_dynamic_fields',
                                 _BINDLESS_MATERIAL_FIELDS)
            for i, (fname, _) in enumerate(bms_fields):
                lines.append(f'OpMemberName {bms_id} {i} "{fname}"')

        # Decorations
        lines.extend(self.decorations)

        # Type + constant declarations (unified, dependency-ordered)
        lines.extend(self.reg.emit_declarations())

        # Global variables
        lines.extend(self.global_vars)

        # NonSemantic debug info declarations (after types/globals, before functions)
        if self._dbg_emitter:
            lines.extend(self._dbg_pre_fn_lines)

        # Functions
        lines.extend(fn_code)

        return "\n".join(lines) + "\n"

    def _declare_globals(self):
        """Declare all global variables, types, and decorations."""

        # --- Input variables ---
        for inp in self.stage.inputs:
            type_id = self.reg.lux_type_to_spirv(inp.type_name)
            ptr_type = self.reg.pointer("Input", type_id)
            var_id = self.reg.next_id()
            self.global_vars.append(f"{var_id} = OpVariable {ptr_type} Input")
            self.decorations.append(f"OpDecorate {var_id} Location {inp.location}")
            self.var_map[inp.name] = var_id
            self.var_types[inp.name] = inp.type_name
            self.var_storage[inp.name] = "Input"
            self.interface_ids.append(var_id)
            self.interface_storage[var_id] = "Input"

        # --- Output variables ---
        # mesh outputs are arrays, handled separately in the mesh-specific block
        for out in (self.stage.outputs if self.stage.stage_type != "mesh" else []):
            type_id = self.reg.lux_type_to_spirv(out.type_name)
            ptr_type = self.reg.pointer("Output", type_id)
            var_id = self.reg.next_id()
            self.global_vars.append(f"{var_id} = OpVariable {ptr_type} Output")
            self.decorations.append(f"OpDecorate {var_id} Location {out.location}")
            self.var_map[out.name] = var_id
            self.var_types[out.name] = out.type_name
            self.var_storage[out.name] = "Output"
            self.interface_ids.append(var_id)
            self.interface_storage[var_id] = "Output"

        # --- gl_PerVertex for vertex shaders (builtin_position) ---
        if self.stage.stage_type == "vertex":
            vec4_id = self.reg.lux_type_to_spirv("vec4")
            float_id = self.reg.float32()
            uint_id = self.reg.uint32()
            # gl_PerVertex struct: { vec4 gl_Position, float gl_PointSize, float gl_ClipDistance[], float gl_CullDistance[] }
            # Minimal: just gl_Position
            arr_1 = self._make_array_type(float_id, 1)
            struct_id = self.reg.struct("gl_PerVertex", [vec4_id, float_id, arr_1, arr_1])
            self.per_vertex_struct_id = struct_id
            ptr_type = self.reg.pointer("Output", struct_id)
            var_id = self.reg.next_id()
            self.global_vars.append(f"{var_id} = OpVariable {ptr_type} Output")
            self.per_vertex_var_id = var_id
            self.interface_ids.append(var_id)
            self.interface_storage[var_id] = "Output"

            # Decorations for gl_PerVertex
            self.decorations.append(f"OpDecorate {struct_id} Block")
            self.decorations.append(f"OpMemberDecorate {struct_id} 0 BuiltIn Position")
            self.decorations.append(f"OpMemberDecorate {struct_id} 1 BuiltIn PointSize")
            self.decorations.append(f"OpMemberDecorate {struct_id} 2 BuiltIn ClipDistance")
            self.decorations.append(f"OpMemberDecorate {struct_id} 3 BuiltIn CullDistance")

            self.var_map["builtin_position"] = var_id
            self.var_types["builtin_position"] = "vec4"
            self.var_storage["builtin_position"] = "Output"

        # --- vertex_index and instance_index builtins for vertex shaders ---
        if self.stage.stage_type == "vertex":
            for bi_name, bi_builtin in [("vertex_index", "VertexIndex"), ("instance_index", "InstanceIndex")]:
                uint_type = self.reg.uint32()
                ptr_type = self.reg.pointer("Input", uint_type)
                var_id = self.reg.next_id()
                self.global_vars.append(f"{var_id} = OpVariable {ptr_type} Input")
                self.decorations.append(f"OpDecorate {var_id} BuiltIn {bi_builtin}")
                self.var_map[bi_name] = var_id
                self.var_types[bi_name] = "uint"
                self.var_storage[bi_name] = "Input"
                self.interface_ids.append(var_id)
                self.interface_storage[var_id] = "Input"

        # --- Uniform blocks ---
        for ub in self.stage.uniforms:
            member_types = []
            for i, field in enumerate(ub.fields):
                tid = self.reg.lux_type_to_spirv(field.type_name)
                member_types.append(tid)
                self.uniform_field_indices[field.name] = i
                self.uniform_block_for_field[field.name] = ub.name
                self.var_types[field.name] = field.type_name

            struct_id = self.reg.struct(ub.name, member_types)
            self.uniform_struct_ids[ub.name] = struct_id
            ptr_type = self.reg.pointer("Uniform", struct_id)
            var_id = self.reg.next_id()
            self.global_vars.append(f"{var_id} = OpVariable {ptr_type} Uniform")
            self.uniform_var_ids[ub.name] = var_id
            self.interface_ids.append(var_id)
            self.interface_storage[var_id] = "Uniform"

            # Decorations
            self.decorations.append(f"OpDecorate {struct_id} Block")
            self.decorations.append(f"OpDecorate {var_id} DescriptorSet {ub.set_number}")
            self.decorations.append(f"OpDecorate {var_id} Binding {ub.binding}")

            # Member offsets (std140)
            offsets = compute_std140_offsets(ub.fields)
            for i, offset in enumerate(offsets):
                self.decorations.append(f"OpMemberDecorate {struct_id} {i} Offset {offset}")
                # MatrixStride for matrix members
                if ub.fields[i].type_name in ("mat2", "mat3", "mat4"):
                    self.decorations.append(f"OpMemberDecorate {struct_id} {i} ColMajor")
                    self.decorations.append(f"OpMemberDecorate {struct_id} {i} MatrixStride 16")

        # --- Push constant blocks ---
        for pb in self.stage.push_constants:
            member_types = []
            for i, field in enumerate(pb.fields):
                tid = self.reg.lux_type_to_spirv(field.type_name)
                member_types.append(tid)
                self.push_field_indices[field.name] = i
                self.push_block_for_field[field.name] = pb.name
                self.var_types[field.name] = field.type_name

            struct_id = self.reg.struct(pb.name, member_types)
            self.push_struct_ids[pb.name] = struct_id
            ptr_type = self.reg.pointer("PushConstant", struct_id)
            var_id = self.reg.next_id()
            self.global_vars.append(f"{var_id} = OpVariable {ptr_type} PushConstant")
            self.push_var_ids[pb.name] = var_id
            self.interface_ids.append(var_id)
            self.interface_storage[var_id] = "PushConstant"

            self.decorations.append(f"OpDecorate {struct_id} Block")
            offsets = compute_std140_offsets(pb.fields)
            for i, offset in enumerate(offsets):
                self.decorations.append(f"OpMemberDecorate {struct_id} {i} Offset {offset}")
                # MatrixStride for matrix members in push constants
                if pb.fields[i].type_name in ("mat2", "mat3", "mat4"):
                    self.decorations.append(f"OpMemberDecorate {struct_id} {i} ColMajor")
                    self.decorations.append(f"OpMemberDecorate {struct_id} {i} MatrixStride 16")

        # --- Samplers (separate sampler + texture for WebGPU compatibility) ---
        for sam in self.stage.samplers:
            # Sampler state variable
            sampler_type = self.reg.sampler_type()
            sampler_ptr = self.reg.pointer("UniformConstant", sampler_type)
            sampler_var = self.reg.next_id()
            self.global_vars.append(f"{sampler_var} = OpVariable {sampler_ptr} UniformConstant")
            self.decorations.append(f"OpDecorate {sampler_var} DescriptorSet {sam.set_number}")
            self.decorations.append(f"OpDecorate {sampler_var} Binding {sam.binding}")
            self.interface_ids.append(sampler_var)
            self.interface_storage[sampler_var] = "UniformConstant"

            # Texture image variable (2D, Cube, 2DArray, or CubeArray depending on sampler type)
            sam_type_name = getattr(sam, 'type_name', 'sampler2d')
            if sam_type_name == 'samplerCube':
                image_type = self.reg.cube_image_type()
            elif sam_type_name == 'sampler2DArray':
                image_type = self.reg.image_2d_array_type()
            elif sam_type_name == 'samplerCubeArray':
                image_type = self.reg.image_cube_array_type()
            else:
                image_type = self.reg.image_type()
            image_ptr = self.reg.pointer("UniformConstant", image_type)
            texture_var = self.reg.next_id()
            self.global_vars.append(f"{texture_var} = OpVariable {image_ptr} UniformConstant")
            self.decorations.append(f"OpDecorate {texture_var} DescriptorSet {sam.set_number}")
            self.decorations.append(f"OpDecorate {texture_var} Binding {sam.texture_binding}")
            self.interface_ids.append(texture_var)
            self.interface_storage[texture_var] = "UniformConstant"

            # Store both var IDs for use in sample() codegen
            self.var_map[sam.name + ".__sampler"] = sampler_var
            self.var_map[sam.name + ".__texture"] = texture_var
            self.var_map[sam.name + ".__image_type"] = image_type
            self.var_types[sam.name] = sam_type_name
            self.var_storage[sam.name] = "UniformConstant"

        # --- RT: Ray payload variables ---
        for i, rp in enumerate(self.stage.ray_payloads):
            type_id = self.reg.lux_type_to_spirv(rp.type_name)
            # Outgoing payloads use RayPayloadKHR, incoming use IncomingRayPayloadKHR
            # In raygen: RayPayloadKHR; in closest_hit/any_hit/miss: IncomingRayPayloadKHR
            if self.stage.stage_type == "raygen":
                storage = "RayPayloadKHR"
            else:
                storage = "IncomingRayPayloadKHR"
            ptr_type = self.reg.pointer(storage, type_id)
            var_id = self.reg.next_id()
            self.global_vars.append(f"{var_id} = OpVariable {ptr_type} {storage}")
            loc = rp.location if rp.location is not None else i
            self.decorations.append(f"OpDecorate {var_id} Location {loc}")
            self.var_map[rp.name] = var_id
            self.var_types[rp.name] = rp.type_name
            # Track payload location -> variable ID for trace_ray
            if not hasattr(self, '_payload_vars'):
                self._payload_vars = {}
            self._payload_vars[loc] = var_id
            self.var_storage[rp.name] = storage
            self.interface_ids.append(var_id)
            self.interface_storage[var_id] = storage

        # --- RT: Hit attribute variables ---
        for ha in self.stage.hit_attributes:
            type_id = self.reg.lux_type_to_spirv(ha.type_name)
            storage = "HitAttributeKHR"
            ptr_type = self.reg.pointer(storage, type_id)
            var_id = self.reg.next_id()
            self.global_vars.append(f"{var_id} = OpVariable {ptr_type} {storage}")
            self.var_map[ha.name] = var_id
            self.var_types[ha.name] = ha.type_name
            self.var_storage[ha.name] = storage
            self.interface_ids.append(var_id)
            self.interface_storage[var_id] = storage

        # --- RT: Callable data variables ---
        for i, cd in enumerate(self.stage.callable_data):
            type_id = self.reg.lux_type_to_spirv(cd.type_name)
            if self.stage.stage_type == "callable":
                storage = "IncomingCallableDataKHR"
            else:
                storage = "CallableDataKHR"
            ptr_type = self.reg.pointer(storage, type_id)
            var_id = self.reg.next_id()
            self.global_vars.append(f"{var_id} = OpVariable {ptr_type} {storage}")
            loc = cd.location if cd.location is not None else i
            self.decorations.append(f"OpDecorate {var_id} Location {loc}")
            self.var_map[cd.name] = var_id
            self.var_types[cd.name] = cd.type_name
            self.var_storage[cd.name] = storage
            self.interface_ids.append(var_id)
            self.interface_storage[var_id] = storage

        # --- RT: Acceleration structure variables ---
        for accel in self.stage.accel_structs:
            type_id = self.reg.acceleration_structure_type()
            storage = "UniformConstant"
            ptr_type = self.reg.pointer(storage, type_id)
            var_id = self.reg.next_id()
            self.global_vars.append(f"{var_id} = OpVariable {ptr_type} {storage}")
            if accel.set_number is not None:
                self.decorations.append(f"OpDecorate {var_id} DescriptorSet {accel.set_number}")
            if accel.binding is not None:
                self.decorations.append(f"OpDecorate {var_id} Binding {accel.binding}")
            self.var_map[accel.name] = var_id
            self.var_types[accel.name] = "acceleration_structure"
            self.var_storage[accel.name] = storage
            self.interface_ids.append(var_id)
            self.interface_storage[var_id] = storage

        # --- RT: Storage image variables ---
        # Map Lux format names to SPIR-V image format enums
        _FMT_MAP = {
            "rgba8": "Rgba8", "rgba16f": "Rgba16f", "rgba32f": "Rgba32f",
            "rg16f": "Rg16f", "rg32f": "Rg32f",
            "r16f": "R16f", "r32f": "R32f", "r32i": "R32i", "r32ui": "R32ui",
            "r11g11b10f": "R11fG11fB10f",
        }
        for si in getattr(self.stage, 'storage_images', []):
            fmt = getattr(si, 'format', 'rgba8') or 'rgba8'
            spv_fmt = _FMT_MAP.get(fmt.lower(), "Rgba8")
            f32 = self.reg.float32()
            img_type_key = f"storage_image_2d_{spv_fmt}"
            if img_type_key not in self.reg._types:
                img_type_id = self.reg.next_id()
                self.reg._types[img_type_key] = img_type_id
                self.reg._decls.append(f"{img_type_id} = OpTypeImage {f32} 2D 0 0 0 2 {spv_fmt}")
            else:
                img_type_id = self.reg._types[img_type_key]
            storage = "UniformConstant"
            ptr_type = self.reg.pointer(storage, img_type_id)
            var_id = self.reg.next_id()
            self.global_vars.append(f"{var_id} = OpVariable {ptr_type} {storage}")
            if si.set_number is not None:
                self.decorations.append(f"OpDecorate {var_id} DescriptorSet {si.set_number}")
            if si.binding is not None:
                self.decorations.append(f"OpDecorate {var_id} Binding {si.binding}")
            self.decorations.append(f"OpDecorate {var_id} NonReadable")
            self.var_map[si.name] = var_id
            self.var_types[si.name] = "storage_image"
            self._storage_image_type_ids[si.name] = img_type_id
            self.var_storage[si.name] = storage
            self.interface_ids.append(var_id)
            self.interface_storage[var_id] = storage

        # --- Storage buffer variables (runtime arrays) ---
        # Struct element type registry for BindlessMaterialData
        if not hasattr(self, '_ssbo_struct_fields'):
            self._ssbo_struct_fields = {}  # buffer name -> list[(field_name, field_type)]

        for sb in getattr(self.stage, 'storage_buffers', []):
            # Check if element type is a known struct
            if sb.element_type == "BindlessMaterialData":
                elem_type_id, stride = self._declare_bindless_material_struct()
                self._ssbo_struct_fields[sb.name] = getattr(
                    self, '_bindless_material_dynamic_fields', _BINDLESS_MATERIAL_FIELDS)
            elif sb.element_type == "LightData":
                elem_type_id, stride = self._declare_light_data_struct()
                self._ssbo_struct_fields[sb.name] = _LIGHT_DATA_FIELDS
            elif sb.element_type == "ShadowEntry":
                elem_type_id, stride = self._declare_shadow_entry_struct()
                self._ssbo_struct_fields[sb.name] = _SHADOW_ENTRY_FIELDS
            else:
                elem_type_id = self.reg.lux_type_to_spirv(sb.element_type)

                # Compute array stride from element type
                _ELEM_BYTE_SIZES = {
                    "scalar": 4, "int": 4, "uint": 4, "bool": 4,
                    "vec2": 8, "vec3": 12, "vec4": 16,
                    "ivec2": 8, "ivec3": 12, "ivec4": 16,
                    "uvec2": 8, "uvec3": 12, "uvec4": 16,
                    "mat2": 32, "mat3": 48, "mat4": 64,
                }
                # std430 array stride: element aligned to its own size,
                # but vec3 has stride 16 (padded to vec4 alignment in arrays)
                _ELEM_STRIDE = dict(_ELEM_BYTE_SIZES)
                _ELEM_STRIDE["vec3"] = 16
                _ELEM_STRIDE["ivec3"] = 16
                _ELEM_STRIDE["uvec3"] = 16
                stride = _ELEM_STRIDE.get(sb.element_type, 16)

            # OpTypeRuntimeArray
            rtarr_key = f"_rtarr_{sb.element_type}"
            if rtarr_key not in self.reg._types:
                rtarr_id = self.reg.next_id()
                self.reg._types[rtarr_key] = rtarr_id
                self.reg._decls.append(f"{rtarr_id} = OpTypeRuntimeArray {elem_type_id}")
                self.decorations.append(f"OpDecorate {rtarr_id} ArrayStride {stride}")
            else:
                rtarr_id = self.reg._types[rtarr_key]

            # Struct wrapping the runtime array
            struct_name = f"_SB_{sb.name}"
            struct_id = self.reg.struct(struct_name, [rtarr_id])

            # Variable in StorageBuffer storage class
            ptr_type = self.reg.pointer("StorageBuffer", struct_id)
            var_id = self.reg.next_id()
            self.global_vars.append(f"{var_id} = OpVariable {ptr_type} StorageBuffer")

            # Decorations
            self.decorations.append(f"OpDecorate {struct_id} Block")
            self.decorations.append(f"OpMemberDecorate {struct_id} 0 Offset 0")
            if self.stage.stage_type != "compute":
                self.decorations.append(f"OpMemberDecorate {struct_id} 0 NonWritable")
                self.decorations.append(f"OpDecorate {var_id} NonWritable")
            if sb.set_number is not None:
                self.decorations.append(f"OpDecorate {var_id} DescriptorSet {sb.set_number}")
            if sb.binding is not None:
                self.decorations.append(f"OpDecorate {var_id} Binding {sb.binding}")

            self.storage_buffer_var_ids[sb.name] = var_id
            self.storage_buffer_element_types[sb.name] = sb.element_type
            self.var_map[sb.name] = var_id
            self.var_types[sb.name] = f"_rtarr_{sb.element_type}"
            self.var_storage[sb.name] = "StorageBuffer"
            self.interface_ids.append(var_id)
            self.interface_storage[var_id] = "StorageBuffer"

        # --- Bindless texture arrays (runtime array of combined image sampler) ---
        for bta in getattr(self.stage, 'bindless_texture_arrays', []):
            # Runtime array of combined image sampler
            cis_type = self.reg.combined_image_sampler_type()
            rtarr_type = self.reg.combined_image_sampler_runtime_array()
            ptr_type = self.reg.pointer("UniformConstant", rtarr_type)
            var_id = self.reg.next_id()
            self.global_vars.append(f"{var_id} = OpVariable {ptr_type} UniformConstant")
            if bta.set_number is not None:
                self.decorations.append(f"OpDecorate {var_id} DescriptorSet {bta.set_number}")
            if bta.binding is not None:
                self.decorations.append(f"OpDecorate {var_id} Binding {bta.binding}")
            self.var_map[bta.name] = var_id
            self.var_types[bta.name] = "_bindless_texture_array"
            self.var_storage[bta.name] = "UniformConstant"
            self.interface_ids.append(var_id)
            self.interface_storage[var_id] = "UniformConstant"

        # --- RT: Built-in variables ---
        if self.stage.stage_type in _RT_STAGES:
            emitted_builtins: dict[str, str] = {}  # spv_builtin -> var_id
            for builtin_name, (btype, spv_builtin, valid_stages) in _RT_BUILTINS.items():
                if self.stage.stage_type in valid_stages:
                    # Deduplicate: if same SPIR-V builtin already emitted, reuse its variable
                    if spv_builtin in emitted_builtins:
                        var_id = emitted_builtins[spv_builtin]
                        self.var_map[builtin_name] = var_id
                        self.var_types[builtin_name] = btype
                        self.var_storage[builtin_name] = "Input"
                        continue
                    type_id = self.reg.lux_type_to_spirv(btype)
                    ptr_type = self.reg.pointer("Input", type_id)
                    var_id = self.reg.next_id()
                    self.global_vars.append(f"{var_id} = OpVariable {ptr_type} Input")
                    self.decorations.append(f"OpDecorate {var_id} BuiltIn {spv_builtin}")
                    self.var_map[builtin_name] = var_id
                    self.var_types[builtin_name] = btype
                    self.var_storage[builtin_name] = "Input"
                    self.interface_ids.append(var_id)
                    self.interface_storage[var_id] = "Input"
                    emitted_builtins[spv_builtin] = var_id

        # --- Workgroup built-in variables (mesh, task, compute) ---
        if self.stage.stage_type in _WORKGROUP_STAGES:
            emitted_builtins: dict[str, str] = {}
            for builtin_name, (btype, spv_builtin, valid_stages) in _WORKGROUP_BUILTINS.items():
                if self.stage.stage_type in valid_stages:
                    if spv_builtin in emitted_builtins:
                        var_id = emitted_builtins[spv_builtin]
                        self.var_map[builtin_name] = var_id
                        self.var_types[builtin_name] = btype
                        self.var_storage[builtin_name] = "Input"
                        continue
                    type_id = self.reg.lux_type_to_spirv(btype)
                    ptr_type = self.reg.pointer("Input", type_id)
                    var_id = self.reg.next_id()
                    self.global_vars.append(f"{var_id} = OpVariable {ptr_type} Input")
                    self.decorations.append(f"OpDecorate {var_id} BuiltIn {spv_builtin}")
                    self.var_map[builtin_name] = var_id
                    self.var_types[builtin_name] = btype
                    self.var_storage[builtin_name] = "Input"
                    self.interface_ids.append(var_id)
                    self.interface_storage[var_id] = "Input"
                    emitted_builtins[spv_builtin] = var_id

        # --- Shared memory variables (Workgroup storage class) ---
        if self.stage.stage_type == "compute":
            for sd in getattr(self.stage, 'shared_decls', []):
                elem_type_id = self.reg.lux_type_to_spirv(sd.type_name)
                if sd.array_size is not None:
                    # Fixed-size array: OpTypeArray
                    arr_type = self._make_array_type(elem_type_id, sd.array_size)
                    ptr_type = self.reg.pointer("Workgroup", arr_type)
                    var_id = self.reg.next_id()
                    self.global_vars.append(f"{var_id} = OpVariable {ptr_type} Workgroup")
                else:
                    # Scalar shared variable
                    ptr_type = self.reg.pointer("Workgroup", elem_type_id)
                    var_id = self.reg.next_id()
                    self.global_vars.append(f"{var_id} = OpVariable {ptr_type} Workgroup")

                self.shared_var_ids[sd.name] = var_id
                self.shared_element_types[sd.name] = sd.type_name
                self.shared_array_sizes[sd.name] = sd.array_size
                self.var_map[sd.name] = var_id
                self.var_types[sd.name] = sd.type_name
                self.var_storage[sd.name] = "Workgroup"
                self.interface_ids.append(var_id)
                self.interface_storage[var_id] = "Workgroup"

        # --- Mesh: Output arrays (gl_MeshPerVertexEXT, PrimitiveTriangleIndicesEXT) ---
        if self.stage.stage_type == "mesh":
            defines = getattr(self.module, '_defines', {})
            max_verts = defines.get('max_vertices', 64)
            max_prims = defines.get('max_primitives', 124)

            # gl_MeshPerVertexEXT block { vec4 gl_Position; }[max_vertices]
            vec4_id = self.reg.lux_type_to_spirv("vec4")
            float_id = self.reg.float32()
            arr_1 = self._make_array_type(float_id, 1)
            per_vert_struct = self.reg.struct("gl_MeshPerVertexEXT", [vec4_id, float_id, arr_1, arr_1])
            per_vert_arr = self._make_array_type(per_vert_struct, max_verts)
            ptr_per_vert_arr = self.reg.pointer("Output", per_vert_arr)
            per_vert_var = self.reg.next_id()
            self.global_vars.append(f"{per_vert_var} = OpVariable {ptr_per_vert_arr} Output")
            self.interface_ids.append(per_vert_var)
            self.interface_storage[per_vert_var] = "Output"

            self.decorations.append(f"OpDecorate {per_vert_struct} Block")
            self.decorations.append(f"OpMemberDecorate {per_vert_struct} 0 BuiltIn Position")
            self.decorations.append(f"OpMemberDecorate {per_vert_struct} 1 BuiltIn PointSize")
            self.decorations.append(f"OpMemberDecorate {per_vert_struct} 2 BuiltIn ClipDistance")
            self.decorations.append(f"OpMemberDecorate {per_vert_struct} 3 BuiltIn CullDistance")

            self.var_map["gl_MeshVerticesEXT"] = per_vert_var
            self.var_types["gl_MeshVerticesEXT"] = "gl_MeshPerVertexEXT_arr"
            self.var_storage["gl_MeshVerticesEXT"] = "Output"
            self.per_vertex_struct_id = per_vert_struct
            self.per_vertex_var_id = per_vert_var
            # Store for use in mesh output position writes
            self._mesh_per_vert_struct = per_vert_struct
            self._mesh_max_vertices = max_verts

            # PrimitiveTriangleIndicesEXT: uvec3[max_primitives]
            uvec3_id = self.reg.lux_type_to_spirv("uvec3")
            tri_idx_arr = self._make_array_type(uvec3_id, max_prims)
            ptr_tri_idx_arr = self.reg.pointer("Output", tri_idx_arr)
            tri_idx_var = self.reg.next_id()
            self.global_vars.append(f"{tri_idx_var} = OpVariable {ptr_tri_idx_arr} Output")
            self.decorations.append(f"OpDecorate {tri_idx_var} BuiltIn PrimitiveTriangleIndicesEXT")
            self.interface_ids.append(tri_idx_var)
            self.interface_storage[tri_idx_var] = "Output"

            self.var_map["gl_PrimitiveTriangleIndicesEXT"] = tri_idx_var
            self.var_types["gl_PrimitiveTriangleIndicesEXT"] = "uvec3_arr"
            self.var_storage["gl_PrimitiveTriangleIndicesEXT"] = "Output"

            # Per-vertex output arrays (from mesh_output declarations, or from stage outputs)
            for out in self.stage.outputs:
                type_id = self.reg.lux_type_to_spirv(out.type_name)
                out_arr = self._make_array_type(type_id, max_verts)
                ptr_out_arr = self.reg.pointer("Output", out_arr)
                var_id = self.reg.next_id()
                self.global_vars.append(f"{var_id} = OpVariable {ptr_out_arr} Output")
                self.decorations.append(f"OpDecorate {var_id} Location {out.location}")
                self.var_map[out.name] = var_id
                self.var_types[out.name] = out.type_name
                self.var_storage[out.name] = "Output"
                self.interface_ids.append(var_id)
                self.interface_storage[var_id] = "Output"

        # --- Task: Task payload output variable ---
        if self.stage.stage_type == "task":
            for tp in getattr(self.stage, 'task_payloads', []):
                type_id = self.reg.lux_type_to_spirv(tp.type_name)
                storage = "TaskPayloadWorkgroupEXT"
                ptr_type = self.reg.pointer(storage, type_id)
                var_id = self.reg.next_id()
                self.global_vars.append(f"{var_id} = OpVariable {ptr_type} {storage}")
                self.var_map[tp.name] = var_id
                self.var_types[tp.name] = tp.type_name
                self.var_storage[tp.name] = storage
                self.interface_ids.append(var_id)
                self.interface_storage[var_id] = storage

        # --- Specialization constants ---
        for i, sc in enumerate(self.module.spec_constants):
            spec_id = sc.spec_id if sc.spec_id is not None else i
            type_id = self.reg.lux_type_to_spirv(sc.type_name)
            var_id = self.reg.next_id()
            # Evaluate default value for the OpSpecConstant
            default_val = self._eval_spec_const_default(sc)
            self.reg._decls.append(f"{var_id} = OpSpecConstant {type_id} {default_val}")
            self.decorations.append(f"OpDecorate {var_id} SpecId {spec_id}")
            self.spec_const_ids[sc.name] = var_id
            self.var_types[sc.name] = sc.type_name

    def _eval_spec_const_default(self, sc: SpecConstDecl) -> str:
        """Evaluate the default value of a specialization constant to a literal."""
        expr = sc.default_value
        if isinstance(expr, NumberLit):
            if sc.type_name in ("int", "uint"):
                return str(int(float(expr.value)))
            return str(float(expr.value))
        if isinstance(expr, UnaryOp) and expr.op == "-" and isinstance(expr.operand, NumberLit):
            if sc.type_name in ("int", "uint"):
                return str(-int(float(expr.operand.value)))
            return str(-float(expr.operand.value))
        if isinstance(expr, BoolLit):
            return "1" if expr.value else "0"
        # Fallback: 0
        return "0"

    def _declare_bindless_material_struct(self) -> tuple[str, int]:
        """Declare the BindlessMaterialData struct type in SPIR-V.

        Returns (struct_type_id, stride_in_bytes).
        """
        key = "BindlessMaterialData"
        if f"struct_{key}" in self.reg._types:
            # Return cached — stride depends on whether extra properties were added
            cached_stride = getattr(self, '_bindless_material_stride', 128)
            return self.reg._types[f"struct_{key}"], cached_stride

        # Start with base fields
        fields = list(_BINDLESS_MATERIAL_FIELDS)

        # Append extra properties if present (from surface properties)
        extra = getattr(self.stage, '_bindless_extra_properties', None)
        if extra:
            for name, type_name in extra:
                if type_name == "vec3":
                    fields.append((f"_pad_{name}", "scalar"))
                fields.append((name, type_name))
            # Pad to 16-byte alignment
            while len(fields) % 4 != 0:
                fields.append((f"_pad_end_{len(fields)}", "scalar"))

        # Store the dynamic fields list for OpMemberName emission
        self._bindless_material_dynamic_fields = fields

        # Build member types
        member_types = []
        for fname, ftype in fields:
            tid = self.reg.lux_type_to_spirv(ftype)
            member_types.append(tid)

        struct_id = self.reg.struct(key, member_types)

        # Compute std430 offsets and add member decorations
        offset = 0
        _STD430 = {
            "scalar": (4, 4), "int": (4, 4), "uint": (4, 4),
            "vec2": (8, 8), "vec3": (12, 16), "vec4": (16, 16),
            "mat4": (64, 16),
        }
        for i, (fname, ftype) in enumerate(fields):
            size, align = _STD430.get(ftype, (4, 4))
            offset = (offset + align - 1) & ~(align - 1)
            self.decorations.append(f"OpMemberDecorate {struct_id} {i} Offset {offset}")
            offset += size

        # Stride = round up to largest alignment (16 for vec4)
        stride = (offset + 15) & ~15
        self._bindless_material_stride = stride
        return struct_id, stride

    def _declare_light_data_struct(self) -> tuple[str, int]:
        """Declare the LightData struct type in SPIR-V.
        Returns (struct_type_id, stride_in_bytes).
        """
        key = "LightData"
        if f"struct_{key}" in self.reg._types:
            return self.reg._types[f"struct_{key}"], 64  # cached

        member_types = []
        for fname, ftype in _LIGHT_DATA_FIELDS:
            tid = self.reg.lux_type_to_spirv(ftype)
            member_types.append(tid)

        struct_id = self.reg.struct(key, member_types)

        offset = 0
        _STD430 = {
            "scalar": (4, 4), "int": (4, 4), "uint": (4, 4),
            "vec2": (8, 8), "vec3": (12, 16), "vec4": (16, 16),
            "mat4": (64, 16),
        }
        for i, (fname, ftype) in enumerate(_LIGHT_DATA_FIELDS):
            size, align = _STD430.get(ftype, (4, 4))
            offset = (offset + align - 1) & ~(align - 1)
            self.decorations.append(f"OpMemberDecorate {struct_id} {i} Offset {offset}")
            if ftype == "mat4":
                self.decorations.append(f"OpMemberDecorate {struct_id} {i} ColMajor")
                self.decorations.append(f"OpMemberDecorate {struct_id} {i} MatrixStride 16")
            offset += size

        stride = (offset + 15) & ~15
        return struct_id, stride

    def _declare_shadow_entry_struct(self) -> tuple[str, int]:
        """Declare the ShadowEntry struct type in SPIR-V.
        Returns (struct_type_id, stride_in_bytes).
        """
        key = "ShadowEntry"
        if f"struct_{key}" in self.reg._types:
            return self.reg._types[f"struct_{key}"], 80  # cached

        member_types = []
        for fname, ftype in _SHADOW_ENTRY_FIELDS:
            tid = self.reg.lux_type_to_spirv(ftype)
            member_types.append(tid)

        struct_id = self.reg.struct(key, member_types)

        offset = 0
        _STD430 = {
            "scalar": (4, 4), "int": (4, 4), "uint": (4, 4),
            "vec2": (8, 8), "vec3": (12, 16), "vec4": (16, 16),
            "mat4": (64, 16),
        }
        for i, (fname, ftype) in enumerate(_SHADOW_ENTRY_FIELDS):
            size, align = _STD430.get(ftype, (4, 4))
            offset = (offset + align - 1) & ~(align - 1)
            self.decorations.append(f"OpMemberDecorate {struct_id} {i} Offset {offset}")
            if ftype == "mat4":
                self.decorations.append(f"OpMemberDecorate {struct_id} {i} ColMajor")
                self.decorations.append(f"OpMemberDecorate {struct_id} {i} MatrixStride 16")
            offset += size

        stride = (offset + 15) & ~15
        return struct_id, stride

    def _sampler_image_type(self, sampler_arg) -> str:
        """Return the OpTypeImage id for a sampler argument, for use with OpImage."""
        sam_name = None
        if isinstance(sampler_arg, VarRef):
            sam_name = sampler_arg.name
        if sam_name:
            img_key = sam_name + ".__image_type"
            if img_key in self.var_map:
                return self.var_map[img_key]
        # Fallback: use the default 2D image type
        return self.reg.image_type()

    def _make_array_type(self, elem_type: str, length: int) -> str:
        length_id = self.reg.const_uint(length)
        return self.reg.array(elem_type, length_id)

    def _gen_function(self, fn: FunctionDef) -> list[str]:
        lines = []
        void_type = self.reg.void()
        fn_type = self.reg.function_type(void_type)

        lines.append(f"%main = OpFunction {void_type} None {fn_type}")
        label = self.reg.next_id()
        lines.append(f"{label} = OpLabel")

        # Local variables (let statements need Function storage class pointers)
        self.local_vars: dict[str, str] = {}  # name -> %id (pointer)
        self.local_types: dict[str, str] = {}  # name -> lux type name

        # SSA value forwarding (release mode only — debug preserves all OpVariable)
        self._ssa_values = {}
        self._mutable_vars = set()
        if not self.debug:
            self._scan_mutable_vars(fn.body)
            self._scan_forward_refs(fn.body)

        # SPIR-V requires all OpVariable in a function to be at the top of
        # the first block. We use a deferred list that _alloc_local adds to.
        self._var_decls: list[str] = []
        self._pre_declare_locals(fn.body)

        body_lines = []

        # NonSemantic debug info: emit DebugFunction + DebugFunctionDefinition + DebugScope
        if self._dbg_emitter:
            fn_line = fn.loc.line if fn.loc else 1
            dbg_fn_id, dbg_fn_lines = self._dbg_emitter.emit_debug_function(fn.name, fn_line)
            body_lines.extend(dbg_fn_lines)
            body_lines.append(self._dbg_emitter.emit_function_definition(dbg_fn_id, "%main"))
            body_lines.append(self._dbg_emitter.emit_debug_scope(dbg_fn_id))

        for stmt in fn.body:
            body_lines.extend(self._gen_stmt(stmt))

        # Emit variable declarations first, then body
        lines.extend(self._var_decls)

        # NonSemantic debug info: emit DebugLocalVariable + DebugDeclare for each local
        if self._dbg_emitter:
            for name, var_id in self.local_vars.items():
                lux_type = self.local_types.get(name, "scalar")
                loc_line = 0
                # Try to find the LetStmt that defines this variable for line info
                for stmt in fn.body:
                    if isinstance(stmt, LetStmt) and stmt.name == name:
                        loc_line = stmt.loc.line if stmt.loc else 0
                        break
                dbg_var_id, dbg_var_instr = self._dbg_emitter.emit_local_variable(name, lux_type, loc_line)
                lines.append(dbg_var_instr)
                lines.append(self._dbg_emitter.emit_debug_declare(dbg_var_id, var_id))

        lines.extend(body_lines)

        # Collect local variable names for debug OpName emission
        if self.debug:
            for name, var_id in self.local_vars.items():
                self._all_local_names.append((name, var_id))

        # If this is main and no explicit return, add OpReturn
        lines.append("OpReturn")
        lines.append("OpFunctionEnd")
        return lines

    def _alloc_local(self, name: str, type_name: str) -> str:
        """Allocate a local variable, adding its OpVariable to the deferred list."""
        type_id = self.reg.lux_type_to_spirv(type_name)
        ptr_type = self.reg.pointer("Function", type_id)
        var_id = self.reg.next_id()
        self._var_decls.append(f"{var_id} = OpVariable {ptr_type} Function")
        self.local_vars[name] = var_id
        self.local_types[name] = type_name
        if self._precision_map.get(name) == "fp16":
            self.decorations.append(f"OpDecorate {var_id} RelaxedPrecision")
        return var_id

    def _pre_declare_locals(self, stmts: list):
        """Pre-declare all local variables at the top of the function block."""
        for stmt in stmts:
            if isinstance(stmt, LetStmt):
                # SSA-eligible vars skip OpVariable allocation (release mode)
                if not self.debug and stmt.name not in self._mutable_vars:
                    continue
                self._alloc_local(stmt.name, stmt.type_name)
            elif isinstance(stmt, IfStmt):
                self._pre_declare_locals(stmt.then_body)
                self._pre_declare_locals(stmt.else_body)
            elif isinstance(stmt, DebugBlock):
                self._pre_declare_locals(stmt.body)
            elif isinstance(stmt, ForStmt):
                self._alloc_local(stmt.loop_var, stmt.loop_var_type)
                self._pre_declare_locals(stmt.body)
            elif isinstance(stmt, WhileStmt):
                self._pre_declare_locals(stmt.body)

    def _emit_debug_line(self, node, lines: list[str]) -> None:
        """Emit OpLine before a statement/expression if debug mode is on and the node has a loc."""
        if not self.debug or self._debug_source_id is None:
            return
        loc = getattr(node, 'loc', None)
        if loc is None:
            return
        # Avoid redundant OpLine for the same source line
        if loc.line != self._debug_current_line:
            self._debug_current_line = loc.line
            col = loc.column if loc.column else 0
            lines.append(f"OpLine {self._debug_source_id} {loc.line} {col}")
            # Dual emission: also emit NonSemantic DebugLine for RenderDoc
            if self._dbg_emitter:
                lines.append(self._dbg_emitter.emit_debug_line(loc.line, col))

    def _gen_stmt(self, stmt) -> list[str]:
        lines = []
        self._emit_debug_line(stmt, lines)

        if isinstance(stmt, LetStmt):
            # Evaluate RHS
            val_id, val_lines = self._gen_expr(stmt.value)
            lines.extend(val_lines)
            # Coerce RHS to declared type if needed (e.g., int -> scalar)
            val_id = self._coerce_to_type(val_id, stmt.value, stmt.type_name, lines)
            if not self.debug and stmt.name not in self._mutable_vars:
                # SSA path: forward value directly, no OpVariable/OpStore
                self._ssa_values[stmt.name] = val_id
                if self._precision_map.get(stmt.name) == "fp16":
                    self.decorations.append(f"OpDecorate {val_id} RelaxedPrecision")
            else:
                # Variable already declared in _pre_declare_locals
                var_id = self.local_vars[stmt.name]
                lines.append(f"OpStore {var_id} {val_id}")

        elif isinstance(stmt, AssignStmt):
            target = stmt.target
            if isinstance(target, AssignTarget):
                target = target.expr
            val_id, val_lines = self._gen_expr(stmt.value)
            lines.extend(val_lines)
            # Coerce value to target type if needed
            target_type_name = self._resolve_store_target_type(target)
            if target_type_name:
                val_id = self._coerce_to_type(val_id, stmt.value, target_type_name, lines)
            ptr_id, ptr_lines = self._gen_store_target(target)
            lines.extend(ptr_lines)
            lines.append(f"OpStore {ptr_id} {val_id}")

        elif isinstance(stmt, ReturnStmt):
            if stmt.value is None:
                lines.append("OpReturn")
            else:
                val_id, val_lines = self._gen_expr(stmt.value)
                lines.extend(val_lines)
                lines.append(f"OpReturnValue {val_id}")

        elif isinstance(stmt, DiscardStmt):
            lines.append("OpKill")

        elif isinstance(stmt, IfStmt):
            cond_id, cond_lines = self._gen_expr(stmt.condition)
            lines.extend(cond_lines)
            then_label = self._next_label()
            else_label = self._next_label()
            merge_label = self._next_label()
            lines.append(f"OpSelectionMerge {merge_label} None")
            if stmt.else_body:
                lines.append(f"OpBranchConditional {cond_id} {then_label} {else_label}")
            else:
                lines.append(f"OpBranchConditional {cond_id} {then_label} {merge_label}")
            lines.append(f"{then_label} = OpLabel")
            # DebugLexicalBlock for then body
            saved_scope = None
            if self._dbg_emitter:
                saved_scope = self._dbg_emitter._current_scope_id
                line_num = stmt.loc.line if stmt.loc else 1
                blk_id, blk_instr = self._dbg_emitter.emit_lexical_block(line_num)
                lines.append(blk_instr)
                lines.append(self._dbg_emitter.emit_debug_scope(blk_id))
            then_lines = []
            for s in stmt.then_body:
                then_lines.extend(self._gen_stmt(s))
            if self._dbg_emitter and saved_scope:
                then_lines.append(self._dbg_emitter.emit_debug_scope(saved_scope))
            lines.extend(then_lines)
            # Skip OpBranch if then-body already terminates (OpReturn/OpReturnValue/OpKill)
            if not self._block_terminated(then_lines):
                lines.append(f"OpBranch {merge_label}")
            if stmt.else_body:
                lines.append(f"{else_label} = OpLabel")
                if self._dbg_emitter:
                    saved_scope2 = self._dbg_emitter._current_scope_id
                    line_num = stmt.loc.line if stmt.loc else 1
                    blk_id2, blk_instr2 = self._dbg_emitter.emit_lexical_block(line_num)
                    lines.append(blk_instr2)
                    lines.append(self._dbg_emitter.emit_debug_scope(blk_id2))
                else_lines = []
                for s in stmt.else_body:
                    else_lines.extend(self._gen_stmt(s))
                if self._dbg_emitter:
                    else_lines.append(self._dbg_emitter.emit_debug_scope(saved_scope2))
                lines.extend(else_lines)
                if not self._block_terminated(else_lines):
                    lines.append(f"OpBranch {merge_label}")
            lines.append(f"{merge_label} = OpLabel")

        elif isinstance(stmt, ExprStmt):
            _, expr_lines = self._gen_expr(stmt.expr)
            lines.extend(expr_lines)

        elif isinstance(stmt, DebugPrintStmt):
            lines.extend(self._gen_debug_print(stmt))

        elif isinstance(stmt, AssertStmt):
            lines.extend(self._gen_assert(stmt))

        elif isinstance(stmt, DebugBlock):
            # Flatten: emit body statements inline
            for s in stmt.body:
                lines.extend(self._gen_stmt(s))

        elif isinstance(stmt, ForStmt):
            # Store initial value
            var_id = self.local_vars[stmt.loop_var]
            init_id, init_lines = self._gen_expr(stmt.init_value)
            lines.extend(init_lines)
            # Coerce init value to loop var type if needed
            init_id = self._coerce_to_type(init_id, stmt.init_value, stmt.loop_var_type, lines)
            lines.append(f"OpStore {var_id} {init_id}")

            header_label = self._next_label()
            body_label = self._next_label()
            continue_label = self._next_label()
            merge_label = self._next_label()

            lines.append(f"OpBranch {header_label}")
            lines.append(f"{header_label} = OpLabel")

            # Condition (must come before OpLoopMerge)
            cond_id, cond_lines = self._gen_expr(stmt.condition)
            lines.extend(cond_lines)

            # Loop merge with optional unroll (must immediately precede branch)
            unroll_ctrl = "Unroll" if stmt.unroll else "None"
            lines.append(f"OpLoopMerge {merge_label} {continue_label} {unroll_ctrl}")
            lines.append(f"OpBranchConditional {cond_id} {body_label} {merge_label}")

            # Body
            lines.append(f"{body_label} = OpLabel")
            self._loop_label_stack.append((merge_label, continue_label))
            for s in stmt.body:
                lines.extend(self._gen_stmt(s))
            self._loop_label_stack.pop()
            lines.append(f"OpBranch {continue_label}")

            # Continue block (update)
            lines.append(f"{continue_label} = OpLabel")
            update_val_id, update_lines = self._gen_expr(stmt.update_value)
            lines.extend(update_lines)
            # Coerce update value to loop var type
            update_val_id = self._coerce_to_type(update_val_id, stmt.update_value, stmt.loop_var_type, lines)
            target = stmt.update_target
            if isinstance(target, AssignTarget):
                target = target.expr
            if isinstance(target, VarRef) and target.name in self.local_vars:
                lines.append(f"OpStore {self.local_vars[target.name]} {update_val_id}")
            else:
                ptr_id, ptr_lines = self._gen_store_target(target)
                lines.extend(ptr_lines)
                lines.append(f"OpStore {ptr_id} {update_val_id}")
            lines.append(f"OpBranch {header_label}")

            # Merge
            lines.append(f"{merge_label} = OpLabel")

        elif isinstance(stmt, WhileStmt):
            header_label = self._next_label()
            body_label = self._next_label()
            continue_label = self._next_label()
            merge_label = self._next_label()

            lines.append(f"OpBranch {header_label}")
            lines.append(f"{header_label} = OpLabel")

            # Condition (must come before OpLoopMerge)
            cond_id, cond_lines = self._gen_expr(stmt.condition)
            lines.extend(cond_lines)

            unroll_ctrl = "Unroll" if stmt.unroll else "None"
            lines.append(f"OpLoopMerge {merge_label} {continue_label} {unroll_ctrl}")
            lines.append(f"OpBranchConditional {cond_id} {body_label} {merge_label}")

            lines.append(f"{body_label} = OpLabel")
            self._loop_label_stack.append((merge_label, continue_label))
            for s in stmt.body:
                lines.extend(self._gen_stmt(s))
            self._loop_label_stack.pop()
            lines.append(f"OpBranch {continue_label}")

            lines.append(f"{continue_label} = OpLabel")
            lines.append(f"OpBranch {header_label}")

            lines.append(f"{merge_label} = OpLabel")

        elif isinstance(stmt, BreakStmt):
            if self._loop_label_stack:
                merge_label, _ = self._loop_label_stack[-1]
                lines.append(f"OpBranch {merge_label}")
                # Dead code label (SPIR-V requires a block after branch)
                dead_label = self._next_label()
                lines.append(f"{dead_label} = OpLabel")

        elif isinstance(stmt, ContinueStmt):
            if self._loop_label_stack:
                _, continue_label = self._loop_label_stack[-1]
                lines.append(f"OpBranch {continue_label}")
                dead_label = self._next_label()
                lines.append(f"{dead_label} = OpLabel")

        return lines

    def _get_or_create_debug_string(self, fmt: str) -> str:
        """Get or create an OpString ID for a debug format string."""
        if fmt not in self._debug_string_ids:
            str_id = self.reg.next_id()
            self._debug_string_ids[fmt] = str_id
        return self._debug_string_ids[fmt]

    def _gen_debug_print(self, stmt: DebugPrintStmt) -> list[str]:
        """Generate SPIR-V for debug_print statement using NonSemantic.DebugPrintf."""
        lines = []
        if not self._debug_printf_ext_id:
            return lines

        # Get format string ID
        fmt_id = self._get_or_create_debug_string(stmt.format_string)

        # Evaluate arguments
        arg_ids = []
        for arg in stmt.args:
            aid, alines = self._gen_expr(arg)
            lines.extend(alines)
            arg_ids.append(aid)

        # Emit OpExtInst for DebugPrintf (instruction number 1)
        void_type = self.reg.void()
        result = self.reg.next_id()
        args_str = " ".join(arg_ids)
        if args_str:
            lines.append(f"{result} = OpExtInst {void_type} {self._debug_printf_ext_id} 1 {fmt_id} {args_str}")
        else:
            lines.append(f"{result} = OpExtInst {void_type} {self._debug_printf_ext_id} 1 {fmt_id}")

        return lines

    def _gen_assert(self, stmt: AssertStmt) -> list[str]:
        """Generate SPIR-V for assert statement: conditional debugPrintf on failure."""
        lines = []
        if not self._debug_printf_ext_id:
            return lines

        # Evaluate condition
        cond_id, cond_lines = self._gen_expr(stmt.condition)
        lines.extend(cond_lines)

        # Create labels for branch
        ok_label = self._next_label()
        fail_label = self._next_label()
        merge_label = self._next_label()

        lines.append(f"OpSelectionMerge {merge_label} None")
        lines.append(f"OpBranchConditional {cond_id} {ok_label} {fail_label}")

        # Fail block: print assertion message
        lines.append(f"{fail_label} = OpLabel")
        msg = stmt.message or "assertion failed"
        loc_info = ""
        if stmt.loc:
            loc_info = f" [{self.source_name}:{stmt.loc.line}]"
        fmt_str = f"ASSERT FAILED{loc_info}: {msg}"
        fmt_id = self._get_or_create_debug_string(fmt_str)
        void_type = self.reg.void()
        result = self.reg.next_id()
        lines.append(f"{result} = OpExtInst {void_type} {self._debug_printf_ext_id} 1 {fmt_id}")
        # Assert kill mode: demote the invocation on failure (fragment only)
        if self.assert_kill and self.stage.stage_type == "fragment":
            lines.append("OpDemoteToHelperInvocation")
        lines.append(f"OpBranch {merge_label}")

        # OK block
        lines.append(f"{ok_label} = OpLabel")
        lines.append(f"OpBranch {merge_label}")

        # Merge
        lines.append(f"{merge_label} = OpLabel")

        return lines

    def _gen_store_target(self, target) -> tuple[str, list[str]]:
        """Return (pointer_id, lines) for a store target."""
        lines = []

        if isinstance(target, VarRef):
            name = target.name
            # builtin_position -> access chain into gl_PerVertex[0]
            if name == "builtin_position" and self.per_vertex_var_id:
                vec4_id = self.reg.lux_type_to_spirv("vec4")
                ptr_type = self.reg.pointer("Output", vec4_id)
                idx = self.reg.const_int(0, signed=True)
                ac_id = self.reg.next_id()
                lines.append(f"{ac_id} = OpAccessChain {ptr_type} {self.per_vertex_var_id} {idx}")
                return ac_id, lines

            if name in self.local_vars:
                return self.local_vars[name], lines
            if name in self.var_map:
                return self.var_map[name], lines
            raise ValueError(f"Unknown store target: {name}")

        elif isinstance(target, SwizzleAccess):
            # For swizzle write, we need the pointer to the whole vector,
            # then do an OpLoad, OpVectorShuffle, OpStore
            # For simplicity in v1, only support full assignment
            return self._gen_store_target(target.object)

        elif isinstance(target, FieldAccess):
            return self._gen_store_target(target.object)

        elif isinstance(target, IndexAccess):
            # Check for shared memory array store: shared_arr[index] = value
            if isinstance(target.object, VarRef) and target.object.name in self.shared_var_ids:
                name = target.object.name
                if self.shared_array_sizes.get(name) is not None:
                    elem_type_name = self.shared_element_types[name]
                    elem_type_id = self.reg.lux_type_to_spirv(elem_type_name)
                    var_id = self.shared_var_ids[name]
                    idx_id, idx_lines = self._gen_expr(target.index)
                    lines.extend(idx_lines)
                    idx_lux_type = self._resolve_expr_lux_type(target.index)
                    if idx_lux_type in ("scalar",):
                        int_type = self.reg.int32()
                        conv = self.reg.next_id()
                        lines.append(f"{conv} = OpConvertFToS {int_type} {idx_id}")
                        idx_id = conv
                    ptr_elem = self.reg.pointer("Workgroup", elem_type_id)
                    ac_id = self.reg.next_id()
                    lines.append(f"{ac_id} = OpAccessChain {ptr_elem} {var_id} {idx_id}")
                    return ac_id, lines

            # Check for storage buffer indexed store: buffer[index] = value
            if isinstance(target.object, VarRef) and target.object.name in self.storage_buffer_var_ids:
                name = target.object.name
                buf_var_id = self.storage_buffer_var_ids[name]
                elem_type_name = self.storage_buffer_element_types[name]
                elem_type_id = self.reg.lux_type_to_spirv(elem_type_name)
                idx_id, idx_lines = self._gen_expr(target.index)
                lines.extend(idx_lines)
                # Convert float index to int
                idx_lux_type = self._resolve_expr_lux_type(target.index)
                if idx_lux_type in ("scalar",):
                    int_type = self.reg.int32()
                    conv = self.reg.next_id()
                    lines.append(f"{conv} = OpConvertFToS {int_type} {idx_id}")
                    idx_id = conv
                ptr_elem = self.reg.pointer("StorageBuffer", elem_type_id)
                const_0 = self.reg.const_int(0, signed=True)
                ac_id = self.reg.next_id()
                lines.append(f"{ac_id} = OpAccessChain {ptr_elem} {buf_var_id} {const_0} {idx_id}")
                return ac_id, lines

            # Check for mesh output array writes: output_name[index]
            if isinstance(target.object, VarRef):
                name = target.object.name
                # Check for gl_MeshVerticesEXT[index] writes FIRST (before generic output handler)
                if name == "gl_MeshVerticesEXT" and hasattr(self, '_mesh_per_vert_struct'):
                    idx_id, idx_lines = self._gen_expr(target.index)
                    lines.extend(idx_lines)
                    idx_type = self._resolve_expr_lux_type(target.index)
                    if idx_type in ("scalar",):
                        int_type = self.reg.int32()
                        conv = self.reg.next_id()
                        lines.append(f"{conv} = OpConvertFToS {int_type} {idx_id}")
                        idx_id = conv
                    vec4_id = self.reg.lux_type_to_spirv("vec4")
                    ptr_type = self.reg.pointer("Output", vec4_id)
                    const_0 = self.reg.const_int(0, signed=True)
                    ac_id = self.reg.next_id()
                    lines.append(f"{ac_id} = OpAccessChain {ptr_type} {self.var_map[name]} {idx_id} {const_0}")
                    return ac_id, lines
                # gl_PrimitiveTriangleIndicesEXT[index]
                if name == "gl_PrimitiveTriangleIndicesEXT":
                    idx_id, idx_lines = self._gen_expr(target.index)
                    lines.extend(idx_lines)
                    idx_type = self._resolve_expr_lux_type(target.index)
                    if idx_type in ("scalar",):
                        int_type = self.reg.int32()
                        conv = self.reg.next_id()
                        lines.append(f"{conv} = OpConvertFToS {int_type} {idx_id}")
                        idx_id = conv
                    uvec3_id = self.reg.lux_type_to_spirv("uvec3")
                    ptr_type = self.reg.pointer("Output", uvec3_id)
                    ac_id = self.reg.next_id()
                    lines.append(f"{ac_id} = OpAccessChain {ptr_type} {self.var_map[name]} {idx_id}")
                    return ac_id, lines
                # Generic output array handler (mesh per-vertex output arrays like frag_color[tid])
                if name in self.var_map and self.var_storage.get(name) == "Output":
                    idx_id, idx_lines = self._gen_expr(target.index)
                    lines.extend(idx_lines)
                    # Convert index to int if float
                    idx_type = self._resolve_expr_lux_type(target.index)
                    if idx_type in ("scalar",):
                        int_type = self.reg.int32()
                        conv = self.reg.next_id()
                        lines.append(f"{conv} = OpConvertFToS {int_type} {idx_id}")
                        idx_id = conv
                    out_type = self.var_types.get(name, "vec4")
                    type_id = self.reg.lux_type_to_spirv(out_type)
                    ptr_type = self.reg.pointer("Output", type_id)
                    ac_id = self.reg.next_id()
                    lines.append(f"{ac_id} = OpAccessChain {ptr_type} {self.var_map[name]} {idx_id}")
                    return ac_id, lines
            obj_ptr, obj_lines = self._gen_store_target(target.object)
            lines.extend(obj_lines)
            idx_id, idx_lines = self._gen_expr(target.index)
            lines.extend(idx_lines)
            result_id = self.reg.next_id()
            return result_id, lines

        raise ValueError(f"Unknown store target type: {type(target).__name__}")

    def _gen_expr(self, expr) -> tuple[str, list[str]]:
        """Generate SPIR-V instructions for an expression.

        Returns (result_id, lines).
        """
        lines = []

        if isinstance(expr, NumberLit):
            val = expr.value
            resolved = getattr(expr, 'resolved_type', None)
            if resolved == "int":
                return self.reg.const_int(int(float(val)), signed=True), lines
            if resolved == "uint":
                return self.reg.const_uint(int(float(val))), lines
            if "." in val or "e" in val.lower():
                result = self.reg.const_float(float(val))
            else:
                result = self.reg.const_float(float(val))
            return result, lines

        elif isinstance(expr, BoolLit):
            return self.reg.const_bool(expr.value), lines

        elif isinstance(expr, VarRef):
            return self._gen_var_load(expr.name, lines)

        elif isinstance(expr, BinaryOp):
            return self._gen_binary(expr, lines)

        elif isinstance(expr, UnaryOp):
            return self._gen_unary(expr, lines)

        elif isinstance(expr, CallExpr):
            return self._gen_call(expr, lines)

        elif isinstance(expr, ConstructorExpr):
            return self._gen_constructor(expr, lines)

        elif isinstance(expr, SwizzleAccess):
            return self._gen_swizzle(expr, lines)

        elif isinstance(expr, FieldAccess):
            return self._gen_field_access(expr, lines)

        elif isinstance(expr, IndexAccess):
            return self._gen_index_access(expr, lines)

        elif isinstance(expr, TernaryExpr):
            return self._gen_ternary(expr, lines)

        raise ValueError(f"Unknown expr type: {type(expr).__name__}")

    def _gen_var_load(self, name: str, lines: list[str]) -> tuple[str, list[str]]:
        """Load a variable value."""
        # Check SSA values first (release mode mem2reg)
        if name in self._ssa_values:
            return self._ssa_values[name], lines

        # Check uniform fields (need OpAccessChain)
        if name in self.uniform_field_indices:
            block_name = self.uniform_block_for_field[name]
            var_id = self.uniform_var_ids[block_name]
            field_idx = self.uniform_field_indices[name]
            field_type = self.var_types[name]
            type_id = self.reg.lux_type_to_spirv(field_type)
            ptr_type = self.reg.pointer("Uniform", type_id)
            idx_id = self.reg.const_int(field_idx, signed=True)
            ac_id = self.reg.next_id()
            lines.append(f"{ac_id} = OpAccessChain {ptr_type} {var_id} {idx_id}")
            result = self.reg.next_id()
            lines.append(f"{result} = OpLoad {type_id} {ac_id}")
            return result, lines

        # Check push constant fields
        if name in self.push_field_indices:
            block_name = self.push_block_for_field[name]
            var_id = self.push_var_ids[block_name]
            field_idx = self.push_field_indices[name]
            field_type = self.var_types[name]
            type_id = self.reg.lux_type_to_spirv(field_type)
            ptr_type = self.reg.pointer("PushConstant", type_id)
            idx_id = self.reg.const_int(field_idx, signed=True)
            ac_id = self.reg.next_id()
            lines.append(f"{ac_id} = OpAccessChain {ptr_type} {var_id} {idx_id}")
            result = self.reg.next_id()
            lines.append(f"{result} = OpLoad {type_id} {ac_id}")
            return result, lines

        # Check local variables
        if name in self.local_vars:
            type_name = self.local_types[name]
            type_id = self.reg.lux_type_to_spirv(type_name)
            result = self.reg.next_id()
            lines.append(f"{result} = OpLoad {type_id} {self.local_vars[name]}")
            if self._precision_map.get(name) == "fp16":
                self.decorations.append(f"OpDecorate {result} RelaxedPrecision")
            return result, lines

        # Check global vars (inputs/outputs) - skip storage buffers and shared arrays (accessed via index)
        if name in self.var_map:
            if name in self.shared_var_ids and self.shared_array_sizes.get(name) is not None:
                # Shared arrays must be accessed via indexing, not direct load
                return self.var_map[name], lines
            if name in self.storage_buffer_var_ids:
                # Storage buffers must be accessed via indexing, not direct load.
                # Return the variable pointer so IndexAccess can use it.
                return self.var_map[name], lines
            type_name = self.var_types[name]
            # Use per-variable image type for storage images (supports custom formats like Rg16f)
            if name in self._storage_image_type_ids:
                type_id = self._storage_image_type_ids[name]
            else:
                type_id = self.reg.lux_type_to_spirv(type_name)
            result = self.reg.next_id()
            lines.append(f"{result} = OpLoad {type_id} {self.var_map[name]}")
            return result, lines

        # Check specialization constants (their IDs are already OpSpecConstant results)
        if name in self.spec_const_ids:
            return self.spec_const_ids[name], lines

        # Check module constants
        for c in self.module.constants:
            if c.name == name:
                return self._gen_expr(c.value)

        raise ValueError(f"Undefined variable in codegen: {name}")

    def _gen_binary(self, expr: BinaryOp, lines: list[str]) -> tuple[str, list[str]]:
        left_id, left_lines = self._gen_expr(expr.left)
        lines.extend(left_lines)
        right_id, right_lines = self._gen_expr(expr.right)
        lines.extend(right_lines)

        result_type = self.reg.lux_type_to_spirv(expr.resolved_type)
        result = self.reg.next_id()

        lt = self._resolve_expr_lux_type(expr.left)
        rt = self._resolve_expr_lux_type(expr.right)

        op = expr.op

        if op in ("==", "!=", "<", ">", "<=", ">="):
            bool_type = self.reg.bool_type()

            lt_type_obj = resolve_type(resolve_alias_chain(lt))
            rt_type_obj = resolve_type(resolve_alias_chain(rt))
            def _is_int_scalar(t):
                return isinstance(t, ScalarType) and t.name in ("int", "uint")

            both_int = _is_int_scalar(lt_type_obj) and _is_int_scalar(rt_type_obj)

            if both_int:
                # Integer comparisons
                either_signed = (lt_type_obj.name == "int" or rt_type_obj.name == "int")
                int_cmp_ops = {
                    "==": "OpIEqual", "!=": "OpINotEqual",
                }
                if either_signed:
                    int_cmp_ops.update({"<": "OpSLessThan", ">": "OpSGreaterThan",
                                        "<=": "OpSLessThanEqual", ">=": "OpSGreaterThanEqual"})
                else:
                    int_cmp_ops.update({"<": "OpULessThan", ">": "OpUGreaterThan",
                                        "<=": "OpULessThanEqual", ">=": "OpUGreaterThanEqual"})
                lines.append(f"{result} = {int_cmp_ops[op]} {bool_type} {left_id} {right_id}")
                return result, lines

            # Convert int/uint operands to float for float comparison ops
            if _is_int_scalar(lt_type_obj):
                conv_id = self.reg.next_id()
                conv_op = "OpConvertSToF" if lt_type_obj.name == "int" else "OpConvertUToF"
                lines.append(f"{conv_id} = {conv_op} {self.reg.float32()} {left_id}")
                left_id = conv_id
            if _is_int_scalar(rt_type_obj):
                conv_id = self.reg.next_id()
                conv_op = "OpConvertSToF" if rt_type_obj.name == "int" else "OpConvertUToF"
                lines.append(f"{conv_id} = {conv_op} {self.reg.float32()} {right_id}")
                right_id = conv_id

            cmp_ops = {
                "==": "OpFOrdEqual", "!=": "OpFOrdNotEqual",
                "<": "OpFOrdLessThan", ">": "OpFOrdGreaterThan",
                "<=": "OpFOrdLessThanEqual", ">=": "OpFOrdGreaterThanEqual",
            }
            lines.append(f"{result} = {cmp_ops[op]} {bool_type} {left_id} {right_id}")
            return result, lines

        if op in ("&&", "||"):
            bool_type = self.reg.bool_type()
            logic_op = "OpLogicalAnd" if op == "&&" else "OpLogicalOr"
            lines.append(f"{result} = {logic_op} {bool_type} {left_id} {right_id}")
            return result, lines

        # Bitwise ops — operands must be integer type
        if op in ("&", "|", "^"):
            bitwise_ops = {"&": "OpBitwiseAnd", "|": "OpBitwiseOr", "^": "OpBitwiseXor"}
            uint_t = self.reg.uint32()
            # Convert float operands to uint (NumberLit defaults to float in Lux)
            lt_resolved = resolve_alias_chain(lt)
            rt_resolved = resolve_alias_chain(rt)
            if lt_resolved == "scalar":
                if isinstance(expr.left, NumberLit):
                    left_id = self.reg.const_uint(int(float(expr.left.value)))
                else:
                    conv = self.reg.next_id()
                    lines.append(f"{conv} = OpConvertFToU {uint_t} {left_id}")
                    left_id = conv
            if rt_resolved == "scalar":
                if isinstance(expr.right, NumberLit):
                    right_id = self.reg.const_uint(int(float(expr.right.value)))
                else:
                    conv = self.reg.next_id()
                    lines.append(f"{conv} = OpConvertFToU {uint_t} {right_id}")
                    right_id = conv
            lines.append(f"{result} = {bitwise_ops[op]} {result_type} {left_id} {right_id}")
            return result, lines

        # Shift ops — operands must be integer type
        if op in ("<<", ">>"):
            shift_ops = {"<<": "OpShiftLeftLogical", ">>": "OpShiftRightLogical"}
            uint_t = self.reg.uint32()
            lt_resolved = resolve_alias_chain(lt)
            rt_resolved = resolve_alias_chain(rt)
            if lt_resolved == "scalar":
                if isinstance(expr.left, NumberLit):
                    left_id = self.reg.const_uint(int(float(expr.left.value)))
                else:
                    conv = self.reg.next_id()
                    lines.append(f"{conv} = OpConvertFToU {uint_t} {left_id}")
                    left_id = conv
            if rt_resolved == "scalar":
                if isinstance(expr.right, NumberLit):
                    right_id = self.reg.const_uint(int(float(expr.right.value)))
                else:
                    conv = self.reg.next_id()
                    lines.append(f"{conv} = OpConvertFToU {uint_t} {right_id}")
                    right_id = conv
            lines.append(f"{result} = {shift_ops[op]} {result_type} {left_id} {right_id}")
            return result, lines

        # Arithmetic ops
        spv_op = self._select_arith_op(op, lt, rt)

        # Handle mixed vec/scalar for +, -, /, % by splatting scalar to vec
        lt_resolved = resolve_alias_chain(lt)
        rt_resolved = resolve_alias_chain(rt)
        lt_type = resolve_type(lt_resolved)
        rt_type = resolve_type(rt_resolved)

        # Implicit int/uint→float conversion for float arithmetic
        def _is_int_type(t):
            if isinstance(t, ScalarType) and t.name in ("int", "uint"):
                return True
            if isinstance(t, VectorType) and t.component in ("int", "uint"):
                return True
            return False

        def _is_float_type(t):
            if isinstance(t, ScalarType) and t.name == "scalar":
                return True
            if isinstance(t, VectorType) and t.component == "scalar":
                return True
            return False

        both_int = _is_int_type(lt_type) and _is_int_type(rt_type)
        if not both_int:
            if _is_int_type(lt_type) and (_is_float_type(rt_type) or not _is_int_type(rt_type)):
                conv_id = self.reg.next_id()
                conv_type = self.reg.float32()
                conv_op = "OpConvertSToF" if (isinstance(lt_type, ScalarType) and lt_type.name == "int") or (isinstance(lt_type, VectorType) and lt_type.component == "int") else "OpConvertUToF"
                lines.append(f"{conv_id} = {conv_op} {conv_type} {left_id}")
                left_id = conv_id
                lt_type = resolve_type("scalar")
                lt_resolved = "scalar"

            if _is_int_type(rt_type) and (_is_float_type(lt_type) or not _is_int_type(lt_type)):
                conv_id = self.reg.next_id()
                conv_type = self.reg.float32()
                conv_op = "OpConvertSToF" if (isinstance(rt_type, ScalarType) and rt_type.name == "int") or (isinstance(rt_type, VectorType) and rt_type.component == "int") else "OpConvertUToF"
                lines.append(f"{conv_id} = {conv_op} {conv_type} {right_id}")
                right_id = conv_id
                rt_type = resolve_type("scalar")
                rt_resolved = "scalar"

        if op in ("+", "-", "/", "%"):
            if isinstance(lt_type, VectorType) and isinstance(rt_type, ScalarType):
                splat_id = self.reg.next_id()
                vec_type = self.reg.lux_type_to_spirv(lt_resolved)
                components = " ".join([right_id] * lt_type.size)
                lines.append(f"{splat_id} = OpCompositeConstruct {vec_type} {components}")
                right_id = splat_id
            elif isinstance(lt_type, ScalarType) and isinstance(rt_type, VectorType):
                splat_id = self.reg.next_id()
                vec_type = self.reg.lux_type_to_spirv(rt_resolved)
                components = " ".join([left_id] * rt_type.size)
                lines.append(f"{splat_id} = OpCompositeConstruct {vec_type} {components}")
                left_id = splat_id

        # OpVectorTimesScalar requires vector as first operand, scalar as second.
        # When expression is scalar * vec, swap operand order.
        if spv_op == "OpVectorTimesScalar" and isinstance(lt_type, ScalarType) and isinstance(rt_type, VectorType):
            lines.append(f"{result} = {spv_op} {result_type} {right_id} {left_id}")
        elif spv_op == "OpMatrixTimesScalar" and isinstance(lt_type, ScalarType) and isinstance(rt_type, MatrixType):
            lines.append(f"{result} = {spv_op} {result_type} {right_id} {left_id}")
        else:
            lines.append(f"{result} = {spv_op} {result_type} {left_id} {right_id}")
        return result, lines

    def _select_arith_op(self, op: str, lt: str, rt: str) -> str:
        from luxc.builtins.types import resolve_alias_chain
        lt_type = resolve_type(resolve_alias_chain(lt))
        rt_type = resolve_type(resolve_alias_chain(rt))

        # Check if both operands are integer types
        def _both_int(lt_t, rt_t):
            if isinstance(lt_t, ScalarType) and isinstance(rt_t, ScalarType):
                return lt_t.name in ("int", "uint") and rt_t.name in ("int", "uint")
            return False

        bi = _both_int(lt_type, rt_type)

        if op == "+":
            return "OpIAdd" if bi else "OpFAdd"
        elif op == "-":
            return "OpISub" if bi else "OpFSub"
        elif op == "/":
            if bi:
                if (isinstance(lt_type, ScalarType) and lt_type.name == "int") or \
                   (isinstance(rt_type, ScalarType) and rt_type.name == "int"):
                    return "OpSDiv"
                return "OpUDiv"
            return "OpFDiv"
        elif op == "%":
            if bi:
                if (isinstance(lt_type, ScalarType) and lt_type.name == "int") or \
                   (isinstance(rt_type, ScalarType) and rt_type.name == "int"):
                    return "OpSMod"
                return "OpUMod"
            return "OpFMod"
        elif op == "*":
            if bi:
                return "OpIMul"
            # Matrix * Vector
            if isinstance(lt_type, MatrixType) and isinstance(rt_type, VectorType):
                return "OpMatrixTimesVector"
            # Vector * Matrix
            if isinstance(lt_type, VectorType) and isinstance(rt_type, MatrixType):
                return "OpVectorTimesMatrix"
            # Matrix * Matrix
            if isinstance(lt_type, MatrixType) and isinstance(rt_type, MatrixType):
                return "OpMatrixTimesMatrix"
            # Matrix * Scalar
            if isinstance(lt_type, MatrixType) and isinstance(rt_type, ScalarType):
                return "OpMatrixTimesScalar"
            if isinstance(lt_type, ScalarType) and isinstance(rt_type, MatrixType):
                return "OpMatrixTimesScalar"
            # Vector * Scalar or Scalar * Vector
            if isinstance(lt_type, VectorType) and isinstance(rt_type, ScalarType):
                return "OpVectorTimesScalar"
            if isinstance(lt_type, ScalarType) and isinstance(rt_type, VectorType):
                return "OpVectorTimesScalar"
            return "OpFMul"

        return "OpFMul"

    def _gen_unary(self, expr: UnaryOp, lines: list[str]) -> tuple[str, list[str]]:
        operand_id, op_lines = self._gen_expr(expr.operand)
        lines.extend(op_lines)
        result_type = self.reg.lux_type_to_spirv(expr.resolved_type)
        result = self.reg.next_id()
        if expr.op == "-":
            operand_type = self._resolve_expr_lux_type(expr.operand)
            if operand_type in ("int", "uint"):
                lines.append(f"{result} = OpSNegate {result_type} {operand_id}")
            else:
                lines.append(f"{result} = OpFNegate {result_type} {operand_id}")
        elif expr.op == "!":
            lines.append(f"{result} = OpLogicalNot {result_type} {operand_id}")
        return result, lines

    def _coerce_to_type(self, val_id: str, expr, target_type_name: str, lines: list[str]) -> str:
        """Coerce a value to the target type if needed (e.g., float literal -> int)."""
        src_type = self._resolve_expr_lux_type(expr)
        if target_type_name in ("int", "uint") and src_type == "scalar":
            # Float -> int/uint
            if isinstance(expr, NumberLit) and "." not in expr.value and "e" not in expr.value.lower():
                # Integer literal stored as float constant -- re-emit as int constant
                int_val = int(expr.value)
                if target_type_name == "int":
                    return self.reg.const_int(int_val, signed=True)
                else:
                    return self.reg.const_uint(int_val)
            int_type = self.reg.int32() if target_type_name == "int" else self.reg.uint32()
            conv = self.reg.next_id()
            conv_op = "OpConvertFToS" if target_type_name == "int" else "OpConvertFToU"
            lines.append(f"{conv} = {conv_op} {int_type} {val_id}")
            return conv
        if target_type_name == "scalar" and src_type in ("int", "uint"):
            float_type = self.reg.float32()
            conv = self.reg.next_id()
            conv_op = "OpConvertSToF" if src_type == "int" else "OpConvertUToF"
            lines.append(f"{conv} = {conv_op} {float_type} {val_id}")
            return conv
        return val_id

    def _resolve_sampler_types(self, sam_type: str) -> tuple[str, str]:
        """Return (image_type, sampled_image_type) for a given sampler type name."""
        if sam_type == "samplerCube":
            return self.reg.cube_image_type(), self.reg.sampled_cube_image_type()
        elif sam_type == "sampler2DArray":
            return self.reg.image_2d_array_type(), self.reg.sampled_image_2d_array_type()
        elif sam_type == "samplerCubeArray":
            return self.reg.image_cube_array_type(), self.reg.sampled_cube_array_type()
        else:
            return self.reg.image_type(), self.reg.sampled_image_type()

    def _gen_call(self, expr: CallExpr, lines: list[str]) -> tuple[str, list[str]]:
        if not isinstance(expr.func, VarRef):
            raise ValueError("Non-simple function calls not supported")

        fname = expr.func.name

        # --- Atomic operations ---
        _ATOMIC_FUNCS = {
            "atomic_add", "atomic_min", "atomic_max",
            "atomic_and", "atomic_or", "atomic_xor",
            "atomic_exchange", "atomic_compare_exchange",
            "atomic_load", "atomic_store",
        }
        if fname in _ATOMIC_FUNCS:
            return self._gen_atomic_call(fname, expr, lines)

        # sample(tex, uv) — handle before general arg generation since tex
        # is not a loadable variable (it's split into sampler + texture)
        if fname == "sample":
            sampler_arg = expr.args[0]
            # Generate UV argument normally
            uv_id, uv_lines = self._gen_expr(expr.args[1])
            lines.extend(uv_lines)
            result_type = self.reg.lux_type_to_spirv(expr.resolved_type)
            result = self.reg.next_id()
            if isinstance(sampler_arg, VarRef) and sampler_arg.name + ".__sampler" in self.var_map:
                sam_name = sampler_arg.name
                # Determine sampler type and resolve image/sampled-image types
                sam_type = self.var_types.get(sam_name, "sampler2d")
                img_type, sampled_img_type = self._resolve_sampler_types(sam_type)
                # Load the texture image
                tex_loaded = self.reg.next_id()
                lines.append(f"{tex_loaded} = OpLoad {img_type} {self.var_map[sam_name + '.__texture']}")
                # Load the sampler
                samp_type = self.reg.sampler_type()
                samp_loaded = self.reg.next_id()
                lines.append(f"{samp_loaded} = OpLoad {samp_type} {self.var_map[sam_name + '.__sampler']}")
                # Combine into sampled image
                combined = self.reg.next_id()
                lines.append(f"{combined} = OpSampledImage {sampled_img_type} {tex_loaded} {samp_loaded}")
                # Sample
                lines.append(f"{result} = OpImageSampleImplicitLod {result_type} {combined} {uv_id}")
            else:
                # Fallback: combined image sampler (legacy)
                tex_id, tex_lines = self._gen_expr(sampler_arg)
                lines.extend(tex_lines)
                lines.append(f"{result} = OpImageSampleImplicitLod {result_type} {tex_id} {uv_id}")
            return result, lines

        # sample_lod(tex, coords, lod) — explicit LOD sampling
        if fname == "sample_lod":
            sampler_arg = expr.args[0]
            # Generate coords and lod arguments
            coords_id, coords_lines = self._gen_expr(expr.args[1])
            lines.extend(coords_lines)
            lod_id, lod_lines = self._gen_expr(expr.args[2])
            lines.extend(lod_lines)
            result_type = self.reg.lux_type_to_spirv(expr.resolved_type)
            result = self.reg.next_id()
            if isinstance(sampler_arg, VarRef) and sampler_arg.name + ".__sampler" in self.var_map:
                sam_name = sampler_arg.name
                sam_type = self.var_types.get(sam_name, "sampler2d")
                img_type, sampled_img_type = self._resolve_sampler_types(sam_type)
                tex_loaded = self.reg.next_id()
                lines.append(f"{tex_loaded} = OpLoad {img_type} {self.var_map[sam_name + '.__texture']}")
                samp_type = self.reg.sampler_type()
                samp_loaded = self.reg.next_id()
                lines.append(f"{samp_loaded} = OpLoad {samp_type} {self.var_map[sam_name + '.__sampler']}")
                combined = self.reg.next_id()
                lines.append(f"{combined} = OpSampledImage {sampled_img_type} {tex_loaded} {samp_loaded}")
                lines.append(f"{result} = OpImageSampleExplicitLod {result_type} {combined} {coords_id} Lod {lod_id}")
            else:
                tex_id, tex_lines = self._gen_expr(sampler_arg)
                lines.extend(tex_lines)
                lines.append(f"{result} = OpImageSampleExplicitLod {result_type} {tex_id} {coords_id} Lod {lod_id}")
            return result, lines

        # sample_grad(tex, coords, ddx, ddy) — explicit gradient sampling
        if fname == "sample_grad":
            sampler_arg = expr.args[0]
            # Generate coords, ddx, ddy arguments
            coords_id, coords_lines = self._gen_expr(expr.args[1])
            lines.extend(coords_lines)
            ddx_id, ddx_lines = self._gen_expr(expr.args[2])
            lines.extend(ddx_lines)
            ddy_id, ddy_lines = self._gen_expr(expr.args[3])
            lines.extend(ddy_lines)
            result_type = self.reg.lux_type_to_spirv(expr.resolved_type)
            result = self.reg.next_id()
            if isinstance(sampler_arg, VarRef) and sampler_arg.name + ".__sampler" in self.var_map:
                sam_name = sampler_arg.name
                sam_type = self.var_types.get(sam_name, "sampler2d")
                img_type, sampled_img_type = self._resolve_sampler_types(sam_type)
                tex_loaded = self.reg.next_id()
                lines.append(f"{tex_loaded} = OpLoad {img_type} {self.var_map[sam_name + '.__texture']}")
                samp_type = self.reg.sampler_type()
                samp_loaded = self.reg.next_id()
                lines.append(f"{samp_loaded} = OpLoad {samp_type} {self.var_map[sam_name + '.__sampler']}")
                combined = self.reg.next_id()
                lines.append(f"{combined} = OpSampledImage {sampled_img_type} {tex_loaded} {samp_loaded}")
                lines.append(f"{result} = OpImageSampleExplicitLod {result_type} {combined} {coords_id} Grad {ddx_id} {ddy_id}")
            else:
                tex_id, tex_lines = self._gen_expr(sampler_arg)
                lines.extend(tex_lines)
                lines.append(f"{result} = OpImageSampleExplicitLod {result_type} {tex_id} {coords_id} Grad {ddx_id} {ddy_id}")
            return result, lines

        # sample_compare(shadow_tex, vec3(uv, layer), depth_ref) → scalar
        if fname == "sample_compare":
            sampler_arg = expr.args[0]
            coords_id, coords_lines = self._gen_expr(expr.args[1])  # vec3(u, v, layer)
            lines.extend(coords_lines)
            dref_id, dref_lines = self._gen_expr(expr.args[2])  # depth reference
            lines.extend(dref_lines)
            result_type = self.reg.float32()  # returns scalar
            result = self.reg.next_id()
            if isinstance(sampler_arg, VarRef) and sampler_arg.name + ".__sampler" in self.var_map:
                sam_name = sampler_arg.name
                sam_type = self.var_types.get(sam_name, "sampler2DArray")
                img_type, sampled_img_type = self._resolve_sampler_types(sam_type)
                tex_loaded = self.reg.next_id()
                lines.append(f"{tex_loaded} = OpLoad {img_type} {self.var_map[sam_name + '.__texture']}")
                samp_loaded = self.reg.next_id()
                lines.append(f"{samp_loaded} = OpLoad {self.reg.sampler_type()} {self.var_map[sam_name + '.__sampler']}")
                combined = self.reg.next_id()
                lines.append(f"{combined} = OpSampledImage {sampled_img_type} {tex_loaded} {samp_loaded}")
                lod_zero = self.reg.const_float(0.0)
                lines.append(f"{result} = OpImageSampleDrefExplicitLod {result_type} {combined} {coords_id} {dref_id} Lod {lod_zero}")
            return result, lines

        # sample_array(tex, uv, layer) → vec4
        if fname == "sample_array":
            sampler_arg = expr.args[0]
            uv_id, uv_lines = self._gen_expr(expr.args[1])   # vec2
            lines.extend(uv_lines)
            layer_id, layer_lines = self._gen_expr(expr.args[2])  # scalar (layer index)
            lines.extend(layer_lines)
            # Build vec3(uv.x, uv.y, layer) coordinate
            float_type = self.reg.float32()
            uv_x = self.reg.next_id()
            lines.append(f"{uv_x} = OpCompositeExtract {float_type} {uv_id} 0")
            uv_y = self.reg.next_id()
            lines.append(f"{uv_y} = OpCompositeExtract {float_type} {uv_id} 1")
            coord3 = self.reg.next_id()
            lines.append(f"{coord3} = OpCompositeConstruct {self.reg.vec(3)} {uv_x} {uv_y} {layer_id}")
            result_type = self.reg.lux_type_to_spirv(expr.resolved_type)
            result = self.reg.next_id()
            if isinstance(sampler_arg, VarRef) and sampler_arg.name + ".__sampler" in self.var_map:
                sam_name = sampler_arg.name
                sam_type = self.var_types.get(sam_name, "sampler2DArray")
                img_type, sampled_img_type = self._resolve_sampler_types(sam_type)
                tex_loaded = self.reg.next_id()
                lines.append(f"{tex_loaded} = OpLoad {img_type} {self.var_map[sam_name + '.__texture']}")
                samp_loaded = self.reg.next_id()
                lines.append(f"{samp_loaded} = OpLoad {self.reg.sampler_type()} {self.var_map[sam_name + '.__sampler']}")
                combined = self.reg.next_id()
                lines.append(f"{combined} = OpSampledImage {sampled_img_type} {tex_loaded} {samp_loaded}")
                lod_zero = self.reg.const_float(0.0)
                lines.append(f"{result} = OpImageSampleExplicitLod {result_type} {combined} {coord3} Lod {lod_zero}")
            return result, lines

        # sample_bindless(texture_array, index, uv) — bindless implicit LOD
        if fname == "sample_bindless":
            arr_arg = expr.args[0]
            idx_id, idx_lines = self._gen_expr(expr.args[1])
            lines.extend(idx_lines)
            uv_id, uv_lines = self._gen_expr(expr.args[2])
            lines.extend(uv_lines)
            result_type = self.reg.lux_type_to_spirv(expr.resolved_type)
            result = self.reg.next_id()
            if isinstance(arr_arg, VarRef) and self.var_types.get(arr_arg.name) == "_bindless_texture_array":
                arr_var = self.var_map[arr_arg.name]
                cis_type = self.reg.combined_image_sampler_type()
                ptr_cis = self.reg.pointer("UniformConstant", cis_type)
                # NonUniform decoration on index
                self.decorations.append(f"OpDecorate {idx_id} NonUniform")
                # Access chain into the runtime array
                ac_id = self.reg.next_id()
                lines.append(f"{ac_id} = OpAccessChain {ptr_cis} {arr_var} {idx_id}")
                self.decorations.append(f"OpDecorate {ac_id} NonUniform")
                # Load combined image sampler
                loaded = self.reg.next_id()
                lines.append(f"{loaded} = OpLoad {cis_type} {ac_id}")
                self.decorations.append(f"OpDecorate {loaded} NonUniform")
                # Sample
                lines.append(f"{result} = OpImageSampleImplicitLod {result_type} {loaded} {uv_id}")
            return result, lines

        # sample_bindless_lod(texture_array, index, uv, lod) — bindless explicit LOD
        if fname == "sample_bindless_lod":
            arr_arg = expr.args[0]
            idx_id, idx_lines = self._gen_expr(expr.args[1])
            lines.extend(idx_lines)
            uv_id, uv_lines = self._gen_expr(expr.args[2])
            lines.extend(uv_lines)
            lod_id, lod_lines = self._gen_expr(expr.args[3])
            lines.extend(lod_lines)
            result_type = self.reg.lux_type_to_spirv(expr.resolved_type)
            result = self.reg.next_id()
            if isinstance(arr_arg, VarRef) and self.var_types.get(arr_arg.name) == "_bindless_texture_array":
                arr_var = self.var_map[arr_arg.name]
                cis_type = self.reg.combined_image_sampler_type()
                ptr_cis = self.reg.pointer("UniformConstant", cis_type)
                # NonUniform decoration on index
                self.decorations.append(f"OpDecorate {idx_id} NonUniform")
                # Access chain into the runtime array
                ac_id = self.reg.next_id()
                lines.append(f"{ac_id} = OpAccessChain {ptr_cis} {arr_var} {idx_id}")
                self.decorations.append(f"OpDecorate {ac_id} NonUniform")
                # Load combined image sampler
                loaded = self.reg.next_id()
                lines.append(f"{loaded} = OpLoad {cis_type} {ac_id}")
                self.decorations.append(f"OpDecorate {loaded} NonUniform")
                # Sample with explicit LOD
                lines.append(f"{result} = OpImageSampleExplicitLod {result_type} {loaded} {uv_id} Lod {lod_id}")
            return result, lines

        # texture_levels(sampler) -> int — handle before general arg generation
        # since sampler is split into __sampler + __texture
        if fname == "texture_levels":
            self._needs_image_query = True
            sampler_arg = expr.args[0]
            result_type = self.reg.int32()
            result = self.reg.next_id()
            if isinstance(sampler_arg, VarRef) and sampler_arg.name + ".__texture" in self.var_map:
                sam_name = sampler_arg.name
                img_type = self._sampler_image_type(sampler_arg)
                tex_loaded = self.reg.next_id()
                lines.append(f"{tex_loaded} = OpLoad {img_type} {self.var_map[sam_name + '.__texture']}")
                lines.append(f"{result} = OpImageQueryLevels {result_type} {tex_loaded}")
            else:
                # Fallback: try loading as combined image sampler
                tex_id, tex_lines = self._gen_expr(sampler_arg)
                lines.extend(tex_lines)
                img_id = self.reg.next_id()
                lines.append(f"{img_id} = OpImage {self._sampler_image_type(sampler_arg)} {tex_id}")
                lines.append(f"{result} = OpImageQueryLevels {result_type} {img_id}")
            return result, lines

        # texture_size(sampler, lod) -> ivec2 — handle before general arg generation
        if fname == "texture_size":
            self._needs_image_query = True
            sampler_arg = expr.args[0]
            lod_id, lod_lines = self._gen_expr(expr.args[1])
            lines.extend(lod_lines)
            # Convert lod to int if needed
            lod_resolved = resolve_alias_chain(getattr(expr.args[1], 'resolved_type', 'int'))
            if lod_resolved == "scalar":
                int_t = self.reg.int32()
                conv = self.reg.next_id()
                lines.append(f"{conv} = OpConvertFToS {int_t} {lod_id}")
                lod_id = conv
            ivec2_type = self.reg.vec(2, "int")
            result = self.reg.next_id()
            if isinstance(sampler_arg, VarRef) and sampler_arg.name + ".__texture" in self.var_map:
                sam_name = sampler_arg.name
                img_type = self._sampler_image_type(sampler_arg)
                tex_loaded = self.reg.next_id()
                lines.append(f"{tex_loaded} = OpLoad {img_type} {self.var_map[sam_name + '.__texture']}")
                lines.append(f"{result} = OpImageQuerySizeLod {ivec2_type} {tex_loaded} {lod_id}")
            else:
                tex_id, tex_lines = self._gen_expr(sampler_arg)
                lines.extend(tex_lines)
                img_id = self.reg.next_id()
                lines.append(f"{img_id} = OpImage {self._sampler_image_type(sampler_arg)} {tex_id}")
                lines.append(f"{result} = OpImageQuerySizeLod {ivec2_type} {img_id} {lod_id}")
            return result, lines

        # Generate arguments
        arg_ids = []
        for arg in expr.args:
            aid, alines = self._gen_expr(arg)
            lines.extend(alines)
            arg_ids.append(aid)

        result_type = self.reg.lux_type_to_spirv(expr.resolved_type)
        result = self.reg.next_id()

        # atan with 2 args -> Atan2 (not Atan)
        if fname == "atan" and len(arg_ids) == 2:
            args_str = " ".join(arg_ids)
            lines.append(f"{result} = OpExtInst {result_type} {self.glsl_ext_id} Atan2 {args_str}")
            return result, lines

        # mod -> OpFMod (core SPIR-V op, not GLSL.std.450)
        if fname == "mod":
            lines.append(f"{result} = OpFMod {result_type} {arg_ids[0]} {arg_ids[1]}")
            return result, lines

        # RT instructions
        if fname == "trace_ray":
            # trace_ray(accel, ray_flags, cull_mask, sbt_offset, sbt_stride, miss_index,
            #           origin, tmin, direction, tmax, payload_loc)
            # OpTraceRayKHR is void, no result
            # Args 1-5 must be uint; last arg is payload variable (not location index)
            accel_id = arg_ids[0]
            uint_type = self.reg.uint32()
            converted = list(arg_ids[1:])
            # Indices 0-4 (ray_flags..miss_index) need uint
            for i in range(5):
                if i < len(converted):
                    conv_id = self.reg.next_id()
                    lines.append(f"{conv_id} = OpConvertFToU {uint_type} {converted[i]}")
                    converted[i] = conv_id
            # Last arg (payload_loc) must be the payload OpVariable, not a location number
            if len(converted) > 9:
                # Get the payload location (as a constant, e.g. 0)
                payload_loc = 0  # default
                last_arg = expr.args[-1]
                if hasattr(last_arg, 'value'):
                    payload_loc = int(last_arg.value)
                # Look up the payload variable
                payload_vars = getattr(self, '_payload_vars', {})
                if payload_loc in payload_vars:
                    converted[9] = payload_vars[payload_loc]
                else:
                    # Fallback: find first payload variable
                    for pv in payload_vars.values():
                        converted[9] = pv
                        break
            args_str = " ".join(converted)
            lines.append(f"OpTraceRayKHR {accel_id} {args_str}")
            # Return a dummy zero for the expression system
            result = self.reg.const_float(0.0)
            return result, lines

        if fname == "report_intersection":
            # report_intersection(hit_t, hit_kind) -> bool
            bool_type = self.reg.bool_type()
            result = self.reg.next_id()
            lines.append(f"{result} = OpReportIntersectionKHR {bool_type} {arg_ids[0]} {arg_ids[1]}")
            return result, lines

        if fname == "execute_callable":
            # execute_callable(sbt_index, callable_data_loc) -> void
            lines.append(f"OpExecuteCallableKHR {arg_ids[0]} {arg_ids[1]}")
            result = self.reg.const_float(0.0)
            return result, lines

        if fname == "ignore_intersection":
            lines.append("OpIgnoreIntersectionKHR")
            result = self.reg.const_float(0.0)
            return result, lines

        if fname == "terminate_ray":
            lines.append("OpTerminateRayKHR")
            result = self.reg.const_float(0.0)
            return result, lines

        # Mesh shader instructions
        if fname == "set_mesh_outputs":
            # set_mesh_outputs(vert_count, prim_count) -> void
            uint_type = self.reg.uint32()
            # Convert args to uint if float
            vert_id = arg_ids[0]
            prim_id = arg_ids[1]
            vert_lux_type = self._resolve_expr_lux_type(expr.args[0])
            prim_lux_type = self._resolve_expr_lux_type(expr.args[1])
            if vert_lux_type == "scalar":
                conv = self.reg.next_id()
                lines.append(f"{conv} = OpConvertFToU {uint_type} {vert_id}")
                vert_id = conv
            if prim_lux_type == "scalar":
                conv = self.reg.next_id()
                lines.append(f"{conv} = OpConvertFToU {uint_type} {prim_id}")
                prim_id = conv
            lines.append(f"OpSetMeshOutputsEXT {vert_id} {prim_id}")
            result = self.reg.const_float(0.0)
            return result, lines

        if fname == "emit_mesh_tasks":
            # emit_mesh_tasks(gx, gy, gz) -> void
            uint_type = self.reg.uint32()
            converted = []
            for i, aid in enumerate(arg_ids):
                arg_lux_type = self._resolve_expr_lux_type(expr.args[i])
                if arg_lux_type == "scalar":
                    conv = self.reg.next_id()
                    lines.append(f"{conv} = OpConvertFToU {uint_type} {aid}")
                    converted.append(conv)
                else:
                    converted.append(aid)
            # Task payload variable (if exists)
            tp_vars = getattr(self.stage, 'task_payloads', [])
            if tp_vars:
                tp_var = self.var_map.get(tp_vars[0].name, converted[0])
                lines.append(f"OpEmitMeshTasksEXT {converted[0]} {converted[1]} {converted[2]} {tp_var}")
            else:
                lines.append(f"OpEmitMeshTasksEXT {converted[0]} {converted[1]} {converted[2]}")
            result = self.reg.const_float(0.0)
            return result, lines

        if fname == "barrier":
            # barrier() — workgroup execution + memory barrier
            scope_wg = self.reg.const_uint(2)       # Workgroup scope
            semantics = self.reg.const_uint(0x108)   # AcquireRelease | WorkgroupMemory
            lines.append(f"OpControlBarrier {scope_wg} {scope_wg} {semantics}")
            result = self.reg.const_float(0.0)  # void return placeholder
            return result, lines

        if fname == "image_store":
            # image_store(image, coord_ivec2, value_vec4)
            # OpImageWrite image coord value
            # The image arg was already loaded by _gen_expr; arg_ids[0] IS the loaded image
            img_loaded = arg_ids[0]
            # Convert coord to ivec2 if needed (launch_id.xy is uvec2 or vec2)
            coord_id = arg_ids[1]
            value_id = arg_ids[2]
            # Coord must be integer — convert from float if needed
            coord_type_name = getattr(expr.args[1], 'resolved_type', 'vec2')
            coord_resolved = resolve_type(resolve_alias_chain(coord_type_name))
            if isinstance(coord_resolved, VectorType) and coord_resolved.component == "scalar":
                # Convert float vec2 to int ivec2
                ivec2_type = self.reg.vec(2, "int")
                conv_id = self.reg.next_id()
                lines.append(f"{conv_id} = OpConvertFToS {ivec2_type} {coord_id}")
                coord_id = conv_id
            elif isinstance(coord_resolved, VectorType) and coord_resolved.component == "uint":
                # Convert uint uvec2 to int ivec2
                ivec2_type = self.reg.vec(2, "int")
                conv_id = self.reg.next_id()
                lines.append(f"{conv_id} = OpBitcast {ivec2_type} {coord_id}")
                coord_id = conv_id
            lines.append(f"OpImageWrite {img_loaded} {coord_id} {value_id}")
            result = self.reg.const_float(0.0)
            return result, lines

        # image_size(image) -> ivec2/ivec3 (OpImageQuerySize)
        if fname == "image_size":
            self._needs_image_query = True
            img_loaded = arg_ids[0]
            ivec2_type = self.reg.vec(2, "int")
            lines.append(f"{result} = OpImageQuerySize {ivec2_type} {img_loaded}")
            return result, lines

        # mix/clamp/smoothstep: splat scalar args to vector when needed
        if fname in ("mix", "clamp", "smoothstep") and len(expr.args) >= 3:
            first_type = resolve_type(resolve_alias_chain(expr.args[0].resolved_type))
            if isinstance(first_type, VectorType):
                for i in range(1, len(arg_ids)):
                    arg_type = resolve_type(resolve_alias_chain(expr.args[i].resolved_type))
                    if isinstance(arg_type, ScalarType):
                        vec_type = self.reg.lux_type_to_spirv(first_type.name)
                        splat_id = self.reg.next_id()
                        parts = " ".join([arg_ids[i]] * first_type.size)
                        lines.append(f"{splat_id} = OpCompositeConstruct {vec_type} {parts}")
                        arg_ids[i] = splat_id

        # Check if it's a GLSL.std.450 function
        glsl_name = LUX_TO_GLSL.get(fname)
        if glsl_name:
            args_str = " ".join(arg_ids)
            lines.append(f"{result} = OpExtInst {result_type} {self.glsl_ext_id} {glsl_name} {args_str}")
            return result, lines

        # dot -> OpDot
        if fname == "dot":
            lines.append(f"{result} = OpDot {result_type} {arg_ids[0]} {arg_ids[1]}")
            return result, lines

        # transpose -> OpTranspose (core SPIR-V, not GLSL.std.450)
        if fname == "transpose":
            lines.append(f"{result} = OpTranspose {result_type} {arg_ids[0]}")
            return result, lines

        # any_nan / any_inf -> OpIsNan / OpIsInf (core SPIR-V ops)
        if fname in ("any_nan", "any_inf"):
            op = "OpIsNan" if fname == "any_nan" else "OpIsInf"
            arg_type_name = resolve_alias_chain(expr.args[0].resolved_type)
            arg_lux_type = resolve_type(arg_type_name)
            bool_type = self.reg.bool_type()
            if isinstance(arg_lux_type, VectorType):
                bvec_type = self.reg.vec(arg_lux_type.size, "bool")
                nan_vec = self.reg.next_id()
                lines.append(f"{nan_vec} = {op} {bvec_type} {arg_ids[0]}")
                lines.append(f"{result} = OpAny {bool_type} {nan_vec}")
            else:
                lines.append(f"{result} = {op} {bool_type} {arg_ids[0]}")
            return result, lines

        # User-defined function — inline it
        fn_def = self._find_user_function(fname)
        if fn_def is not None:
            return self._gen_inline_call(fn_def, arg_ids, result_type, lines)

        raise ValueError(f"Unknown function in codegen: {fname}")

    def _find_user_function(self, name: str) -> FunctionDef | None:
        for fn in self.module.functions:
            if fn.name == name:
                return fn
        for fn in self.stage.functions:
            if fn.name == name and fn.name != "main":
                return fn
        return None

    def _gen_inline_call(
        self, fn: FunctionDef, arg_ids: list[str], result_type_id: str,
        lines: list[str]
    ) -> tuple[str, list[str]]:
        """Inline a user-defined function call by evaluating its body."""
        # Save and create new local var scope for the inlined function
        saved_locals = self.local_vars
        saved_local_types = self.local_types
        saved_ssa = self._ssa_values
        saved_mutable = self._mutable_vars
        self.local_vars = dict(self.local_vars)
        self.local_types = dict(self.local_types)
        self._ssa_values = dict(self._ssa_values)

        # Use unique names to avoid collisions with outer scope
        inline_id = self.reg.next_id().replace("%", "")

        # Scan inlined function body for mutable vars (release mode)
        inline_mutable = set()
        if not self.debug:
            old_mutable = self._mutable_vars
            self._mutable_vars = set()
            self._scan_mutable_vars(fn.body)
            self._scan_forward_refs(fn.body)
            inline_mutable = self._mutable_vars
            self._mutable_vars = inline_mutable

        # Bind parameters to argument SSA ids
        for param, arg_id in zip(fn.params, arg_ids):
            if not self.debug and param.name not in inline_mutable:
                # SSA path: forward argument value directly
                self._ssa_values[param.name] = arg_id
            else:
                # OpVariable path for mutable params
                unique_name = f"_inline_{inline_id}_{param.name}"
                self._alloc_local(unique_name, param.type_name)
                lines.append(f"OpStore {self.local_vars[unique_name]} {arg_id}")
                self.local_vars[param.name] = self.local_vars[unique_name]
                self.local_types[param.name] = param.type_name

        # Pre-declare locals for the inlined function body
        for stmt in fn.body:
            if isinstance(stmt, LetStmt):
                if not self.debug and stmt.name not in inline_mutable:
                    continue  # SSA-eligible, handled in _gen_stmt
                unique_name = f"_inline_{inline_id}_{stmt.name}"
                self._alloc_local(unique_name, stmt.type_name)
                self.local_vars[stmt.name] = self.local_vars[unique_name]
                self.local_types[stmt.name] = stmt.type_name

        # Execute body statements, capture return value
        return_val = None
        for stmt in fn.body:
            if isinstance(stmt, ReturnStmt):
                val_id, val_lines = self._gen_expr(stmt.value)
                lines.extend(val_lines)
                return_val = val_id
                break
            else:
                lines.extend(self._gen_stmt(stmt))

        # Restore scope
        self.local_vars = saved_locals
        self.local_types = saved_local_types
        self._ssa_values = saved_ssa
        self._mutable_vars = saved_mutable

        if return_val is None:
            raise ValueError(f"Inlined function '{fn.name}' has no return statement")
        return return_val, lines

    def _gen_constructor(self, expr: ConstructorExpr, lines: list[str]) -> tuple[str, list[str]]:
        type_name = expr.type_name
        target_type = resolve_type(type_name)
        result_type_id = self.reg.lux_type_to_spirv(type_name)

        # Evaluate all args
        arg_ids = []
        arg_type_names = []
        for arg in expr.args:
            aid, alines = self._gen_expr(arg)
            lines.extend(alines)
            arg_ids.append(aid)
            arg_type_names.append(self._resolve_expr_type_name(arg))

        if isinstance(target_type, VectorType):
            target_size = target_type.size
            target_comp = target_type.component  # "scalar"/"float", "int", or "uint"

            # Constant vector hoisting: if ALL args are NumberLit, emit OpConstantComposite
            if all(isinstance(a, NumberLit) for a in expr.args):
                if target_comp in ("scalar", "float"):
                    const_ids = [self.reg.const_float(float(a.value)) for a in expr.args]
                elif target_comp == "int":
                    const_ids = [self.reg.const_int(int(float(a.value)), signed=True) for a in expr.args]
                elif target_comp == "uint":
                    const_ids = [self.reg.const_int(int(float(a.value)), signed=False) for a in expr.args]
                else:
                    const_ids = None
                if const_ids is not None:
                    if len(const_ids) == 1 and target_size > 1:
                        const_ids = const_ids * target_size
                    if len(const_ids) == target_size:
                        return self.reg.const_composite(result_type_id, const_ids), lines

            # Expand args: e.g., vec4(vec3, scalar) -> 4 components
            components = []
            for aid, atn in zip(arg_ids, arg_type_names):
                at = resolve_type(atn)
                if isinstance(at, VectorType):
                    # Extract each component using the source vector's component type
                    comp_type = self.reg._component_type(
                        at.component if at.component != "scalar" else "float"
                    )
                    for i in range(at.size):
                        c = self.reg.next_id()
                        lines.append(f"{c} = OpCompositeExtract {comp_type} {aid} {i}")
                        # Convert to float if source is integer but target is float vector
                        if at.component in ("int", "uint") and target_comp in ("scalar", "float"):
                            fc = self.reg.next_id()
                            float_type = self.reg.float32()
                            if at.component == "int":
                                lines.append(f"{fc} = OpConvertSToF {float_type} {c}")
                            else:
                                lines.append(f"{fc} = OpConvertUToF {float_type} {c}")
                            components.append(fc)
                        # Convert to int/uint if source is float but target is integer vector
                        elif at.component in ("scalar", "float") and target_comp in ("int", "uint"):
                            ic = self.reg.next_id()
                            int_type = self.reg._component_type(target_comp)
                            conv_op = "OpConvertFToS" if target_comp == "int" else "OpConvertFToU"
                            lines.append(f"{ic} = {conv_op} {int_type} {c}")
                            components.append(ic)
                        else:
                            components.append(c)
                else:
                    # Scalar arg — may need conversion if target is int/uint vector
                    # and the source is a float (which is the default for NumberLit)
                    src_type = resolve_type(resolve_alias_chain(atn)) if atn else None
                    src_is_float = (
                        src_type is None
                        or (isinstance(src_type, ScalarType) and src_type.name in ("scalar", "float"))
                    )
                    src_is_int = isinstance(src_type, ScalarType) and src_type.name in ("int", "uint")
                    if target_comp in ("int", "uint") and src_is_float:
                        ic = self.reg.next_id()
                        int_type = self.reg._component_type(target_comp)
                        conv_op = "OpConvertFToS" if target_comp == "int" else "OpConvertFToU"
                        lines.append(f"{ic} = {conv_op} {int_type} {aid}")
                        components.append(ic)
                    elif target_comp in ("scalar", "float") and src_is_int:
                        fc = self.reg.next_id()
                        float_type = self.reg.float32()
                        conv_op = "OpConvertSToF" if src_type.name == "int" else "OpConvertUToF"
                        lines.append(f"{fc} = {conv_op} {float_type} {aid}")
                        components.append(fc)
                    else:
                        components.append(aid)

            # If single scalar arg, splat to all components
            if len(components) == 1 and target_size > 1:
                components = components * target_size

            if len(components) != target_size:
                raise ValueError(
                    f"Constructor {type_name} expects {target_size} components, got {len(components)}"
                )

            result = self.reg.next_id()
            parts = " ".join(components)
            lines.append(f"{result} = OpCompositeConstruct {result_type_id} {parts}")
            return result, lines

        elif isinstance(target_type, MatrixType):
            # mat4(col0, col1, col2, col3) or mat4(scalar) for identity-ish
            result = self.reg.next_id()
            parts = " ".join(arg_ids)
            lines.append(f"{result} = OpCompositeConstruct {result_type_id} {parts}")
            return result, lines

        raise ValueError(f"Unknown constructor type: {type_name}")

    def _gen_swizzle(self, expr: SwizzleAccess, lines: list[str]) -> tuple[str, list[str]]:
        obj_id, obj_lines = self._gen_expr(expr.object)
        lines.extend(obj_lines)

        components = expr.components
        indices = [_swizzle_index(c) for c in components]

        # Determine the component type from the source vector
        obj_type_name = getattr(expr.object, 'resolved_type', 'vec4')
        obj_type = resolve_type(resolve_alias_chain(obj_type_name))
        component_kind = "float"
        if isinstance(obj_type, VectorType):
            component_kind = obj_type.component if obj_type.component != "scalar" else "float"

        # Determine expected result type from the expression's resolved_type
        expr_resolved = getattr(expr, 'resolved_type', 'scalar')
        needs_float_conv = component_kind in ("uint", "int") and expr_resolved in ("scalar", "vec2", "vec3", "vec4")

        if len(indices) == 1:
            # Single component extract — use source type, then convert if needed
            extract_type = self.reg._component_type(component_kind)
            extract_id = self.reg.next_id()
            lines.append(f"{extract_id} = OpCompositeExtract {extract_type} {obj_id} {indices[0]}")

            if needs_float_conv:
                result = self.reg.next_id()
                float_type = self.reg.float32()
                conv_op = "OpConvertSToF" if component_kind == "int" else "OpConvertUToF"
                lines.append(f"{result} = {conv_op} {float_type} {extract_id}")
                return result, lines
            return extract_id, lines
        else:
            # Multi-component shuffle
            n = len(indices)
            if component_kind == "uint":
                shuffle_type = self.reg.lux_type_to_spirv(f"uvec{n}")
            elif component_kind == "int":
                shuffle_type = self.reg.lux_type_to_spirv(f"ivec{n}")
            else:
                shuffle_type = self.reg.lux_type_to_spirv(f"vec{n}")
            shuffle_id = self.reg.next_id()
            idx_str = " ".join(str(i) for i in indices)
            lines.append(f"{shuffle_id} = OpVectorShuffle {shuffle_type} {obj_id} {obj_id} {idx_str}")

            if needs_float_conv:
                result = self.reg.next_id()
                float_vec_type = self.reg.lux_type_to_spirv(f"vec{n}")
                conv_op = "OpConvertSToF" if component_kind == "int" else "OpConvertUToF"
                lines.append(f"{result} = {conv_op} {float_vec_type} {shuffle_id}")
                return result, lines
            return shuffle_id, lines

    def _gen_field_access(self, expr: FieldAccess, lines: list[str]) -> tuple[str, list[str]]:
        # Handle field access on struct-typed storage buffer elements:
        # materials[mat_idx].baseColorFactor -> OpAccessChain into SSBO struct
        if (isinstance(expr.object, IndexAccess)
            and isinstance(expr.object.object, VarRef)
            and expr.object.object.name in getattr(self, '_ssbo_struct_fields', {})):
            buf_name = expr.object.object.name
            buf_var_id = self.storage_buffer_var_ids[buf_name]
            field_name = expr.field
            fields = self._ssbo_struct_fields[buf_name]

            # Find field index and type
            field_idx = None
            field_type = None
            for i, (fname, ftype) in enumerate(fields):
                if fname == field_name:
                    field_idx = i
                    field_type = ftype
                    break
            if field_idx is None:
                raise ValueError(f"Unknown field '{field_name}' in SSBO struct '{buf_name}'")

            # Generate array index expression
            idx_id, idx_lines = self._gen_expr(expr.object.index)
            lines.extend(idx_lines)

            # Convert float index to int if needed
            idx_lux_type = self._resolve_expr_lux_type(expr.object.index)
            if idx_lux_type in ("scalar", "vec2", "vec3", "vec4"):
                conv_id = self.reg.next_id()
                int_type = self.reg.int32()
                lines.append(f"{conv_id} = OpConvertFToS {int_type} {idx_id}")
                idx_id = conv_id

            # OpAccessChain: buf_var -> 0 (runtime array wrapper) -> idx -> field_idx
            type_id = self.reg.lux_type_to_spirv(field_type)
            ptr_type = self.reg.pointer("StorageBuffer", type_id)
            const_0 = self.reg.const_int(0, signed=True)
            field_const = self.reg.const_int(field_idx, signed=True)
            ac_id = self.reg.next_id()
            lines.append(f"{ac_id} = OpAccessChain {ptr_type} {buf_var_id} {const_0} {idx_id} {field_const}")
            result = self.reg.next_id()
            lines.append(f"{result} = OpLoad {type_id} {ac_id}")
            return result, lines

        # Handle qualified access to uniform block fields: Material.roughness_factor
        if isinstance(expr.object, VarRef) and expr.object.name in self.uniform_var_ids:
            block_name = expr.object.name
            var_id = self.uniform_var_ids[block_name]
            field_name = expr.field
            # Find the field index in this block
            for ub in self.stage.uniforms:
                if ub.name == block_name:
                    for i, f in enumerate(ub.fields):
                        if f.name == field_name:
                            field_type = f.type_name
                            type_id = self.reg.lux_type_to_spirv(field_type)
                            ptr_type = self.reg.pointer("Uniform", type_id)
                            idx_id = self.reg.const_int(i, signed=True)
                            ac_id = self.reg.next_id()
                            lines.append(f"{ac_id} = OpAccessChain {ptr_type} {var_id} {idx_id}")
                            result = self.reg.next_id()
                            lines.append(f"{result} = OpLoad {type_id} {ac_id}")
                            return result, lines
            raise ValueError(f"Unknown field '{field_name}' in uniform block '{block_name}'")

        # Handle qualified access to push constant fields
        if isinstance(expr.object, VarRef) and expr.object.name in self.push_var_ids:
            block_name = expr.object.name
            var_id = self.push_var_ids[block_name]
            field_name = expr.field
            for pb in self.stage.push_constants:
                if pb.name == block_name:
                    for i, f in enumerate(pb.fields):
                        if f.name == field_name:
                            field_type = f.type_name
                            type_id = self.reg.lux_type_to_spirv(field_type)
                            ptr_type = self.reg.pointer("PushConstant", type_id)
                            idx_id = self.reg.const_int(i, signed=True)
                            ac_id = self.reg.next_id()
                            lines.append(f"{ac_id} = OpAccessChain {ptr_type} {var_id} {idx_id}")
                            result = self.reg.next_id()
                            lines.append(f"{result} = OpLoad {type_id} {ac_id}")
                            return result, lines
            raise ValueError(f"Unknown field '{field_name}' in push constant block '{block_name}'")

        # Default: struct.field via composite extract
        obj_id, obj_lines = self._gen_expr(expr.object)
        lines.extend(obj_lines)

        # Resolve correct field index and type from struct metadata
        field_idx = 0
        result_type = self.reg.float32()
        struct_type_name = self._resolve_expr_lux_type(expr.object)
        # Also check local_types for VarRef to CSE-generated struct locals
        if struct_type_name == "scalar" and isinstance(expr.object, VarRef):
            struct_type_name = self.local_types.get(expr.object.name, struct_type_name)
        struct_fields_map = {
            "LightData": _LIGHT_DATA_FIELDS,
            "BindlessMaterialData": getattr(self, '_bindless_material_dynamic_fields',
                                            _BINDLESS_MATERIAL_FIELDS),
            "ShadowEntry": _SHADOW_ENTRY_FIELDS,
        }
        fields = struct_fields_map.get(struct_type_name)
        if fields:
            for i, (fname, ftype) in enumerate(fields):
                if fname == expr.field:
                    field_idx = i
                    result_type = self.reg.lux_type_to_spirv(ftype)
                    break

        result = self.reg.next_id()
        lines.append(f"{result} = OpCompositeExtract {result_type} {obj_id} {field_idx}")
        return result, lines

    def _gen_index_access(self, expr: IndexAccess, lines: list[str]) -> tuple[str, list[str]]:
        # Check for shared memory array access: shared_arr[index]
        if isinstance(expr.object, VarRef) and expr.object.name in self.shared_var_ids:
            name = expr.object.name
            if self.shared_array_sizes.get(name) is not None:
                # It's a shared array — OpAccessChain into Workgroup array
                elem_type_name = self.shared_element_types[name]
                elem_type_id = self.reg.lux_type_to_spirv(elem_type_name)
                var_id = self.shared_var_ids[name]

                idx_id, idx_lines = self._gen_expr(expr.index)
                lines.extend(idx_lines)

                # Convert index to int if float
                idx_lux_type = self._resolve_expr_lux_type(expr.index)
                if idx_lux_type in ("scalar",):
                    int_type = self.reg.int32()
                    conv = self.reg.next_id()
                    lines.append(f"{conv} = OpConvertFToS {int_type} {idx_id}")
                    idx_id = conv

                ptr_elem = self.reg.pointer("Workgroup", elem_type_id)
                ac_id = self.reg.next_id()
                lines.append(f"{ac_id} = OpAccessChain {ptr_elem} {var_id} {idx_id}")
                result = self.reg.next_id()
                lines.append(f"{result} = OpLoad {elem_type_id} {ac_id}")
                return result, lines

        # Check for storage buffer access: buffer[index]
        if isinstance(expr.object, VarRef) and expr.object.name in self.storage_buffer_var_ids:
            buf_name = expr.object.name
            buf_var_id = self.storage_buffer_var_ids[buf_name]
            elem_type_name = self.storage_buffer_element_types[buf_name]
            elem_type_id = self.reg.lux_type_to_spirv(elem_type_name)

            # Generate index expression
            idx_id, idx_lines = self._gen_expr(expr.index)
            lines.extend(idx_lines)

            # Convert index to int if it's a float
            idx_lux_type = self._resolve_expr_lux_type(expr.index)
            if idx_lux_type in ("scalar", "vec2", "vec3", "vec4"):
                conv_id = self.reg.next_id()
                int_type = self.reg.int32()
                lines.append(f"{conv_id} = OpConvertFToS {int_type} {idx_id}")
                idx_id = conv_id

            # OpAccessChain: struct_ptr -> member[0] -> array[idx]
            ptr_elem = self.reg.pointer("StorageBuffer", elem_type_id)
            const_0 = self.reg.const_int(0, signed=True)
            ac_id = self.reg.next_id()
            lines.append(f"{ac_id} = OpAccessChain {ptr_elem} {buf_var_id} {const_0} {idx_id}")
            result = self.reg.next_id()
            lines.append(f"{result} = OpLoad {elem_type_id} {ac_id}")
            return result, lines

        # Default: composite extract (vectors, matrices)
        obj_id, obj_lines = self._gen_expr(expr.object)
        lines.extend(obj_lines)
        idx_id, idx_lines = self._gen_expr(expr.index)
        lines.extend(idx_lines)
        result_type = self.reg.lux_type_to_spirv(expr.resolved_type)
        result = self.reg.next_id()
        lines.append(f"{result} = OpCompositeExtract {result_type} {obj_id} 0")  # simplified
        return result, lines

    def _gen_ternary(self, expr: TernaryExpr, lines: list[str]) -> tuple[str, list[str]]:
        cond_id, cond_lines = self._gen_expr(expr.condition)
        lines.extend(cond_lines)
        then_id, then_lines = self._gen_expr(expr.then_expr)
        lines.extend(then_lines)
        else_id, else_lines = self._gen_expr(expr.else_expr)
        lines.extend(else_lines)
        result_type = self.reg.lux_type_to_spirv(expr.resolved_type)
        result = self.reg.next_id()
        lines.append(f"{result} = OpSelect {result_type} {cond_id} {then_id} {else_id}")
        return result, lines

    def _resolve_expr_lux_type(self, expr) -> str:
        if hasattr(expr, "resolved_type") and expr.resolved_type:
            return expr.resolved_type
        return "scalar"

    def _resolve_expr_type_name(self, expr) -> str:
        return self._resolve_expr_lux_type(expr)

    def _resolve_store_target_type(self, target) -> str | None:
        """Resolve the lux type name for a store target (variable)."""
        if isinstance(target, VarRef):
            name = target.name
            if name in self.local_types:
                return self.local_types[name]
            if name in self.var_types:
                return self.var_types[name]
        elif isinstance(target, IndexAccess) and isinstance(target.object, VarRef):
            name = target.object.name
            if name in self.shared_element_types:
                return self.shared_element_types[name]
            if name in self.storage_buffer_element_types:
                return self.storage_buffer_element_types[name]
        return None

    def _coerce_store_value(self, target, value_expr, val_id: str, lines: list[str]) -> str:
        """Coerce a store value to match the target type (e.g., float -> uint for shared/SSBO)."""
        val_type = self._resolve_expr_lux_type(value_expr)
        if val_type not in ("scalar",):
            return val_id  # No coercion needed if not float

        # Determine target element type
        target_type_name = None
        if isinstance(target, IndexAccess) and isinstance(target.object, VarRef):
            name = target.object.name
            if name in self.shared_element_types:
                target_type_name = self.shared_element_types[name]
            elif name in self.storage_buffer_element_types:
                target_type_name = self.storage_buffer_element_types[name]
        elif isinstance(target, VarRef):
            name = target.name
            if name in self.shared_element_types:
                target_type_name = self.shared_element_types[name]

        if target_type_name in ("uint", "int"):
            target_type_id = self.reg.lux_type_to_spirv(target_type_name)
            conv = self.reg.next_id()
            conv_op = "OpConvertFToU" if target_type_name == "uint" else "OpConvertFToS"
            lines.append(f"{conv} = {conv_op} {target_type_id} {val_id}")
            return conv

        return val_id

    # --- Atomic operations codegen ---

    def _gen_atomic_call(self, fname: str, expr: CallExpr, lines: list[str]) -> tuple[str, list[str]]:
        """Generate SPIR-V for atomic operations on shared memory or SSBOs."""
        # First arg is the memory location — need pointer, not value
        first_arg = expr.args[0]
        ptr_id, ptr_lines = self._gen_atomic_pointer(first_arg)
        lines.extend(ptr_lines)

        # Determine scope and semantics
        is_shared = self._is_shared_ref(first_arg)
        scope_const = self.reg.const_uint(2) if is_shared else self.reg.const_uint(1)  # Workgroup=2, Device=1
        semantics_const = self.reg.const_uint(0x108)  # AcquireRelease | WorkgroupMemory

        # Determine result type from the element type
        elem_type_name = self._resolve_atomic_element_type(first_arg)
        elem_type_id = self.reg.lux_type_to_spirv(elem_type_name)

        # Load remaining args normally
        remaining_arg_ids = []
        for arg in expr.args[1:]:
            aid, alines = self._gen_expr(arg)
            lines.extend(alines)
            # Coerce float literals to int/uint if element type is integer
            if elem_type_name in ("int", "uint") and self._resolve_expr_lux_type(arg) == "scalar":
                if isinstance(arg, NumberLit) and "." not in arg.value:
                    if elem_type_name == "uint":
                        aid = self.reg.const_uint(int(arg.value))
                    else:
                        aid = self.reg.const_int(int(arg.value), signed=True)
                else:
                    conv_type = self.reg.uint32() if elem_type_name == "uint" else self.reg.int32()
                    conv = self.reg.next_id()
                    conv_op = "OpConvertFToU" if elem_type_name == "uint" else "OpConvertFToS"
                    lines.append(f"{conv} = {conv_op} {conv_type} {aid}")
                    aid = conv
            remaining_arg_ids.append(aid)

        result = self.reg.next_id()

        if fname == "atomic_add":
            lines.append(f"{result} = OpAtomicIAdd {elem_type_id} {ptr_id} {scope_const} {semantics_const} {remaining_arg_ids[0]}")
            return result, lines

        if fname == "atomic_min":
            op = "OpAtomicSMin" if elem_type_name == "int" else "OpAtomicUMin"
            lines.append(f"{result} = {op} {elem_type_id} {ptr_id} {scope_const} {semantics_const} {remaining_arg_ids[0]}")
            return result, lines

        if fname == "atomic_max":
            op = "OpAtomicSMax" if elem_type_name == "int" else "OpAtomicUMax"
            lines.append(f"{result} = {op} {elem_type_id} {ptr_id} {scope_const} {semantics_const} {remaining_arg_ids[0]}")
            return result, lines

        if fname in ("atomic_and", "atomic_or", "atomic_xor"):
            _ATOMIC_OP_MAP = {
                "atomic_and": "OpAtomicAnd",
                "atomic_or": "OpAtomicOr",
                "atomic_xor": "OpAtomicXor",
            }
            op = _ATOMIC_OP_MAP[fname]
            lines.append(f"{result} = {op} {elem_type_id} {ptr_id} {scope_const} {semantics_const} {remaining_arg_ids[0]}")
            return result, lines

        if fname == "atomic_exchange":
            lines.append(f"{result} = OpAtomicExchange {elem_type_id} {ptr_id} {scope_const} {semantics_const} {remaining_arg_ids[0]}")
            return result, lines

        if fname == "atomic_compare_exchange":
            # OpAtomicCompareExchange result_type pointer scope equal_semantics unequal_semantics value comparator
            # Note: SPIR-V order is (value, comparator) but Lux order is (ref, comparator, value)
            semantics_neq = self.reg.const_uint(0)  # Relaxed for unequal
            lines.append(f"{result} = OpAtomicCompareExchange {elem_type_id} {ptr_id} {scope_const} {semantics_const} {semantics_neq} {remaining_arg_ids[1]} {remaining_arg_ids[0]}")
            return result, lines

        if fname == "atomic_load":
            lines.append(f"{result} = OpAtomicLoad {elem_type_id} {ptr_id} {scope_const} {semantics_const}")
            return result, lines

        if fname == "atomic_store":
            # OpAtomicStore is void
            lines.append(f"OpAtomicStore {ptr_id} {scope_const} {semantics_const} {remaining_arg_ids[0]}")
            result = self.reg.const_float(0.0)  # void return placeholder
            return result, lines

        raise ValueError(f"Unknown atomic function: {fname}")

    def _gen_atomic_pointer(self, expr) -> tuple[str, list[str]]:
        """Generate a pointer to the atomic target (shared var, shared array element, SSBO element)."""
        lines = []

        if isinstance(expr, IndexAccess):
            # shared_arr[idx] or ssbo[idx]
            if isinstance(expr.object, VarRef):
                name = expr.object.name

                # Shared array
                if name in self.shared_var_ids and self.shared_array_sizes.get(name) is not None:
                    elem_type_name = self.shared_element_types[name]
                    elem_type_id = self.reg.lux_type_to_spirv(elem_type_name)
                    var_id = self.shared_var_ids[name]
                    idx_id, idx_lines = self._gen_expr(expr.index)
                    lines.extend(idx_lines)
                    idx_lux_type = self._resolve_expr_lux_type(expr.index)
                    if idx_lux_type in ("scalar",):
                        int_type = self.reg.int32()
                        conv = self.reg.next_id()
                        lines.append(f"{conv} = OpConvertFToS {int_type} {idx_id}")
                        idx_id = conv
                    ptr_elem = self.reg.pointer("Workgroup", elem_type_id)
                    ac_id = self.reg.next_id()
                    lines.append(f"{ac_id} = OpAccessChain {ptr_elem} {var_id} {idx_id}")
                    return ac_id, lines

                # SSBO
                if name in self.storage_buffer_var_ids:
                    buf_var_id = self.storage_buffer_var_ids[name]
                    elem_type_name = self.storage_buffer_element_types[name]
                    elem_type_id = self.reg.lux_type_to_spirv(elem_type_name)
                    idx_id, idx_lines = self._gen_expr(expr.index)
                    lines.extend(idx_lines)
                    idx_lux_type = self._resolve_expr_lux_type(expr.index)
                    if idx_lux_type in ("scalar",):
                        int_type = self.reg.int32()
                        conv = self.reg.next_id()
                        lines.append(f"{conv} = OpConvertFToS {int_type} {idx_id}")
                        idx_id = conv
                    ptr_elem = self.reg.pointer("StorageBuffer", elem_type_id)
                    const_0 = self.reg.const_int(0, signed=True)
                    ac_id = self.reg.next_id()
                    lines.append(f"{ac_id} = OpAccessChain {ptr_elem} {buf_var_id} {const_0} {idx_id}")
                    return ac_id, lines

        elif isinstance(expr, VarRef):
            name = expr.name
            # Shared scalar variable — just return its pointer
            if name in self.shared_var_ids and self.shared_array_sizes.get(name) is None:
                return self.shared_var_ids[name], lines

        raise ValueError(f"Cannot generate atomic pointer for expression: {type(expr).__name__}")

    def _is_shared_ref(self, expr) -> bool:
        """Check if an expression references shared memory."""
        if isinstance(expr, VarRef):
            return expr.name in self.shared_var_ids
        if isinstance(expr, IndexAccess) and isinstance(expr.object, VarRef):
            return expr.object.name in self.shared_var_ids
        return False

    def _resolve_atomic_element_type(self, expr) -> str:
        """Determine the element type for an atomic operation's target."""
        if isinstance(expr, VarRef):
            name = expr.name
            if name in self.shared_element_types:
                return self.shared_element_types[name]
        if isinstance(expr, IndexAccess) and isinstance(expr.object, VarRef):
            name = expr.object.name
            if name in self.shared_element_types:
                return self.shared_element_types[name]
            if name in self.storage_buffer_element_types:
                return self.storage_buffer_element_types[name]
        return "uint"  # default


def _swizzle_index(c: str) -> int:
    return {"x": 0, "y": 1, "z": 2, "w": 3,
            "r": 0, "g": 1, "b": 2, "a": 3}[c]
