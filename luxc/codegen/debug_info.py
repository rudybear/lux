"""NonSemantic.Shader.DebugInfo.100 emission for RenderDoc step-through debugging.

Emits extended debug instructions that enable:
- Source display with embedded full text
- Line-level stepping (richer than OpLine with start/end columns)
- Named variable inspection with types
- Scope-aware variable visibility
- Lexical block scoping for if/for/while

All instructions use: OpExtInst %void %dbg <opcode> <operands>
All numeric operands must be OpConstant IDs (not literals).

Reference: https://github.com/KhronosGroup/SPIRV-Registry/blob/main/nonsemantic/NonSemantic.Shader.DebugInfo.100.asciidoc
"""

from __future__ import annotations

from typing import Optional


# NonSemantic.Shader.DebugInfo.100 opcode numbers
class DebugOp:
    DebugInfoNone = 0
    DebugCompilationUnit = 1
    DebugTypeBasic = 2
    DebugTypePointer = 3
    DebugTypeQualifier = 4
    DebugTypeArray = 5
    DebugTypeVector = 6
    DebugTypedef = 7
    DebugTypeFunction = 8
    DebugTypeEnum = 9
    DebugTypeComposite = 10
    DebugTypeMember = 11
    DebugTypeInheritance = 12
    DebugTypePtrToMember = 13
    DebugTypeTemplate = 14
    DebugTypeTemplateParameter = 15
    DebugTypeTemplateTemplateParameter = 16
    DebugTypeTemplateParameterPack = 17
    DebugGlobalVariable = 18
    DebugFunctionDeclaration = 19
    DebugFunction = 20
    DebugLexicalBlock = 21
    DebugLexicalBlockDiscriminator = 22
    DebugScope = 23
    DebugNoScope = 24
    DebugInlinedAt = 25
    DebugLocalVariable = 26
    DebugInlinedVariable = 27
    DebugDeclare = 28
    DebugValue = 29
    DebugOperation = 30
    DebugExpression = 31
    DebugMacroDef = 32
    DebugMacroUndef = 33
    DebugImportedEntity = 34
    DebugSource = 35
    DebugFunctionDefinition = 101
    DebugSourceContinued = 102
    DebugLine = 103
    DebugNoLine = 104
    DebugBuildIdentifier = 105
    DebugStoragePath = 106
    DebugEntryPoint = 107


# DebugInfoFlags
FLAG_NONE = 0
FLAG_IS_PROTECTED = 1
FLAG_IS_PRIVATE = 2
FLAG_IS_LOCAL = 4
FLAG_IS_DEFINITION = 8
FLAG_FWDECL = 16
FLAG_ARTIFICIAL = 32
FLAG_EXPLICIT = 64
FLAG_PROTOTYPED = 128
FLAG_OBJECT_POINTER = 256
FLAG_STATIC_MEMBER = 512
FLAG_INDIRECT_VARIABLE = 1024
FLAG_LVALUE_REFERENCE = 2048
FLAG_RVALUE_REFERENCE = 4096
FLAG_IS_OPTIMIZED = 8192
FLAG_IS_ENUM_CLASS = 16384
FLAG_IS_TYPE_PASS_BY_VALUE = 32768
FLAG_IS_TYPE_PASS_BY_REFERENCE = 65536

# DebugBaseTypeAttributeEncoding
ENCODING_UNSPECIFIED = 0
ENCODING_ADDRESS = 1
ENCODING_BOOLEAN = 2
ENCODING_FLOAT = 3
ENCODING_SIGNED = 4
ENCODING_SIGNED_CHAR = 5
ENCODING_UNSIGNED = 6
ENCODING_UNSIGNED_CHAR = 7


class DebugInfoEmitter:
    """Helper that generates NonSemantic.Shader.DebugInfo.100 extended instructions.

    Keeps the main SpvGenerator clean by encapsulating all debug info logic.
    """

    def __init__(self, reg, source_name: str, source_text: str):
        """
        Args:
            reg: TypeRegistry instance (for next_id, const_int, etc.)
            source_name: e.g., "shader.lux"
            source_text: full shader source code
        """
        self.reg = reg
        self.source_name = source_name
        self.source_text = source_text
        self.ext_id: str | None = None  # %dbg = OpExtInstImport

        # Pre-allocated IDs
        self._source_id: str | None = None
        self._comp_unit_id: str | None = None
        self._void_id: str | None = None

        # Debug type cache: lux_type -> debug type ID
        self._debug_types: dict[str, str] = {}

        # Debug function IDs: fn_name -> debug function ID
        self._debug_functions: dict[str, str] = {}

        # Pre-built constant IDs (needed because operands must be OpConstant)
        self._const_cache: dict[int, str] = {}

        # Declarations that go before function bodies
        self.pre_fn_lines: list[str] = []

        # Tracks current function's debug scope for DebugDeclare
        self._current_scope_id: str | None = None
        self._current_fn_debug_id: str | None = None

        # Expression ID for empty DebugExpression (used in DebugDeclare)
        self._empty_expression_id: str | None = None

    def init(self, void_type_id: str) -> str:
        """Initialize the debug info emitter. Returns the extension import ID.

        Must be called during generate() header assembly.
        """
        self.ext_id = self.reg.next_id()
        self._void_id = void_type_id
        return self.ext_id

    def _const_u32(self, value: int) -> str:
        """Get or create an OpConstant %uint for a numeric operand."""
        if value in self._const_cache:
            return self._const_cache[value]
        cid = self.reg.const_int(value, signed=False)
        self._const_cache[value] = cid
        return cid

    def _emit(self, result_id: str, opcode: int, operands: list[str]) -> str:
        """Generate an OpExtInst line."""
        if operands:
            ops = " ".join(operands)
            return f"{result_id} = OpExtInst {self._void_id} {self.ext_id} {opcode} {ops}"
        return f"{result_id} = OpExtInst {self._void_id} {self.ext_id} {opcode}"

    def _emit_typed(self, result_id: str, result_type: str, opcode: int, operands: list[str]) -> str:
        """Generate an OpExtInst line with a custom result type."""
        if operands:
            ops = " ".join(operands)
            return f"{result_id} = OpExtInst {result_type} {self.ext_id} {opcode} {ops}"
        return f"{result_id} = OpExtInst {result_type} {self.ext_id} {opcode}"

    def emit_header_lines(self) -> list[str]:
        """Emit capability + extension + import lines for the header section."""
        lines = []
        lines.append('OpExtension "SPV_KHR_non_semantic_info"')
        lines.append(
            f'{self.ext_id} = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"'
        )
        return lines

    def emit_pre_function_decls(self) -> list[str]:
        """Emit all declarations that must appear before function bodies.

        Includes: DebugSource, DebugCompilationUnit, DebugTypeBasic/Vector/Matrix,
        DebugTypeFunction, DebugFunction.
        """
        lines = []

        # DebugSource — embed full source text
        source_string_id = self.reg.next_id()
        # Escape source text for OpString
        escaped = self.source_text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        lines.append(f'{source_string_id} = OpString "{escaped}"')

        name_string_id = self.reg.next_id()
        lines.append(f'{name_string_id} = OpString "{self.source_name}"')

        self._source_id = self.reg.next_id()
        lines.append(self._emit_typed(
            self._source_id, self._void_id,
            DebugOp.DebugSource,
            [name_string_id, source_string_id],
        ))

        # DebugCompilationUnit (version=1, DWARF=5, source=DebugSource, language=GLSL)
        self._comp_unit_id = self.reg.next_id()
        lines.append(self._emit_typed(
            self._comp_unit_id, self._void_id,
            DebugOp.DebugCompilationUnit,
            [
                self._const_u32(1),   # version
                self._const_u32(5),   # DWARF version
                self._source_id,      # source
                self._const_u32(2),   # language: GLSL
            ],
        ))

        # Empty DebugExpression (needed for DebugDeclare)
        self._empty_expression_id = self.reg.next_id()
        lines.append(self._emit_typed(
            self._empty_expression_id, self._void_id,
            DebugOp.DebugExpression,
            [],
        ))

        # DebugInfoNone for optional args
        self._debug_none_id = self.reg.next_id()
        lines.append(self._emit_typed(
            self._debug_none_id, self._void_id,
            DebugOp.DebugInfoNone,
            [],
        ))

        # Basic types
        self._emit_basic_types(lines)

        self.pre_fn_lines = lines
        return lines

    def _emit_basic_types(self, lines: list[str]) -> None:
        """Emit DebugTypeBasic for float, int, uint, bool."""
        type_defs = [
            ("scalar", "float", 32, ENCODING_FLOAT),
            ("int", "int", 32, ENCODING_SIGNED),
            ("uint", "uint", 32, ENCODING_UNSIGNED),
            ("bool", "bool", 32, ENCODING_BOOLEAN),
        ]
        for lux_type, display_name, bits, encoding in type_defs:
            name_id = self.reg.next_id()
            lines.append(f'{name_id} = OpString "{display_name}"')

            type_id = self.reg.next_id()
            lines.append(self._emit_typed(
                type_id, self._void_id,
                DebugOp.DebugTypeBasic,
                [name_id, self._const_u32(bits), self._const_u32(encoding)],
            ))
            self._debug_types[lux_type] = type_id
            if lux_type == "scalar":
                self._debug_types["float"] = type_id

        # Vector types
        float_type = self._debug_types["scalar"]
        int_type = self._debug_types["int"]
        uint_type = self._debug_types["uint"]

        for size in (2, 3, 4):
            # float vectors
            vec_id = self.reg.next_id()
            lines.append(self._emit_typed(
                vec_id, self._void_id,
                DebugOp.DebugTypeVector,
                [float_type, self._const_u32(size)],
            ))
            self._debug_types[f"vec{size}"] = vec_id

            # int vectors
            ivec_id = self.reg.next_id()
            lines.append(self._emit_typed(
                ivec_id, self._void_id,
                DebugOp.DebugTypeVector,
                [int_type, self._const_u32(size)],
            ))
            self._debug_types[f"ivec{size}"] = ivec_id

            # uint vectors
            uvec_id = self.reg.next_id()
            lines.append(self._emit_typed(
                uvec_id, self._void_id,
                DebugOp.DebugTypeVector,
                [uint_type, self._const_u32(size)],
            ))
            self._debug_types[f"uvec{size}"] = uvec_id

        # Matrix types (mat2, mat3, mat4) — use DebugTypeArray of vec columns
        # NonSemantic doesn't have a DebugTypeMatrix, so we represent as composite
        for size in (2, 3, 4):
            vec_type = self._debug_types[f"vec{size}"]
            mat_id = self.reg.next_id()
            lines.append(self._emit_typed(
                mat_id, self._void_id,
                DebugOp.DebugTypeArray,
                [vec_type, self._const_u32(size)],
            ))
            self._debug_types[f"mat{size}"] = mat_id

    def get_debug_type(self, lux_type: str) -> str | None:
        """Get the debug type ID for a Lux type name."""
        return self._debug_types.get(lux_type)

    def emit_debug_function(self, fn_name: str, line: int = 1) -> tuple[str, list[str]]:
        """Emit DebugTypeFunction + DebugFunction for a function. Returns (debug_fn_id, lines)."""
        lines = []

        # DebugTypeFunction: void(void) for main
        fn_type_id = self.reg.next_id()
        void_dbg = self._debug_types.get("void", self._debug_none_id)
        lines.append(self._emit_typed(
            fn_type_id, self._void_id,
            DebugOp.DebugTypeFunction,
            [self._const_u32(0), void_dbg],  # flags=0, return_type
        ))

        # Function name string
        fn_name_id = self.reg.next_id()
        lines.append(f'{fn_name_id} = OpString "{fn_name}"')

        # DebugFunction
        debug_fn_id = self.reg.next_id()
        lines.append(self._emit_typed(
            debug_fn_id, self._void_id,
            DebugOp.DebugFunction,
            [
                fn_name_id,           # name
                fn_type_id,           # type
                self._source_id,      # source
                self._const_u32(line), # line
                self._const_u32(0),   # column
                self._comp_unit_id,   # scope (compilation unit)
                fn_name_id,           # linkage name
                self._const_u32(FLAG_IS_DEFINITION | FLAG_IS_LOCAL),
                self._const_u32(line), # scope line
            ],
        ))

        self._debug_functions[fn_name] = debug_fn_id
        self._current_fn_debug_id = debug_fn_id
        self._current_scope_id = debug_fn_id
        return debug_fn_id, lines

    def emit_function_definition(self, debug_fn_id: str, spv_fn_id: str) -> str:
        """Emit DebugFunctionDefinition at the start of a function's entry block."""
        result_id = self.reg.next_id()
        return self._emit_typed(
            result_id, self._void_id,
            DebugOp.DebugFunctionDefinition,
            [debug_fn_id, spv_fn_id],
        )

    def emit_debug_scope(self, scope_id: str) -> str:
        """Emit DebugScope to set the current debug scope."""
        result_id = self.reg.next_id()
        self._current_scope_id = scope_id
        return self._emit_typed(
            result_id, self._void_id,
            DebugOp.DebugScope,
            [scope_id],
        )

    def emit_debug_no_scope(self) -> str:
        """Emit DebugNoScope."""
        result_id = self.reg.next_id()
        return self._emit_typed(
            result_id, self._void_id,
            DebugOp.DebugNoScope,
            [],
        )

    def emit_debug_line(self, line: int, col_start: int = 0, line_end: int = 0,
                        col_end: int = 0) -> str:
        """Emit DebugLine with start/end line and column info."""
        if line_end == 0:
            line_end = line
        result_id = self.reg.next_id()
        return self._emit_typed(
            result_id, self._void_id,
            DebugOp.DebugLine,
            [
                self._source_id,
                self._const_u32(line),
                self._const_u32(line_end),
                self._const_u32(col_start),
                self._const_u32(col_end),
            ],
        )

    def emit_debug_no_line(self) -> str:
        """Emit DebugNoLine."""
        result_id = self.reg.next_id()
        return self._emit_typed(
            result_id, self._void_id,
            DebugOp.DebugNoLine,
            [],
        )

    def emit_local_variable(self, name: str, lux_type: str, line: int) -> tuple[str, str]:
        """Emit DebugLocalVariable. Returns (debug_var_id, instruction_line)."""
        debug_type = self.get_debug_type(lux_type)
        if debug_type is None:
            debug_type = self._debug_none_id

        name_id = self.reg.next_id()
        name_line = f'{name_id} = OpString "{name}"'

        var_id = self.reg.next_id()
        instr = self._emit_typed(
            var_id, self._void_id,
            DebugOp.DebugLocalVariable,
            [
                name_id,              # name
                debug_type,           # type
                self._source_id,      # source
                self._const_u32(line), # line
                self._const_u32(0),   # column
                self._current_scope_id or self._debug_none_id,  # scope
            ],
        )
        return var_id, f"{name_line}\n{instr}"

    def emit_debug_declare(self, debug_var_id: str, spv_var_id: str) -> str:
        """Emit DebugDeclare linking a DebugLocalVariable to its OpVariable."""
        result_id = self.reg.next_id()
        return self._emit_typed(
            result_id, self._void_id,
            DebugOp.DebugDeclare,
            [debug_var_id, spv_var_id, self._empty_expression_id],
        )

    def emit_lexical_block(self, line: int, col: int = 0) -> tuple[str, str]:
        """Emit DebugLexicalBlock for if/for/while bodies. Returns (block_id, instruction)."""
        block_id = self.reg.next_id()
        instr = self._emit_typed(
            block_id, self._void_id,
            DebugOp.DebugLexicalBlock,
            [
                self._source_id,
                self._const_u32(line),
                self._const_u32(col),
                self._current_scope_id or self._comp_unit_id,
            ],
        )
        return block_id, instr

    def emit_inlined_at(self, line: int, scope_id: str,
                        inlined_parent: str | None = None) -> tuple[str, str]:
        """Emit DebugInlinedAt for inline function call sites. Returns (id, instruction)."""
        inlined_id = self.reg.next_id()
        operands = [
            self._const_u32(line),
            scope_id,
        ]
        if inlined_parent:
            operands.append(inlined_parent)
        instr = self._emit_typed(
            inlined_id, self._void_id,
            DebugOp.DebugInlinedAt,
            operands,
        )
        return inlined_id, instr
