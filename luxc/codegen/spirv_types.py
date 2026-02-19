"""SPIR-V type registry with deduplication."""

from __future__ import annotations


class TypeRegistry:
    """Manages SPIR-V type and constant declarations, ensuring each is declared once.

    All type and constant declarations go into a single ordered list (_decls)
    so that dependencies are always satisfied (e.g., array length constants
    appear before the OpTypeArray that references them).
    """

    def __init__(self):
        self._next_id = 1
        self._types: dict[str, str] = {}      # type_key -> %id
        self._const_cache: dict[str, str] = {} # const_key -> %id
        self._decls: list[str] = []            # all type + constant declarations in order

    def next_id(self) -> str:
        name = f"%{self._next_id}"
        self._next_id += 1
        return name

    def set_next_id(self, n: int):
        self._next_id = n

    # --- Core types ---

    def void(self) -> str:
        return self._ensure_type("void", "OpTypeVoid")

    def bool_type(self) -> str:
        return self._ensure_type("bool", "OpTypeBool")

    def float32(self) -> str:
        return self._ensure_type("float32", "OpTypeFloat 32")

    def int32(self) -> str:
        return self._ensure_type("int32_signed", "OpTypeInt 32 1")

    def uint32(self) -> str:
        return self._ensure_type("uint32", "OpTypeInt 32 0")

    def vec(self, n: int, component: str = "float") -> str:
        comp_id = self._component_type(component)
        key = f"vec{n}_{component}"
        return self._ensure_type(key, f"OpTypeVector {comp_id} {n}")

    def mat(self, n: int) -> str:
        col = self.vec(n)
        key = f"mat{n}"
        return self._ensure_type(key, f"OpTypeMatrix {col} {n}")

    def mat_cols_rows(self, cols: int, rows: int) -> str:
        """Non-square matrix: cols columns of rows-component vectors."""
        col = self.vec(rows)
        key = f"mat{cols}x{rows}"
        return self._ensure_type(key, f"OpTypeMatrix {col} {cols}")

    def image_type(self) -> str:
        f32 = self.float32()
        key = "image2d"
        return self._ensure_type(key, f"OpTypeImage {f32} 2D 0 0 0 1 Unknown")

    def sampler_type(self) -> str:
        key = "sampler"
        return self._ensure_type(key, "OpTypeSampler")

    def sampled_image_type(self) -> str:
        img = self.image_type()
        key = "sampled_image"
        return self._ensure_type(key, f"OpTypeSampledImage {img}")

    def acceleration_structure_type(self) -> str:
        key = "accel_struct"
        return self._ensure_type(key, "OpTypeAccelerationStructureKHR")

    def _storage_image_type(self) -> str:
        f32 = self.float32()
        key = "storage_image_2d"
        return self._ensure_type(key, f"OpTypeImage {f32} 2D 0 0 0 2 Rgba8")

    def pointer(self, storage_class: str, pointee: str) -> str:
        key = f"ptr_{storage_class}_{pointee}"
        return self._ensure_type(key, f"OpTypePointer {storage_class} {pointee}")

    def function_type(self, return_type: str, param_types: list[str] | None = None) -> str:
        params = param_types or []
        param_str = " ".join(params)
        key = f"fn_{return_type}_{'_'.join(params) if params else 'void'}"
        decl = f"OpTypeFunction {return_type}"
        if params:
            decl += " " + param_str
        return self._ensure_type(key, decl)

    def struct(self, name: str, member_types: list[str]) -> str:
        members_str = " ".join(member_types)
        key = f"struct_{name}"
        return self._ensure_type(key, f"OpTypeStruct {members_str}")

    def array(self, element_type: str, length_id: str) -> str:
        key = f"array_{element_type}_{length_id}"
        return self._ensure_type(key, f"OpTypeArray {element_type} {length_id}")

    def runtime_array(self, element_type: str) -> str:
        key = f"rtarray_{element_type}"
        return self._ensure_type(key, f"OpTypeRuntimeArray {element_type}")

    # --- Constants ---

    def const_float(self, value: float) -> str:
        f32 = self.float32()
        key = f"const_f32_{value}"
        if key in self._const_cache:
            return self._const_cache[key]
        cid = self.next_id()
        self._const_cache[key] = cid
        self._decls.append(f"{cid} = OpConstant {f32} {value}")
        return cid

    def const_int(self, value: int, signed: bool = True) -> str:
        it = self.int32() if signed else self.uint32()
        prefix = "si" if signed else "ui"
        key = f"const_{prefix}32_{value}"
        if key in self._const_cache:
            return self._const_cache[key]
        cid = self.next_id()
        self._const_cache[key] = cid
        self._decls.append(f"{cid} = OpConstant {it} {value}")
        return cid

    def const_uint(self, value: int) -> str:
        return self.const_int(value, signed=False)

    def const_bool(self, value: bool) -> str:
        bt = self.bool_type()
        key = f"const_bool_{value}"
        if key in self._const_cache:
            return self._const_cache[key]
        cid = self.next_id()
        self._const_cache[key] = cid
        op = "OpConstantTrue" if value else "OpConstantFalse"
        self._decls.append(f"{cid} = {op} {bt}")
        return cid

    def const_composite(self, type_id: str, constituents: list[str]) -> str:
        key = f"const_composite_{type_id}_{'_'.join(constituents)}"
        if key in self._const_cache:
            return self._const_cache[key]
        cid = self.next_id()
        self._const_cache[key] = cid
        parts = " ".join(constituents)
        self._decls.append(f"{cid} = OpConstantComposite {type_id} {parts}")
        return cid

    # --- Helpers ---

    def lux_type_to_spirv(self, type_name: str) -> str:
        from luxc.builtins.types import resolve_alias_chain
        # Resolve type aliases before mapping to SPIR-V
        resolved = resolve_alias_chain(type_name)
        mapping = {
            "void": self.void,
            "bool": self.bool_type,
            "scalar": self.float32,
            "int": self.int32,
            "uint": self.uint32,
            "vec2": lambda: self.vec(2),
            "vec3": lambda: self.vec(3),
            "vec4": lambda: self.vec(4),
            "ivec2": lambda: self.vec(2, "int"),
            "ivec3": lambda: self.vec(3, "int"),
            "ivec4": lambda: self.vec(4, "int"),
            "uvec2": lambda: self.vec(2, "uint"),
            "uvec3": lambda: self.vec(3, "uint"),
            "uvec4": lambda: self.vec(4, "uint"),
            "mat2": lambda: self.mat(2),
            "mat3": lambda: self.mat(3),
            "mat4": lambda: self.mat(4),
            "mat4x3": lambda: self.mat_cols_rows(4, 3),
            "sampler2d": self.sampled_image_type,
            "acceleration_structure": self.acceleration_structure_type,
            "storage_image": self._storage_image_type,
        }
        factory = mapping.get(resolved)
        if factory is None:
            raise ValueError(f"Unknown Lux type: {type_name} (resolved to: {resolved})")
        return factory()

    def emit_declarations(self) -> list[str]:
        """Return all type + constant declarations in dependency order."""
        return list(self._decls)

    # --- Internal ---

    def _component_type(self, component: str) -> str:
        if component == "float":
            return self.float32()
        elif component == "int":
            return self.int32()
        elif component == "uint":
            return self.uint32()
        raise ValueError(f"Unknown component type: {component}")

    def _ensure_type(self, key: str, decl: str) -> str:
        if key in self._types:
            return self._types[key]
        tid = self.next_id()
        self._types[key] = tid
        self._decls.append(f"{tid} = {decl}")
        return tid
