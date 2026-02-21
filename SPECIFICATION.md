# Lux Shader Language Specification

**Version 0.2** -- February 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Lexical Structure](#2-lexical-structure)
3. [Type System](#3-type-system)
4. [Declarations](#4-declarations)
5. [Expressions](#5-expressions)
6. [Statements](#6-statements)
7. [Shader Stages](#7-shader-stages)
8. [Built-in Variables](#8-built-in-variables)
9. [Built-in Functions](#9-built-in-functions)
10. [Import System](#10-import-system)
11. [Standard Library Reference](#11-standard-library-reference)
12. [Declarative Syntax](#12-declarative-syntax)
13. [Automatic Differentiation](#13-automatic-differentiation)
14. [Compilation Pipeline](#14-compilation-pipeline)
15. [SPIR-V Output](#15-spir-v-output)
16. [Limitations](#16-limitations)
17. [Examples](#17-examples)

---

## 1. Introduction

### 1.1 Language Goals and Philosophy

Lux is a **math-first shader language** designed for humans and large language models alike. Its core design philosophy is to let programmers write rendering math directly -- surfaces, materials, lighting equations -- while the compiler handles all GPU translation to SPIR-V for Vulkan.

Lux eliminates the boilerplate that dominates conventional shader languages: no `layout(set=0, binding=1)`, no `gl_Position`, no manual stage wiring. Instead, the programmer writes mathematical expressions and declarative material definitions, and the compiler generates fully typed, validated SPIR-V binaries.

### 1.2 Target

Lux compiles to **SPIR-V** for use with the **Vulkan** graphics API. The compiler emits SPIR-V 1.0 for rasterization stages (vertex, fragment) and SPIR-V 1.4 with the `SPV_KHR_ray_tracing` extension for ray tracing stages.

### 1.3 Design Principles

- **Mathematical vocabulary**: `scalar` instead of `float`, `builtin_position` instead of `gl_Position`.
- **No layout qualifiers**: Locations, descriptor sets, and bindings are auto-assigned by declaration order.
- **Explicit types**: All variables require type annotations; there is no type inference for declarations.
- **No semicolons after blocks**: Block-delimited constructs (functions, stages, if/else) do not require trailing semicolons.
- **One file, multi-stage**: Vertex, fragment, and ray tracing stages coexist in a single `.lux` file.
- **Declarative materials**: High-level `surface`, `geometry`, and `pipeline` blocks expand to full shader stages.
- **Function inlining**: All user-defined function calls are inlined at the call site; no SPIR-V `OpFunctionCall` is emitted.
- **No loops**: Iterative algorithms (FBM, Voronoi) must be manually unrolled.

---

## 2. Lexical Structure

### 2.1 Character Set

Lux source files are encoded in **UTF-8**. The language uses only the ASCII subset for all tokens; non-ASCII characters may appear only in comments.

### 2.2 Whitespace

Whitespace consists of spaces (U+0020), horizontal tabs (U+0009), form feeds (U+000C), carriage returns (U+000D), and newlines (U+000A). Whitespace is insignificant and serves only to separate tokens. Any amount of whitespace may appear between tokens.

### 2.3 Comments

Lux supports single-line comments beginning with `//`. Everything from `//` to the end of the line is ignored.

```
// This is a comment
let x: scalar = 1.0; // inline comment
```

There are no multi-line or block comments.

### 2.4 Identifiers

An identifier begins with a letter (a-z, A-Z) or underscore (`_`), followed by zero or more letters, digits (0-9), or underscores.

```
IDENT = /[a-zA-Z_][a-zA-Z0-9_]*/
```

Identifiers are case-sensitive. The identifier `foo` is distinct from `Foo`.

### 2.5 Keywords

The following identifiers are reserved as keywords and may not be used as user-defined names:

| Category | Keywords |
|---|---|
| Declarations | `fn`, `let`, `const`, `type`, `struct`, `import` |
| Control flow | `return`, `if`, `else` |
| Stage types | `vertex`, `fragment`, `raygen`, `closest_hit`, `any_hit`, `miss`, `intersection`, `callable` |
| Stage items | `in`, `out`, `uniform`, `push`, `sampler2d`, `samplerCube` |
| RT items | `ray_payload`, `hit_attribute`, `callable_data`, `acceleration_structure` |
| Declarative | `surface`, `geometry`, `pipeline`, `schedule`, `environment`, `procedural`, `layers` |
| Boolean | `true`, `false` |

### 2.6 Literals

#### 2.6.1 Numeric Literals

Numeric literals may be integers or floating-point numbers. All numeric literals are treated as `scalar` (32-bit float) by default.

```
NUMBER = /[0-9]+\.[0-9]*([eE][+-]?[0-9]+)?/
       | /[0-9]+[eE][+-]?[0-9]+/
       | /[0-9]*\.[0-9]+([eE][+-]?[0-9]+)?/
       | /[0-9]+/
```

Examples of valid numeric literals:

| Literal | Form |
|---|---|
| `42` | Integer form (still `scalar` type) |
| `3.14` | Floating-point |
| `0.5` | Floating-point |
| `.5` | Floating-point (leading dot) |
| `1.0e10` | Scientific notation |
| `2.5E-3` | Scientific notation |
| `1e5` | Scientific notation without decimal point |

There are no hexadecimal, octal, or binary literal forms. There are no literal suffixes (no `f`, `u`, `i` suffixes).

#### 2.6.2 Boolean Literals

```
true
false
```

Boolean literals produce values of type `bool`.

### 2.7 Operators

The following operators are defined, listed from lowest to highest precedence:

| Precedence | Operators | Associativity | Description |
|---|---|---|---|
| 1 (lowest) | `? :` | Right | Ternary conditional |
| 2 | `\|\|` | Left | Logical OR |
| 3 | `&&` | Left | Logical AND |
| 4 | `==` `!=` | Left | Equality |
| 5 | `<` `>` `<=` `>=` | Left | Comparison |
| 6 | `+` `-` | Left | Addition, subtraction |
| 7 | `*` `/` `%` | Left | Multiplication, division, modulo |
| 8 | `-` `!` | Right (prefix) | Unary negation, logical NOT |
| 9 (highest) | `.` `()` `[]` | Left (postfix) | Member access, call, index |

### 2.8 Swizzle Patterns

Vector swizzle patterns use the component sets `{x, y, z, w}` or `{r, g, b, a}`. A swizzle consists of 1 to 4 characters from a single component set.

```
SWIZZLE = /[xyzw]{1,4}/ | /[rgba]{1,4}/
```

The two component sets are aliases:

| Position | Set 1 | Set 2 |
|---|---|---|
| 0 | `x` | `r` |
| 1 | `y` | `g` |
| 2 | `z` | `b` |
| 3 | `w` | `a` |

Examples: `.x`, `.xy`, `.xyz`, `.xyzw`, `.rgb`, `.rgba`, `.yx`, `.zzz`.

The number of components in the swizzle determines the result type:
- 1 component: `scalar`
- 2 components: `vec2`
- 3 components: `vec3`
- 4 components: `vec4`

Mixing component sets (e.g., `.xg`) is not permitted.

### 2.9 Punctuation and Delimiters

```
{  }  (  )  [  ]  ;  :  ,  .  =  @  ->  ?
```

### 2.10 Assignment Operators

The language supports simple assignment (`=`). Compound assignment operators (`+=`, `-=`, `*=`, `/=`) are desugared to their expanded forms during parsing. For example, `a += b;` is equivalent to `a = a + b;`.

---

## 3. Type System

### 3.1 Primitive Types

| Lux Type | Description | SPIR-V Mapping | Size |
|---|---|---|---|
| `scalar` | 32-bit floating-point | `OpTypeFloat 32` | 4 bytes |
| `int` | 32-bit signed integer | `OpTypeInt 32 1` | 4 bytes |
| `uint` | 32-bit unsigned integer | `OpTypeInt 32 0` | 4 bytes |
| `bool` | Boolean value | `OpTypeBool` | 4 bytes |
| `void` | No value | `OpTypeVoid` | 0 bytes |

### 3.2 Vector Types

| Lux Type | Component | Size | SPIR-V Mapping |
|---|---|---|---|
| `vec2` | `scalar` | 2 | `OpTypeVector %float 2` |
| `vec3` | `scalar` | 3 | `OpTypeVector %float 3` |
| `vec4` | `scalar` | 4 | `OpTypeVector %float 4` |
| `ivec2` | `int` | 2 | `OpTypeVector %int 2` |
| `ivec3` | `int` | 3 | `OpTypeVector %int 3` |
| `ivec4` | `int` | 4 | `OpTypeVector %int 4` |
| `uvec2` | `uint` | 2 | `OpTypeVector %uint 2` |
| `uvec3` | `uint` | 3 | `OpTypeVector %uint 3` |
| `uvec4` | `uint` | 4 | `OpTypeVector %uint 4` |

### 3.3 Matrix Types

| Lux Type | Columns | Column Type | SPIR-V Mapping |
|---|---|---|---|
| `mat2` | 2 | `vec2` | `OpTypeMatrix %vec2 2` |
| `mat3` | 3 | `vec3` | `OpTypeMatrix %vec3 3` |
| `mat4` | 4 | `vec4` | `OpTypeMatrix %vec4 4` |

All matrices are column-major. Matrices use `ColMajor` layout with a `MatrixStride` of 16 in uniform buffers.

### 3.4 Opaque Types

| Lux Type | Description | SPIR-V Mapping |
|---|---|---|
| `sampler2d` | 2D texture sampler | `OpTypeSampledImage` (split into `OpTypeSampler` + `OpTypeImage` for codegen) |
| `samplerCube` | Cube map texture sampler | `OpTypeSampledImage` with `Cube` dimensionality (split into `OpTypeSampler` + `OpTypeImage Cube` for codegen) |
| `acceleration_structure` | RT top-level acceleration structure | `OpTypeAccelerationStructureKHR` |

### 3.5 Type Aliases

Type aliases create alternate names for existing types:

```
type Radiance = vec3;
type Direction = vec3;
```

Type aliases are resolved transitively: if `type A = B;` and `type B = vec3;`, then `A` resolves to `vec3`. Aliases do not create new types; they are purely syntactic substitutions. Aliased names may be used anywhere a type name is expected.

### 3.6 Type Constructors

Vector and matrix types may be constructed using constructor syntax:

```
vec3(1.0, 2.0, 3.0)   // explicit components
vec3(1.0)              // splat: all components = 1.0
vec4(pos, 1.0)         // pack: vec3 + scalar -> vec4
vec4(v.xy, 0.0, 1.0)  // pack: vec2 + scalar + scalar -> vec4
mat4(c0, c1, c2, c3)  // from 4 column vectors
```

**Splat rule**: When a constructor receives a single scalar argument, that value is broadcast to all components.

**Pack rule**: When a constructor receives a mix of vector and scalar arguments, components are extracted and concatenated. The total number of scalar components must match the target type's size.

### 3.7 Type Promotion

Numeric literals of type `scalar` may be implicitly promoted to `int` or `uint` when passed as arguments to built-in functions that expect integer parameters. This promotion applies only at function call boundaries and does not create implicit conversions between declared variable types.

### 3.8 No Implicit Conversions

Lux does not perform implicit type conversions between:
- Different vector sizes (e.g., `vec2` to `vec3`)
- Different component types (e.g., `vec3` to `ivec3`)
- Scalars and vectors in variable declarations (explicit constructors required)

### 3.9 Arithmetic Type Rules

Binary arithmetic operators follow these rules:

| Left Type | Operator | Right Type | Result Type |
|---|---|---|---|
| `scalar` | `+` `-` `*` `/` `%` | `scalar` | `scalar` |
| `vecN` | `+` `-` `*` `/` `%` | `vecN` | `vecN` (component-wise) |
| `scalar` | `*` | `vecN` | `vecN` (scalar broadcast) |
| `vecN` | `*` | `scalar` | `vecN` (scalar broadcast) |
| `scalar` | `+` `-` `/` `%` | `vecN` | `vecN` (scalar broadcast) |
| `vecN` | `+` `-` `/` `%` | `scalar` | `vecN` (scalar broadcast) |
| `matN` | `*` | `vecN` | `vecN` (matrix-vector multiply) |
| `vecN` | `*` | `matN` | `vecN` (vector-matrix multiply) |
| `matN` | `*` | `matN` | `matN` (matrix-matrix multiply) |
| `scalar` | `*` | `matN` | `matN` (scalar-matrix multiply) |
| `matN` | `*` | `scalar` | `matN` (matrix-scalar multiply) |

Comparison and equality operators always produce `bool`. Logical operators (`&&`, `||`) operate on `bool` and produce `bool`.

---

## 4. Declarations

### 4.1 Constants

```
const_decl = "const" IDENT ":" type "=" expr ";" ;
```

Constants are module-level immutable bindings. They must have an explicit type annotation and an initializer expression. Constants are evaluated at compile time when possible (see [Section 14, Constant Folding](#14-compilation-pipeline)).

```
const PI: scalar = 3.14159265358979;
const EPSILON: scalar = 0.00001;
```

Constants are visible to all functions and stages within the same module, and are exported when the module is imported.

### 4.2 Variables

```
let_stmt = "let" IDENT ":" type "=" expr ";" ;
```

Variables are declared within function bodies using `let`. They must have an explicit type annotation and an initializer expression. Variables are mutable and may be reassigned.

```
let x: scalar = 1.0;
let color: vec3 = vec3(0.5, 0.7, 1.0);
```

### 4.3 Functions

```
function_def = attribute* "fn" IDENT "(" param_list? ")" ("->" type)? "{" statement* "}" ;
param_list   = param ("," param)* ;
param        = IDENT ":" type ;
attribute    = "@" IDENT ;
```

Functions are defined with the `fn` keyword. Parameters require explicit type annotations. The return type follows `->` and may be omitted for void functions.

```
fn fresnel_schlick(cos_theta: scalar, f0: vec3) -> vec3 {
    return f0 + (vec3(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}
```

Functions defined at module level are visible to all stages and are exported when the module is imported. Functions defined inside a stage block are local to that stage.

All user-defined function calls are **inlined** at the call site. There is no SPIR-V `OpFunctionCall` emitted. Recursion is not supported (except for recursive `trace_ray` in RT stages, which is handled by the hardware).

**Annotations**: Functions may be prefixed with `@` annotations. Supported annotations:

| Annotation | Purpose | Reference |
|---|---|---|
| `@differentiable` | Auto-generate gradient functions via forward-mode autodiff | Section 13 |
| `@layer` | Register as a custom layer function for use in `layers [...]` blocks | Section 12.2.2 |

### 4.4 Type Aliases

```
type_alias = "type" IDENT "=" type ";" ;
```

Type aliases create alternate names for existing types:

```
type Radiance = vec3;
type Reflectance = vec3;
```

Aliases are resolved transitively and are exported when the module is imported.

### 4.5 Struct Definitions

```
struct_def   = "struct" IDENT "{" struct_field ("," struct_field)* ","? "}" ;
struct_field = IDENT ":" type ;
```

Struct definitions declare named aggregate types. In the current version, struct support is limited; uniform and push constant blocks serve the same purpose for GPU data layout.

```
struct Material {
    albedo: vec3,
    roughness: scalar,
    metallic: scalar,
}
```

### 4.6 Import Declarations

```
import_decl = "import" IDENT ";" ;
```

Imports bring all exported symbols (functions, constants, type aliases, schedules) from the named module into the current scope. See [Section 10](#10-import-system) for details.

```
import brdf;
import noise;
```

### 4.7 Features Declarations

```
features_decl: "features" "{" feature_field ("," feature_field)* ","? "}"
feature_field: IDENT ":" "bool"
```

Features blocks declare compile-time boolean flags. Multiple blocks are merged.

### 4.8 Conditional Blocks

```
conditional_block: "if" feature_expr "{" module_item* "}"
```

Module-level `if` blocks conditionally include top-level declarations.

### 4.9 Feature Expressions

```
?feature_expr: feature_or
?feature_or: feature_and ("||" feature_and)*
?feature_and: feature_not ("&&" feature_not)*
?feature_not: "!" feature_not -> feature_negate
            | feature_primary
?feature_primary: IDENT -> feature_ref
                | "(" feature_expr ")"
```

Feature expressions evaluate at compile time against the active feature set. They support `&&` (and), `||` (or), `!` (not), and parenthesized grouping.

### 4.10 `if` Guard Suffix

The `("if" feature_expr)?` suffix is supported on:
- `surface_sampler` — conditional texture bindings
- `layer_call` — conditional material layers
- `geometry_field` — conditional vertex attributes
- `output_binding` — conditional vertex outputs
- `schedule_member` — conditional schedule overrides
- `pipeline_member` — conditional pipeline configuration

---

## 5. Expressions

### 5.1 Primary Expressions

```
primary = NUMBER          -> number_lit
        | "true"          -> bool_true
        | "false"         -> bool_false
        | IDENT           -> var_ref
        | "(" expr ")"   ;
```

A primary expression is a numeric literal, boolean literal, variable reference, or parenthesized expression.

### 5.2 Arithmetic Expressions

```
expr + expr     // addition
expr - expr     // subtraction
expr * expr     // multiplication
expr / expr     // division
expr % expr     // modulo (floating-point remainder)
```

All arithmetic is floating-point (`OpFAdd`, `OpFSub`, `OpFMul`, `OpFDiv`, `OpFMod`). Mixed scalar/vector operations broadcast the scalar to all components. Matrix multiplication uses the appropriate SPIR-V instructions (`OpMatrixTimesVector`, `OpVectorTimesMatrix`, `OpMatrixTimesMatrix`, `OpMatrixTimesScalar`, `OpVectorTimesScalar`).

### 5.3 Comparison Expressions

```
expr == expr    // equal
expr != expr    // not equal
expr <  expr    // less than
expr >  expr    // greater than
expr <= expr    // less than or equal
expr >= expr    // greater than or equal
```

All comparisons produce `bool` and use ordered floating-point comparison (`OpFOrdEqual`, `OpFOrdNotEqual`, `OpFOrdLessThan`, `OpFOrdGreaterThan`, `OpFOrdLessThanEqual`, `OpFOrdGreaterThanEqual`).

### 5.4 Logical Expressions

```
expr && expr    // logical AND
expr || expr    // logical OR
!expr           // logical NOT
```

Logical operators produce `bool` (`OpLogicalAnd`, `OpLogicalOr`, `OpLogicalNot`).

### 5.5 Unary Expressions

```
-expr           // arithmetic negation (OpFNegate)
!expr           // logical NOT (OpLogicalNot)
```

Unary negation requires a numeric operand (scalar, vector, or matrix). Logical NOT requires a `bool` operand and produces `bool`.

### 5.6 Ternary Conditional

```
condition ? then_expr : else_expr
```

The condition must evaluate to `bool`. Both branches must have the same type. The ternary operator is compiled to `OpSelect`.

### 5.7 Function Calls

```
call_expr = postfix_expr "(" arg_list? ")" ;
arg_list  = expr ("," expr)* ;
```

Function calls invoke built-in functions, standard library functions, or user-defined functions. Overload resolution is performed based on argument count and types.

```
let d: scalar = dot(n, l);
let c: vec3 = normalize(v + l);
let s: vec4 = sample(albedo_tex, uv);
```

### 5.8 Constructor Expressions

```
constructor_expr = TYPE_CONSTRUCTOR "(" arg_list? ")" ;
TYPE_CONSTRUCTOR = "vec2" | "vec3" | "vec4"
                 | "ivec2" | "ivec3" | "ivec4"
                 | "uvec2" | "uvec3" | "uvec4"
                 | "mat2" | "mat3" | "mat4" ;
```

Constructors create vector or matrix values. See [Section 3.6](#36-type-constructors) for rules.

### 5.9 Member Access

```
field_access = postfix_expr "." IDENT ;
```

Member access is used for struct field access and uniform block field access.

### 5.10 Swizzle Access

```
swizzle_access = postfix_expr "." SWIZZLE ;
```

Swizzle access extracts or rearranges vector components. See [Section 2.8](#28-swizzle-patterns) for the swizzle pattern syntax. Swizzle access on the left-hand side of an assignment writes to the selected components.

Single-component swizzle uses `OpCompositeExtract`. Multi-component swizzle uses `OpVectorShuffle`.

### 5.11 Index Access

```
index_access = postfix_expr "[" expr "]" ;
```

Index access extracts elements from vectors or matrices:

- `vecN[i]` produces a `scalar` (or the vector's component type)
- `matN[i]` produces a `vecN` (the i-th column)

### 5.12 Operator Precedence Summary

From lowest to highest precedence:

1. **Ternary**: `? :`
2. **Logical OR**: `||`
3. **Logical AND**: `&&`
4. **Equality**: `==`, `!=`
5. **Comparison**: `<`, `>`, `<=`, `>=`
6. **Additive**: `+`, `-`
7. **Multiplicative**: `*`, `/`, `%`
8. **Unary**: `-` (negation), `!` (logical NOT)
9. **Postfix**: `.field`, `.swizzle`, `(args)`, `[index]`

Parentheses may be used to override precedence.

---

## 6. Statements

### 6.1 Variable Declaration

```
let name: type = expr;
```

Declares a new mutable variable with the given type and initial value. The variable is in scope from its declaration to the end of the enclosing function. In SPIR-V, local variables are allocated as `Function`-storage-class pointers at the top of the function's first block.

### 6.2 Assignment

```
name = expr;
```

Assigns a new value to a previously declared variable, output, or built-in variable.

### 6.3 Swizzle/Member/Index Assignment

```
name.x = expr;
name.xyz = expr;
name.field = expr;
name[index] = expr;
```

Assignment targets may be swizzle patterns, field accesses, or index accesses. The left-hand side must refer to a mutable location.

### 6.4 Return Statement

```
return expr;
```

Returns a value from the enclosing function. The type of the expression must match the function's declared return type. Void functions do not require a return statement; an implicit `OpReturn` is emitted at the end.

### 6.5 If/Else Statement

```
if (condition) {
    statements...
}

if (condition) {
    statements...
} else {
    statements...
}
```

The condition must be enclosed in parentheses and must produce a `bool` value. Both branches are enclosed in braces. The else clause is optional.

In SPIR-V, if/else is compiled to `OpSelectionMerge` and `OpBranchConditional` with distinct labeled blocks.

### 6.6 Expression Statement

```
expr;
```

An expression followed by a semicolon is a statement. This is used for function calls with side effects (e.g., `trace_ray(...)`, `ignore_intersection()`).

---

## 7. Shader Stages

### 7.1 Stage Block Syntax

```
stage_block = STAGE_TYPE "{" stage_item* "}" ;
STAGE_TYPE  = "vertex" | "fragment" | "raygen" | "closest_hit"
            | "any_hit" | "miss" | "intersection" | "callable" ;
```

A stage block defines a single shader stage. Multiple stage blocks may appear in a single `.lux` file. Each stage block compiles to a separate SPIR-V module.

### 7.2 Stage Items

Within a stage block, the following declarations are permitted:

#### 7.2.1 Input Variables

```
in name: type;
```

Declares a stage input variable. Inputs are automatically assigned `Location` decorations by declaration order, starting from 0. In SPIR-V, inputs use the `Input` storage class.

#### 7.2.2 Output Variables

```
out name: type;
```

Declares a stage output variable. Outputs are automatically assigned `Location` decorations by declaration order, starting from 0. In SPIR-V, outputs use the `Output` storage class.

#### 7.2.3 Uniform Blocks

```
uniform BlockName {
    field1: type,
    field2: type,
    ...
}
```

Declares a uniform buffer object. Fields are laid out using the **std140** layout rules. The block is automatically assigned a descriptor set and binding number. Uniform block fields are accessible directly by name within the stage (no block prefix needed).

In SPIR-V, uniform blocks use the `Uniform` storage class, and field access is compiled to `OpAccessChain` with integer field indices.

**std140 Layout Rules**:

| Type | Size (bytes) | Alignment (bytes) |
|---|---|---|
| `scalar`, `int`, `uint`, `bool` | 4 | 4 |
| `vec2` | 8 | 8 |
| `vec3` | 12 | 16 |
| `vec4` | 16 | 16 |
| `mat2` | 32 | 16 |
| `mat3` | 48 | 16 |
| `mat4` | 64 | 16 |

#### 7.2.4 Push Constant Blocks

```
push BlockName {
    field1: type,
    field2: type,
    ...
}
```

Declares a push constant block. Fields follow std140 layout. Push constant fields are accessible directly by name. In SPIR-V, push constant blocks use the `PushConstant` storage class.

#### 7.2.5 Sampler Declarations

```
sampler2d name;
samplerCube name;
```

Declares a 2D texture sampler or a cube map texture sampler. Each sampler declaration generates two bindings: one for the sampler state and one for the texture image (for WebGPU/Vulkan compatibility). Samplers are automatically assigned descriptor set and binding numbers.

A `sampler2d` samples a 2D texture image with `vec2` UV coordinates. A `samplerCube` samples a cube map texture with a `vec3` direction vector.

#### 7.2.6 Functions

```
fn name(params...) -> return_type { body }
```

Stage-local functions. Each stage must contain exactly one function named `main` with no parameters and no return type (void). Additional helper functions may be defined within the stage.

#### 7.2.7 RT-Specific Declarations

The following declarations are valid only in ray tracing stages:

```
ray_payload name: type;          // Ray payload variable
hit_attribute name: type;        // Hit attribute variable
callable_data name: type;        // Callable data variable
acceleration_structure name;     // Acceleration structure binding
```

**Ray payloads** use `RayPayloadKHR` storage class in `raygen`/`miss` stages and `IncomingRayPayloadKHR` in `closest_hit`/`any_hit` stages.

**Hit attributes** use the `HitAttributeKHR` storage class.

**Callable data** uses `CallableDataKHR` storage class in calling stages and `IncomingCallableDataKHR` in `callable` stages.

**Acceleration structures** use `UniformConstant` storage class and are automatically assigned descriptor set and binding numbers.

### 7.3 Rasterization Stages

#### 7.3.1 Vertex Stage

```
vertex { ... }
```

Execution model: `Vertex`. Processes per-vertex data. The special built-in variable `builtin_position` (type `vec4`) is available for writing the clip-space position. This maps to the `gl_Position` member of the `gl_PerVertex` output block.

#### 7.3.2 Fragment Stage

```
fragment { ... }
```

Execution model: `Fragment`. Execution mode: `OriginUpperLeft`. Processes per-fragment data from rasterized primitives.

### 7.4 Ray Tracing Stages

#### 7.4.1 Ray Generation Stage

```
raygen { ... }
```

Execution model: `RayGenerationKHR`. Entry point for ray tracing pipelines. Typically computes ray origins and directions, calls `trace_ray`, and stores results.

#### 7.4.2 Closest Hit Stage

```
closest_hit { ... }
```

Execution model: `ClosestHitKHR`. Invoked for the closest intersection along a ray. Has access to hit information builtins and incoming ray payload.

#### 7.4.3 Any Hit Stage

```
any_hit { ... }
```

Execution model: `AnyHitKHR`. Invoked for each potential intersection. Can call `ignore_intersection()` to reject the hit or `terminate_ray()` to accept it and stop traversal.

#### 7.4.4 Miss Stage

```
miss { ... }
```

Execution model: `MissKHR`. Invoked when a ray does not intersect any geometry. Typically writes environment/sky color to the ray payload.

#### 7.4.5 Intersection Stage

```
intersection { ... }
```

Execution model: `IntersectionKHR`. Implements custom ray-geometry intersection tests for procedural geometry. Calls `report_intersection()` to report hits.

#### 7.4.6 Callable Stage

```
callable { ... }
```

Execution model: `CallableKHR`. A general-purpose shader callable from other RT stages via `execute_callable()`.

---

## 8. Built-in Variables

### 8.1 Rasterization Built-ins

| Name | Type | Stage | Description |
|---|---|---|---|
| `builtin_position` | `vec4` | `vertex` | Clip-space output position (maps to `gl_Position`) |

### 8.2 Ray Tracing Built-ins

The following built-in variables are automatically available in their valid stages:

| Name | Type | Valid Stages | SPIR-V BuiltIn | Description |
|---|---|---|---|---|
| `launch_id` | `uvec3` | `raygen` | `LaunchIdKHR` | Current pixel/invocation index |
| `launch_size` | `uvec3` | `raygen` | `LaunchSizeKHR` | Total dispatch dimensions |
| `world_ray_origin` | `vec3` | `closest_hit`, `any_hit`, `miss`, `intersection` | `WorldRayOriginKHR` | Ray origin in world space |
| `world_ray_direction` | `vec3` | `closest_hit`, `any_hit`, `miss`, `intersection` | `WorldRayDirectionKHR` | Ray direction in world space |
| `ray_tmin` | `scalar` | `closest_hit`, `any_hit`, `miss`, `intersection` | `RayTminKHR` | Minimum ray parameter |
| `ray_tmax` | `scalar` | `closest_hit`, `any_hit`, `miss`, `intersection` | `RayTmaxKHR` | Maximum ray parameter |
| `hit_t` | `scalar` | `closest_hit`, `any_hit` | `RayTmaxKHR` | Hit distance (alias for `ray_tmax` at hit point) |
| `instance_id` | `int` | `closest_hit`, `any_hit`, `intersection` | `InstanceCustomIndexKHR` | Custom instance index |
| `primitive_id` | `int` | `closest_hit`, `any_hit`, `intersection` | `PrimitiveId` | Primitive index within geometry |
| `hit_kind` | `uint` | `closest_hit`, `any_hit` | `HitKindKHR` | Type of hit (triangle front/back) |
| `object_to_world` | `mat4` | `closest_hit`, `any_hit`, `intersection` | `ObjectToWorldKHR` | Object-to-world transform matrix |
| `world_to_object` | `mat4` | `closest_hit`, `any_hit`, `intersection` | `WorldToObjectKHR` | World-to-object transform matrix |
| `incoming_ray_flags` | `uint` | `closest_hit`, `any_hit`, `miss`, `intersection` | `IncomingRayFlagsKHR` | Flags passed to `trace_ray` |

Note: `hit_t` and `ray_tmax` share the same SPIR-V variable (`RayTmaxKHR`). In `closest_hit` and `any_hit` stages, `ray_tmax` holds the parametric distance to the current hit point, which is aliased as `hit_t` for readability.

---

## 9. Built-in Functions

All built-in functions are available in every stage without import. Overloads are provided for `scalar`, `vec2`, `vec3`, and `vec4` where applicable.

### 9.1 Math Functions

| Function | Signature(s) | Description | SPIR-V Mapping |
|---|---|---|---|
| `abs(x)` | `T -> T` | Absolute value | `GLSL.std.450 FAbs` |
| `sign(x)` | `T -> T` | Sign (-1, 0, or 1) | `GLSL.std.450 FSign` |
| `floor(x)` | `T -> T` | Floor (round toward negative infinity) | `GLSL.std.450 Floor` |
| `ceil(x)` | `T -> T` | Ceiling (round toward positive infinity) | `GLSL.std.450 Ceil` |
| `fract(x)` | `T -> T` | Fractional part (`x - floor(x)`) | `GLSL.std.450 Fract` |
| `mod(x, y)` | `T, T -> T` | Floating-point modulo | `OpFMod` |
| `sqrt(x)` | `T -> T` | Square root | `GLSL.std.450 Sqrt` |
| `inversesqrt(x)` | `T -> T` | Inverse square root (`1 / sqrt(x)`) | `GLSL.std.450 InverseSqrt` |
| `pow(x, y)` | `T, T -> T` | Power (`x^y`) | `GLSL.std.450 Pow` |
| `exp(x)` | `T -> T` | Natural exponential (`e^x`) | `GLSL.std.450 Exp` |
| `exp2(x)` | `T -> T` | Base-2 exponential (`2^x`) | `GLSL.std.450 Exp2` |
| `log(x)` | `T -> T` | Natural logarithm | `GLSL.std.450 Log` |
| `log2(x)` | `T -> T` | Base-2 logarithm | `GLSL.std.450 Log2` |
| `min(x, y)` | `T, T -> T` | Minimum | `GLSL.std.450 FMin` |
| `max(x, y)` | `T, T -> T` | Maximum | `GLSL.std.450 FMax` |
| `clamp(x, lo, hi)` | `T, T, T -> T` | Clamp to range [lo, hi] | `GLSL.std.450 FClamp` |
| `clamp(x, lo, hi)` | `vecN, scalar, scalar -> vecN` | Clamp vector with scalar bounds | `GLSL.std.450 FClamp` |
| `mix(a, b, t)` | `T, T, T -> T` | Linear interpolation | `GLSL.std.450 FMix` |
| `mix(a, b, t)` | `vecN, vecN, scalar -> vecN` | Interpolate vectors with scalar t | `GLSL.std.450 FMix` |
| `step(edge, x)` | `T, T -> T` | Step function (0 if x < edge, else 1) | `GLSL.std.450 Step` |
| `smoothstep(e0, e1, x)` | `T, T, T -> T` | Smooth Hermite interpolation | `GLSL.std.450 SmoothStep` |
| `fma(a, b, c)` | `T, T, T -> T` | Fused multiply-add (`a*b + c`) | `GLSL.std.450 Fma` |

Where `T` is any of `scalar`, `vec2`, `vec3`, `vec4`.

### 9.2 Trigonometric Functions

| Function | Signature(s) | Description | SPIR-V Mapping |
|---|---|---|---|
| `sin(x)` | `T -> T` | Sine (radians) | `GLSL.std.450 Sin` |
| `cos(x)` | `T -> T` | Cosine (radians) | `GLSL.std.450 Cos` |
| `tan(x)` | `T -> T` | Tangent (radians) | `GLSL.std.450 Tan` |
| `asin(x)` | `T -> T` | Arc sine | `GLSL.std.450 Asin` |
| `acos(x)` | `T -> T` | Arc cosine | `GLSL.std.450 Acos` |
| `atan(x)` | `T -> T` | Arc tangent (1 argument) | `GLSL.std.450 Atan` |
| `atan(y, x)` | `T, T -> T` | Arc tangent (2 arguments, atan2) | `GLSL.std.450 Atan2` |

Where `T` is any of `scalar`, `vec2`, `vec3`, `vec4`.

### 9.3 Vector Functions

| Function | Signature | Description | SPIR-V Mapping |
|---|---|---|---|
| `length(v)` | `T -> scalar` | Vector magnitude | `GLSL.std.450 Length` |
| `distance(a, b)` | `vecN, vecN -> scalar` | Euclidean distance | `GLSL.std.450 Distance` |
| `dot(a, b)` | `vecN, vecN -> scalar` | Dot product | `OpDot` |
| `cross(a, b)` | `vec3, vec3 -> vec3` | Cross product | `GLSL.std.450 Cross` |
| `normalize(v)` | `T -> T` | Unit vector | `GLSL.std.450 Normalize` |
| `reflect(I, N)` | `T, T -> T` | Reflection vector | `GLSL.std.450 Reflect` |
| `refract(I, N, eta)` | `T, T, scalar -> T` | Refraction vector | `GLSL.std.450 Refract` |

Where `T` is any of `scalar`, `vec2`, `vec3`, `vec4`. `distance` and `dot` are defined for `vec2`, `vec3`, `vec4` only.

### 9.4 Texture Functions

| Function | Signature | Description | SPIR-V Mapping |
|---|---|---|---|
| `sample(tex, uv)` | `sampler2d, vec2 -> vec4` | Sample a 2D texture at UV coordinates | `OpImageSampleImplicitLod` |
| `sample(tex, dir)` | `samplerCube, vec3 -> vec4` | Sample a cube map texture with a direction vector | `OpImageSampleImplicitLod` |
| `sample_lod(tex, uv, lod)` | `sampler2d, vec2, scalar -> vec4` | Sample a 2D texture at an explicit mip level | `OpImageSampleExplicitLod` with `Lod` operand |
| `sample_lod(tex, dir, lod)` | `samplerCube, vec3, scalar -> vec4` | Sample a cube map at an explicit mip level | `OpImageSampleExplicitLod` with `Lod` operand |

The `sample` function internally loads the separate sampler and texture image, combines them with `OpSampledImage`, then performs `OpImageSampleImplicitLod`. The `sample_lod` variant uses `OpImageSampleExplicitLod` with the `Lod` operand, which is required in ray tracing stages where implicit derivatives are not available.

**RT auto-rewrite**: When a surface declaration containing `sample()` calls is compiled for a ray tracing pipeline, the compiler automatically rewrites all `sample()` calls to `sample_lod()` with LOD 0. This allows the same surface declaration to work in both raster and RT modes without modification.

### 9.5 Ray Tracing Functions

| Function | Signature | Description | SPIR-V Mapping |
|---|---|---|---|
| `trace_ray(accel, ray_flags, cull_mask, sbt_offset, sbt_stride, miss_index, origin, tmin, direction, tmax, payload_loc)` | `acceleration_structure, uint, uint, uint, uint, uint, vec3, scalar, vec3, scalar, int -> void` | Trace a ray | `OpTraceRayKHR` |
| `report_intersection(hit_t, hit_kind)` | `scalar, uint -> bool` | Report a hit from intersection shader | `OpReportIntersectionKHR` |
| `execute_callable(sbt_index, callable_data_loc)` | `uint, int -> void` | Execute a callable shader | `OpExecuteCallableKHR` |
| `ignore_intersection()` | `-> void` | Ignore current intersection (any-hit only) | `OpIgnoreIntersectionKHR` |
| `terminate_ray()` | `-> void` | Accept hit and stop traversal (any-hit only) | `OpTerminateRayKHR` |

For `trace_ray`, the integer parameters (`ray_flags`, `cull_mask`, `sbt_offset`, `sbt_stride`, `miss_index`) are automatically converted from `scalar` to `uint` via `OpConvertFToU`, and `payload_loc` is converted from `scalar` to `int` via `OpConvertFToS`, since Lux treats all numeric literals as `scalar`.

---

## 10. Import System

### 10.1 Syntax

```
import module_name;
```

### 10.2 Module Resolution

The compiler searches for the module file in the following order:

1. **Standard library directory**: `luxc/stdlib/<module_name>.lux`
2. **Source file directory**: The directory containing the importing `.lux` file

The first file found is used. If no matching file is found, a compilation error is reported.

### 10.3 Symbol Merging

All exported symbols from the imported module are merged into the importing module's scope:

- **Functions**: All function definitions
- **Constants**: All constant declarations
- **Type aliases**: All type alias declarations
- **Schedules**: All schedule declarations

The following are NOT imported:
- Stage blocks
- Surface declarations
- Geometry declarations
- Pipeline declarations
- Environment declarations
- Procedural declarations

### 10.4 Namespace Rules

Imported symbols are merged directly into the caller's scope with no namespacing. Functions are available by their unqualified name:

```
import brdf;

// fresnel_schlick is now available directly
let f: vec3 = fresnel_schlick(cos_theta, f0);
```

### 10.5 Transitive Imports

If an imported module itself contains `import` declarations, those are resolved recursively. Symbols from transitively imported modules are available in the original importing module.

### 10.6 Available Standard Library Modules

| Module | Description |
|---|---|
| `brdf` | Physically-based BRDF functions |
| `sdf` | Signed distance field primitives and operations |
| `noise` | Procedural noise and FBM functions |
| `color` | Color space conversion and tonemapping |
| `colorspace` | HSV conversion and artistic color controls |
| `texture` | Normal mapping, triplanar projection, UV utilities |

---

## 11. Standard Library Reference

### 11.1 Module: brdf

Physically-based rendering building blocks. Includes type aliases and constants.

#### Exported Type Aliases

| Alias | Target |
|---|---|
| `Radiance` | `vec3` |
| `Reflectance` | `vec3` |
| `Direction` | `vec3` |
| `Normal` | `vec3` |
| `Irradiance` | `vec3` |

#### Exported Constants

| Name | Type | Value |
|---|---|---|
| `PI` | `scalar` | `3.14159265358979` |
| `INV_PI` | `scalar` | `0.31830988618379` |
| `EPSILON` | `scalar` | `0.00001` |

#### Fresnel Functions

| Function | Signature | Description |
|---|---|---|
| `fresnel_schlick` | `(cos_theta: scalar, f0: vec3) -> vec3` | Schlick's Fresnel approximation |
| `fresnel_schlick_roughness` | `(cos_theta: scalar, f0: vec3, roughness: scalar) -> vec3` | Roughness-aware Schlick Fresnel |

#### Normal Distribution Functions

| Function | Signature | Description |
|---|---|---|
| `ggx_ndf` | `(n_dot_h: scalar, roughness: scalar) -> scalar` | GGX/Trowbridge-Reitz NDF |
| `ggx_ndf_fast` | `(n_dot_h: scalar, roughness: scalar) -> scalar` | Approximate GGX NDF (no alpha squared) |

#### Geometry / Masking Functions

| Function | Signature | Description |
|---|---|---|
| `smith_ggx_g1` | `(n_dot_v: scalar, roughness: scalar) -> scalar` | Smith GGX single-direction masking |
| `smith_ggx` | `(n_dot_v: scalar, n_dot_l: scalar, roughness: scalar) -> scalar` | Smith GGX combined masking-shadowing |
| `smith_ggx_fast` | `(n_dot_v: scalar, n_dot_l: scalar, roughness: scalar) -> scalar` | Approximate height-correlated Smith GGX |
| `v_ggx_correlated` | `(n_dot_l: scalar, n_dot_v: scalar, roughness: scalar) -> scalar` | Height-correlated Smith G (glTF spec-compliant) |

#### Diffuse Models

| Function | Signature | Description |
|---|---|---|
| `lambert_brdf` | `(albedo: vec3, n_dot_l: scalar) -> vec3` | Lambertian diffuse BRDF |
| `oren_nayar_diffuse` | `(albedo: vec3, roughness: scalar, n_dot_l: scalar, n_dot_v: scalar) -> vec3` | Oren-Nayar rough diffuse |
| `burley_diffuse` | `(albedo: vec3, roughness: scalar, n_dot_l: scalar, n_dot_v: scalar, v_dot_h: scalar) -> vec3` | Disney/Burley diffuse |

#### Complete BRDF Evaluation

| Function | Signature | Description |
|---|---|---|
| `microfacet_brdf` | `(n: vec3, v: vec3, l: vec3, roughness: scalar, f0: vec3) -> vec3` | Full microfacet specular BRDF (GGX + Smith + Schlick) |
| `microfacet_brdf_fast` | `(n: vec3, v: vec3, l: vec3, roughness: scalar, f0: vec3) -> vec3` | Approximate microfacet BRDF |
| `microfacet_brdf_roughness` | `(n: vec3, v: vec3, l: vec3, roughness: scalar, f0: vec3) -> vec3` | Microfacet BRDF with roughness-aware Fresnel |

#### Composite BRDF

| Function | Signature | Description |
|---|---|---|
| `pbr_brdf` | `(n: vec3, v: vec3, l: vec3, albedo: vec3, roughness: scalar, metallic: scalar) -> vec3` | Full PBR metallic-roughness BRDF |
| `pbr_brdf_fast` | `(n: vec3, v: vec3, l: vec3, albedo: vec3, roughness: scalar, metallic: scalar) -> vec3` | Approximate PBR BRDF |
| `gltf_pbr` | `(n: vec3, v: vec3, l: vec3, albedo: vec3, roughness: scalar, metallic: scalar) -> vec3` | glTF 2.0 spec-compliant PBR (height-correlated Smith) |

#### Conductor Fresnel

| Function | Signature | Description |
|---|---|---|
| `conductor_fresnel` | `(f0: vec3, f82: vec3, v_dot_h: scalar) -> vec3` | Conductor Fresnel with Lazanyi correction |

#### Sheen

| Function | Signature | Description |
|---|---|---|
| `charlie_ndf` | `(roughness: scalar, n_dot_h: scalar) -> scalar` | Charlie NDF for sheen |
| `sheen_visibility` | `(n_dot_l: scalar, n_dot_v: scalar) -> scalar` | Sheen visibility term |
| `sheen_brdf` | `(sheen_color: vec3, roughness: scalar, n_dot_h: scalar, n_dot_l: scalar, n_dot_v: scalar) -> vec3` | Complete sheen BRDF |

#### Clearcoat

| Function | Signature | Description |
|---|---|---|
| `clearcoat_brdf` | `(n: vec3, v: vec3, l: vec3, clearcoat: scalar, clearcoat_roughness: scalar) -> scalar` | Clearcoat specular layer |

#### Anisotropic GGX

| Function | Signature | Description |
|---|---|---|
| `anisotropic_ggx_ndf` | `(n_dot_h: scalar, t_dot_h: scalar, b_dot_h: scalar, at: scalar, ab: scalar) -> scalar` | Anisotropic GGX NDF |
| `anisotropic_v_ggx` | `(n_dot_l: scalar, n_dot_v: scalar, t_dot_v: scalar, b_dot_v: scalar, t_dot_l: scalar, b_dot_l: scalar, at: scalar, ab: scalar) -> scalar` | Anisotropic visibility term |

#### Transmission / Volume

| Function | Signature | Description |
|---|---|---|
| `transmission_btdf` | `(n: vec3, v: vec3, l: vec3, roughness: scalar, ior: scalar) -> scalar` | Thin-surface transmission BTDF |
| `transmission_color` | `(base_color: vec3, btdf: scalar, transmission_factor: scalar) -> vec3` | Apply transmission factor to base color |
| `diffuse_transmission` | `(albedo: vec3, n_dot_l: scalar) -> vec3` | Back-face Lambert for transmitted light |
| `volumetric_btdf` | `(n: vec3, v: vec3, l: vec3, roughness: scalar, eta_i: scalar, eta_o: scalar) -> scalar` | Walter 2007 volumetric refraction BTDF |
| `volume_attenuation` | `(distance: scalar, attenuation_color: vec3, attenuation_distance: scalar) -> vec3` | Beer-Lambert volume attenuation |
| `ior_to_f0` | `(ior: scalar) -> scalar` | Convert IOR to Fresnel F0 |

#### Iridescence

| Function | Signature | Description |
|---|---|---|
| `iridescence_f0_to_ior` | `(f0: scalar) -> scalar` | Convert F0 to IOR |
| `iridescence_ior_to_f0` | `(n_t: scalar, n_i: scalar) -> scalar` | Interface F0 from two IORs |
| `iridescence_sensitivity` | `(opd: scalar, shift: vec3) -> vec3` | Spectral sensitivity via CIE XYZ Gaussians |
| `iridescence_fresnel` | `(outside_ior: scalar, film_ior: scalar, base_f0: vec3, thickness: scalar, cos_theta: scalar) -> vec3` | Thin-film iridescence Fresnel (Belcour 2017) |

#### Dispersion

| Function | Signature | Description |
|---|---|---|
| `dispersion_ior` | `(base_ior: scalar, dispersion: scalar) -> vec3` | Per-channel IOR via Cauchy/Abbe number |
| `dispersion_refract` | `(v: vec3, n: vec3, base_ior: scalar, dispersion: scalar) -> vec3` | Per-channel refraction |
| `dispersion_f0` | `(base_ior: scalar, dispersion: scalar) -> vec3` | Per-channel F0 from dispersed IOR |

### 11.2 Module: sdf

Signed distance field primitives, CSG operators, transforms, and utilities.

#### SDF Primitives

| Function | Signature | Description |
|---|---|---|
| `sdf_sphere` | `(p: vec3, radius: scalar) -> scalar` | Sphere centered at origin |
| `sdf_box` | `(p: vec3, half_extents: vec3) -> scalar` | Axis-aligned box |
| `sdf_round_box` | `(p: vec3, half_extents: vec3, radius: scalar) -> scalar` | Rounded box |
| `sdf_plane` | `(p: vec3, normal: vec3, offset: scalar) -> scalar` | Infinite plane |
| `sdf_torus` | `(p: vec3, major: scalar, minor: scalar) -> scalar` | Torus (in XZ plane) |
| `sdf_cylinder` | `(p: vec3, radius: scalar, height: scalar) -> scalar` | Capped cylinder |
| `sdf_capsule` | `(p: vec3, a: vec3, b: vec3, radius: scalar) -> scalar` | Capsule between two points |

#### CSG Operators

| Function | Signature | Description |
|---|---|---|
| `sdf_union` | `(d1: scalar, d2: scalar) -> scalar` | Union (min) |
| `sdf_intersection` | `(d1: scalar, d2: scalar) -> scalar` | Intersection (max) |
| `sdf_subtraction` | `(d1: scalar, d2: scalar) -> scalar` | Subtraction |
| `sdf_smooth_union` | `(d1: scalar, d2: scalar, k: scalar) -> scalar` | Smooth union with blending factor k |
| `sdf_smooth_subtraction` | `(d1: scalar, d2: scalar, k: scalar) -> scalar` | Smooth subtraction with blending factor k |

#### Transform Helpers

| Function | Signature | Description |
|---|---|---|
| `sdf_translate` | `(p: vec3, offset: vec3) -> vec3` | Translate point |
| `sdf_scale` | `(p: vec3, factor: scalar) -> vec3` | Scale point |
| `sdf_repeat` | `(p: vec3, spacing: vec3) -> vec3` | Infinite repetition |

#### Utilities

| Function | Signature | Description |
|---|---|---|
| `sdf_round` | `(d: scalar, radius: scalar) -> scalar` | Round edges |
| `sdf_onion` | `(d: scalar, thickness: scalar) -> scalar` | Hollow shell |
| `sdf_elongate` | `(p: vec3, h: vec3) -> vec3` | Elongate shape |

### 11.3 Module: noise

Procedural noise and fractal functions. Uses sin-based arithmetic hashing (no bitwise operations required).

#### Hash Functions

| Function | Signature | Description |
|---|---|---|
| `hash21` | `(p: vec2) -> scalar` | 2D to 1D hash |
| `hash22` | `(p: vec2) -> vec2` | 2D to 2D hash |
| `hash31` | `(p: vec3) -> scalar` | 3D to 1D hash |
| `hash33` | `(p: vec3) -> vec3` | 3D to 3D hash |

#### Value Noise

| Function | Signature | Description |
|---|---|---|
| `value_noise2d` | `(p: vec2) -> scalar` | 2D value noise with cubic interpolation |
| `value_noise3d` | `(p: vec3) -> scalar` | 3D value noise with cubic interpolation |

#### Gradient Noise

| Function | Signature | Description |
|---|---|---|
| `gradient_noise2d` | `(p: vec2) -> scalar` | 2D Perlin-style gradient noise |
| `gradient_noise3d` | `(p: vec3) -> scalar` | 3D Perlin-style gradient noise |

#### Fractal Brownian Motion (FBM)

| Function | Signature | Description |
|---|---|---|
| `fbm2d_4` | `(p: vec2, lacunarity: scalar, gain: scalar) -> scalar` | 2D FBM, 4 octaves (unrolled) |
| `fbm2d_6` | `(p: vec2, lacunarity: scalar, gain: scalar) -> scalar` | 2D FBM, 6 octaves (unrolled) |
| `fbm3d_4` | `(p: vec3, lacunarity: scalar, gain: scalar) -> scalar` | 3D FBM, 4 octaves (unrolled) |
| `fbm3d_6` | `(p: vec3, lacunarity: scalar, gain: scalar) -> scalar` | 3D FBM, 6 octaves (unrolled) |

#### Voronoi

| Function | Signature | Description |
|---|---|---|
| `voronoi2d` | `(p: vec2) -> vec2` | 2D Voronoi (returns vec2: cell distance, edge distance). Unrolled 3x3 grid search. |

### 11.4 Module: color

Color space conversions and tonemapping operators.

| Function | Signature | Description |
|---|---|---|
| `linear_to_srgb` | `(c: vec3) -> vec3` | Convert linear color to sRGB gamma space |
| `srgb_to_linear` | `(c: vec3) -> vec3` | Convert sRGB color to linear space |
| `luminance` | `(c: vec3) -> scalar` | Compute perceptual luminance (Rec. 709 weights) |
| `tonemap_reinhard` | `(hdr: vec3) -> vec3` | Reinhard tonemapping operator |
| `tonemap_aces` | `(hdr: vec3) -> vec3` | ACES filmic tonemapping operator |

### 11.5 Module: colorspace

HSV color space conversion and artistic color controls.

| Function | Signature | Description |
|---|---|---|
| `rgb_to_hsv` | `(c: vec3) -> vec3` | Convert RGB to HSV (H in [0,1], S in [0,1], V in [0,1]) |
| `hsv_to_rgb` | `(c: vec3) -> vec3` | Convert HSV to RGB |
| `contrast` | `(c: vec3, pivot: scalar, amount: scalar) -> vec3` | Adjust contrast around a pivot point |
| `saturate_color` | `(c: vec3, amount: scalar) -> vec3` | Adjust color saturation |
| `hue_shift` | `(c: vec3, shift: scalar) -> vec3` | Rotate hue by shift amount (in [0,1]) |
| `brightness` | `(c: vec3, amount: scalar) -> vec3` | Multiply brightness |
| `gamma_correct` | `(c: vec3, gamma: scalar) -> vec3` | Apply gamma correction |

### 11.6 Module: texture

Normal mapping, triplanar projection, parallax mapping, and UV utilities.

#### TBN / Normal Mapping

| Function | Signature | Description |
|---|---|---|
| `tbn_perturb_normal` | `(normal_sample: vec3, world_normal: vec3, world_tangent: vec3, world_bitangent: vec3) -> vec3` | Transform tangent-space normal to world space |
| `tbn_from_tangent` | `(normal: vec3, tangent: vec4) -> vec3` | Compute bitangent from normal and tangent (w = handedness) |
| `unpack_normal` | `(encoded: vec3) -> vec3` | Decode normal from [0,1] to [-1,1] range |
| `unpack_normal_strength` | `(encoded: vec3, strength: scalar) -> vec3` | Decode normal with adjustable strength |

#### Triplanar Projection

| Function | Signature | Description |
|---|---|---|
| `triplanar_weights` | `(normal: vec3, sharpness: scalar) -> vec3` | Compute blend weights for triplanar projection |
| `triplanar_uv_x` | `(world_pos: vec3) -> vec2` | UV coordinates for X-axis projection |
| `triplanar_uv_y` | `(world_pos: vec3) -> vec2` | UV coordinates for Y-axis projection |
| `triplanar_uv_z` | `(world_pos: vec3) -> vec2` | UV coordinates for Z-axis projection |
| `triplanar_blend` | `(sample_x: vec3, sample_y: vec3, sample_z: vec3, weights: vec3) -> vec3` | Blend three color samples by triplanar weights |
| `triplanar_blend_scalar` | `(sample_x: scalar, sample_y: scalar, sample_z: scalar, weights: vec3) -> scalar` | Blend three scalar samples by triplanar weights |

#### Parallax Mapping

| Function | Signature | Description |
|---|---|---|
| `parallax_offset` | `(height: scalar, scale: scalar, view_dir_ts: vec3) -> vec2` | Simple parallax offset mapping |

#### UV Utilities

| Function | Signature | Description |
|---|---|---|
| `rotate_uv` | `(uv: vec2, angle: scalar, center: vec2) -> vec2` | Rotate UV coordinates around center |
| `tile_uv` | `(uv: vec2, scale: vec2) -> vec2` | Tile UV coordinates with scale |

### 11.7 Module: toon

Custom `@layer` functions for non-photorealistic rendering.

| Function | Signature | Description |
|---|---|---|
| `cartoon` | `(base: vec3, n: vec3, v: vec3, l: vec3, bands: scalar, rim_power: scalar, rim_color: vec3) -> vec3` | Cel-shading with quantized NdotL lighting bands and Fresnel rim lighting. Annotated with `@layer` for use in `layers [...]` blocks. |

---

## 12. Declarative Syntax

Lux provides high-level declarative constructs that expand into full shader stage blocks. This separates material definition from GPU plumbing.

### 12.1 Geometry Block

A geometry block declares the vertex layout, transform uniforms, and output bindings for a vertex shader.

```
geometry Name {
    field1: type,
    field2: type,
    ...
    transform: BlockName {
        uniform_field1: type,
        uniform_field2: type,
        ...
    }
    outputs {
        output_name: expression,
        ...
        clip_pos: expression,
    }
}
```

**Fields** become vertex input attributes (in declaration order).

**Transform block** becomes a uniform buffer. The block name and fields become the uniform block name and fields.

**Outputs block** defines the values passed to the fragment shader:
- Named outputs (e.g., `world_pos`, `world_normal`, `frag_uv`) become vertex outputs and fragment inputs.
- The special name `clip_pos` maps to `builtin_position` (clip-space position).

Output expressions may reference input fields, transform uniforms, and built-in functions.

**Expansion**: The compiler generates a `vertex` stage block with:
- `in` declarations for each geometry field
- A `uniform` block from the transform
- `out` declarations for each output binding (except `clip_pos`)
- A `fn main()` that evaluates output expressions and writes to outputs/`builtin_position`

### 12.2 Surface Block

A surface block declares material properties and the BRDF evaluation function. There are two syntax forms: the **member syntax** (using `brdf:`) and the **layered syntax** (using `layers [...]`).

#### 12.2.1 Member Syntax (Basic)

```
surface Name {
    sampler2d texture_name,
    member_name: expression,
    ...
}
```

**Sampler declarations** (`sampler2d name` or `samplerCube name`) register texture samplers that will be available in the generated fragment shader. Both 2D and cube map samplers are supported.

**Members** define material properties. The key member is `brdf`, which specifies the BRDF evaluation to use:

```
surface CopperMetal {
    brdf: pbr(vec3(0.95, 0.64, 0.54), 0.3, 0.9),
}
```

Supported BRDF shorthand names and their expansions:

| BRDF Name | Expansion |
|---|---|
| `lambert(albedo)` | `lambert_brdf(albedo, max(dot(n, l), 0.0))` |
| `microfacet_ggx(roughness, f0)` | `microfacet_brdf(n, v, l, roughness, f0)` |
| `pbr(albedo, roughness, metallic)` | `pbr_brdf(n, v, l, albedo, roughness, metallic)` |

The generated fragment shader provides `n` (normalized surface normal), `v` (view direction), and `l` (light direction) automatically.

**Expansion**: The compiler generates a `fragment` stage block with:
- `in` declarations for interpolated vertex outputs
- `out color: vec4;` output
- `uniform Light { light_dir: vec3, view_pos: vec3 }` block
- Sampler declarations
- A `fn main()` that normalizes the surface normal, computes view and light directions, evaluates the BRDF, applies ambient and exposure, and writes the final color

#### 12.2.2 Layered Syntax

The layered syntax provides a composable way to build complex materials from individual lighting layers:

```
surface Name {
    sampler2d texture_2d_name,
    samplerCube cubemap_name,
    member_name: expression,
    ...
    layers [
        layer_name(arg_name: expression, ...),
        layer_name(arg_name: expression, ...),
    ]
}
```

**Layer ordering**: Layers are listed bottom-to-top (base layer first). The compiler evaluates them top-to-bottom with albedo-scaling for energy conservation. Each layer consumes a fraction of the remaining light energy before passing the rest to the layer below it.

**Built-in layer types**:

| Layer | Arguments | Description |
|---|---|---|
| `base(albedo, roughness, metallic)` | `albedo: vec3`, `roughness: scalar`, `metallic: scalar` | PBR direct lighting via `gltf_pbr`. This is the foundation layer that computes Cook-Torrance specular + Lambertian diffuse. |
| `normal_map(map)` | `map: sampler2d` | TBN normal perturbation. Samples the normal map, transforms from tangent space to world space using the TBN matrix, and perturbs the surface normal used by subsequent layers. |
| `ibl(specular_map, irradiance_map, brdf_lut)` | `specular_map: samplerCube`, `irradiance_map: samplerCube`, `brdf_lut: sampler2d` | Image-based lighting with multi-scattering energy compensation. Samples the irradiance map for diffuse IBL and the pre-filtered specular map at a roughness-dependent mip level for specular IBL. Applies the split-sum approximation using the BRDF LUT and adds multi-scattering energy compensation. |
| `emission(color)` | `color: vec3` | Additive emissive contribution. Adds the given color directly to the final output, unaffected by lighting. |

**Example**:

```
surface PBRMaterial {
    sampler2d albedo_tex,
    sampler2d normal_tex,
    sampler2d metalrough_tex,
    sampler2d brdf_lut,
    samplerCube specular_map,
    samplerCube irradiance_map,
    layers [
        base(
            albedo: sample(albedo_tex, frag_uv).rgb,
            roughness: sample(metalrough_tex, frag_uv).g,
            metallic: sample(metalrough_tex, frag_uv).b,
        ),
        normal_map(map: normal_tex),
        ibl(
            specular_map: specular_map,
            irradiance_map: irradiance_map,
            brdf_lut: brdf_lut,
        ),
    ]
}
```

**Energy conservation**: The layer system ensures energy conservation automatically. The `ibl` layer adds indirect illumination that is scaled by the surface's albedo and Fresnel response, while the `base` layer provides direct illumination. The compiler evaluates layers from top to bottom, with each layer modifying the surface appearance in a physically-consistent manner.

**Custom layers via `@layer`**: Users can define custom layer functions using the `@layer` annotation on a regular function. The function must have at least 4 parameters (`base: vec3, n: vec3, v: vec3, l: vec3`) and return `vec3`. Additional parameters are mapped by name from `LayerArg` entries in the `layers [...]` block:

```
@layer
fn cartoon(base: vec3, n: vec3, v: vec3, l: vec3,
           bands: scalar, rim_power: scalar, rim_color: vec3) -> vec3 {
    // Cel-shading: quantized NdotL + rim lighting
    let n_dot_l: scalar = max(dot(n, l), 0.0);
    let quantized: scalar = floor(n_dot_l * bands + 0.5) / bands;
    let cel: vec3 = base * quantized;
    let n_dot_v: scalar = max(dot(n, v), 0.0);
    let rim: scalar = pow(1.0 - n_dot_v, rim_power);
    return cel + rim_color * rim;
}

surface ToonSurface {
    sampler2d albedo_tex,
    layers [
        base(albedo: sample(albedo_tex, uv).xyz, roughness: 0.8, metallic: 0.0),
        cartoon(bands: 4.0, rim_power: 3.0, rim_color: vec3(0.3, 0.3, 0.5)),
    ]
}
```

Custom layers are validated at compile time: the function must have ≥4 parameters with the correct types, must return `vec3`, and must not collide with a built-in layer name (`base`, `normal_map`, `ibl`, `emission`, `coat`, `sheen`, `transmission`). Custom layers are inserted in declaration order after all built-in layers and before `emission`. In RT mode, any `sample()` calls within custom layer arguments are automatically rewritten to `sample_lod()`.

#### 12.2.3 RT Unification

The same surface declaration (whether using member or layered syntax) compiles to both raster fragment shaders and RT closest-hit shaders. When compiling for a ray tracing pipeline:

- All `sample()` calls are automatically rewritten to `sample_lod()` with LOD 0, since implicit derivatives are not available in RT stages.
- The fragment shader `main()` logic is adapted into a `closest_hit` stage that reads hit attributes and writes to the incoming ray payload.
- No changes to the surface declaration are required to switch between raster and RT modes.

#### 12.2.4 AST Representation

The layered surface syntax introduces three new AST node types:

| AST Node | Fields | Description |
|---|---|---|
| `SurfaceSampler` | `sampler_type: str`, `name: str` | A sampler declaration within a surface block. The `sampler_type` field is `"sampler2d"` or `"samplerCube"`. |
| `LayerCall` | `name: str`, `args: list[LayerArg]` | A single layer invocation within a `layers` block. |
| `LayerArg` | `name: str`, `value: Expr` | A named argument to a layer call. |

### 12.3 Schedule Block

A schedule block selects algorithm variants for BRDF evaluation and post-processing, separating "what to render" from "how to render it".

```
schedule Name {
    slot: variant,
    ...
}
```

**Valid slots and variants**:

| Slot | Default | Variants |
|---|---|---|
| `fresnel` | `schlick` | `schlick`, `schlick_roughness` |
| `distribution` | `ggx` | `ggx`, `ggx_fast` |
| `geometry_term` | `smith_ggx` | `smith_ggx`, `smith_ggx_fast` |
| `tonemap` | `none` | `none`, `aces`, `reinhard` |

The schedule affects how `microfacet_ggx` and `pbr` BRDF calls are expanded:
- `distribution: ggx_fast` or `geometry_term: smith_ggx_fast` select `microfacet_brdf_fast` / `pbr_brdf_fast`
- `fresnel: schlick_roughness` selects `microfacet_brdf_roughness`
- `tonemap: aces` or `tonemap: reinhard` append tonemapping to the fragment shader output

### 12.4 Environment Block

An environment block defines the background color for miss shaders in RT pipelines.

```
environment Name {
    color: expression,
}
```

The `color` member specifies the environment color expression. It may reference RT built-in variables such as `world_ray_direction` for directional sky gradients.

**Expansion**: The compiler generates a `miss` stage block with an incoming ray payload that writes the evaluated color to the payload.

### 12.5 Procedural Block

A procedural block defines an SDF-based procedural geometry for RT intersection testing.

```
procedural Name {
    sdf: expression,
}
```

The `sdf` member specifies the signed distance function expression.

**Expansion**: The compiler generates an `intersection` stage block that performs ray marching (sphere tracing) by unrolling a fixed number of march steps (8), evaluating the SDF at each step, and calling `report_intersection` when the distance is below a threshold.

### 12.6 Pipeline Block

A pipeline block ties together geometry, surface, schedule, and environment declarations.

```
pipeline Name {
    geometry: GeometryName,
    surface: SurfaceName,
    schedule: ScheduleName,
    environment: EnvironmentName,
    procedural: ProceduralName,
    mode: raytrace,
    max_bounces: N,
}
```

All members are optional except `surface`. Members reference declarations by name.

| Member | Required | Description |
|---|---|---|
| `geometry` | No | Vertex layout and transform (rasterization only) |
| `surface` | Yes | Material definition |
| `schedule` | No | Algorithm variant selection |
| `environment` | No | Background for RT miss shader |
| `procedural` | No | SDF for RT intersection shader |
| `mode` | No | `rasterize` (default) or `raytrace` |
| `max_bounces` | No | Maximum ray recursion depth (RT only, default 1) |

**Rasterization mode** (`mode: rasterize` or omitted): Generates `vertex` and `fragment` stages from the geometry and surface.

**Ray tracing mode** (`mode: raytrace`): Generates:
- `raygen` stage with camera uniforms, acceleration structure, and ray payload
- `closest_hit` stage from the surface BRDF
- `miss` stage from the environment (if provided)
- `any_hit` stage if the surface has an `opacity` member
- `intersection` stage from the procedural (if provided)

---

## 13. Automatic Differentiation

### 13.1 The @differentiable Annotation

Functions annotated with `@differentiable` trigger automatic generation of gradient functions using forward-mode automatic differentiation.

```
@differentiable
fn energy(x: scalar) -> scalar {
    return x * x + sin(x);
}
```

### 13.2 Generated Functions

For each `scalar` parameter of a `@differentiable` function, a gradient function is generated with the naming convention:

```
{function_name}_d_{parameter_name}
```

For the example above, the compiler generates:

```
fn energy_d_x(x: scalar) -> scalar {
    return x * 1.0 + 1.0 * x + cos(x) * 1.0;
}
```

Which simplifies to `2*x + cos(x)` after constant folding.

### 13.3 Differentiation Rules

The following derivative rules are applied:

#### Constants and Variables
- `d/dx(constant)` = `0.0`
- `d/dx(x)` = `1.0` (for the differentiation variable)
- `d/dx(y)` = `0.0` (for other variables) or the tracked derivative if `y` was computed from `x`

#### Arithmetic Operations
- **Sum rule**: `d/dx(a + b)` = `da + db`
- **Difference rule**: `d/dx(a - b)` = `da - db`
- **Product rule**: `d/dx(a * b)` = `a * db + da * b`
- **Quotient rule**: `d/dx(a / b)` = `(da * b - a * db) / (b * b)`

#### Unary Operations
- `d/dx(-a)` = `-da`
- `d/dx(!a)` = `0.0`

#### Built-in Function Derivatives (1-argument)

| Function | Derivative |
|---|---|
| `sin(u)` | `cos(u) * du` |
| `cos(u)` | `-sin(u) * du` |
| `tan(u)` | `du / (cos(u) * cos(u))` |
| `exp(u)` | `exp(u) * du` |
| `exp2(u)` | `exp2(u) * log(2.0) * du` |
| `log(u)` | `du / u` |
| `log2(u)` | `du / (u * log(2.0))` |
| `sqrt(u)` | `du / (2.0 * sqrt(u))` |
| `inversesqrt(u)` | `-inversesqrt(u) / (2.0 * u) * du` |
| `abs(u)` | `sign(u) * du` |
| `sign(u)` | `0.0` (piecewise constant) |
| `floor(u)` | `0.0` (piecewise constant) |
| `ceil(u)` | `0.0` (piecewise constant) |
| `step(u)` | `0.0` (piecewise constant) |
| `fract(u)` | `du` |
| `normalize(v)` | `(dv - n * dot(n, dv)) / length(v)` |
| `length(v)` | `dot(v, dv) / length(v)` |

#### Built-in Function Derivatives (2-argument)

| Function | Derivative |
|---|---|
| `pow(u, v)` (v constant) | `v * pow(u, v-1) * du` |
| `pow(u, v)` (general) | `pow(u,v) * (v*du/u + log(u)*dv)` |
| `min(u, v)` | `u < v ? du : dv` |
| `max(u, v)` | `u > v ? du : dv` |
| `step(e, x)` | `0.0` |
| `dot(a, b)` | `dot(da, b) + dot(a, db)` |
| `mod(u, v)` | `du` |
| `reflect(I, N)` | `dI - 2*N*dot(N, dI)` (simplified when N constant) |

#### Built-in Function Derivatives (3-argument)

| Function | Derivative |
|---|---|
| `mix(a, b, t)` | `da*(1-t) + db*t + (b-a)*dt` |
| `clamp(x, lo, hi)` | `(x > lo && x < hi) ? dx : 0.0` |
| `smoothstep(e0, e1, x)` | `6*t*(1-t)*dx / (e1-e0)` where `t = clamp((x-e0)/(e1-e0), 0, 1)` |

#### User-Defined Function Calls

For calls to user-defined functions, the chain rule is applied by generating calls to the gradient function for each active parameter:

```
d/dx f(a(x), b(x)) = f_d_p0(a, b) * da/dx + f_d_p1(a, b) * db/dx
```

### 13.4 Simplification

Generated derivative expressions undergo algebraic simplification:
- `0.0 + x` simplifies to `x`
- `x + 0.0` simplifies to `x`
- `0.0 * x` simplifies to `0.0`
- `1.0 * x` simplifies to `x`
- `x * 1.0` simplifies to `x`
- `-0.0` simplifies to `0.0`

### 13.5 Constructor and Swizzle Differentiation

- `d/dx vec3(a, b, c)` = `vec3(da, db, dc)` (component-wise)
- `d/dx expr.xyz` = `(d/dx expr).xyz` (swizzle propagation)
- Ternary: condition unchanged, both branches differentiated

### 13.6 Limitations

- Only `scalar`-typed parameters are differentiated
- Non-scalar parameters have zero derivative
- Functions in both module scope and stage scope are processed
- No branching differentiation through if/else control flow (condition is kept, both branches are independently differentiated)

---

## 14. Compilation Pipeline

The Lux compiler processes source code through a 9-stage pipeline:

### Stage 1: Parse

The Lark parser reads the `.lux` source file using the grammar defined in `lux.lark`. The Lark parse tree is transformed into typed AST dataclasses by the tree builder.

**Input**: Lux source text
**Output**: `Module` AST with all top-level declarations

### Stage 2: Import Resolution

All `import` declarations are resolved by searching for `.lux` files in the standard library directory (`luxc/stdlib/`) and the source file's directory. Imported modules are parsed recursively, and their exported symbols (functions, constants, type aliases, schedules) are merged into the importing module.

**Input**: `Module` with import declarations
**Output**: `Module` with merged symbols from all imports

### Stage 3: Surface Expansion

Declarative blocks (`surface`, `geometry`, `pipeline`, `environment`, `procedural`) are expanded into concrete `StageBlock` instances. Pipeline declarations drive the expansion, selecting rasterization or ray tracing mode and wiring together the appropriate declarations.

**Input**: `Module` with declarative blocks
**Output**: `Module` with generated stage blocks appended

### Stage 4: Autodiff Expansion

Functions annotated with `@differentiable` are processed to generate gradient functions using forward-mode automatic differentiation. Generated functions are added to the module's function list.

**Input**: `Module` with `@differentiable` functions
**Output**: `Module` with generated gradient functions appended

### Stage 5: Type Checking

All functions and stage blocks are type-checked. This includes:
- Resolving type aliases
- Registering constants, inputs, outputs, uniforms, push constants, samplers, and RT variables in the symbol table
- Checking expression types, operator compatibility, and function call overloads
- Verifying return type consistency
- Annotating AST nodes with resolved type information

**Input**: `Module` with all functions and stages
**Output**: Type-annotated `Module` (or error)

### Stage 6: Constant Folding

Compile-time evaluation of constant expressions:
- Arithmetic on numeric literals (`1.0 + 2.0` becomes `3.0`)
- Constant variable inlining (`const PI` replaced with literal value)
- Known built-in function calls on literals (`sin(0.0)` becomes `0.0`)
- Dead branch elimination: `if(true)` keeps only the then-branch, `if(false)` keeps only the else-branch
- Ternary with constant condition selects the appropriate branch

Foldable built-in functions (1-arg): `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `exp`, `exp2`, `log`, `log2`, `sqrt`, `abs`, `floor`, `ceil`, `fract`, `sign`.

Foldable built-in functions (2-arg): `min`, `max`, `pow`, `step`.

Foldable built-in functions (3-arg): `mix`, `clamp`, `smoothstep`.

**Input**: Type-annotated `Module`
**Output**: `Module` with folded constants

### Stage 7: Layout Assignment

Automatic assignment of:
- **Input locations**: By declaration order, starting from 0
- **Output locations**: By declaration order, starting from 0
- **Uniform block descriptor set and binding**: Auto-assigned sequentially
- **Sampler bindings**: Each sampler gets two consecutive bindings (sampler state + texture image)
- **RT payload locations**: Auto-assigned by declaration order
- **Callable data locations**: Auto-assigned by declaration order
- **Acceleration structure bindings**: Auto-assigned sequentially after uniforms and samplers

Fragment stages in a pipeline with geometry get a descriptor set offset of 1 to avoid clashing with vertex stage uniforms.

**Input**: `Module` with stages
**Output**: `Module` with all layout numbers assigned

### Stage 8: SPIR-V Generation

Each stage block is compiled to SPIR-V assembly text. The generator:
- Declares all types, constants, and global variables
- Emits decorations (locations, bindings, offsets, built-ins)
- Generates the `main` entry point function
- Inlines all user-defined function calls
- Pre-declares all local variables at the top of the function (SPIR-V requirement)

**Input**: Layout-assigned `Module` + `StageBlock`
**Output**: SPIR-V assembly text (`.spvasm`)

### Stage 9: Assembly and Validation

The SPIR-V assembly text is processed by external tools:
1. `spirv-as` assembles the text to a binary `.spv` file
2. `spirv-val` validates the binary (optional, can be skipped with `--no-validate`)

**Input**: `.spvasm` text
**Output**: Validated `.spv` binary

### Output Files

| Stage Type | File Extension |
|---|---|
| `vertex` | `.vert.spv` |
| `fragment` | `.frag.spv` |
| `raygen` | `.rgen.spv` |
| `closest_hit` | `.rchit.spv` |
| `any_hit` | `.rahit.spv` |
| `miss` | `.rmiss.spv` |
| `intersection` | `.rint.spv` |
| `callable` | `.rcall.spv` |

### CLI Flags

| Flag | Description |
|---|---|
| `--no-validate` | Skip SPIR-V validation after assembly. |
| `--pipeline <name>` | Compile only the named pipeline from a multi-pipeline file. When a `.lux` file contains multiple `pipeline` declarations, this flag selects a single pipeline by name for compilation. Only the stage blocks generated by the selected pipeline (and its referenced geometry, surface, environment, and procedural declarations) are emitted. If omitted, all pipelines in the file are compiled. |
| `--features <list>` | Comma-separated list of feature flags to enable (e.g., `--features has_normal_map,has_clearcoat`). |
| `--all-permutations` | Compile all 2^N feature combinations and emit a permutation manifest. |

### Compile-Time Feature Stripping

When a module declares `features { ... }`, the compiler:

1. Collects all feature flag names from `features_decl` blocks
2. Receives the active feature set from `--features` CLI flag (or programmatic API)
3. Evaluates all `if` guard conditions against the active set
4. Strips (removes) items whose conditions evaluate to false
5. Inlines `conditional_block` contents whose conditions are true
6. Clears all condition fields — downstream passes see a clean AST

This is a preprocessing step, not runtime branching. The generated SPIR-V contains only code for enabled features.

#### Output Naming

Feature-enabled outputs include a sorted suffix:
- Base (no features): `shader.frag.spv`
- With features: `shader+emission+normal_map.frag.spv`

#### Reflection Metadata

When features are active, the reflection JSON includes:

```json
{
    "features": {
        "has_normal_map": true,
        "has_clearcoat": false
    },
    "feature_suffix": "+normal_map"
}
```

#### Permutation Generation

`--all-permutations` compiles all 2^N combinations and emits a manifest:

```json
{
    "pipeline": "GltfForward",
    "features": ["has_normal_map", "has_clearcoat"],
    "permutations": [
        { "suffix": "", "features": {"has_normal_map": false, "has_clearcoat": false} },
        { "suffix": "+normal_map", "features": {"has_normal_map": true, "has_clearcoat": false} }
    ]
}
```

---

## 15. SPIR-V Output

### 15.1 SPIR-V Version

- **Rasterization stages**: SPIR-V 1.0
- **Ray tracing stages**: SPIR-V 1.4

### 15.2 Extensions

- `SPV_KHR_ray_tracing` (for RT stages only)

### 15.3 Capabilities

- `Shader` (all stages)
- `RayTracingKHR` (RT stages only)

### 15.4 Extended Instruction Set

- `GLSL.std.450` (imported via `OpExtInstImport`)

### 15.5 Memory Model

```
OpMemoryModel Logical GLSL450
```

### 15.6 Execution Models

| Stage | Execution Model |
|---|---|
| `vertex` | `Vertex` |
| `fragment` | `Fragment` |
| `raygen` | `RayGenerationKHR` |
| `closest_hit` | `ClosestHitKHR` |
| `any_hit` | `AnyHitKHR` |
| `miss` | `MissKHR` |
| `intersection` | `IntersectionKHR` |
| `callable` | `CallableKHR` |

### 15.7 Execution Modes

- Fragment shaders: `OriginUpperLeft`
- All other stages: no additional execution modes

### 15.8 Layout Rules

- **Uniform buffers**: std140 layout with `Block` decoration
- **Push constants**: std140 layout with `Block` decoration, `PushConstant` storage class
- **Input/Output variables**: Location-based, auto-assigned by declaration order
- **Samplers**: Separate sampler + texture pattern (two bindings per sampler)
- **Matrices**: `ColMajor` layout, `MatrixStride 16`

### 15.9 Vertex Shader Output Block

The vertex stage emits a `gl_PerVertex` output block containing:
- `gl_Position` (vec4, BuiltIn Position) -- mapped from `builtin_position`
- `gl_PointSize` (scalar, BuiltIn PointSize)
- `gl_ClipDistance[]` (float array, BuiltIn ClipDistance)
- `gl_CullDistance[]` (float array, BuiltIn CullDistance)

Only `gl_Position` is written; the others are present for interface completeness.

### 15.10 Function Inlining

All user-defined function calls are inlined at the call site. The inliner:
- Creates unique local variable names to avoid collisions (`_inline_{id}_{param}`)
- Stores argument values into parameter variables
- Executes the function body statements
- Captures the return value

No `OpFunctionCall` instructions are emitted for user-defined functions.

### 15.11 SPIR-V Instruction Mapping

| Lux Operation | SPIR-V Instruction |
|---|---|
| `a + b` (float) | `OpFAdd` |
| `a - b` (float) | `OpFSub` |
| `a * b` (float) | `OpFMul` |
| `a / b` (float) | `OpFDiv` |
| `a % b` (float) | `OpFMod` |
| `a * b` (vec * scalar) | `OpVectorTimesScalar` |
| `a * b` (mat * vec) | `OpMatrixTimesVector` |
| `a * b` (vec * mat) | `OpVectorTimesMatrix` |
| `a * b` (mat * mat) | `OpMatrixTimesMatrix` |
| `a * b` (mat * scalar) | `OpMatrixTimesScalar` |
| `-a` | `OpFNegate` |
| `!a` | `OpLogicalNot` |
| `a && b` | `OpLogicalAnd` |
| `a \|\| b` | `OpLogicalOr` |
| `a == b` | `OpFOrdEqual` |
| `a != b` | `OpFOrdNotEqual` |
| `a < b` | `OpFOrdLessThan` |
| `a > b` | `OpFOrdGreaterThan` |
| `a <= b` | `OpFOrdLessThanEqual` |
| `a >= b` | `OpFOrdGreaterThanEqual` |
| `cond ? a : b` | `OpSelect` |
| `v.xyz` (swizzle) | `OpVectorShuffle` or `OpCompositeExtract` |
| `vec3(a, b, c)` | `OpCompositeConstruct` |
| `let x = val` | `OpVariable` (Function) + `OpStore` |
| `x = val` | `OpStore` |
| `if (...) { } else { }` | `OpSelectionMerge` + `OpBranchConditional` |
| `return val` | `OpReturnValue` |
| `dot(a, b)` | `OpDot` |
| `mod(a, b)` | `OpFMod` |
| Built-in functions | `OpExtInst %glsl Instruction args` |
| `sample(tex, uv)` | `OpLoad` + `OpSampledImage` + `OpImageSampleImplicitLod` |
| `trace_ray(...)` | `OpTraceRayKHR` |
| `report_intersection(...)` | `OpReportIntersectionKHR` |
| `execute_callable(...)` | `OpExecuteCallableKHR` |
| `ignore_intersection()` | `OpIgnoreIntersectionKHR` |
| `terminate_ray()` | `OpTerminateRayKHR` |

---

## 16. Limitations

The following features are intentionally absent from Lux v0.2:

| Limitation | Rationale / Workaround |
|---|---|
| **No loops** (`for`, `while`, `do`) | Iterative algorithms must be manually unrolled. FBM and Voronoi in the stdlib use fixed-count unrolling. Ensures deterministic performance. |
| **No bitwise operators** (`&`, `\|`, `^`, `~`, `<<`, `>>`) | Noise functions use sin-based arithmetic hashing instead. |
| **No user-defined structs in stages** | Uniform and push constant blocks serve the same purpose. Type aliases provide named types. |
| **No arrays** | Array-like patterns are expressed via manual unrolling or separate variables. |
| **No switch statements** | Use chained ternary expressions or if/else chains. |
| **Single entry point per stage** | Each stage must have exactly one `fn main()` with no parameters and void return. |
| **No recursion** | All function calls are inlined. The only recursive call is `trace_ray` (handled by RT hardware). |
| **No string types** | No string literals or string operations. |
| **No preprocessor** | No `#define`, `#include`, or conditional compilation. Use the import system instead. |
| **No integer arithmetic ops** | All arithmetic uses floating-point SPIR-V instructions. Integer parameters (e.g., for `trace_ray`) are converted from float. |
| **No explicit layout qualifiers** | All locations, sets, and bindings are auto-assigned. There is no syntax for manual layout control. |
| **No multi-line comments** | Only `//` single-line comments are supported. |
| **No compound assignment** | `+=`, `-=`, `*=`, `/=` are not currently in the grammar; use `x = x + y;` instead. |

---

## 17. Examples

### 17.1 Minimal Vertex + Fragment (Hello Triangle)

A minimal Lux program that renders a colored triangle with per-vertex colors.

```
// Hello Triangle -- simplest working Lux program
// Produces a colored triangle with per-vertex colors

vertex {
    in position: vec3;
    in color: vec3;

    out frag_color: vec3;

    fn main() {
        frag_color = color;
        builtin_position = vec4(position, 1.0);
    }
}

fragment {
    in frag_color: vec3;
    out color: vec4;

    fn main() {
        color = vec4(frag_color, 1.0);
    }
}
```

This produces two SPIR-V modules:
- `hello_triangle.vert.spv` -- vertex shader with 2 inputs (position at location 0, color at location 1), 1 output (frag_color at location 0), and builtin_position output
- `hello_triangle.frag.spv` -- fragment shader with 1 input (frag_color at location 0) and 1 output (color at location 0)

### 17.2 Fragment-Only with Imports (SDF Shapes)

A fragment shader that evaluates combined SDF primitives using the `sdf` standard library module.

```
// SDF Shapes -- render 2D cross-section of combined SDF primitives

import sdf;

fragment {
    in uv: vec2;
    out color: vec4;

    fn main() {
        // Map UV [0,1] to centered coordinates [-2, 2]
        let p: vec3 = vec3((uv - vec2(0.5)) * 4.0, 0.0);

        // Sphere at origin
        let d_sphere: scalar = sdf_sphere(p, 0.8);

        // Box offset to the right
        let p_box: vec3 = sdf_translate(p, vec3(1.2, 0.0, 0.0));
        let d_box: scalar = sdf_box(p_box, vec3(0.5, 0.5, 0.5));

        // Torus offset to the left
        let p_torus: vec3 = sdf_translate(p, vec3(-1.2, 0.0, 0.0));
        let d_torus: scalar = sdf_torus(p_torus, 0.5, 0.2);

        // Combine with smooth union
        let d_combined: scalar = sdf_smooth_union(d_sphere, d_box, 0.3);
        let d_final: scalar = sdf_smooth_union(d_combined, d_torus, 0.3);

        // Round the result
        let d_rounded: scalar = sdf_round(d_final, 0.02);

        // Color based on distance
        let inside: scalar = smoothstep(0.01, 0.0, d_rounded);
        let band: scalar = smoothstep(0.02, 0.0, abs(d_rounded));
        let warm: vec3 = vec3(0.9, 0.4, 0.1) * inside;
        let cool: vec3 = vec3(0.1, 0.2, 0.4) * (1.0 - inside);
        let edge: vec3 = vec3(1.0) * band;

        color = vec4(warm + cool + edge, 1.0);
    }
}
```

### 17.3 Declarative PBR (Surface + Geometry + Pipeline)

A fully declarative PBR material using the surface/geometry/pipeline system.

```
// PBR material with texture sampling using declarative surface syntax

import brdf;

// Declare vertex transform pipeline with UV passthrough
geometry StandardMesh {
    position: vec3,
    normal: vec3,
    uv: vec2,
    transform: MVP {
        model: mat4,
        view: mat4,
        projection: mat4,
    }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        frag_uv: uv,
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

// Declare material -- texture-sampled albedo, dielectric, moderate roughness
surface TexturedPBR {
    sampler2d albedo_tex,
    brdf: pbr(sample(albedo_tex, frag_uv).xyz, 0.5, 0.0),
}

// Tie them together -- compiler generates both vertex + fragment SPIR-V
pipeline PBRForward {
    geometry: StandardMesh,
    surface: TexturedPBR,
}
```

This single file produces `pbr_surface.vert.spv` and `pbr_surface.frag.spv`. The compiler automatically:
- Generates the vertex shader from the geometry declaration
- Generates the fragment shader with PBR BRDF evaluation, light uniforms, and texture sampling
- Assigns all layout bindings

### 17.4 Ray Tracing (Declarative RT Pipeline)

A declarative ray tracing pipeline that reuses the same surface BRDF for closest-hit shading and adds an environment for the miss shader.

```
// Lux Ray Tracing Example: Simple Path Tracer

import brdf;

// Surface is UNCHANGED -- same BRDF works for both raster and RT
surface CopperMetal {
    brdf: pbr(vec3(0.95, 0.64, 0.54), 0.3, 0.9),
}

// New: environment for miss shader (sky gradient)
environment GradientSky {
    color: mix(vec3(1.0), vec3(0.5, 0.7, 1.0), 0.5),
}

// Pipeline with mode: raytrace
pipeline PathTracer {
    mode: raytrace,
    surface: CopperMetal,
    environment: GradientSky,
    max_bounces: 1,
}
```

This produces three SPIR-V modules:
- `rt_pathtracer.rgen.spv` -- ray generation shader with camera uniforms and acceleration structure
- `rt_pathtracer.rchit.spv` -- closest-hit shader with PBR BRDF evaluation
- `rt_pathtracer.rmiss.spv` -- miss shader with sky gradient

The same `surface CopperMetal` declaration can be used with both a rasterization pipeline and an RT pipeline without modification.

### 17.5 Layered PBR with IBL (Surface Layers)

A fully declarative PBR material using the layered surface syntax with image-based lighting, normal mapping, and multi-pipeline compilation.

```
// glTF PBR with IBL using layered surface syntax
// Compiles to both raster and RT pipelines from the same surface declaration

import brdf;

geometry StandardMesh {
    position: vec3,
    normal: vec3,
    uv: vec2,
    tangent: vec4,
    transform: MVP {
        model: mat4,
        view: mat4,
        projection: mat4,
    }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        frag_uv: uv,
        frag_tangent: tangent,
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

// Layered surface: base PBR + normal map + IBL
surface GltfPBR {
    sampler2d albedo_tex,
    sampler2d normal_tex,
    sampler2d metalrough_tex,
    sampler2d brdf_lut,
    samplerCube specular_map,
    samplerCube irradiance_map,
    layers [
        base(
            albedo: sample(albedo_tex, frag_uv).rgb,
            roughness: sample(metalrough_tex, frag_uv).g,
            metallic: sample(metalrough_tex, frag_uv).b,
        ),
        normal_map(map: normal_tex),
        ibl(
            specular_map: specular_map,
            irradiance_map: irradiance_map,
            brdf_lut: brdf_lut,
        ),
    ]
}

environment GradientSky {
    color: mix(vec3(1.0), vec3(0.5, 0.7, 1.0), 0.5),
}

// Raster pipeline
pipeline RasterForward {
    geometry: StandardMesh,
    surface: GltfPBR,
}

// RT pipeline -- same surface, different mode
pipeline RTPathTracer {
    mode: raytrace,
    surface: GltfPBR,
    environment: GradientSky,
    max_bounces: 1,
}
```

Compile only the raster pipeline:

```
luxc gltf_pbr.lux --pipeline RasterForward
```

Compile only the RT pipeline:

```
luxc gltf_pbr.lux --pipeline RTPathTracer
```

The same `surface GltfPBR` declaration works for both pipelines. For the RT pipeline, the compiler automatically rewrites `sample()` calls to `sample_lod()` with LOD 0.

---

## Grammar Reference (EBNF)

For completeness, the full formal grammar of Lux is reproduced here in a notation close to EBNF. This is derived from the Lark grammar file `lux.lark`.

```ebnf
start           = module_item* ;

module_item     = const_decl
                | function_def
                | stage_block
                | struct_def
                | type_alias
                | import_decl
                | features_decl
                | conditional_block
                | surface_decl
                | geometry_decl
                | pipeline_decl
                | schedule_decl
                | environment_decl
                | procedural_decl ;

(* -- Top-level declarations -- *)
const_decl      = "const" IDENT ":" type "=" expr ";" ;
struct_def      = "struct" IDENT "{" struct_field ("," struct_field)* ","? "}" ;
struct_field    = IDENT ":" type ;
type_alias      = "type" IDENT "=" type ";" ;
import_decl     = "import" IDENT ";" ;

(* -- Features -- *)
features_decl   = "features" "{" feature_field ("," feature_field)* ","? "}" ;
feature_field   = IDENT ":" "bool" ;
conditional_block = "if" feature_expr "{" module_item* "}" ;
feature_expr    = feature_or ;
feature_or      = feature_and ("||" feature_and)* ;
feature_and     = feature_not ("&&" feature_not)* ;
feature_not     = "!" feature_not | feature_primary ;
feature_primary = IDENT | "(" feature_expr ")" ;

(* -- Surface -- *)
surface_decl    = "surface" IDENT "{" surface_item ("," surface_item)* ","? "}" ;
surface_item    = surface_sampler | surface_member | surface_layers ;
surface_sampler = SAMPLER_KW IDENT ;
SAMPLER_KW      = "sampler2d" | "samplerCube" ;
surface_member  = IDENT ":" expr ;
surface_layers  = "layers" "[" layer_call ("," layer_call)* ","? "]" ;
layer_call      = IDENT "(" layer_arg ("," layer_arg)* ","? ")" ;
layer_arg       = IDENT ":" expr ;

(* -- Geometry -- *)
geometry_decl   = "geometry" IDENT "{" geometry_item* "}" ;
geometry_item   = geometry_field | geometry_transform | geometry_outputs ;
geometry_field  = IDENT ":" type "," ;
geometry_transform = "transform" ":" IDENT "{" block_field ("," block_field)* ","? "}" ;
geometry_outputs = "outputs" "{" output_binding ("," output_binding)* ","? "}" ;
output_binding  = IDENT ":" expr ;

(* -- Pipeline -- *)
pipeline_decl   = "pipeline" IDENT "{" pipeline_member ("," pipeline_member)* ","? "}" ;
pipeline_member = IDENT ":" expr ;

(* -- Schedule -- *)
schedule_decl   = "schedule" IDENT "{" schedule_member ("," schedule_member)* ","? "}" ;
schedule_member = IDENT ":" IDENT ;

(* -- Environment -- *)
environment_decl = "environment" IDENT "{" surface_item ("," surface_item)* ","? "}" ;

(* -- Procedural -- *)
procedural_decl = "procedural" IDENT "{" procedural_member ("," procedural_member)* ","? "}" ;
procedural_member = IDENT ":" expr ;

(* -- Stage blocks -- *)
stage_block     = STAGE_TYPE "{" stage_item* "}" ;
STAGE_TYPE      = "vertex" | "fragment" | "raygen" | "closest_hit"
                | "any_hit" | "miss" | "intersection" | "callable" ;

stage_item      = in_decl | out_decl | uniform_block | push_block
                | sampler_decl | function_def
                | ray_payload_decl | hit_attribute_decl
                | callable_data_decl | accel_decl ;

in_decl         = "in" IDENT ":" type ";" ;
out_decl        = "out" IDENT ":" type ";" ;
uniform_block   = "uniform" IDENT "{" block_field ("," block_field)* ","? "}" ;
push_block      = "push" IDENT "{" block_field ("," block_field)* ","? "}" ;
block_field     = IDENT ":" type ;
sampler_decl    = SAMPLER_KW IDENT ";" ;

ray_payload_decl     = "ray_payload" IDENT ":" type ";" ;
hit_attribute_decl   = "hit_attribute" IDENT ":" type ";" ;
callable_data_decl   = "callable_data" IDENT ":" type ";" ;
accel_decl           = "acceleration_structure" IDENT ";" ;

(* -- Functions -- *)
function_def    = attribute* "fn" IDENT "(" param_list? ")" ("->" type)? "{" statement* "}" ;
attribute       = "@" IDENT ;
param_list      = param ("," param)* ;
param           = IDENT ":" type ;

(* -- Statements -- *)
statement       = let_stmt | assign_stmt | return_stmt | if_stmt | expr_stmt ;
let_stmt        = "let" IDENT ":" type "=" expr ";" ;
assign_stmt     = assign_target "=" expr ";" ;
return_stmt     = "return" expr ";" ;
if_stmt         = "if" "(" expr ")" "{" statement* "}" ("else" "{" statement* "}")? ;
expr_stmt       = expr ";" ;

assign_target   = swizzle_access | field_access | index_access | IDENT ;

(* -- Types -- *)
type            = TYPE_NAME ;
TYPE_NAME       = "scalar" | "int" | "uint" | "bool" | "void"
                | "vec2" | "vec3" | "vec4"
                | "ivec2" | "ivec3" | "ivec4"
                | "uvec2" | "uvec3" | "uvec4"
                | "mat2" | "mat3" | "mat4"
                | "sampler2d" | "samplerCube" | "acceleration_structure"
                | IDENT (* user-defined type alias *) ;

(* -- Expressions (precedence climbing) -- *)
expr            = ternary_expr ;
ternary_expr    = or_expr ("?" expr ":" ternary_expr)? ;
or_expr         = and_expr ("||" and_expr)* ;
and_expr        = equality_expr ("&&" equality_expr)* ;
equality_expr   = comparison_expr (("==" | "!=") comparison_expr)* ;
comparison_expr = additive_expr (("<=" | ">=" | "<" | ">") additive_expr)* ;
additive_expr   = multiplicative_expr (("+" | "-") multiplicative_expr)* ;
multiplicative_expr = unary_expr (("*" | "/" | "%") unary_expr)* ;
unary_expr      = ("-" | "!") unary_expr | postfix_expr ;

postfix_expr    = primary
                | call_expr
                | constructor_expr
                | swizzle_access
                | field_access
                | index_access ;

call_expr       = postfix_expr "(" arg_list? ")" ;
constructor_expr = TYPE_CONSTRUCTOR "(" arg_list? ")" ;
swizzle_access  = postfix_expr "." SWIZZLE ;
field_access    = postfix_expr "." IDENT ;
index_access    = postfix_expr "[" expr "]" ;

TYPE_CONSTRUCTOR = "vec2" | "vec3" | "vec4"
                 | "ivec2" | "ivec3" | "ivec4"
                 | "uvec2" | "uvec3" | "uvec4"
                 | "mat2" | "mat3" | "mat4" ;

arg_list        = expr ("," expr)* ;

SWIZZLE         = /[xyzw]{1,4}/ | /[rgba]{1,4}/ ;

primary         = NUMBER | "true" | "false" | IDENT | "(" expr ")" ;

NUMBER          = /[0-9]+\.[0-9]*([eE][+-]?[0-9]+)?/
                | /[0-9]+[eE][+-]?[0-9]+/
                | /[0-9]*\.[0-9]+([eE][+-]?[0-9]+)?/
                | /[0-9]+/ ;

IDENT           = /[a-zA-Z_][a-zA-Z0-9_]*/ ;
```

---

*End of Specification*
