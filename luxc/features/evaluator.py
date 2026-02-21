"""Feature expression evaluation and AST stripping.

Evaluates compile-time feature expressions and removes AST items
whose feature conditions are not met. Runs after parsing but before
import resolution and surface expansion.
"""

from __future__ import annotations
from luxc.parser.ast_nodes import (
    Module, FeaturesDecl, ConditionalBlock,
    FeatureRef, FeatureAnd, FeatureOr, FeatureNot,
    SurfaceDecl, GeometryDecl, PipelineDecl, ScheduleDecl,
    EnvironmentDecl, ProceduralDecl,
    ConstDecl, FunctionDef, StructDef, TypeAlias, ImportDecl,
    StageBlock,
)


def evaluate_feature_expr(expr: object, active: set[str]) -> bool:
    """Evaluate a feature expression against a set of active features.

    Args:
        expr: A FeatureRef, FeatureAnd, FeatureOr, or FeatureNot node.
        active: Set of feature names that are enabled.

    Returns:
        True if the condition is satisfied.
    """
    if isinstance(expr, FeatureRef):
        return expr.name in active
    elif isinstance(expr, FeatureAnd):
        return evaluate_feature_expr(expr.left, active) and evaluate_feature_expr(expr.right, active)
    elif isinstance(expr, FeatureOr):
        return evaluate_feature_expr(expr.left, active) or evaluate_feature_expr(expr.right, active)
    elif isinstance(expr, FeatureNot):
        return not evaluate_feature_expr(expr.operand, active)
    else:
        raise TypeError(f"Unknown feature expression type: {type(expr)}")


def collect_feature_names(module: Module) -> list[str]:
    """Collect all declared feature names from the module's features blocks.

    Returns a sorted list of unique feature names.
    """
    names = []
    seen = set()
    for fd in module.features_decls:
        for name in fd.features:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def strip_features(module: Module, active: set[str]) -> None:
    """Remove all items whose feature condition evaluates to false.

    Mutates the module in place:
    - Removes conditional samplers, layers, fields, outputs, members
      whose condition evaluates to false
    - Inlines conditional_block contents when condition is true
    - Removes conditional_blocks when condition is false
    - Clears all condition fields so downstream code sees a clean AST

    Args:
        module: The parsed module to strip.
        active: Set of feature names that are enabled.
    """
    # 1. Process module-level conditional blocks (stored in features_decls
    #    handling — they arrive as items in the start() transformer)
    #    ConditionalBlocks are stored as items during parsing and need
    #    to be inlined or removed. They are dispatched in the tree builder's
    #    start() method into a temporary holding list.
    #    We process them from module._conditional_blocks if present.
    _process_conditional_blocks(module, active)

    # 2. Strip conditional items from surfaces
    for surface in module.surfaces:
        # Strip samplers
        surface.samplers = [
            s for s in surface.samplers
            if s.condition is None or evaluate_feature_expr(s.condition, active)
        ]
        for s in surface.samplers:
            s.condition = None

        # Strip layers
        if surface.layers is not None:
            surface.layers = [
                layer for layer in surface.layers
                if layer.condition is None or evaluate_feature_expr(layer.condition, active)
            ]
            for layer in surface.layers:
                layer.condition = None

    # 3. Strip conditional items from geometries
    for geo in module.geometries:
        # Strip fields
        geo.fields = [
            f for f in geo.fields
            if f.condition is None or evaluate_feature_expr(f.condition, active)
        ]
        for f in geo.fields:
            f.condition = None

        # Strip output bindings
        if geo.outputs is not None:
            geo.outputs.bindings = [
                b for b in geo.outputs.bindings
                if b.condition is None or evaluate_feature_expr(b.condition, active)
            ]
            for b in geo.outputs.bindings:
                b.condition = None

    # 4. Strip conditional items from schedules
    for sched in module.schedules:
        sched.members = [
            m for m in sched.members
            if m.condition is None or evaluate_feature_expr(m.condition, active)
        ]
        for m in sched.members:
            m.condition = None

    # 5. Strip conditional items from pipelines
    for pipeline in module.pipelines:
        pipeline.members = [
            m for m in pipeline.members
            if m.condition is None or evaluate_feature_expr(m.condition, active)
        ]
        for m in pipeline.members:
            m.condition = None


def _process_conditional_blocks(module: Module, active: set[str]) -> None:
    """Process ConditionalBlock items stored during parsing.

    ConditionalBlocks whose condition is true have their contents
    inlined into the module. Those whose condition is false are discarded.
    """
    cond_blocks = getattr(module, '_conditional_blocks', [])
    if not cond_blocks:
        return

    for block in cond_blocks:
        if evaluate_feature_expr(block.condition, active):
            # Inline the block's items into the module
            for item in block.items:
                if isinstance(item, ConstDecl):
                    module.constants.append(item)
                elif isinstance(item, FunctionDef):
                    module.functions.append(item)
                elif isinstance(item, StructDef):
                    module.structs.append(item)
                elif isinstance(item, StageBlock):
                    module.stages.append(item)
                elif isinstance(item, TypeAlias):
                    module.type_aliases.append(item)
                elif isinstance(item, ImportDecl):
                    module.imports.append(item)
                elif isinstance(item, SurfaceDecl):
                    module.surfaces.append(item)
                elif isinstance(item, GeometryDecl):
                    module.geometries.append(item)
                elif isinstance(item, PipelineDecl):
                    module.pipelines.append(item)
                elif isinstance(item, ScheduleDecl):
                    module.schedules.append(item)
                elif isinstance(item, EnvironmentDecl):
                    module.environments.append(item)
                elif isinstance(item, ProceduralDecl):
                    module.procedurals.append(item)
                elif isinstance(item, FeaturesDecl):
                    module.features_decls.append(item)

    # Clear the conditional blocks list
    module._conditional_blocks = []
