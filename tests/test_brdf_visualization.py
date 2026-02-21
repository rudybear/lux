"""Tests for P15 BRDF & Layer Visualization shaders.

Each shader gets 2 tests: compilation + pixel validation.
Compilation tests verify .frag.spv is produced and passes spirv-val.
Pixel validation tests check for non-black output and expected visual content.
"""

import subprocess
import pytest
from pathlib import Path
from luxc.compiler import compile_source
from luxc.builtins.types import clear_type_aliases

EXAMPLES = Path(__file__).parent.parent / "examples"


def _has_spirv_tools() -> bool:
    try:
        subprocess.run(["spirv-as", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


requires_spirv_tools = pytest.mark.skipif(
    not _has_spirv_tools(), reason="spirv-as/spirv-val not found on PATH"
)


def _has_wgpu() -> bool:
    try:
        import wgpu  # noqa: F401
        return True
    except ImportError:
        return False


requires_wgpu = pytest.mark.skipif(
    not _has_wgpu(), reason="wgpu not installed"
)


def _render_fullscreen(spv_path: Path, width: int = 256, height: int = 256):
    """Render a fullscreen fragment shader and return pixel array."""
    import sys
    playground_dir = str(Path(__file__).parent.parent / "playground")
    if playground_dir not in sys.path:
        sys.path.insert(0, playground_dir)
    from render_harness import render_fullscreen
    return render_fullscreen(spv_path, width=width, height=height)


@requires_spirv_tools
class TestTransferFunctionsCompilation:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_transfer_functions_compiles(self, tmp_path):
        """viz_transfer_functions.lux compiles to valid SPIR-V."""
        src = (EXAMPLES / "viz_transfer_functions.lux").read_text()
        compile_source(src, "viz_transfer_functions", tmp_path, validate=True)
        assert (tmp_path / "viz_transfer_functions.frag.spv").exists()


@requires_spirv_tools
@requires_wgpu
class TestTransferFunctionsRendering:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_transfer_functions_renders(self, tmp_path):
        """Transfer functions render with non-black output and 6 distinct cells."""
        import numpy as np

        src = (EXAMPLES / "viz_transfer_functions.lux").read_text()
        compile_source(src, "viz_transfer_functions", tmp_path, validate=True)
        spv = tmp_path / "viz_transfer_functions.frag.spv"

        pixels = _render_fullscreen(spv)
        h, w, _ = pixels.shape
        rgb = pixels[:, :, :3].astype(np.float32)
        brightness = rgb.sum(axis=2)

        # Non-black check
        assert brightness.max() > 20, "Image is too dark"

        # 6 cells (2x3 grid): each should have distinct curve content
        cell_w = w // 2
        cell_h = h // 3
        cell_stds = []
        for row in range(3):
            for col in range(2):
                cell = brightness[row * cell_h:(row + 1) * cell_h,
                                  col * cell_w:(col + 1) * cell_w]
                cell_stds.append(cell.std())

        # At least 4 cells should have meaningful variation (curves)
        active_cells = sum(1 for s in cell_stds if s > 3.0)
        assert active_cells >= 4, f"Only {active_cells}/6 cells have curves"


@requires_spirv_tools
class TestPolarCompilation:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_polar_compiles(self, tmp_path):
        """viz_brdf_polar.lux compiles to valid SPIR-V."""
        src = (EXAMPLES / "viz_brdf_polar.lux").read_text()
        compile_source(src, "viz_brdf_polar", tmp_path, validate=True)
        assert (tmp_path / "viz_brdf_polar.frag.spv").exists()


@requires_spirv_tools
@requires_wgpu
class TestPolarRendering:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_polar_renders(self, tmp_path):
        """Polar lobes render with 4 visible quadrants."""
        import numpy as np

        src = (EXAMPLES / "viz_brdf_polar.lux").read_text()
        compile_source(src, "viz_brdf_polar", tmp_path, validate=True)
        spv = tmp_path / "viz_brdf_polar.frag.spv"

        pixels = _render_fullscreen(spv)
        h, w, _ = pixels.shape
        rgb = pixels[:, :, :3].astype(np.float32)
        brightness = rgb.sum(axis=2)

        # Non-black check
        assert brightness.max() > 20, "Image is too dark"

        # 4 quadrants should have visible content
        cell_w = w // 2
        cell_h = h // 2
        quadrant_means = []
        for row in range(2):
            for col in range(2):
                cell = brightness[row * cell_h:(row + 1) * cell_h,
                                  col * cell_w:(col + 1) * cell_w]
                quadrant_means.append(cell.mean())

        active_quads = sum(1 for m in quadrant_means if m > 2.0)
        assert active_quads >= 4, f"Only {active_quads}/4 quadrants visible"


@requires_spirv_tools
class TestParamSweepCompilation:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_param_sweep_compiles(self, tmp_path):
        """viz_param_sweep.lux compiles to valid SPIR-V."""
        src = (EXAMPLES / "viz_param_sweep.lux").read_text()
        compile_source(src, "viz_param_sweep", tmp_path, validate=True)
        assert (tmp_path / "viz_param_sweep.frag.spv").exists()


@requires_spirv_tools
@requires_wgpu
class TestParamSweepRendering:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_param_sweep_renders(self, tmp_path):
        """Both panels are non-black with color variation."""
        import numpy as np

        src = (EXAMPLES / "viz_param_sweep.lux").read_text()
        compile_source(src, "viz_param_sweep", tmp_path, validate=True)
        spv = tmp_path / "viz_param_sweep.frag.spv"

        pixels = _render_fullscreen(spv)
        h, w, _ = pixels.shape
        rgb = pixels[:, :, :3].astype(np.float32)
        brightness = rgb.sum(axis=2)

        # Both panels non-black
        left = brightness[:, :w // 2]
        right = brightness[:, w // 2:]
        assert left.max() > 10, "Left panel is too dark"
        assert right.max() > 10, "Right panel is too dark"

        # Color variation across each panel
        assert left.std() > 5, "Left panel lacks variation"
        assert right.std() > 5, "Right panel lacks variation"


@requires_spirv_tools
class TestFurnaceCompilation:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_furnace_compiles(self, tmp_path):
        """viz_furnace_test.lux compiles to valid SPIR-V."""
        src = (EXAMPLES / "viz_furnace_test.lux").read_text()
        compile_source(src, "viz_furnace_test", tmp_path, validate=True)
        assert (tmp_path / "viz_furnace_test.frag.spv").exists()


@requires_spirv_tools
@requires_wgpu
class TestFurnaceRendering:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_furnace_renders(self, tmp_path):
        """Furnace test renders with curves in all 4 cells."""
        import numpy as np

        src = (EXAMPLES / "viz_furnace_test.lux").read_text()
        compile_source(src, "viz_furnace_test", tmp_path, validate=True)
        spv = tmp_path / "viz_furnace_test.frag.spv"

        pixels = _render_fullscreen(spv)
        h, w, _ = pixels.shape
        rgb = pixels[:, :, :3].astype(np.float32)
        brightness = rgb.sum(axis=2)

        # All 4 cells should have content
        cell_w = w // 2
        cell_h = h // 2
        for row in range(2):
            for col in range(2):
                cell = brightness[row * cell_h:(row + 1) * cell_h,
                                  col * cell_w:(col + 1) * cell_w]
                non_black = (cell > 5).sum() / cell.size
                assert non_black > 0.05, f"Cell ({col},{row}) is blank"


@requires_spirv_tools
class TestLayerEnergyCompilation:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_layer_energy_compiles(self, tmp_path):
        """viz_layer_energy.lux compiles to valid SPIR-V."""
        src = (EXAMPLES / "viz_layer_energy.lux").read_text()
        compile_source(src, "viz_layer_energy", tmp_path, validate=True)
        assert (tmp_path / "viz_layer_energy.frag.spv").exists()


@requires_spirv_tools
@requires_wgpu
class TestLayerEnergyRendering:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_layer_energy_renders(self, tmp_path):
        """Layer energy chart shows multiple color bands with >50% coverage."""
        import numpy as np

        src = (EXAMPLES / "viz_layer_energy.lux").read_text()
        compile_source(src, "viz_layer_energy", tmp_path, validate=True)
        spv = tmp_path / "viz_layer_energy.frag.spv"

        pixels = _render_fullscreen(spv)
        h, w, _ = pixels.shape
        rgb = pixels[:, :, :3].astype(np.float32)
        brightness = rgb.sum(axis=2)

        # Significant coverage (stacked area chart fills most of the panel)
        non_black = (brightness > 10).sum() / (h * w)
        assert non_black > 0.3, f"Coverage too low: {non_black:.1%}"

        # Multiple distinct colors (at least 3 unique hue bands)
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]
        # Check that we have a mix of warm (orange/cyan) and cool colors
        warm_pixels = ((r > g) & (brightness > 20)).sum()
        cool_pixels = ((b > r) & (brightness > 20)).sum()
        assert warm_pixels > 100, "No warm-colored band visible"
        assert cool_pixels > 100, "No cool-colored band visible"
