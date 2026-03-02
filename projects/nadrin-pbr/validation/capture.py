"""Capture screenshots from PBR renderers for visual comparison.

Usage:
    python capture.py original   # Capture reference from original renderer
    python capture.py lux        # Capture from lux-variant renderer
    python capture.py compare    # Compare captures (PSNR/SSIM)
"""

import subprocess
import sys
import time
from pathlib import Path

try:
    from PIL import ImageGrab, Image
    import numpy as np
except ImportError:
    print("Requires: pip install Pillow numpy")
    sys.exit(1)

BASE_DIR = Path(__file__).parent.parent
ORIGINAL_DIR = BASE_DIR / "upstream" / "data"
LUX_DIR = BASE_DIR / "lux-variant" / "data"
REF_DIR = Path(__file__).parent / "reference_images"
REPORTS_DIR = Path(__file__).parent / "reports"


def find_window_by_title(title_substring):
    """Find a window by partial title match (Windows-only)."""
    try:
        import ctypes
        from ctypes import wintypes

        EnumWindows = ctypes.windll.user32.EnumWindows
        GetWindowTextW = ctypes.windll.user32.GetWindowTextW
        GetWindowTextLengthW = ctypes.windll.user32.GetWindowTextLengthW
        GetWindowRect = ctypes.windll.user32.GetWindowRect
        IsWindowVisible = ctypes.windll.user32.IsWindowVisible

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        result = []

        def enum_callback(hwnd, lparam):
            if IsWindowVisible(hwnd):
                length = GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    GetWindowTextW(hwnd, buf, length + 1)
                    if title_substring.lower() in buf.value.lower():
                        rect = wintypes.RECT()
                        GetWindowRect(hwnd, ctypes.byref(rect))
                        result.append((hwnd, buf.value, (rect.left, rect.top, rect.right, rect.bottom)))
            return True

        EnumWindows(WNDENUMPROC(enum_callback), 0)
        return result
    except Exception as e:
        print(f"Window search error: {e}")
        return []


def capture_window_printwindow(hwnd):
    """Capture window using PrintWindow API (works even if occluded)."""
    import ctypes
    from ctypes import wintypes
    import ctypes.wintypes

    # Get client rect (rendering area, excludes title bar and borders)
    client_rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(client_rect))
    w = client_rect.right - client_rect.left
    h = client_rect.bottom - client_rect.top

    if w <= 0 or h <= 0:
        return None

    # Create device context and bitmap
    hwndDC = ctypes.windll.user32.GetDC(hwnd)
    memDC = ctypes.windll.gdi32.CreateCompatibleDC(hwndDC)
    bitmap = ctypes.windll.gdi32.CreateCompatibleBitmap(hwndDC, w, h)
    ctypes.windll.gdi32.SelectObject(memDC, bitmap)

    # PrintWindow with PW_CLIENTONLY flag (2) to capture just client area
    ctypes.windll.user32.PrintWindow(hwnd, memDC, 2)

    # Read bitmap data
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32),
            ("biWidth", ctypes.c_long),
            ("biHeight", ctypes.c_long),
            ("biPlanes", ctypes.c_uint16),
            ("biBitCount", ctypes.c_uint16),
            ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32),
            ("biXPelsPerMeter", ctypes.c_long),
            ("biYPelsPerMeter", ctypes.c_long),
            ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h  # Top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0  # BI_RGB

    buf = ctypes.create_string_buffer(w * h * 4)
    ctypes.windll.gdi32.GetDIBits(memDC, bitmap, 0, h, buf, ctypes.byref(bmi), 0)

    # Cleanup
    ctypes.windll.gdi32.DeleteObject(bitmap)
    ctypes.windll.gdi32.DeleteDC(memDC)
    ctypes.windll.user32.ReleaseDC(hwnd, hwndDC)

    # Convert BGRA to RGB PIL image
    img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1)
    return img.convert("RGB")


def capture_window(title_substring, output_path, wait_seconds=10):
    """Capture a screenshot of a specific window."""
    print(f"Waiting {wait_seconds}s for rendering to stabilize...")
    time.sleep(wait_seconds)

    windows = find_window_by_title(title_substring)
    if not windows:
        print(f"Warning: No window with '{title_substring}' found, using full screen")
        img = ImageGrab.grab()
    else:
        hwnd, title, rect = windows[0]
        print(f"Found window: '{title}' at {rect}")

        # Try PrintWindow API first (captures without occlusion)
        img = capture_window_printwindow(hwnd)
        if img is None:
            # Fallback to screen capture
            import ctypes
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            time.sleep(0.5)
            left, top, right, bottom = rect
            img = ImageGrab.grab(bbox=(left + 8, top + 31, right - 8, bottom - 8))

    img.save(str(output_path))
    print(f"Saved screenshot: {output_path} ({img.size[0]}x{img.size[1]})")
    return img


def run_and_capture(exe_dir, exe_name, output_name, window_title="Physically Based Rendering"):
    """Run a PBR executable and capture a screenshot."""
    exe_path = exe_dir / exe_name
    if not exe_path.exists():
        print(f"Error: {exe_path} not found")
        sys.exit(1)

    print(f"Launching {exe_name} from {exe_dir}...")
    proc = subprocess.Popen(
        [str(exe_path), "-vulkan"],
        cwd=str(exe_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for initialization
        output_path = REF_DIR / output_name
        img = capture_window(window_title, output_path, wait_seconds=12)
        return img
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Process terminated.")


def compare_images(ref_path, test_path):
    """Compare two images using PSNR and SSIM."""
    ref = np.array(Image.open(ref_path).convert("RGB"), dtype=np.float64)
    test = np.array(Image.open(test_path).convert("RGB"), dtype=np.float64)

    # Resize if dimensions differ
    if ref.shape != test.shape:
        print(f"Warning: Image dimensions differ ({ref.shape} vs {test.shape}), resizing test to match ref")
        test_img = Image.open(test_path).convert("RGB").resize(
            (ref.shape[1], ref.shape[0]), Image.LANCZOS
        )
        test = np.array(test_img, dtype=np.float64)

    # PSNR
    mse = np.mean((ref - test) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255.0 ** 2 / mse)

    # Simple structural similarity (mean-based)
    # Using a simplified version since we may not have scipy
    mu_ref = np.mean(ref)
    mu_test = np.mean(test)
    sigma_ref = np.std(ref)
    sigma_test = np.std(test)
    sigma_cross = np.mean((ref - mu_ref) * (test - mu_test))

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ssim = ((2 * mu_ref * mu_test + C1) * (2 * sigma_cross + C2)) / \
           ((mu_ref ** 2 + mu_test ** 2 + C1) * (sigma_ref ** 2 + sigma_test ** 2 + C2))

    # Generate diff heatmap
    diff = np.abs(ref - test)
    diff_normalized = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    diff_img = Image.fromarray(diff_normalized)
    diff_path = REPORTS_DIR / "diff_heatmap.png"
    diff_img.save(str(diff_path))

    return psnr, ssim, diff_path


def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "original":
        run_and_capture(ORIGINAL_DIR, "PBR_Original.exe", "reference.png")

    elif mode == "lux":
        run_and_capture(LUX_DIR, "PBR_Lux.exe", "lux_variant.png")

    elif mode == "compare":
        ref = REF_DIR / "reference.png"
        test = REF_DIR / "lux_variant.png"
        if not ref.exists() or not test.exists():
            print("Error: Need both reference.png and lux_variant.png")
            print("Run 'python capture.py original' and 'python capture.py lux' first")
            sys.exit(1)

        psnr, ssim, diff_path = compare_images(ref, test)

        print(f"\n{'='*50}")
        print(f"Visual Parity Results")
        print(f"{'='*50}")
        print(f"PSNR:  {psnr:.2f} dB  (target: > 35 dB)")
        print(f"SSIM:  {ssim:.4f}     (target: > 0.97)")
        print(f"Diff:  {diff_path}")
        print(f"{'='*50}")

        psnr_pass = psnr > 35
        ssim_pass = ssim > 0.97
        overall = "PASS" if (psnr_pass and ssim_pass) else "FAIL"

        print(f"PSNR:  {'PASS' if psnr_pass else 'FAIL'}")
        print(f"SSIM:  {'PASS' if ssim_pass else 'FAIL'}")
        print(f"Overall: {overall}")

        # Write report
        report = {
            "psnr": psnr,
            "ssim": ssim,
            "psnr_pass": psnr_pass,
            "ssim_pass": ssim_pass,
            "overall": overall,
        }
        import json
        report_path = REPORTS_DIR / "comparison_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved: {report_path}")
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
