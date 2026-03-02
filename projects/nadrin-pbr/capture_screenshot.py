"""Capture screenshots from both original and lux-variant apps."""
import subprocess
import time
import ctypes
import ctypes.wintypes
from PIL import Image
import sys
import os

def find_window(title_substring, timeout=10):
    """Find window by partial title match."""
    EnumWindows = ctypes.windll.user32.EnumWindows
    GetWindowTextW = ctypes.windll.user32.GetWindowTextW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible
    
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    result = [None]
    
    def callback(hwnd, lparam):
        if IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            GetWindowTextW(hwnd, buf, 256)
            if title_substring.lower() in buf.value.lower():
                result[0] = hwnd
                return False
        return True
    
    start = time.time()
    while time.time() - start < timeout:
        result[0] = None
        EnumWindows(WNDENUMPROC(callback), 0)
        if result[0]:
            return result[0]
        time.sleep(0.5)
    return None

def capture_window(hwnd, output_path):
    """Capture window client area to PNG."""
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    
    rect = ctypes.wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    
    if w <= 0 or h <= 0:
        print(f"  Window client area is {w}x{h}, skipping")
        return False
    
    hdc = user32.GetDC(hwnd)
    memdc = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(memdc, bmp)
    
    PW_CLIENTONLY = 1
    PW_RENDERFULLCONTENT = 2
    user32.PrintWindow(hwnd, memdc, PW_CLIENTONLY | PW_RENDERFULLCONTENT)
    
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ('biSize', ctypes.c_uint32),
            ('biWidth', ctypes.c_int32),
            ('biHeight', ctypes.c_int32),
            ('biPlanes', ctypes.c_uint16),
            ('biBitCount', ctypes.c_uint16),
            ('biCompression', ctypes.c_uint32),
            ('biSizeImage', ctypes.c_uint32),
            ('biXPelsPerMeter', ctypes.c_int32),
            ('biYPelsPerMeter', ctypes.c_int32),
            ('biClrUsed', ctypes.c_uint32),
            ('biClrImportant', ctypes.c_uint32),
        ]
    
    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0
    
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(memdc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    
    img = Image.frombytes('RGBA', (w, h), buf.raw, 'raw', 'BGRA')
    img = img.convert('RGB')
    img.save(output_path)
    
    # Quick stats
    import numpy as np
    arr = np.array(img)
    print(f"  Saved {output_path} ({w}x{h})")
    print(f"  Mean RGB: ({arr[:,:,0].mean():.1f}, {arr[:,:,1].mean():.1f}, {arr[:,:,2].mean():.1f})")
    print(f"  Min: {arr.min()}, Max: {arr.max()}")
    white_pct = (arr == 255).all(axis=2).mean() * 100
    black_pct = (arr == 0).all(axis=2).mean() * 100
    print(f"  White pixels: {white_pct:.1f}%, Black pixels: {black_pct:.1f}%")
    
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(memdc)
    user32.ReleaseDC(hwnd, hdc)
    return True

def run_and_capture(exe_path, working_dir, output_path):
    """Launch app, wait for rendering, capture screenshot, kill."""
    print(f"\nLaunching: {os.path.basename(exe_path)}")
    
    proc = subprocess.Popen(
        [exe_path],
        cwd=working_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(5)
    
    hwnd = find_window("Physically Based Rendering")
    if not hwnd:
        print("  ERROR: Could not find window!")
        proc.kill()
        proc.wait()
        return False
    
    print(f"  Found window: hwnd=0x{hwnd:X}")
    time.sleep(2)
    
    success = capture_window(hwnd, output_path)
    
    proc.kill()
    proc.wait()
    return success

if __name__ == "__main__":
    captures_dir = "D:/shaderlang/projects/nadrin-pbr/captures"
    os.makedirs(captures_dir, exist_ok=True)
    
    print("=== Capturing reference from ORIGINAL ===")
    run_and_capture(
        "D:/shaderlang/projects/nadrin-pbr/upstream/build/Release/PBR_Original.exe",
        "D:/shaderlang/projects/nadrin-pbr/upstream/data",
        f"{captures_dir}/reference_original.png"
    )
    
    time.sleep(3)
    
    print("\n=== Capturing from LUX-VARIANT ===")
    run_and_capture(
        "D:/shaderlang/projects/nadrin-pbr/lux-variant/build/Release/PBR_Lux.exe",
        "D:/shaderlang/projects/nadrin-pbr/lux-variant/data",
        f"{captures_dir}/reference_lux.png"
    )
