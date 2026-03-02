"""Capture a single GPU frame from Nadrin/PBR using RenderDoc Python API."""
import sys, os, time, shutil

sys.path.insert(0, 'D:/renderdoc/module')
import renderdoc as rd

# Parse args: capture_frame.py [original|lux]
variant = sys.argv[1] if len(sys.argv) > 1 else 'lux'

if variant == 'original':
    cap_path = 'D:/shaderlang/projects/nadrin-pbr/captures/original.rdc'
    exe = r'D:\shaderlang\projects\nadrin-pbr\upstream\build\Release\PBR_Original.exe'
    workdir = r'D:\shaderlang\projects\nadrin-pbr\upstream\data'
elif variant == 'lux':
    cap_path = 'D:/shaderlang/projects/nadrin-pbr/captures/lux_variant.rdc'
    exe = r'D:\shaderlang\projects\nadrin-pbr\lux-variant\build\Release\PBR_Lux.exe'
    workdir = r'D:\shaderlang\projects\nadrin-pbr\lux-variant\data'
else:
    print(f'Unknown variant: {variant}')
    sys.exit(1)

os.makedirs(os.path.dirname(cap_path), exist_ok=True)

opts = rd.CaptureOptions()
opts.apiValidation = False
opts.captureCallstacks = False
opts.refAllResources = False

env = []

print(f'Capturing {variant}: {exe}')
print(f'CWD: {workdir}')
result = rd.ExecuteAndInject(exe, workdir, '', env, cap_path, opts, False)
print(f'Ident: {result.ident}')

if result.ident == 0:
    print('Failed to inject')
    sys.exit(1)

print('Connecting to target...')
target = rd.CreateTargetControl('', result.ident, 'claude-code', True)

if target is None or not target.Connected():
    print('Failed to connect')
    sys.exit(1)

# Wait for app to initialize (Nadrin loads assets, processes env map)
print('Connected. Init pump (8s for compute shaders)...')
for _ in range(80):
    target.ReceiveMessage(None)
    time.sleep(0.1)

print('Triggering capture...')
target.TriggerCapture(1)

print('Waiting for capture...')
cap_received = False
for i in range(100):  # 10 seconds max
    msg = target.ReceiveMessage(None)
    t = msg.type

    if t == rd.TargetControlMessageType.NewCapture:
        cap = msg.newCapture
        print(f'  Capture received: path={cap.path}')
        if cap.local and cap.path and cap.path != cap_path:
            shutil.copy2(cap.path, cap_path)
        elif not cap.local:
            target.CopyCapture(cap.captureId, cap_path)
        cap_received = True
        break
    elif t == rd.TargetControlMessageType.CaptureProgress:
        print(f'  Progress: {msg.capProgress:.0%}')

    time.sleep(0.1)

if not cap_received:
    print('No capture after 10s')

target.Shutdown()

if os.path.exists(cap_path):
    print(f'OK: {cap_path} ({os.path.getsize(cap_path):,} bytes)')
else:
    print(f'FAILED: no capture file')
