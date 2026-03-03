@echo off
REM ============================================================
REM  Lux Shader Debugger — Interactive Playground
REM ============================================================
REM
REM  This script launches the CPU-side shader debugger on
REM  examples/debug_playground.lux — a PBR fragment shader
REM  with intentional edge cases for debugging practice.
REM
REM  Usage:
REM    debug_playground.bat              — interactive REPL
REM    debug_playground.bat batch        — batch mode, show output
REM    debug_playground.bat nan          — detect NaN/Inf sources
REM    debug_playground.bat trace        — trace all variable values
REM    debug_playground.bat break        — breakpoints at BRDF stages
REM    debug_playground.bat grazing      — simulate grazing angle pixel
REM    debug_playground.bat zero         — test roughness=0 edge case
REM    debug_playground.bat custom       — use your own inputs.json
REM    debug_playground.bat help         — show this help
REM
REM ============================================================

setlocal

set SHADER=examples/debug_playground.lux
set STAGE=fragment

if "%1"=="" goto interactive
if "%1"=="help" goto help
if "%1"=="batch" goto batch
if "%1"=="nan" goto nan
if "%1"=="trace" goto trace
if "%1"=="break" goto breakpoints
if "%1"=="grazing" goto grazing
if "%1"=="zero" goto zero
if "%1"=="custom" goto custom
goto help

:interactive
echo.
echo  === Lux Shader Debugger — Interactive Mode ===
echo.
echo  Commands:  start, step, next, continue, run, break ^<line^>,
echo             print ^<var^>, locals, source, output, quit
echo.
echo  Try this:
echo    start           (stop at first statement)
echo    step            (advance one statement)
echo    step            (advance again)
echo    print albedo    (inspect a variable)
echo    locals          (see all variables)
echo    break 113       (set breakpoint at BRDF)
echo    continue        (run to breakpoint)
echo    continue        (run to completion)
echo.
python -m luxc %SHADER% --debug-run --stage %STAGE%
goto end

:batch
echo.
echo  === Batch Mode — Run and show output ===
echo.
python -m luxc %SHADER% --debug-run --stage %STAGE% --batch
goto end

:nan
echo.
echo  === NaN Detection — Find where bad values originate ===
echo.
python -m luxc %SHADER% --debug-run --stage %STAGE% --batch --check-nan
goto end

:trace
echo.
echo  === Variable Trace — Every intermediate value ===
echo.
python -m luxc %SHADER% --debug-run --stage %STAGE% --batch --dump-vars
goto end

:breakpoints
echo.
echo  === Breakpoint Inspection — BRDF stages ===
echo.
echo  Line 100: after material setup (f0, diffuse_color)
echo  Line 113: after BRDF denominator (specular term)
echo  Line 135: before tonemap (HDR color)
echo.
python -m luxc %SHADER% --debug-run --stage %STAGE% --batch --break 100 --break 113 --break 135 --dump-at-break
goto end

:grazing
echo.
echo  === Grazing Angle — Metallic surface at steep angle ===
echo.
echo  Inputs: normal=(0,1,0), position=(10,0,0), roughness=0.1, metallic=0.9, exposure=2.0
echo.
python -m luxc %SHADER% --debug-run --stage %STAGE% --input examples/debug_grazing_angle.json --batch --check-nan --dump-vars
goto end

:zero
echo.
echo  === Zero Roughness — Perfect mirror edge case ===
echo.
echo  Inputs: roughness=0.0, metallic=0.0
echo.
python -m luxc %SHADER% --debug-run --stage %STAGE% --input examples/debug_zero_roughness.json --batch --check-nan --dump-vars
goto end

:custom
echo.
echo  === Custom Inputs — Edit and replay ===
echo.
if not exist "inputs.json" (
    echo  No inputs.json found. Exporting defaults...
    python -m luxc %SHADER% --debug-run --export-inputs inputs.json
    echo.
    echo  Created inputs.json — edit it with your values, then run:
    echo    debug_playground.bat custom
    echo.
) else (
    echo  Using inputs.json
    echo.
    python -m luxc %SHADER% --debug-run --stage %STAGE% --input inputs.json --batch --check-nan --dump-vars
)
goto end

:help
echo.
echo  Lux Shader Debugger — Interactive Playground
echo  =============================================
echo.
echo  Usage:
echo    debug_playground.bat              Interactive REPL (gdb-style)
echo    debug_playground.bat batch        Run and show output color
echo    debug_playground.bat nan          Detect NaN/Inf with source line
echo    debug_playground.bat trace        Trace all variable assignments
echo    debug_playground.bat break        Breakpoints at BRDF stages
echo    debug_playground.bat grazing      Metallic surface at grazing angle
echo    debug_playground.bat zero         Test roughness=0 edge case
echo    debug_playground.bat custom       Use/create inputs.json
echo    debug_playground.bat help         Show this help
echo.
echo  The shader: examples/debug_playground.lux
echo  A PBR fragment shader with 8 labeled stages and an intentional
echo  NaN trap for debugging practice.
echo.
echo  Full docs: docs/debugger-guide.md
echo  Session transcript: docs/debug-session-transcript.md
echo.
goto end

:end
endlocal
