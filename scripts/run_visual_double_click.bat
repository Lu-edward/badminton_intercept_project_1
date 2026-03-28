@echo off
chcp 65001 >nul
echo ========================================
echo   运行羽毛球拦截可视化
echo ========================================
echo.

cd /d "%~dp0"

powershell -ExecutionPolicy Bypass -File "%~dp0run_control_chain_visual.ps1" -NumEnvs 1 -Speed 0.2 %*

if errorlevel 1 (
    echo.
    echo [ERROR] 脚本执行失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo [INFO] 脚本执行完成
pause
