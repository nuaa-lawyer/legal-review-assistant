@echo off
cd /d "%~dp0"
title 法审助手 - 本地演示
echo.
echo ================================================
echo    法审助手 - 本地演示启动器
echo ================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [无法启动] 没有找到 .venv 运行环境。
  echo.
  echo 请先在项目目录下依次执行这两条命令：
  echo     python -m venv .venv
  echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

if not exist ".env" (
  echo [无法启动] 没有找到 .env 配置文件（里面是 API Key）。
  echo 请找项目负责人索取 .env 并放在项目根目录。
  echo.
  pause
  exit /b 1
)

echo 正在启动，浏览器会自动打开 localhost:8501
echo 关闭本窗口即可停止服务。
echo.
".venv\Scripts\python.exe" -m streamlit run app.py 
pause
