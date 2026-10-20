#!/usr/bin/env python
"""One-click reproduce: train the tuned model, then bring up the visualization stack.

Steps:
  1. environment check (project venv, data file)
  2. train  (ml/train.py --config ml/configs/tuned.json -> ml/registry/model.pkl, logged to MLflow)
  3. start MLflow UI   (http://127.0.0.1:5000)
  4. start FastAPI app (http://127.0.0.1:8000, serves the digit-drawing demo page at "/")
  5. open the browser on the demo page and the MLflow UI

Usage:
  python reproduce.py            # full flow
  python reproduce.py --stop     # stop the background services started before
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
VENV_DVC = os.path.join(ROOT, ".venv", "Scripts", "dvc.exe")
DATA_CSV = os.path.join(ROOT, "data", "digits.csv")
LOGS_DIR = os.path.join(ROOT, "logs")
PID_FILE = os.path.join(LOGS_DIR, "services.json")

MLFLOW_PORT = 5000
APP_PORT = 8000
MLFLOW_UI = f"http://127.0.0.1:{MLFLOW_PORT}"
APP_UI = f"http://127.0.0.1:{APP_PORT}"

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # MLflow 3.x: local file backend


# --------------------------------------------------------------------------- helpers
def say(msg):
    print(f"\n[mlops] {msg}")


def step(n, msg):
    print(f"\n=== 步骤 {n}：{msg} ===")


def run_sync(cmd, cwd=ROOT, env=None):
    """Run a command in the foreground and stream its output."""
    print("$", " ".join(cmd))
    env = dict(os.environ, **(env or {}))
    proc = subprocess.run(cmd, check=False, cwd=cwd, env=env)
    if proc.returncode != 0:
        print(f"[mlops] 命令失败，退出码 {proc.returncode}")
        sys.exit(1)


def port_busy(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_http(url, timeout=60, interval=1.0):
    """Wait until url returns HTTP 200 (or any response)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status < 500:
                    return True
        except OSError:
            time.sleep(interval)
    return False


def start_detached(cmd, log_name):
    """Start a long-running service detached from this process, keep its PID."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, log_name)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with open(log_path, "wb") as out:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
            creationflags=flags, env=dict(os.environ),
        )
    return proc.pid


def save_pids(pids):
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(PID_FILE, "w") as f:
        json.dump(pids, f, indent=2)


def load_pids():
    try:
        with open(PID_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------- steps
def ensure_data():
    if os.path.exists(DATA_CSV):
        size = os.path.getsize(DATA_CSV)
        say(f"数据已就绪：data/digits.csv（{size:,} 字节）")
        return
    say("data/digits.csv 缺失，尝试从 DVC 恢复…")
    if os.path.exists(VENV_DVC):
        run_sync([VENV_DVC, "pull", "-r", "myazure"])
    if not os.path.exists(DATA_CSV):
        say("DVC 恢复失败或未配置远程，改用 sklearn 重新导出（原始 0-16 版本）")
        run_sync([VENV_PY, "ml/data_pipeline.py"])
    if not os.path.exists(DATA_CSV):
        say("数据准备失败，请检查 data/digits.csv")
        sys.exit(1)


def train():
    say("开始训练 RandomForest（tuned: n_estimators=200, max_depth=12）…")
    run_sync([
        VENV_PY, "ml/train.py",
        "--config", "ml/configs/tuned.json",
        "--experiment", "digits-tuned",
        "--run-name", "tuned_rf",
    ])
    say("训练完成，模型已保存到 ml/registry/model.pkl 并记录到 MLflow")


def start_services():
    pids = {}
    if port_busy(MLFLOW_PORT):
        say(f"MLflow UI 已在 {MLFLOW_UI} 运行，跳过启动")
    else:
        say("启动 MLflow UI…")
        pids["mlflow"] = start_detached(
            [VENV_PY, "-m", "mlflow", "ui", "--host", "127.0.0.1",
             "--port", str(MLFLOW_PORT), "--backend-store-uri", os.path.join(ROOT, "mlruns")],
            "mlflow-ui.log",
        )
    if port_busy(APP_PORT):
        say(f"FastAPI 服务已在 {APP_UI} 运行，跳过启动")
    else:
        say("启动 FastAPI 服务…")
        pids["uvicorn"] = start_detached(
            [VENV_PY, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", str(APP_PORT)],
            "app.log",
        )
    if pids:
        save_pids({**load_pids(), **pids})


def wait_and_open():
    ok_app = wait_http(APP_UI + "/health", timeout=60)
    if ok_app:
        say(f"FastAPI 服务已就绪：{APP_UI}/health")
    else:
        say("等待 FastAPI 服务超时，请查看 logs/app.log")
    ok_mlflow = wait_http(MLFLOW_UI, timeout=30)
    if ok_mlflow:
        say(f"MLflow UI 已就绪：{MLFLOW_UI}")

    say("正在打开浏览器…")
    if ok_app:
        webbrowser.open(APP_UI + "/")
    if ok_mlflow:
        time.sleep(0.6)
        webbrowser.open(MLFLOW_UI)


def stop_services():
    pids = load_pids()
    if not pids:
        say("没有找到之前启动的服务记录（logs/services.json），无需停止")
        return
    for name, pid in pids.items():
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               check=False, capture_output=True, timeout=15)
            else:
                subprocess.run(["kill", str(pid)], check=False, capture_output=True, timeout=15)
            say(f"已停止 {name}（PID {pid}）")
        except (OSError, subprocess.TimeoutExpired):
            say(f"停止 {name}（PID {pid}）失败，可手动结束该进程")
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


# --------------------------------------------------------------------------- main
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--stop":
        stop_services()
        return

    if not os.path.exists(VENV_PY):
        print("[mlops] 未找到 .venv，请先执行：python -m venv .venv && pip install -r requirements.txt -r requirements-dev.txt")
        sys.exit(1)

    say("工程根目录：" + ROOT)
    step(1, "检查数据")
    ensure_data()
    step(2, "训练模型")
    train()
    step(3, "启动服务（MLflow UI + FastAPI）")
    start_services()
    step(4, "等待服务就绪并打开浏览器")
    wait_and_open()

    print("\n" + "=" * 60)
    print(" 全部完成，常用地址：")
    print(f"   可视化演示页 : {APP_UI}/   （手写数字识别）")
    print(f"   API 文档     : {APP_UI}/docs")
    print(f"   MLflow UI    : {MLFLOW_UI}")
    print(f"   服务日志     : {LOGS_DIR}")
    print(" 停止后台服务：python reproduce.py --stop")
    print("=" * 60)


if __name__ == "__main__":
    main()
