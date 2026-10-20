"""FastAPI app: handwritten digit recognition API."""
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = "ml/registry/model.pkl"

app = FastAPI(title="Digit Recognition API")


class PredictRequest(BaseModel):
    pixels: list[float] = Field(..., min_length=64, max_length=64)


def _load_model():
    return joblib.load(MODEL_PATH)


model = _load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if any(p < 0 or p > 16 for p in req.pixels):
        raise HTTPException(status_code=422, detail="pixels must be in [0, 16]")
    # 模型在 v2 全局 min-max 归一化数据（[0,1]）上训练，输入必须同口径
    X = np.array(req.pixels).reshape(1, -1) / 16.0
    proba = model.predict_proba(X)[0]
    pred = int(np.argmax(proba))
    return {"prediction": pred, "confidence": round(float(proba[pred]), 4)}


DIGIT_DEMO_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>手写数字识别 · 可视化演示</title>
<style>
  :root { --ink:#1f2430; --muted:#6b7280; --accent:#0f766e; --accent2:#ea580c; --bg:#f4f6f8; --card:#ffffff; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; background:var(--bg); color:var(--ink); }
  .wrap { max-width:880px; margin:0 auto; padding:32px 20px 48px; }
  header { text-align:center; margin-bottom:24px; }
  header h1 { font-size:26px; margin:0 0 6px; }
  header p { color:var(--muted); margin:0; font-size:14px; }
  .grid { display:grid; grid-template-columns: 280px 1fr; gap:28px; align-items:start; }
  @media (max-width:720px) { .grid { grid-template-columns:1fr; } }
  .card { background:var(--card); border-radius:14px; box-shadow:0 1px 4px rgba(15,23,42,.08); padding:20px; }
  .card h2 { font-size:15px; margin:0 0 12px; color:var(--muted); font-weight:600; }
  canvas { display:block; width:280px; height:280px; border:2px solid #d7dbe2; border-radius:12px; cursor:crosshair; background:#fff; touch-action:none; }
  .btns { margin-top:14px; display:flex; gap:10px; }
  .btns button { flex:1; padding:10px 0; font-size:15px; border:none; border-radius:9px; cursor:pointer; font-weight:600; }
  #btnPredict { background:var(--accent); color:#fff; }
  #btnClear { background:#eef0f3; color:var(--ink); }
  .result { text-align:center; }
  .result .digit { font-size:88px; font-weight:800; line-height:1.1; color:var(--accent2); min-height:100px; }
  .result .hint { color:var(--muted); font-size:14px; }
  .status { margin-top:14px; font-size:13px; color:var(--muted); }
  .status .ok { color:#16a34a; font-weight:600; }
  .preview { margin-top:14px; }
  .preview canvas { display:block; margin:0 auto; border:1px solid #e2e5ea; border-radius:8px; }
  footer { margin-top:28px; font-size:13px; color:var(--muted); text-align:center; line-height:1.8; }
  footer a { color:var(--accent); text-decoration:none; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>手写数字识别</h1>
    <p>在左侧画板上写一个数字（0-9），模型会实时预测 · 训练数据 scikit-learn digits (8×8)</p>
  </header>
  <div class="grid">
    <div class="card">
      <h2>画板</h2>
      <canvas id="board" width="280" height="280"></canvas>
      <div class="btns">
        <button id="btnPredict" type="button">预测</button>
        <button id="btnClear" type="button">清空</button>
      </div>
      <div class="status" id="status">检查服务状态…</div>
    </div>
    <div class="card result">
      <h2>预测结果</h2>
      <div class="digit" id="digit">—</div>
      <div class="hint" id="hint">写一个数字，然后点击「预测」</div>
      <div class="preview"><canvas id="gridView" width="88" height="88"></canvas><div class="hint" style="margin-top:6px">模型看到的 8×8 网格</div></div>
      <div class="status">API 文档：<a href="/docs" target="_blank">/docs</a> · 健康检查：<a href="/health" target="_blank">/health</a> · 实验跟踪：<a href="http://127.0.0.1:5000" target="_blank">MLflow UI</a></div>
    </div>
  </div>
  <footer>MLops Final · RandomForest · DVC + MLflow · FastAPI</footer>
</div>
<script>
(function () {
  var canvas = document.getElementById('board');
  var ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, 280, 280);
  ctx.lineWidth = 16;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#000';

  var drawing = false;
  function pos(e) {
    var r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * 280 / r.width, y: (e.clientY - r.top) * 280 / r.height };
  }
  canvas.addEventListener('mousedown', function (e) { drawing = true; ctx.beginPath(); ctx.moveTo(pos(e).x, pos(e).y); });
  canvas.addEventListener('mousemove', function (e) { if (!drawing) return; var p = pos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); });
  window.addEventListener('mouseup', function () { drawing = false; });
  canvas.addEventListener('touchstart', function (e) { e.preventDefault(); drawing = true; var t = e.touches[0]; ctx.beginPath(); ctx.moveTo((t.clientX - canvas.getBoundingClientRect().left) * 280 / canvas.getBoundingClientRect().width, (t.clientY - canvas.getBoundingClientRect().top) * 280 / canvas.getBoundingClientRect().height); }, { passive: false });
  canvas.addEventListener('touchmove', function (e) { e.preventDefault(); if (!drawing) return; var t = e.touches[0]; var r = canvas.getBoundingClientRect(); ctx.lineTo((t.clientX - r.left) * 280 / r.width, (t.clientY - r.top) * 280 / r.height); ctx.stroke(); }, { passive: false });
  canvas.addEventListener('touchend', function () { drawing = false; });

  document.getElementById('btnClear').addEventListener('click', function () {
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, 280, 280);
    ctx.strokeStyle = '#000';
    document.getElementById('digit').textContent = '—';
    document.getElementById('hint').textContent = '已清空，请重新书写';
  });

  function gridToPixels() {
    // 1) 读取画布，找到笔迹包围盒（非纯白像素）
    var src = ctx.getImageData(0, 0, 280, 280).data;
    var minX = 280, minY = 280, maxX = -1, maxY = -1;
    for (var y = 0; y < 280; y++) {
      for (var x = 0; x < 280; x++) {
        if (src[(y * 280 + x) * 4] < 200) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return null; // 没有笔迹
    var bw = maxX - minX + 1, bh = maxY - minY + 1;
    // 2) 保持纵横比，缩放到 8x8 网格的 80%，居中（与训练数据形态一致）
    var scale = Math.min(6.4 / bw, 6.4 / bh);
    var w8 = Math.max(1, Math.round(bw * scale));
    var h8 = Math.max(1, Math.round(bh * scale));
    var ox = Math.floor((8 - w8) / 2);
    var oy = Math.floor((8 - h8) / 2);
    // 3) 从画布裁剪笔迹区域并缩放绘制到 8x8 离屏画布
    var off = document.createElement('canvas');
    off.width = 8; off.height = 8;
    var octx = off.getContext('2d');
    octx.fillStyle = '#fff';
    octx.fillRect(0, 0, 8, 8);
    octx.drawImage(canvas, minX, minY, bw, bh, ox, oy, w8, h8);
    var data = octx.getImageData(0, 0, 8, 8).data;
    // 4) 灰度 -> 像素值：黑=16，白=0（与训练数据口径一致）
    var px = [];
    for (var i = 0; i < 64; i++) {
      var gray = data[i * 4];
      px.push(+(16 * (255 - gray) / 255).toFixed(4));
    }
    return px;
  }

  function drawPreview(pixels) {
    var view = document.getElementById('gridView');
    var vctx = view.getContext('2d');
    vctx.fillStyle = '#fff';
    vctx.fillRect(0, 0, 88, 88);
    for (var i = 0; i < 64; i++) {
      var v = 255 - Math.round(pixels[i] / 16 * 255);
      vctx.fillStyle = 'rgb(' + v + ',' + v + ',' + v + ')';
      vctx.fillRect((i % 8) * 11, Math.floor(i / 8) * 11, 11, 11);
    }
  }

  var busy = false;
  document.getElementById('btnPredict').addEventListener('click', function () {
    var pixels = gridToPixels();
    if (pixels === null) {
      document.getElementById('hint').textContent = '画板上还没有内容，请先书写一个数字';
      return;
    }
    if (busy) return;
    busy = true;
    document.getElementById('hint').textContent = '预测中…';
    fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pixels: pixels })
    }).then(function (r) { return r.json(); }).then(function (d) {
      document.getElementById('digit').textContent = d.prediction;
      var pct = Math.round((d.confidence || 0) * 100);
      var msg = '预测完成 · 置信度 ' + pct + '%';
      if (d.confidence < 0.6) msg += '（置信度偏低：建议写粗、写大、居中）';
      document.getElementById('hint').textContent = msg;
      drawPreview(pixels);
    }).catch(function () {
      document.getElementById('hint').textContent = '请求失败，请确认服务已启动';
    }).finally(function () { busy = false; });
  });

  fetch('/health').then(function (r) { return r.json(); }).then(function (d) {
    var s = document.getElementById('status');
    s.innerHTML = '服务状态：<span class="ok">' + d.status + '</span>';
  }).catch(function () {
    document.getElementById('status').textContent = '服务状态：无法连接';
  });
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def demo_page():
    return DIGIT_DEMO_HTML


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
