#!/usr/bin/env python
"""Simulate REAL free-hand writing on the 280x280 canvas using rendered fonts
(random size / position / font style), then compare accuracy of the OLD
preprocessing (whole-canvas shrink) vs the NEW one (bbox crop + aspect-preserving
scale + center into 8x8) with the production model.
"""
import os
import random

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
CANVAS = 280
FONT_DIR = r"C:\Windows\Fonts"
FONTS = ["arial.ttf", "arialbd.ttf", "times.ttf", "simhei.ttf", "simsun.ttc"]


def old_pipeline(canvas):
    """改进前：整张画布直接缩放到 8x8（不裁剪、不居中）"""
    img = Image.fromarray(np.uint8(canvas)).resize((8, 8), Image.BILINEAR)
    a = np.asarray(img, dtype=float)
    return ((255 - a) / 255 * 16).ravel()


def new_pipeline(canvas):
    """改进后：bbox 裁剪 -> 保持纵横比缩放到 8x8 的 80% -> 居中（与页面 JS 一致）"""
    mask = canvas < 200
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    crop = Image.fromarray(np.uint8(canvas[y0:y1 + 1, x0:x1 + 1]))
    scale = min(6.4 / bw, 6.4 / bh)
    w8, h8 = max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))
    im8 = crop.resize((w8, h8), Image.BILINEAR)
    a8 = np.full((8, 8), 255.0)
    ox, oy = (8 - w8) // 2, (8 - h8) // 2
    a8[oy:oy + h8, ox:ox + w8] = np.asarray(im8, dtype=float)
    return ((255 - a8) / 255 * 16).ravel()


def render_digit(d, size, ox, oy, font_path, stroke_width=0):
    """在 280x280 白底画布上以给定字号/位置/字体渲染一个数字（黑色抗锯齿墨迹）。
    stroke_width>0 时用描边加粗笔画，模拟真实手写的粗笔。"""
    img = Image.new("L", (CANVAS, CANVAS), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size)
    if stroke_width:
        draw.text((ox, oy), str(d), fill=0, font=font,
                  stroke_width=stroke_width, stroke_fill=0)
    else:
        draw.text((ox, oy), str(d), fill=0, font=font)
    return np.asarray(img, dtype=float)


def evaluate(model, canvas, label, total, old_correct, new_correct):
    # 模型在 [0,1] 归一化数据上训练，预测前把 0~16 的像素值除以 16 对齐口径
    old_px = old_pipeline(canvas) / 16.0
    new_px = new_pipeline(canvas) / 16.0
    if new_px is None:
        return total, old_correct, new_correct
    total += 1
    if int(model.predict(old_px.reshape(1, -1))[0]) == label:
        old_correct += 1
    if int(model.predict(new_px.reshape(1, -1))[0]) == label:
        new_correct += 1
    return total, old_correct, new_correct


def run(rng, model, available_fonts, stroke_width, tag, complete=True, standard=False):
    old_correct = new_correct = total = 0
    for d in range(10):
        for font_name in available_fonts:
            font_path = os.path.join(FONT_DIR, font_name)
            for _ in range(20):
                if standard:
                    # 对照：固定居中、标准大小（约画布 60%）
                    size, ox, oy = 170, 55, 55
                else:
                    size = rng.randint(90, 220)
                    margin = 10
                    ox = rng.randint(margin, CANVAS - size - margin) if complete else rng.randint(30, 220)
                    oy = rng.randint(margin, CANVAS - size - margin) if complete else rng.randint(30, 220)
                canvas = render_digit(d, size, ox, oy, font_path, stroke_width)
                total, old_correct, new_correct = evaluate(
                    model, canvas, d, total, old_correct, new_correct)
    print(f"[{tag}] 样本数: {total}")
    print(f"  改进前（整图直接缩放）: {old_correct}/{total} = {old_correct / total * 100:.1f}%")
    print(f"  改进后（bbox 裁剪+居中）: {new_correct}/{total} = {new_correct / total * 100:.1f}%")


def main():
    rng = random.Random(42)
    model = joblib.load(os.path.join(ROOT, "ml", "registry", "model.pkl"))
    available_fonts = [f for f in FONTS if os.path.exists(os.path.join(FONT_DIR, f))]
    print("可用字体:", available_fonts)
    run(rng, model, available_fonts, stroke_width=0, tag="对照：标准大小居中（细笔画）", standard=True)
    run(rng, model, available_fonts, stroke_width=0, tag="随机大小/位置·数字完整在画布内（细笔画）")
    run(rng, model, available_fonts, stroke_width=25, tag="随机大小/位置·数字完整在画布内（粗笔画25px）")


if __name__ == "__main__":
    main()
