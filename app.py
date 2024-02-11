import gradio as gr
from PIL import Image, ImageDraw
import numpy as np

def draw_lines(x, y):
    # キャンバスのサイズ
    width, height = 1024, 1024
    # 白背景のキャンバスを作成
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # 直線を描画する関数
    def draw_rotated_line(angle):
        rad = np.radians(angle)
        cos_val = np.cos(rad)
        sin_val = np.sin(rad)
        # キャンバスの対角線の長さを計算
        diagonal_length = np.sqrt(width**2 + height**2)
        # 中心から端までの線を計算
        line_length = diagonal_length
        # 線の始点と終点を計算
        start_x = x - line_length * cos_val
        start_y = y - line_length * sin_val
        end_x = x + line_length * cos_val
        end_y = y + line_length * sin_val
        # 線を描画
        draw.line((start_x, start_y, end_x, end_y), fill="blue", width=1)

    # 5度ずつ回転させながら直線を描画
    for angle in range(0, 360, 5):
        draw_rotated_line(angle)

    # PIL Imageをnumpy配列に変換
    return np.array(image)

# Gradioインターフェース
iface = gr.Interface(fn=draw_lines,
                     inputs=[gr.Slider(minimum=0, maximum=1024, value=512, label="X座標"),
                             gr.Slider(minimum=0, maximum=1024, value=512, label="Y座標")],
                     outputs="image",
                     live=True,
                     title="VP preprocess")

iface.launch()
