import torch
import numpy as np
from lrpcontorl import LRPControl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 比例制御モデルを読み出す
lr=LRPControl(2,2, mode=True, model_path='./model/parameter.txt')

# 台車の初期位置と目標位置
start_pos = np.array([-100.0, -10.0])
target_pos = np.array([20.0, 20.0])
velocity = np.array([0.0, 0.0])

# 制御ゲイン
k_p = np.array(0.5)

# 停車誤差
eps = np.array(1.0)

# アニメーションの設定
fig, ax = plt.subplots()
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.plot([target_pos[0]], [target_pos[1]], 'ro', markersize=10)
line, = ax.plot([], [], 'bo', markersize=10)

# アニメーションの初期化
def init():
    line.set_data([], [])
    return [line]

# アニメーションの更新
def update(frame):
    global start_pos, velocity

    # 台車の速度を更新
    error = target_pos - start_pos
    velocity=lr.pred(x=torch.tensor(error))

    # 台車の位置を更新
    start_pos += velocity.detach().numpy()

    # 台車とライダーの位置をプロット
    line.set_data([start_pos[0]], [start_pos[1]])

    # 目標位置に到達したらアニメーションを停止
    if np.allclose(start_pos, target_pos):
        ani.event_source.stop()

    return [line]

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=np.arange(100), init_func=init, interval=100, blit=True)

# アスペクト比の調整
ax.set_aspect('equal')

# アニメーションの表示
plt.show()