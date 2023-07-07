import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 台車の初期位置と目標位置
start_pos = np.array([-50.0, -50.0])
target_pos = np.array([50.0, 50.0])
velocity = np.array([0.0, 0.0])

# 制御ゲイン
k_p = np.array(0.5)

# 停車誤差
eps = np.array(1.0)

# アニメーションの設定
fig, ax = plt.subplots()
ax.set_title('LRPControl Data')
ax.set_xlabel('Postions X')
ax.set_ylabel('Postions Y')
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
line, = ax.plot([], [], 'ro', markersize=10)

# 速度系列
velocitys = []
# 誤差系列
errors = []

# アニメーションの初期化
def init():
    line.set_data([], [])
    return [line]

# アニメーションの更新
def update(frame):
    global start_pos, velocity

    # 台車の速度を更新
    error = target_pos - start_pos
    velocity = k_p * error

    # 台車の位置を更新
    start_pos += velocity

    # 台車とライダーの位置をプロット
    line.set_data([start_pos[0]], [start_pos[1]])

    # 目標位置に到達したらアニメーションを停止
    if np.allclose(start_pos, target_pos):
        ani.event_source.stop()
    else:
        velocitys.append(velocity)
        errors.append(error)

    return [line]

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=np.arange(100), init_func=init, interval=100, blit=True)
ani.save('./figs/data_action.gif', writer='imagemagick')
# アスペクト比の調整
ax.set_aspect('equal')
# アニメーションの表示
# plt.show()
if not os.path.exists('./data'):
    os.mkdir('./data')

np.savetxt('./data/v.txt', velocitys)
np.savetxt('./data/e.txt', errors)
