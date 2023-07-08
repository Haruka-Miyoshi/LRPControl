import os
import numpy as np
import matplotlib.pyplot as plt
from pcontrol import PController
from matplotlib.animation import FuncAnimation

# 台車の初期位置と目標位置
start_pos = np.array([0.0, 0.0])
target_pos = np.array([500.0, 500.0])

# 制御ゲイン
k_p = np.array(0.5)

# P制御
pc=PController(k_p, target_pos)

# アニメーションの設定
fig, ax = plt.subplots()
ax.set_title('LRPControl Data')
ax.set_xlabel('Postions X')
ax.set_ylabel('Postions Y')
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
line, = ax.plot([], [], 'ro', markersize=10)

# 誤差
errors=None
# 速度
velocitys=None

# アニメーションの初期化
def init():
    line.set_data([], [])
    return [line]

# アニメーションの更新
def update(frame):
    global start_pos, errors, velocitys

    # 台車の位置を更新
    start_pos += pc.update(start_pos)

    # 台車とライダーの位置をプロット
    line.set_data([start_pos[0]], [start_pos[1]])

    # 目標位置に到達したらアニメーションを停止
    if np.allclose(start_pos, target_pos):
        ani.event_source.stop()
        errors, velocitys=pc.get_param()

    return [line]

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=np.arange(100), init_func=init, interval=10, blit=True)

def main():
    global errors, velocitys, ani
    ani.save('./figs/data_action.gif', writer='imagemagick')
    # アスペクト比の調整
    ax.set_aspect('equal')
    # アニメーションの表示
    # plt.show()
    if not os.path.exists('./data'):
        os.mkdir('./data')

    np.savetxt('./data/v.txt', velocitys)
    np.savetxt('./data/e.txt', errors)

if __name__=='__main__':
    main()
