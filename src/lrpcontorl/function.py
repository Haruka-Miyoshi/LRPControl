import os
import torch
import numpy as np
import torch.nn as nn
from .model import *

"""線形回帰による比例制御"""
class LRPControl(object):
    def __init__(self, i_dim, o_dim, mode=False, model_path=''):
        # デバイス設定 GPU or CPU
        self.__device="cuda" if torch.cuda.is_available() else "cpu"
        # モデル定義
        self.__model=Model(i_dim, o_dim).to(device=self.__device)

        # 学習済みモデル
        if mode:
            # 学習済みモデル読み込み
            self.__model.load_state_dict(torch.load(model_path))
        
        # 学習係数
        self.__lr=1e-3
        # 損失関数:最小二乗法
        self.__loss_func=nn.MSELoss()
        # 最適化アルゴリズム:Adam
        self.__opt=torch.optim.Adam(self.__model.parameters(), lr=self.__lr)

        # save file path
        self.FILE_PATH=os.path.join('./model')

        # フォルダを生成
        if not os.path.exists(self.FILE_PATH):
            os.mkdir(self.FILE_PATH)

    """fit:フィッティング処理"""
    def fit(self, X, Y, mode=False, epoch=1000):
        # 損失を格納変数
        losses=torch.zeros(epoch)
        X=X.to(device=self.__device)
        Y=Y.to(device=self.__device)

        for e in range(epoch):
            sum_loss=0.0
            for x, y in zip(X,Y):
                # 予測
                y_hat=self.__model(x.float())
                # 損失計算
                loss=self.__loss_func(torch.abs(y.float()), y_hat)
                
                # 勾配を初期化
                self.__opt.zero_grad()
                
                # 逆伝播を計算
                loss.backward()
                
                # 次のステップ
                self.__opt.step()

                sum_loss+=loss.item()

            # 損失を格納
            losses[e]=sum_loss

        # 損失保存
        if mode:
            """汎用的な保存方法を検討中"""
            # ファイル path
            LOSS_SAVE=os.path.join(self.FILE_PATH, 'loss.txt')
            # 損失結果 保存
            np.savetxt(LOSS_SAVE, losses)
            # パラメータ保存
            PARAM_SAVE=os.path.join(self.FILE_PATH, 'parameter.txt')
            # 学習したパラメータを保存
            torch.save(self.__model.state_dict(), PARAM_SAVE)
        
    """pred:予測処理"""
    def pred(self, x):
        return self.__model(x.float())
    
    """get_params:モデルパラメータを呼び出す"""
    def get_params(self):
        [w, b]=self.__model.parameters()
        return (w[0][0].item(), b[0].item())
    