"""P制御"""
class PController:
    def __init__(self, kp, setpoint):
        self.kp = kp  # 比例ゲイン
        self.setpoint = setpoint  # 目標値

        # 誤差系列
        self.errors=[]
        # 速度系列
        self.controls=[]
    
    def update(self, measured_value):
        error = self.setpoint - measured_value  # 目標値と現在値の誤差

        # 制御量を計算
        control = self.kp * error

        # 保存
        self.errors.append(error)
        self.controls.append(control)

        return control
    
    def get_param(self):
        return self.errors, self.controls