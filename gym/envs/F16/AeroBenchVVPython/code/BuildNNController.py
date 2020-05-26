# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-06-21 14:05:09
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-06-21 15:12:06
# -------------------------------
import tensorflow as tf
print("Tensorflow Version:{}".format(tf.__version__))

import numpy as np

class NeuralNetworkContorller:

    def __init__(self):
        print("NeuralNetworkContorller.__init__")
        # Longitudinal Gains
        K_lqr_long = np.array([[-156.8801506723475, -31.037008068526642, -38.72983346216317]], dtype=float)

        # Lateral Gains
        K_lqr_lat = np.array([[30.511411060051355, -5.705403676148551, -9.310178739319714, \
                                                    -33.97951344944365, -10.652777306717681], \
                              [-22.65901530645282, 1.3193739204719577, -14.2051751789712, \
                                                    6.7374079391328845, -53.726328142239225]], dtype=float)

        self.K_lqr = np.zeros((3, 8))
        self.K_lqr[:1, :3] = K_lqr_long
        self.K_lqr[1:, 3:] = K_lqr_lat

    def train(self):
        pass

    def predict(self, x_ctrl):
        u_deg_123 = np.dot(-self.K_lqr, x_ctrl) # Full Control
        return u_deg_123

