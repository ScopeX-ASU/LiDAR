"""
Description: 
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-10-13 15:07:03
LastEditors: JeremieMelo jqgu1996@163.com
LastEditTime: 2023-10-13 21:08:35
"""

import numpy as np
import matplotlib.pyplot as plt


def test_angle():
    r = 5
    x1, y1 = 0, 0
    x2, y2 = 11, 12
    y2 = np.linspace(-50, 50, 100)

    for x2 in np.linspace(-15, 15, 31):
        dy = np.abs(y2 - y1)
        dx = x2 - x1
        tan_theta = (2 * dx - (4 * dx**2 - 4 * (4 * r - dy) * dy) ** 0.5) / (
            8 * r - 2 * dy
        )
        # tan_theta = np.where(dy == 4*r, dy/(2* abs(dx)), tan_theta)
        print(tan_theta)
        theta = (
            np.arctan(
                (2 * dx - (4 * dx**2 - 4 * (4 * r - dy) * dy) ** 0.5) / (8 * r - 2 * dy)
            )
            * 2
        )
        # theta = np.arctan2(2*dx - (4*dx**2 - 4*(4*r - dy)*dy)**0.5, 8*r - 2*dy) * 2
        theta = np.where((dy <= 4 * r) & (dx < 2 * r), 2 * np.pi, theta)
        # theta = np.where(dy > 4*r, np.arctan2(dy, 2* dx) * 2, theta)
        # print(theta)
        plt.plot(y2, theta / np.pi * 180, label=f"{x2}")
    plt.xlabel("dy")
    plt.ylabel("Phi (degree)")
    plt.legend()
    plt.show()


test_angle()
