import numpy as np


def sort_points_anticlockwise(points: np.ndarray) -> np.ndarray:
    """逆時針排序多邊形頂點

    Args:
        points (np.ndarray): 多邊形頂點 numpy array

    Returns:
        np.ndarray: 排序後的多邊形頂點 numpy array(左上、右上、右下、左下)
    """

    # 1. 計算中點
    cen_x, cen_y = np.mean(points, axis=0)

    # 2. 計算各個座標的tan值，並按照tan值排序
    d2s = []
    for i in range(len(points)):

        o_x = points[i][0] - cen_x
        o_y = points[i][1] - cen_y
        atan2 = np.arctan2(o_y, o_x)
        if atan2 < 0:
            atan2 += np.pi * 2
        d2s.append([points[i], atan2])

    d2s = sorted(d2s, key=lambda x: x[1])
    order_2ds = np.array([x[0] for x in d2s])

    # 3. 整理資料
    new_points = list()
    new_points.append(order_2ds[2])
    new_points.append(order_2ds[3])
    new_points.append(order_2ds[0])
    new_points.append(order_2ds[1])

    return new_points