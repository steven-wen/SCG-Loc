import numpy as np
import math
import cv2


def evaluate(opt, pred_XY, label_XY):
    pred_X, pred_Y = pred_XY
    label_X, label_Y = label_XY
    x_rate = (pred_X - label_X) / opt.data_config["Satellitehw"][0]
    y_rate = (pred_Y - label_Y) / opt.data_config["Satellitehw"][1]
    distance = np.sqrt((np.square(x_rate) + np.square(y_rate)) / 2)  # take the distance to the 0-1
    result = np.exp(-1 * 10 * distance)
    return result


def decode_heatmap_prediction(score_map, satellite_hw, loc_bias=None):
    satellite_h, satellite_w = satellite_hw
    satellite_map = cv2.resize(score_map, (satellite_w, satellite_h))
    pred_id = int(np.argmax(satellite_map))
    raw_pred_xy = np.array(
        [pred_id // satellite_w, pred_id % satellite_w],
        dtype=np.float32)

    refined_pred_xy = raw_pred_xy.copy()
    if loc_bias is not None:
        map_h, map_w = score_map.shape[-2], score_map.shape[-1]
        map_id = int(np.argmax(score_map))
        map_x = int(map_id // map_w)
        map_y = int(map_id % map_w)
        refined_pred_xy = np.array([map_x, map_y], dtype=np.float32)
        refined_pred_xy = refined_pred_xy + loc_bias[:, map_x, map_y]
        refined_pred_xy *= np.array(
            [satellite_h / map_h, satellite_w / map_w],
            dtype=np.float32)

    return raw_pred_xy, refined_pred_xy


def pixel_to_geo(pred_xy, satellite_hw, tl_E, tl_N, br_E, br_N):
    satellite_h, satellite_w = satellite_hw
    pred_x, pred_y = pred_xy
    gps_x = pred_x / satellite_h
    gps_y = pred_y / satellite_w
    pred_gps_e = tl_E + (br_E - tl_E) * gps_y
    pred_gps_n = tl_N - (tl_N - br_N) * gps_x
    return pred_gps_n, pred_gps_e


def Distance(lata, loga, latb, logb):
    # EARTH_RADIUS = 6371.0
    EARTH_RADIUS = 6378.137
    PI = math.pi
    # // 转弧度
    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a)
                                  * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    return distance


def distance(lat1, lon1, lat2, lon2):
    # 将经纬度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # 计算两个经纬度坐标之间的球面距离
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = 6371 * c * 1000  # 6371km 为地球半径，转换为米

    return distance


if __name__ == "__main__":
    l1 = Distance(100.111111, 40.222222, 120.232322, 44.2132311)
    l2 = distance(100.111111, 40.222222, 120.232322, 44.2132311)
    print(l1, l2)
