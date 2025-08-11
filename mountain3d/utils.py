from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def get_d_max(h_aim: float, h_antenna: float) -> float:
    # Возвращает значение дальности прямой видимости (дальности обнаружения)
    # по формуле Введенского, км.
    return 4.12 * (h_aim**0.5 + h_antenna**0.5)


def get_D_max(config: Dict[str, Any]) -> float:
    # Возвращает значение максимальной дальности обнаружения РЛС, м.

    q = config["q"]
    K = config["K"]
    Pr_min = get_Pr_min(q, K)

    P_t = config["P_t"]
    G_t = G_r = config[
        "G_t"
    ]  # Коэффициент усиления передающей антенны = --/-- принимающей антенны
    tau = config["tau"]
    lambd = config["lambda"]
    sigma = config["sigma"]

    D_max = (
        P_t * tau * G_t * G_r * lambd**2 * sigma / (4 * np.pi) ** 3 / Pr_min
    ) ** 0.25
    return D_max


def get_Pr_min(q: float, K: float, k: float = 1.38e-23, T: float = 290) -> float:
    # Возвращает значение минимальной энергии сигнала, принимаемого приемником.
    v = q**2 / 2
    Pr_min = v * k * T * K
    return Pr_min


def get_D(
    h: float, config: Dict[str, Any], threshold: float = 1000
) -> Tuple[float, float]:
    # Возвращает значение абсциссы граничной точки зоны обнаружения
    # по заданному значению высоты h.

    # alpha = config["Dead_angle"]
    lambd = config["lambda"]
    H_max = config["H_max"]
    h_antenna = config["h_antenna"]
    dead_angle = config["Dead_angle"] / 180 * np.pi

    D_max = get_D_max(config)

    E = h / H_max * dead_angle
    D = np.sqrt(np.pi * h_antenna * h / lambd * D_max)

    if D_max * np.cos(E) >= D:  # & (h < threshold):
        ans = (np.tan(dead_angle) * h, D)
    else:
        ans = (np.tan(dead_angle) * h, D_max * np.cos(E))
    return ans


def get_rocket_right_D(h: float, config: Dict[str, Any]) -> float:
    # Возвращает значение абсциссы правой граничной точки зоны поражения ракеты
    # по заданному значению высоты h.

    D_max = config["D_max"]
    return np.sqrt(D_max**2 - h**2)


def get_rocket_left_D(h: float, config: Dict[str, Any]) -> float:
    # Возвращает значение абсциссы левой граничной точки зоны поражения ракеты
    # по заданному значению высоты h.

    D_min = config["D_min"]
    return np.sqrt(D_min**2 - h**2)


def get_statistics(config: Dict[str, Any]) -> None:
    # Печатает в консоль статистику по заданному набору параметров.

    Pr_min = get_Pr_min(q=config["q"], K=config["K"])
    D_max = get_D_max(config)

    r = config["H_max"] * np.tan(np.deg2rad(config["alpha"]))
    E = config["P_t"] * config["tau"]
    d = get_d_max(config["h_antenna"], config["H_max"])

    print(f"Максимальная дальность обнаружения РЛС: {D_max / 1000.:.2f} км")
    print(f"Радиус мертвой воронки: {r / 1000.:.2f} км")
    print(f"Энергия зондирующего сигнала: {E:.2f} Вт * с")
    print(f"Дальность прямой видимости: {d :.2f} км")
    print(f"Минимальная энергия принимаемого сигнала приемником: {Pr_min} Вт * с")


def get_coordinates(config: Dict[str, Any]) -> pd.DataFrame:
    # Возвращает датафрейм, задающий границу зоны обнаружения.

    coordinates = [
        (get_D(h, config), h) for h in np.linspace(1, config["H_max"], config["N_h"])
    ]
    df = pd.DataFrame(coordinates, columns=["D", "h"])

    intersection = pd.DataFrame(
        {
            "h": [config["H_max"], 0],
            "D": [config["H_max"] * np.tan(np.deg2rad(config["alpha"])), 0],
        }
    )

    df = pd.concat([df, intersection, df.iloc[[0]]])
    return df


def get_rocket_coordinates(config: Dict[str, Any]) -> pd.DataFrame:
    # Возвращает датафрейм, задающий границу зоны огня.

    right_coordinates = [
        (get_rocket_right_D(h, config), h)
        for h in np.linspace(config["H_min"], config["H_max"], config["N_h"])
    ]
    left_coordinates = [
        (get_rocket_left_D(h, config), h)
        for h in np.linspace(
            config["D_min"] * np.cos(np.deg2rad(config["alpha"])),
            config["H_min"],
            config["N_h"],
        )
    ]
    right_df = pd.DataFrame(right_coordinates, columns=["D", "h"])
    left_df = pd.DataFrame(left_coordinates, columns=["D", "h"])

    top_intersection = pd.DataFrame(
        {
            "h": [config["H_max"]],
            "D": [config["H_max"] * np.tan(np.deg2rad(config["alpha"]))],
        }
    )

    bottom_intersection = pd.DataFrame({"h": [config["H_min"]], "D": [config["D_min"]]})

    df = pd.concat(
        [right_df, top_intersection, left_df, bottom_intersection, right_df.iloc[[0]]]
    )

    return df


def get_h1_h2(
    r: float, config: Dict[str, Any]
) -> Tuple[Optional[float], Optional[float]]:
    # Возвращает ординаты точек пересечения прямой x = r
    # с графиком границы зоны обнаружения.

    lambd = config["lambda"]

    h_antenna = config["h_antenna"]
    alpha = config["alpha"]  # угол места
    H_max = config["H_max"]

    D_max = get_D_max(config)

    h1 = r**2 / (np.pi * h_antenna * D_max) * lambd
    x_a = H_max * np.tan(np.deg2rad(alpha))
    x_b = D_max * np.cos(np.deg2rad(alpha))

    if r < 0 or r > D_max:
        return None, None

    if r < x_a:
        h2 = r / np.tan(np.deg2rad(alpha))
    elif r >= x_a and r <= x_b:
        h2 = H_max
    else:
        h2 = np.arccos(r / D_max) / (np.deg2rad(alpha) / H_max)

    return h1, h2


def get_rocket_h1_h2(
    r: float, config: Dict[str, Any]
) -> Tuple[Optional[float], Optional[float]]:
    #  Возвращает ординаты точек пересечения прямой x = r с графиком границы зоны огня.
    D_min = config["D_min"]
    D_max = config["D_max"]
    H_min = config["H_min"]
    H_max = config["H_max"]
    alpha = config["alpha"]

    r_a = D_min * np.sin(np.deg2rad(alpha))
    r_b = H_max * np.tan(np.deg2rad(alpha))
    r_c = np.sqrt(D_max**2 - H_max**2)

    if r < r_a or r > D_max:
        return None, None

    if r <= D_min:
        h1 = np.sqrt(D_min**2 - r**2)
    else:
        h1 = H_min

    if r <= r_b:
        h2 = r / np.tan(np.deg2rad(alpha))
    elif r <= r_c:
        h2 = H_max
    else:
        h2 = np.sqrt(D_max**2 - r**2)

    return h1, h2
