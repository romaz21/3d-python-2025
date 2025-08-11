from pathlib import Path

DEFAULT_SLICE_HEIGHT_2D = 1000
DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 800
DEFAULT_RATIO = (1, 1, 0.1)
REPO_ROOT = Path(__file__).parents[1]


def ALARM_HEIGHT_COLOR_MAPPING(h: int):
    if h <= 50:
        return "yellow"
    elif h <= 100:
        return "blue"
    elif h <= 300:
        return "pink"
    elif h <= 500:
        return "green"
    elif h <= 1000:
        return "purple"
    elif h <= 4000:
        return "orange"
    elif h <= 10000:
        return "brown"
    return "black"


def FIRE_HEIGHT_COLOR_MAPPING(h: int):
    if h <= 20:
        return "pink"
    elif h <= 30:
        return "red"
    return "orange"
