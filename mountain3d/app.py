import base64
import io
import json
from typing import Any, Dict, List

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, no_update
from loguru import logger

from .const import (
    DEFAULT_HEIGHT,
    DEFAULT_RATIO,
    DEFAULT_SLICE_HEIGHT_2D,
    DEFAULT_WIDTH,
    REPO_ROOT,
)
from .graphics import (
    add_aircraft,
    create_figure,
    draw_alarm_fire_stl_model,
    draw_detection_dome,
    draw_detection_dome_slices,
)
from .math import get_zone

MODE_2D = False


def make_rooted_path(paths: List[str]):
    return [REPO_ROOT / path for path in paths]


def run_app():  # noqa
    with open(REPO_ROOT / "3d-python-media/example_configs/ui_config.json", "rb") as fp:
        ui_config: Dict[str, Any] = json.load(fp)

    if ui_config["cache"]:
        cahce_path = REPO_ROOT / "3d-python-media/maps/cache"
    else:
        logger.warning("Cache is disabled!")
        cahce_path = None

    fig, heightmap, scale_coefs, spacing = create_figure(
        heightmap_paths=REPO_ROOT / "3d-python-media/maps/preprocessed",
        lat=ui_config["lat"],
        long=ui_config["long"],
        size_km=ui_config["size_km"],
        map_size_px=ui_config["map_size_px"],
        ratio=DEFAULT_RATIO,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        cahce_path=cahce_path,
    )

    with (
        open(REPO_ROOT / "config_annotation/alarm_config.json") as alarm_conf,
        open(REPO_ROOT / "config_annotation/fire_config.json") as fire_conf,
        open(REPO_ROOT / "config_annotation/control_config.json") as control_conf,
        open(REPO_ROOT / "config_annotation/object_config.json") as object_conf,
        open(REPO_ROOT / "config_annotation/dron_config.json") as dron_conf,
    ):
        loc_ann: Dict = json.load(alarm_conf)
        launch_ann: Dict = json.load(fire_conf)
        control_ann: Dict = json.load(control_conf)
        object_ann: Dict = json.load(object_conf)
        dron_ann: Dict = json.load(dron_conf)

    app = Dash(__name__, prevent_initial_callbacks=True)

    def create_input_groups(config: Dict, prefix: str) -> List[dbc.InputGroup]:
        return [
            construct_input_group(prefix, key, val)
            for key, val in config.items()
        ]

    def construct_input_group(arma_type, key, val):
        input_id = f"{arma_type}_inp_{key}"
        if isinstance(val, str):
            return dbc.InputGroup(
                [dbc.InputGroupText(f"{val}:"), dbc.Input(id=input_id)]
            )
        else:
            return dbc.InputGroup(
                [
                    dbc.InputGroupText(f"{val[0]}:"),
                    dbc.Input(id=input_id),
                    dbc.InputGroupText(f", {val[1]}"),
                ]
            )

    locator_data_input = create_input_groups(loc_ann, "loc")
    launcher_data_input = create_input_groups(launch_ann, "launch")
    control_data_input = create_input_groups(control_ann, "control")
    object_data_input = create_input_groups(object_ann, "object")
    dron_data_input = create_input_groups(dron_ann, "dron")

    app.layout = dbc.Container(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader("Выбор техники"),
                    create_body(locator_data_input, launcher_data_input, control_data_input, object_data_input, dron_data_input),
                    dbc.ModalFooter(
                        [
                            dcc.Upload(
                                id="config_upload",
                                children=dbc.Button("Считать из файла"),
                            ),
                            dcc.Upload(
                                id="image_upload",
                                children=dbc.Button("Добавить обозначение"),
                            ),
                            dbc.Button("Готово", id="add_arma_button"),
                        ]
                    ),
                ],
                id="modal",
            ),
            dcc.Graph(figure=fig, id="main_graph", style={"height": "100%"}),
            create_controls(),
            dcc.Store(id="coordinate_store"),
        ],
        style={"height": "100%"},
    )

    @app.callback(
        Output("modal", "is_open", allow_duplicate=True),
        Output("coordinate_store", "data"),
        Input("main_graph", "clickData"),
    )
    def open_arma_settings(clickData):
        clicked_point = clickData["points"][-1]
        logger.info(f"Clicked point: {clickData}")
        x, y, z = (
            int(clicked_point["x"]),
            int(clicked_point["y"]),
            int(clicked_point["z"]),
        )
        logger.info(f"Clicked point: {x=}, {y=}, {z=}")
        return True, json.dumps((x, y, z))

    @app.callback(
        output=dict(
            **{f"loc_{key}": Output(f"loc_inp_{key}", "value") for key in loc_ann},
            **{
                f"launch_{key}": Output(f"launch_inp_{key}", "value")
                for key in launch_ann
            },
            **{
                f"control_{key}": Output(f"control_inp_{key}", "value")
                for key in control_ann
            },
            **{
                f"object_{key}": Output(f"object_inp_{key}", "value")
                for key in object_ann
            },
            active_tab=Output("input_tabs", "active_tab"),
        ),
        inputs=Input("config_upload", "contents"),
    )
    def config_upload_callback(config_contents: str):
        config: Dict = json.loads(base64.b64decode(config_contents.split(",")[1]))
        result_dict = dict(
            **{f"launch_{key}": no_update for key in launch_ann},
            **{f"loc_{key}": no_update for key in loc_ann},
            **{f"control_{key}": no_update for key in control_ann},
            **{f"object_{key}": no_update for key in object_ann},
            active_tab=no_update,
        )

        if set(config.keys()) == set(loc_ann.keys()):
            result_dict.update({f"loc_{key}": config[key] for key in loc_ann})
            result_dict["active_tab"] = "locator_input"
        elif set(config.keys()) == set(launch_ann.keys()):
            result_dict.update({f"launch_{key}": config[key] for key in launch_ann})
            result_dict["active_tab"] = "launcher_input"
        elif set(config.keys()) == set(control_ann.keys()):
            result_dict.update({f"control_{key}": config[key] for key in control_ann})
            result_dict["active_tab"] = "control_input"
        elif set(config.keys()) == set(object_ann.keys()):
            result_dict.update({f"object_{key}": config[key] for key in object_ann})
            result_dict["active_tab"] = "object_input"
        else:
            launch_diff = set(launch_ann.keys()) ^ set(config.keys())
            loc_diff = set(loc_ann.keys()) ^ set(config.keys())

            raise ValueError(
                "Keys in config don't match expected keys!",
                f"mismatch: {min(launch_diff, loc_diff, key=len)}",
            )

        return result_dict

    @app.callback(
        Output("modal", "is_open", allow_duplicate=True),
        Output("main_graph", "figure"),
        State("coordinate_store", "data"),
        State("image_upload", "contents"),
        State("main_graph", "figure"),
        {key: State(f"loc_inp_{key}", "value") for key in loc_ann},
        {key: State(f"launch_inp_{key}", "value") for key in launch_ann},
        {key: State(f"control_inp_{key}", "value") for key in control_ann},
        {key: State(f"object_inp_{key}", "value") for key in object_ann},
        State("slice_height_2d", "value"),
        State("input_tabs", "active_tab"),
        Input("add_arma_button", "n_clicks"),
    )
    def close_arma_settings(
        coords,
        image_contents,
        fig,
        loc_config,
        launch_config,
        control_config,
        object_config,
        slice_height_2d,
        active_tab,
        *_,
    ):
        if "data" in fig.keys():
            for object in fig["data"]:
                if object.get("hoverinfo") == "text":
                    object["text"] = np.array(object["text"])

        fig = go.Figure(fig)
        if (
            ctx.triggered_id == "add_arma_button"
            or ctx.triggered_id == "close_modal_button"
        ):
            if active_tab == "locator_input":
                arma_type = "alarm"
                config = loc_config
                config = {k: float(v) for k, v in config.items()}
            elif active_tab == "launcher_input":
                arma_type = "fire"
                config = launch_config
                config = {k: float(v) for k, v in config.items()}
            elif active_tab == "control_input":
                arma_type = "control"
                config = control_config
            elif active_tab == "object_input":
                arma_type = "object"
                config = object_config
            elif active_tab == "dron_input":
                arma_type = "dron"
                config = object_config
            else:
                raise ValueError(f"Invalid tab id! {active_tab=}")

            center = json.loads(coords)
            center_pos = center[:-1]

            if image_contents:
                image_file = io.BytesIO(base64.b64decode(image_contents.split(",")[1]))
                image_file.name = "image_file.stl"
                draw_alarm_fire_stl_model(fig, image_file, center, scale_coefs)

            if arma_type == "control" or arma_type == "object":
                return False, fig
            
            logger.info(f"Drawing {arma_type} detection dome...")
            if arma_type == "dron":
                print("center: ", center)
                print("center_pos: ", center_pos)
                add_aircraft(fig, center, scale_coefs=scale_coefs)
                return False, fig

            heights, layers_dots_inner, layers_dots_outer = get_zone(
                zone_type=arma_type,
                azimut_from=config["azimuth_from"] * np.pi / 180,
                azimut_to=config["azimuth_to"] * np.pi / 180,
                center=center_pos,
                heightmap=heightmap,
                bin_size=config["sector_size"] * np.pi / 180,
                height_amount=int(config["height_num"]),
                config=config,
                length_coefficient_x=spacing[0],
                length_coefficient_y=spacing[1],
            )

            color = "magenta" if arma_type == "fire" else "cyan"

            layers_dots = np.concatenate(
                [layers_dots_inner, layers_dots_outer[::-1]], axis=0
            )

            global MODE_2D
            if MODE_2D:
                draw_detection_dome_slices(
                    layers_dots,
                    np.concatenate([heights, heights[::-1]]),
                    fig,
                    arma_type,
                    slice_height_2d,
                )
            else:
                draw_detection_dome(
                    layers_dots, np.concatenate([heights, heights[::-1]]), fig, color
                )

            return False, fig

        return no_update, no_update

    @app.callback(
        Output("main_graph", "figure", allow_duplicate=True),
        State("graph_width", "value"),
        State("graph_height", "value"),
        State("x_ratio", "value"),
        State("y_ratio", "value"),
        State("z_ratio", "value"),
        Input("clean_map", "n_clicks"),
    )
    def clean_map(width, height, x_ratio, y_ratio, z_ratio, _):
        return create_figure(
            heightmap_paths=REPO_ROOT / "3d-python-media/maps/preprocessed",
            lat=ui_config["lat"],
            long=ui_config["long"],
            size_km=ui_config["size_km"],
            map_size_px=ui_config["map_size_px"],
            ratio=(x_ratio, y_ratio, z_ratio),
            width=width,
            height=height,
            mode_2d=MODE_2D,
            cahce_path=cahce_path,
        )[0]

    @app.callback(
        Output("main_graph", "figure", allow_duplicate=True),
        State("graph_width", "value"),
        State("graph_height", "value"),
        State("x_ratio", "value"),
        State("y_ratio", "value"),
        State("z_ratio", "value"),
        Input("2d_mode", "value"),
    )
    def mode(width, height, x_ratio, y_ratio, z_ratio, _):
        global MODE_2D
        MODE_2D = not MODE_2D
        return create_figure(
            heightmap_paths=REPO_ROOT / "3d-python-media/maps/preprocessed",
            lat=ui_config["lat"],
            long=ui_config["long"],
            size_km=ui_config["size_km"],
            map_size_px=ui_config["map_size_px"],
            ratio=(x_ratio, y_ratio, z_ratio),
            width=width,
            height=height,
            mode_2d=MODE_2D,
            cahce_path=cahce_path,
        )[0]

    app.run(debug=True)

def create_body(locator_data_input, launcher_data_input, control_data_input, object_data_input, dron_data_input) -> dbc.ModalBody:
    return dbc.ModalBody(
        [
            dbc.Card(
                [
                    dbc.CardHeader("Параметры техники"),
                    dbc.CardBody(
                        [
                            dbc.Tabs(
                                id="input_tabs",
                                children=[
                                    dbc.Tab(
                                        label="РЛС",
                                        tab_id="locator_input",
                                        children=locator_data_input,
                                    ),
                                    dbc.Tab(
                                        label="ЗРК",
                                        tab_id="launcher_input",
                                        children=launcher_data_input,
                                    ),
                                    dbc.Tab(
                                        label="КП",
                                        tab_id="control_input",
                                        children=control_data_input,
                                    ),
                                    dbc.Tab(
                                        label="ОО",
                                        tab_id="object_input",
                                        children=object_data_input,
                                    ),
                                    dbc.Tab(
                                        label="Дрон",
                                        tab_id="dron_input",
                                        children=dron_data_input,
                                    ),
                                ],
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )

def create_controls() -> dbc.Row:
    return dbc.Row([
        dbc.Row([
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Ширина"), 
                dbc.Input(id="graph_width", type="number", value=DEFAULT_WIDTH)
            ]), md=3),
            
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Высота"),
                dbc.Input(id="graph_height", type="number", value=DEFAULT_HEIGHT)
            ]), md=3),
            
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("2D срез (м)"),
                dbc.Input(id="slice_height_2d", type="number", value=DEFAULT_SLICE_HEIGHT_2D)
            ]), md=3),
            
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Режим"),
                dbc.Switch(id="2d_mode", label="2D/3D", value=False)
            ]), md=3)
        ], className="g-2 mb-3"),
        
        dbc.Row([
            dbc.InputGroupText("Соотношение"),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("X"),
                dbc.Input(id="x_ratio", type="number", value=DEFAULT_RATIO[0])
            ]), md=2),
            
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Y"),
                dbc.Input(id="y_ratio", type="number", value=DEFAULT_RATIO[1])
            ]), md=2),
            
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Z"),
                dbc.Input(id="z_ratio", type="number", value=DEFAULT_RATIO[2])
            ]), md=2),
            
            dbc.Col(dbc.Button("Обновить карту", id="clean_map", color="primary"), md=6)
        ], className="g-2")
    ])
