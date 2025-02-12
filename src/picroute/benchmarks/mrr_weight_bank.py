"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-07-16 18:25:30
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-07-16 19:33:30
"""

import os
import sys

import matplotlib.pyplot as plt
from bokeh.io import output_notebook

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import gdsfactory as gf
import numpy as np
from func import (
    add_N_devices,
    add_N_modulators_mzi1x2,
    add_splitter_tree,
    connect_ports,
)
from gdsfactory.config import rich_output
from gdsfactory.generic_tech import get_generic_pdk

from picroute.benchmarks.schematic import CustomSchematic

sys.path.pop(0)
gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

# %env BOKEH_ALLOW_WS_ORIGIN=127.0.0.1:8888,localhost:8888

# output_notebook()
# rich_output()

# %% [markdown]
# First you initialize a session of the schematic editor.
# The editor is synced to a file.
# If file exist, it loads the schematic for editing. If it does not exist, it creates it.
# The schematic file is continuously auto-saved as you edit the schematic in your notebook, so you can track changes with GIT.


# %%
def generate_netlist(N: int = 4, die_area=[1000, 1000]) -> CustomSchematic:
    name = f"mrr_weight_bank_{N}x{N}"
    path = os.path.join(os.path.dirname(__file__), f"{name}/{name}.yml")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    placement_halo = [30, 30, 30, 30]

    se = CustomSchematic(path)
    se.update_settings(design=name, die_area=[[0, 0], die_area], wg_radius=5)
    # cell definition

    # input light GC, fixed cell
    placement_cfg = dict(
        offset=[0, die_area[1] / 2],
        mirror=None,
        pitch=[30, 100],
        placement_halo=placement_halo,
        orient="N",
    )
    gc_names, gc_array_in_ports, _, placement_cfg = add_N_devices(
        se,
        prefix="gc_array_in",
        devices=[gf.components.grating_coupler_elliptical_lumerical()] * N,
        placement_cfg=placement_cfg,
    )
    for idx, gc_name in enumerate(gc_names):
        se.schematic.placements[gc_name].mirror = True
        se.update_placement(
            gc_name,
            placement=[
                "FIXED",
                [0, float(se.schematic.placements[gc_name].y)],
                "FN",
                placement_halo,
            ],
        )

    # four add adrop MRR modulators
    placement_cfg["orient"] = "W"
    (
        mrr_mod_names,
        mrr_mod_array_in_ports,
        mrr_mod_array_out_ports,
        placement_cfg,
    ) = add_N_devices(
        se,
        prefix="mod_array_in",
        devices=[
            gf.components.ring_double_pn(
                add_gap=0.3,
                drop_gap=0.3,
                radius=5.0,
                doping_angle=85,
                doped_heater=True,
                doped_heater_angle_buffer=10,
                doped_heater_layer="NPP",
                doped_heater_width=0.5,
                doped_heater_waveguide_offset=2.175,
            )
        ]
        * N,
        placement_cfg=placement_cfg,
    )
    print(mrr_mod_array_in_ports)
    print(mrr_mod_array_out_ports)
    # exit(0)
    # exit(0)
    """
     4----------3
        ------
        |    |
        ------
     1----------2
        after R90 (W)
        3        2
        | ------ |
        | |    | |
        | |    | |
        | ------ |
        |        |
        4        1 
    """
    for idx, mrr_mod_name in enumerate(mrr_mod_names):
        se.schematic.placements[mrr_mod_name].rotation = 90  # R90
        # se.update_placement(
        #     mrr_mod_name,
        #     placement=[
        #         "UNPLACED",
        #         [0, float(se.schematic.placements[mrr_mod_name].y)],
        #         "W",  # R90
        #         placement_halo,
        #     ],
        # )
    se.add_constraints(
        constraints_cfg=dict(  # it means after flipping, the left side of the object is aligned with the left side of the anchor
            type="alignment", settings={"anchor": "left"}, objects=mrr_mod_names
        )
    )

    placement_cfg["orient"] = "N"
    ## add splitter tree to split signal to N rows
    _, fanout_in_ports, fanout_out_ports, placement_cfg = add_splitter_tree(
        se, prefix="fanout", N=N, placement_cfg=placement_cfg
    )

    ## add N layers of MRR weight banks
    mrr_array_ports_list = [
        gc_array_in_ports,
        mrr_mod_array_in_ports[1::2],
        mrr_mod_array_out_ports[:-2:2],
        mrr_mod_array_in_ports[2::2],
        mrr_mod_array_in_ports[0:1],
        fanout_in_ports,
        fanout_out_ports,
    ]
    mrr_list = []
    for layer in range(N):
        (
            mrr_names,
            mrr_array_in_ports,
            mrr_array_out_ports,
            placement_cfg,
        ) = add_N_devices(
            se,
            prefix=f"mrr_array_{layer}",
            devices=[
                gf.components.ring_single_pn(
                    add_gap=0.3,
                    drop_gap=0.3,
                    radius=5.0,
                    doping_angle=85,
                    doped_heater=True,
                    doped_heater_angle_buffer=10,
                    doped_heater_layer="NPP",
                    doped_heater_width=0.5,
                    doped_heater_waveguide_offset=2.175,
                )
            ]
            * N,
            placement_cfg=placement_cfg,
        )
        mrr_list.append(mrr_names)
        mrr_array_ports_list.append(mrr_array_in_ports)
        mrr_array_ports_list.append(mrr_array_out_ports)

    for row in range(N):
        mrr_names = [mrr_list[col][row] for col in range(N)]
        se.add_constraints(
            constraints_cfg=dict(
                type="alignment", settings={"anchor": "lower"}, objects=mrr_names
            )
        )
    for col in range(N):
        mrr_names = [mrr_list[col][row] for row in range(N)]
        se.add_constraints(
            constraints_cfg=dict(
                type="alignment", settings={"anchor": "left"}, objects=mrr_names
            )
        )

    ## add output gc array
    gc = gf.components.grating_coupler_elliptical_lumerical()
    placement_cfg["offset"][0] = float(
        die_area[0] - gc.size[0] - placement_cfg["pitch"][0]
    )
    gc_names, gc_array_in_ports, _, _ = add_N_devices(
        se,
        prefix="gc_array_out",
        devices=[gf.components.grating_coupler_elliptical_lumerical()] * N,
        placement_cfg=placement_cfg,
    )
    for idx, gc_name in enumerate(gc_names):
        se.update_placement(
            gc_name,
            placement=[
                "FIXED",
                [
                    float(die_area[0] - gc.size[0]),
                    float(se.schematic.placements[gc_name].y),
                ],
                "N",
                placement_halo,
            ],
        )
    mrr_array_ports_list.append(gc_array_in_ports)

    ### connection definition
    print("ports to be connected")
    for ports in mrr_array_ports_list:
        print(ports)
    ## input gc to input mrr mod
    # connect_ports(se, ports1=gc_array_in_ports, ports2=mrr_mod_array_in_ports[1::2]) # port 4 for add-drop mrr

    ## connect mrr mods as a WDM MUX BUS
    # connect_ports(se, ports1=mrr_mod_array_out_ports[:-1:2], ports2=mrr_mod_array_in_ports[::2]) # 2->1, 2->1, 2->1

    for layer in range(0, len(mrr_array_ports_list), 2):
        ports1 = mrr_array_ports_list[layer]
        ports2 = mrr_array_ports_list[layer + 1]
        unconnected_ports1, unconnected_ports2 = connect_ports(
            se, ports1=ports1, ports2=ports2
        )
        for i, port in enumerate(unconnected_ports1):
            if port is not None:
                mrr_array_ports_list[layer + 2][i] = port
            if unconnected_ports2[i] is not None:
                raise ValueError

    # se.schematic.placements["gc1"].mirror = True
    # se.schematic.placements["mzi_in_1"].x = 50
    # se.schematic.placements["mzi_in_1"].y = 50

    # se.schematic.placements["mzi_in_2"].x = 50
    # se.schematic.placements["mzi_in_2"].y = -50

    # se.schematic.placements["mzi_11"].x = 200
    # se.schematic.placements["gc2"].x = 500
    # se.schematic.placements["gc2"].y = 50
    # se.schematic.placements["gc3"].x = 500
    # se.schematic.placements["gc3"].y = -50
    # se.schematic.placements["mzi_in_2"].mirror=True
    # print(se.get_netlist())

    # print(dir(se.schematic.instances["fanout_yb_1_1"]))
    # print(se.schematic.instances["fanout_yb_1_1"].dict)
    # print(dir(mzi))
    se.commit()
    # se.plot_netlist()
    # plt.show()
    return se
    # print(se.port_widget)


if __name__ == "__main__":
    generate_netlist(4, die_area=[800, 800])
    generate_netlist(8, die_area=[1600, 1600])
    generate_netlist(16, die_area=[3200, 3200])
