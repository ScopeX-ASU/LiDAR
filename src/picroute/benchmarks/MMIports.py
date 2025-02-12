"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-07-16 18:25:30
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-07-16 19:33:30
"""

import os
import sys

# import matplotlib.pyplot as plt
# from bokeh.io import output_notebook

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import gdsfactory as gf
import numpy as np
from func import (
    add_N_devices,
    add_splitter_tree,
    connect_crossing_ports,
    connect_ports,
    generate_matrix,
)
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
def generate_netlist(N: int = 4, die_area=[1000, 1000], seed=4321) -> CustomSchematic:
    name = f"multiportmmi_{N}x{N}"
    path = os.path.join(os.path.dirname(__file__), f"{name}/{name}.yml")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    placement_halo = [10, 10, 10, 10]

    se = CustomSchematic(path)
    se.update_settings(design=name, die_area=[[0, 0], die_area], wg_radius=5)
    # cell definition
    assert 2 ** int(np.log2(N)) == N, (
        f"Only support input size to be a power of 2, but got {N}"
    )

    # gf8
    y_offset = gf.components.grating_coupler_elliptical_lumerical().dymin
    # y_offset  = float(gf.components.grating_coupler_elliptical_lumerical().bbox()[0][1])
    # input light GC, fixed cell
    se.add_instance(
        "gc1", gf.components.grating_coupler_elliptical_lumerical(), iloss=2.0
    )
    se.schematic.placements["gc1"].mirror = True
    se.update_placement(
        "gc1",
        placement=["FIXED", [0, die_area[1] / 2 + y_offset], "FN", placement_halo],
    )

    placement_cfg = dict(
        offset=[80, die_area[1] / 2],
        mirror=None,
        pitch=[50, 100],
        orient="N",
        placement_halo=placement_halo,
    )

    _, fanout_in_ports, fanout_out_ports, placement_cfg = add_splitter_tree(
        se, prefix="fanout", N=N, placement_cfg=placement_cfg
    )

    crossing_layer = set()
    count = 0
    wg_type = gf.components.straight
    mzi_type = gf.components.mzi_phase_shifter
    phase_shifter = gf.components.straight_heater_metal_undercut

    ports_list = [
        [("gc1", "o1")],
        fanout_in_ports,
        fanout_out_ports,
    ]
    count += 3

    pitch_x = int(120 * N / 8)
    placement_cfg["pitch"] = [pitch_x, 100]

    devices = [mzi_type()] * N
    device_names, mzi_array_in_ports, mzi_array_out_ports, placement_cfg, _ = (
        add_N_devices(
            se,
            prefix=f"mol_array_{0}",
            devices=devices,
            placement_cfg=placement_cfg,
        )
    )

    ports_list.append(mzi_array_in_ports)
    ports_list.append(mzi_array_out_ports)
    count += 2
    se.add_constraints(
        constraints_cfg=dict(
            type="alignment", settings={"anchor": "left"}, objects=device_names
        )
    )

    device_names, count, placement_cfg = generate_matrix(
        se,
        ports_list,
        count,
        crossing_layer,
        N,
        seed,
        prefix="mmi0",
        placement_cfg=placement_cfg,
    )

    devices = [mzi_type()] * N
    device_names, mzi_array_in_ports, mzi_array_out_ports, placement_cfg, _ = (
        add_N_devices(
            se,
            prefix=f"mol_array_{1}",
            devices=devices,
            placement_cfg=placement_cfg,
        )
    )

    ports_list.append(mzi_array_in_ports)
    ports_list.append(mzi_array_out_ports)
    count += 2

    se.add_constraints(
        constraints_cfg=dict(
            type="alignment", settings={"anchor": "left"}, objects=device_names
        )
    )

    seed += 42
    device_names, count, placement_cfg = generate_matrix(
        se,
        ports_list,
        count,
        crossing_layer,
        N,
        seed,
        prefix="mmi1",
        placement_cfg=placement_cfg,
    )

    placement_cfg["pitch"] = [60 + 2 * pitch_x, 100]

    gc = gf.components.grating_coupler_elliptical_lumerical()
    # placement_cfg["offset"][0] = float(
    #     die_area[0] - gc.size[0] - placement_cfg["pitch"][0]
    # )
    gc_names, gc_array_in_ports, _, _, _ = add_N_devices(
        se,
        prefix="gc_array_out",
        devices=[gf.components.grating_coupler_elliptical_lumerical()] * N,
        placement_cfg=placement_cfg,
    )
    # for idx, gc_name in enumerate(gc_names):
    #     se.update_placement(
    #         gc_name,
    #         placement=[
    #             "FIXED",
    #             [
    #                 float(die_area[0] - gc.size[0]),
    #                 float(se.schematic.placements[gc_name].y),
    #             ],
    #             "N",
    #             placement_halo,
    #         ],
    #     )
    ports_list.append(gc_array_in_ports)
    count += 1

    print(ports_list)
    for layer in range(0, len(ports_list), 2):
        ports1 = ports_list[layer]
        ports2 = ports_list[layer + 1]
        if layer in crossing_layer:
            seed += 42
            unconnected_ports1, unconnected_ports2 = connect_crossing_ports(
                se, ports1=ports1, ports2=ports2, target_inversions=6, seed=seed
            )
        else:
            unconnected_ports1, unconnected_ports2 = connect_ports(
                se, ports1=ports1, ports2=ports2
            )

    se.commit()

    # se.plot_netlist()
    # plt.show()
    return se
    # print(se.port_widget)


if __name__ == "__main__":
    # generate_netlist(4, die_area=[1600, 800])
    # generate_netlist(8, die_area=[4520, 1600], seed=4321)
    # generate_netlist(16, die_area=[6910, 3200], seed=1234)
    generate_netlist(32, die_area=[13000, 6400], seed=1234)
    # generate_netlist(16, die_area=[6400, 3200])
