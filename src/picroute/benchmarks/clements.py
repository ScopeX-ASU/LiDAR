"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-07-16 18:25:30
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-07-16 19:33:30
"""

import os
import sys


# from bokeh.io import output_notebook

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
    name = f"clements_{N}x{N}"
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
    # input light GC, fixed cell
    se.add_instance(
        "gc1", gf.components.grating_coupler_elliptical_lumerical(), iloss=2.0
    )
    se.schematic.placements["gc1"].mirror = True

    se.update_placement(
        "gc1", placement=["FIXED", [0, die_area[1] / 2], "FN", placement_halo]
    )

    placement_cfg = dict(
        offset=[100, die_area[1] / 2],
        mirror=None,
        pitch=[50, 100],
        orient="N",
        placement_halo=placement_halo,
    )

    _, fanout_in_ports, fanout_out_ports, placement_cfg = add_splitter_tree(
        se, prefix="fanout", N=N, placement_cfg=placement_cfg
    )

    placement_cfg["pitch"] = [100, 100]
    mod_names, mod_in_ports, mod_out_ports, placement_cfg = add_N_modulators_mzi1x2(
        se, prefix="mod", N=N, placement_cfg=placement_cfg
    )
    se.add_constraints(
        constraints_cfg=dict(
            type="alignment", settings={"anchor": "left"}, objects=mod_names
        )
    )

    placement_cfg["pitch"] = [80, 100]
    wg_type = gf.components.straight
    mzi_type = gf.components.mzi2x2_2x2_phase_shifter
    mzi = mzi_type()

    mzi_array_ports_list = [
        [("gc1", "o1")],
        fanout_in_ports,
        fanout_out_ports,
        mod_in_ports,
        mod_out_ports,
    ]
    io_list = []

    for layer in range(N):
        if layer % 2 == 0:  # even index layer
            devices = [mzi_type()] * (N // 2)
            (
                device_names,
                mzi_array_in_ports,
                mzi_array_out_ports,
                placement_cfg,
                electrical_ports,
            ) = add_N_devices(
                se,
                prefix=f"mzi_array_{layer}",
                devices=devices,
                placement_cfg=placement_cfg,
            )

            if N % 2 == 1:  # N is odd number
                # devices.append(wg_type(length=float(mzi.size[0])))
                mzi_array_in_ports.append(None)
                mzi_array_out_ports.append(None)
        else:
            devices = [mzi_type()] * ((N - 1) // 2)
            (
                device_names,
                mzi_array_in_ports,
                mzi_array_out_ports,
                placement_cfg,
                electrical_ports,
            ) = add_N_devices(
                se,
                prefix=f"mzi_array_{layer}",
                devices=devices,
                placement_cfg=placement_cfg,
            )
            if N % 2 == 0:
                mzi_array_in_ports.append(None)
                mzi_array_in_ports.insert(0, None)
                mzi_array_out_ports.append(None)
                mzi_array_out_ports.insert(0, None)
                # devices = [wg_type(length=float(mzi.size[0]))] + [mzi_type()] * ((N - 1) // 2) + [wg_type(length=float(mzi.size[0]))]
            else:
                mzi_array_in_ports.insert(0, None)
                mzi_array_out_ports.insert(0, None)
                # devices = [wg_type(length=float(mzi.size[0]))] + [mzi_type()] * ((N - 1) // 2)

        mzi_array_ports_list.append(mzi_array_in_ports)
        mzi_array_ports_list.append(mzi_array_out_ports)
        se.add_constraints(
            constraints_cfg=dict(
                type="alignment", settings={"anchor": "left"}, objects=device_names
            )
        )
        io_list.extend(electrical_ports)

    gc = gf.components.grating_coupler_elliptical_lumerical()
    placement_cfg["offset"][0] = float(
        die_area[0] - gc.xsize - placement_cfg["pitch"][0]
    )
    gc_names, gc_array_in_ports, _, _, _ = add_N_devices(
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
                    float(die_area[0] - gc.xsize),
                    float(se.schematic.placements[gc_name].y),
                ],
                "N",
                placement_halo,
            ],
        )
    mzi_array_ports_list.append(gc_array_in_ports)

    print(mzi_array_ports_list)
    for layer in range(0, len(mzi_array_ports_list), 2):
        ports1 = mzi_array_ports_list[layer]
        ports2 = mzi_array_ports_list[layer + 1]
        unconnected_ports1, unconnected_ports2 = connect_ports(
            se, ports1=ports1, ports2=ports2
        )
        for i, port in enumerate(unconnected_ports1):
            if port is not None:
                mzi_array_ports_list[layer + 2][i] = port
            if unconnected_ports2[i] is not None:
                raise ValueError

    se.commit()

    # se.plot_netlist()
    # plt.show()
    return se
    # print(se.port_widget)


if __name__ == "__main__":
    # generate_netlist(4, die_area=[1600, 800])
    # generate_netlist(8, die_area=[4800, 1600])
    generate_netlist(16, die_area=[8000, 3200])
