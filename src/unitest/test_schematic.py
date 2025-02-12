import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from picroute.benchmarks.schematic_with_region import CustomSchematic

sys.path.pop(0)
import gdsfactory as gf

from gdsfactory.generic_tech import get_generic_pdk
import matplotlib.pyplot as plt

gf.config.rich_output()
PDK = get_generic_pdk()


def test_se_load_netlist():
    N = 4
    name = f"clements_{N}x{N}"
    # name = f"mrr_weight_bank_{N}x{N}"
    path = os.path.join(
        os.path.dirname(__file__), f"../picroute/benchmarks/{name}/{name}.yml"
    )
    se = CustomSchematic(path)
    # se.load_netlist(path[:-4] + ".gp.yml")
    se.load_netlist()
    # print(se.schematic)
    layout_filename = path[:-4] + ".layout.yml"
    print(layout_filename)
    se.instantiate_layout(
        output_filename=layout_filename, default_router="get_bundle_all_angle"
    )
    # se.instantiate_layout(output_filename=layout_filename, default_router="get_bundle")
    # se.instantiate_layout(output_filename=layout_filename, default_router="get_bundle_path_length_match")
    c = se.to_component(layout_filename)
    se.show(filename=layout_filename, show_ports=True)
    # c = gf.read.from_yaml(layout_filename)
    # c.show(show_ports=True)
    # se.plot_netlist()
    # plt.show()


if __name__ == "__main__":
    test_se_load_netlist()
