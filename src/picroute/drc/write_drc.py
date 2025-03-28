"""Write DRC rule decks in KLayout.

TODO:
- define derived layers (composed rules)

More DRC examples:
- https://www.klayout.de/doc-qt5/about/drc_ref.html
- http://klayout.de/doc/manual/drc_basic.html
- https://github.com/usnistgov/SOEN-PDK/tree/master/tech/OLMAC
- https://github.com/google/globalfoundries-pdk-libs-gf180mcu_fd_pr/tree/main/rules/klayout
"""

from __future__ import annotations

import pathlib
from dataclasses import asdict, is_dataclass

import gdsfactory as gf
from gdsfactory.install import get_klayout_path
from gdsfactory.typings import CrossSectionSpec, Layer, PathType

layer_name_to_min_width: dict[str, float]


def get_drc_script_start(name, shortcut) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<klayout-macro>
 <description>{name} DRC</description>
 <version/>
 <category>drc</category>
 <prolog/>
 <epilog/>
 <doc/>
 <autorun>false</autorun>
 <autorun-early>false</autorun-early>
 <shortcut>{shortcut}</shortcut>
 <show-in-menu>true</show-in-menu>
 <group-name>drc_scripts</group-name>
 <menu-path>tools_menu.drc.end</menu-path>
 <interpreter>dsl</interpreter>
 <dsl-interpreter-name>drc-dsl-xml</dsl-interpreter-name>
 <text># {name} DRC

# Read about Klayout DRC scripts in the User Manual under "Design Rule Check (DRC)"
# Based on https://gdsfactory.github.io/gdsfactory/notebooks/_2_klayout.html#Klayout-DRC
# and https://gdsfactory.github.io/gdsfactory/api.html#klayout-drc

report("{name} DRC")
time_start = Time.now
"""


drc_script_end = r"""
time_end = Time.now
print "run time #{(time_end-time_start).round(3)} seconds \n"
</text>
</klayout-macro>
"""


def rule_not_inside(layer: str, not_inside: str) -> str:
    """Checks for that a layer is not inside another layer."""
    error = f"{layer} not inside {not_inside}"
    return f"{layer}.not_inside({not_inside})" f".output({error!r}, {error!r})"


def rule_width(value: float, layer: str, angle_limit: float = 90) -> str:
    """Min feature size."""
    category = "width"
    error = f"{layer} {category} {value}um"
    return (
        f"{layer}.{category}({value}, angle_limit({angle_limit}))"
        f".output({error!r}, {error!r})"
    )


def rule_space(value: float, layer: str, angle_limit: float = 90) -> str:
    """Min Space between shapes of layer."""
    category = "space"
    error = f"{layer} {category} {value}um"
    return (
        f"{layer}.{category}({value}, angle_limit({angle_limit}))"
        f".output({error!r}, {error!r})"
    )


def rule_separation(value: float, layer1: str, layer2: str) -> str:
    """Min space between different layers."""
    error = f"min {layer1} {layer2} separation {value}um"
    return f"{layer1}.separation({layer2}, {value}).output({error!r}, {error!r})"


def rule_enclosing(
    value: float, layer1: str, layer2: str, angle_limit: float = 90
) -> str:
    """Checks if layer1 encloses (is bigger than) layer2 by value."""
    error = f"{layer1} enclosing {layer2} by {value}um"
    return (
        f"{layer1}.enclosing({layer2}, angle_limit({angle_limit}), {value})"
        f".output({error!r}, {error!r})"
    )


def rule_area(layer: str, min_area_um2: float = 2.0) -> str:
    """Return script for min area checking."""
    return f"""

min_{layer}_a = {min_area_um2}.um2
r_{layer}_a = {layer}.with_area(0, min_{layer}_a)
r_{layer}_a.output("{layer.upper()}_A: {layer} area &lt; min_{layer}_a um2")
"""


def rule_density(
    layer: str = "metal1",
    layer_floorplan: str = "FLOORPLAN",
    min_density=0.2,
    max_density=0.8,
) -> str:
    """Return script to ensure density of layer is within min and max.

    based on https://github.com/klayoutmatthias/si4all

    """
    return f"""
min_density = {min_density}
max_density = {max_density}

area = {layer}.area
border_area = {layer_floorplan}.area
if border_area &gt;= 1.dbu * 1.dbu

  r_min_dens = polygon_layer
  r_max_dens = polygon_layer

  dens = area / border_area

  if dens &lt; min_density
    # copy border as min density marker
    r_min_dens = {layer_floorplan}
  end

  if dens &gt; max_density
    # copy border as max density marker
    r_max_dens = {layer_floorplan}
  end

  r_min_dens.output("{layer}_Xa: {layer} density below threshold of {min_density}")
  r_max_dens.output("{layer}: {layer} density above threshold of {max_density}")

end

"""


def connectivity_checks(
    WG_cross_sections: list[CrossSectionSpec], pin_widths: list[float] | float
) -> str:
    """Return script for photonic port connectivity check. Assumes the photonic port pins are inside the Component.

    Args:
        WG_cross_sections: list of waveguide layers to run check for.
        pin_widths: list of port pin widths or a single port pin width/
    """
    connectivity_check = ""
    for i, layer_name in enumerate(WG_cross_sections):
        layer = gf.pdk.get_cross_section(layer_name).width
        layer_name = gf.pdk.get_cross_section(layer_name).layer
        connectivity_check = connectivity_check.join(
            f"""{layer_name}_PIN2 = {layer_name}_PIN.sized(0.0).merged\n
{layer_name}_PIN2 = {layer_name}_PIN2.rectangles.without_area({layer} * {pin_widths if isinstance(pin_widths, float) else pin_widths[i]}) - {layer_name}_PIN2.rectangles.with_area({layer} * 2 * {pin_widths if isinstance(pin_widths, float) else pin_widths[i]})\n
{layer_name}_PIN2.output(\"port alignment error\")\n
{layer_name}_PIN2 = {layer_name}_PIN.sized(0.0).merged\n
{layer_name}_PIN2.non_rectangles.output(\"port width check\")\n\n"""
        )

    return connectivity_check


def write_layer_definition(layers: dict[str, Layer]) -> list[str]:
    """Returns layers definition script for KLayout.

    Args:
        layers: layer definitions can be dict, dataclass or pydantic BaseModel.

    """
    layers = asdict(layers) if is_dataclass(layers) else layers
    layers = dict(layers)
    return [f"{key} = input({value[0]}, {value[1]})" for key, value in layers.items()]


def write_drc_deck(rules: list[str], layers: dict[str, Layer] | None = None) -> str:
    """Returns drc_rule_deck for KLayout.

    based on https://github.com/klayoutmatthias/si4all

    Args:
        rules: list of rules.
        layers: layer definitions can be dict, dataclass or pydantic BaseModel.

    """
    script = []
    if layers:
        script += write_layer_definition(layers=layers)
    script += ["\n"]
    script += rules
    return "\n".join(script)


modes = ["tiled", "default", "deep"]


def write_drc_deck_macro(
    rules: list[str],
    layers: dict[str, Layer] | None = None,
    name: str = "generic",
    filepath: PathType | None = None,
    shortcut: str = "Ctrl+Shift+D",
    mode: str = "tiled",
    threads: int = 4,
    tile_size: int = 500,
    tile_borders: int | None = None,
) -> str:
    """Write KLayout DRC macro.

    You can customize the shortcut to run the DRC macro from the Klayout GUI.

    Args:
        rules: list of rules.
        layers: layer definitions can be dict or dataclass.
        name: drc rule deck name.
        filepath: Optional macro path (defaults to .klayout/drc/name.lydrc).
        shortcut: to run macro from KLayout GUI.
        mode: tiled, default or deep (hierarchical).
        threads: number of threads.
        tile_size: in um for tile mode.
        tile_borders: sides for each. Defaults None to automatic.

    modes:

    - default
        - flat polygon handling
        - single threaded
        - no overhead
        - use for small layouts
        - no side effects
    - tiled
        - need to optimize tile size (maybe 500x500um). Works of each tile individually.
        - finite lookup range
        - output is flat
        - multithreading enable
        - scales with number of CPUs
        - scales with layout area
        - predictable runtime and and memory footprint
    - deep
        - hierarchical mode
        - preserves hierarchy in many cases
        - does not predictably scale with number of CPUs
        - experimental (either very fast of very slow)
        - mainly used for LVS layer preparation

    Klayout supports to switch modes and tile parameters during execution.
    However this function does support switching modes.

    .. code::

        import gdsfactory as gf
        from gdsfactory.geometry.write_drc import (
            write_drc_deck_macro,
            rule_enclosing,
            rule_width,
            rule_space,
            rule_separation,
            rule_area,
            rule_density,
        )
        rules = [
            rule_width(layer="WG", value=0.2),
            rule_space(layer="WG", value=0.2),
            rule_separation(layer1="HEATER", layer2="M1", value=1.0),
            rule_enclosing(layer1="VIAC", layer2="M1", value=0.2),
            rule_area(layer="WG", min_area_um2=0.05),
            rule_density(
                layer="WG", layer_floorplan="FLOORPLAN", min_density=0.5, max_density=0.6
            ),
            rule_not_inside(layer="VIAC", not_inside="NPP"),
        ]

        drc_rule_deck = write_drc_deck_macro(rules=rules, layers=gf.LAYER, mode="tiled")
        print(drc_rule_deck)

    """
    if mode not in modes:
        raise ValueError(f"{mode!r} not in {modes}")

    script = get_drc_script_start(name=name, shortcut=shortcut)

    if mode == "tiled":
        script += f"""
threads({threads})
tiles({tile_size})
"""
        if tile_borders:
            script += f"""
tile_borders({tile_borders})
"""
    elif mode == "deep":
        script += """
deep
"""

    script += write_drc_deck(rules=rules, layers=layers)

    script += drc_script_end
    filepath = filepath or get_klayout_path() / "drc" / f"{name}.lydrc"
    filepath = pathlib.Path(filepath)
    dirpath = filepath.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    filepath = pathlib.Path(filepath)
    filepath.write_text(script, encoding="UTF-8")
    print(f"Wrote DRC deck to {str(filepath)!r} with shortcut {shortcut!r}")
    return script


from gdsfactory.generic_tech import LAYER

if __name__ == "__main__":
    rules = [
        rule_width(layer="WG", value=0.5),
        rule_space(layer="WG", value=1),
        rule_space(layer="M1", value=4),
        rule_separation(layer1="WG", layer2="WG", value=2),
        # rule_separation(layer1="HEATER", layer2="M1", value=1.0),
        # rule_enclosing(layer1="VIAC", layer2="M1", value=0.2),
        # rule_area(layer="WG", min_area_um2=0.05),
        # rule_not_inside(layer="VIAC", not_inside="NPP"),
    ]

    layers = LAYER.dict()
    layers.update({"WG_PIN": (1, 10)})

    drc_rule_deck = write_drc_deck_macro(rules=rules, layers=layers, mode="tiled")
    print(drc_rule_deck)
