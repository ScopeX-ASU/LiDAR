"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-07-16 18:31:58
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-07-16 18:43:06
"""

import random
from copy import deepcopy
from typing import List

import gdsfactory as gf
import numpy as np
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

from gdsfactory.components.mmis.mmi import mmi
from gdsfactory.components.mzis.mzi import mzi1x2

__all__ = [
    "add_splitter_tree",
    "add_N_modulators_mzi1x2",
    "connect_ports",
    "add_N_devices",
]


def add_splitter_tree(
    se,
    prefix: str = "fanout",
    N: int = 4,
    placement_cfg=dict(offset=(0, 0), mirror=None, pitch=(30, 100)),
) -> None:
    # cell definition
    assert 2 ** int(np.log2(N)) == N, (
        f"Only support input size to be a power of 2, but got {N}"
    )
    # splitter tree
    yb_prefix = f"{prefix}_yb"
    yb_name = f"{yb_prefix}_0_0"
    se.add_instance(yb_name, gf.components.mmi1x2(), iloss=0.3)
    input_ports = [(yb_name, "o1")]
    output_ports = []
    device = gf.components.mmi1x2()
    # y_offset = float(device.bbox[0][1])
    y_offset = device.ymin
    device_names = []
    se.schematic.placements[yb_name].x = placement_cfg["offset"][0]
    se.schematic.placements[yb_name].y = placement_cfg["offset"][1] + y_offset
    se.schematic.instances[yb_name].settings["placement"][1] = [
        se.schematic.placements[yb_name].x,
        se.schematic.placements[yb_name].y,
    ]
    se.schematic.instances[yb_name].settings["placement"][3] = placement_cfg[
        "placement_halo"
    ]
    for level in range(1, int(np.log2(N))):
        yb_groups = []
        for branch in range(2**level):
            yb_name = f"{yb_prefix}_{level}_{branch}"
            se.add_instance(yb_name, gf.components.mmi1x2(), iloss=0.3)
            device_names.append(yb_name)
            se.add_net(
                inst1=f"{yb_prefix}_{level - 1}_{branch // 2}",
                port1=["o3", "o2"][branch % 2],
                inst2=yb_name,
                port2="o1",
            )
            if level == int(np.log2(N)) - 1:
                output_ports.append((yb_name, "o3"))
                output_ports.append((yb_name, "o2"))

            se.schematic.placements[yb_name].x = placement_cfg["offset"][
                0
            ] + placement_cfg["pitch"][0] * (level + 1)
            se.schematic.placements[yb_name].y = (
                placement_cfg["offset"][1]
                + placement_cfg["pitch"][1] * (branch - 2**level / 2 + 0.5)
                + y_offset
            )
            se.schematic.instances[yb_name].settings["placement"][1] = [
                se.schematic.placements[yb_name].x,
                se.schematic.placements[yb_name].y,
            ]
            yb_groups.append(yb_name)
            se.schematic.instances[yb_name].settings["placement"][3] = placement_cfg[
                "placement_halo"
            ]
        se.add_constraints(
            constraints_cfg=dict(
                type="alignment", settings={"anchor": "left"}, objects=yb_groups
            )
        )

        placement_offset = [
            float(se.schematic.placements[yb_name].x + device.xsize),
            placement_cfg["offset"][1],
        ]
        placement_cfg["offset"] = placement_offset

    return device_names, input_ports, output_ports, placement_cfg


def add_N_modulators_mzi1x2(
    se,
    prefix: str = "mod",
    N: int = 4,
    placement_cfg=dict(offset=(0, 0), mirror=None, pitch=(30, 100)),
):
    mzi_prefix = f"{prefix}_mzi1x2"
    input_ports = []
    output_ports = []
    device = mzi1x2()
    device_names = []
    for i in range(N):
        mzi_name = f"{mzi_prefix}_{i}"
        se.add_instance(mzi_name, mzi1x2(), iloss=1.2)
        device_names.append(mzi_name)
        input_ports.append((mzi_name, "o1"))
        output_ports.append((mzi_name, "o2"))

        se.schematic.placements[mzi_name].x = (
            placement_cfg["offset"][0] + placement_cfg["pitch"][0]
        )
        se.schematic.placements[mzi_name].y = placement_cfg["offset"][
            1
        ] + placement_cfg["pitch"][1] * (i - N / 2 + 0.5)
        se.schematic.instances[mzi_name].settings["placement"][1] = [
            se.schematic.placements[mzi_name].x,
            se.schematic.placements[mzi_name].y,
        ]
        se.schematic.instances[mzi_name].settings["placement"][3] = placement_cfg[
            "placement_halo"
        ]

    placement_offset = [
        float(se.schematic.placements[mzi_name].x + device.xsize),
        placement_cfg["offset"][1],
    ]
    placement_cfg["offset"] = placement_offset

    return device_names, input_ports, output_ports, placement_cfg


def add_multiport_mmi(
    se,
    prefix: str = "mmi",
    N: int = 4,
    placement_cfg=dict(offset=(0, 0), mirror=None, pitch=(30, 100)),
    type=1,
):
    if type == 1:
        port_num = int(N / 2)
        length_mmi = 36 * port_num / 2
        width_mmi = 10 * port_num / 2
        device = mmi(
            inputs=port_num,
            outputs=port_num,
            length_mmi=length_mmi,
            width_mmi=width_mmi,
            gap_input_tapers=4,
            gap_output_tapers=4,
        )
        input_ports = []
        output_ports = []
        device_names = []

        mmi_name = f"{prefix}_multiport_{0}"
        se.add_instance(
            mmi_name,
            mmi(
                inputs=port_num,
                outputs=port_num,
                length_mmi=length_mmi,
                width_mmi=width_mmi,
                gap_input_tapers=4,
                gap_output_tapers=4,
            ),
            iloss=0.1,
        )
        device_names.append(mmi_name)
        input_ports.extend(
            [
                (mmi_name, p[0].name)
                for p in sorted(
                    [
                        (port, port.center[1])
                        for port in device.get_ports_list(
                            port_type="optical", orientation=180.0
                        )
                    ],
                    key=lambda x: x[1],
                )
            ]
        )
        output_ports.extend(
            [
                (mmi_name, p[0].name)
                for p in sorted(
                    [
                        (port, port.center[1])
                        for port in device.get_ports_list(
                            port_type="optical", orientation=0.0
                        )
                    ],
                    key=lambda x: x[1],
                )
            ]
        )

        se.schematic.placements[mmi_name].x = (
            placement_cfg["offset"][0] + placement_cfg["pitch"][0]
        )
        se.schematic.placements[mmi_name].y = placement_cfg["offset"][
            1
        ] + placement_cfg["pitch"][1] * (-port_num / 2)
        se.schematic.instances[mmi_name].settings["placement"][1] = [
            se.schematic.placements[mmi_name].x,
            se.schematic.placements[mmi_name].y,
        ]
        se.schematic.instances[mmi_name].settings["placement"][3] = placement_cfg[
            "placement_halo"
        ]

        mmi_name = f"{prefix}_multiport_{1}"
        se.add_instance(
            mmi_name,
            mmi(
                inputs=port_num,
                outputs=port_num,
                length_mmi=length_mmi,
                width_mmi=width_mmi,
                gap_input_tapers=4,
                gap_output_tapers=4,
            ),
            iloss=0.1,
        )
        device_names.append(mmi_name)
        input_ports.extend(
            [
                (mmi_name, p[0].name)
                for p in sorted(
                    [
                        (port, port.center[1])
                        for port in device.get_ports_list(
                            port_type="optical", orientation=180.0
                        )
                    ],
                    key=lambda x: x[1],
                )
            ]
        )
        output_ports.extend(
            [
                (mmi_name, p[0].name)
                for p in sorted(
                    [
                        (port, port.center[1])
                        for port in device.get_ports_list(
                            port_type="optical", orientation=0.0
                        )
                    ],
                    key=lambda x: x[1],
                )
            ]
        )

        se.schematic.placements[mmi_name].x = (
            placement_cfg["offset"][0] + placement_cfg["pitch"][0]
        )
        se.schematic.placements[mmi_name].y = placement_cfg["offset"][
            1
        ] + placement_cfg["pitch"][1] * (port_num / 2)
        se.schematic.instances[mmi_name].settings["placement"][1] = [
            se.schematic.placements[mmi_name].x,
            se.schematic.placements[mmi_name].y,
        ]
        se.schematic.instances[mmi_name].settings["placement"][3] = placement_cfg[
            "placement_halo"
        ]

        placement_offset = [
            float(se.schematic.placements[mmi_name].x + device.size[0]),
            placement_cfg["offset"][1],
        ]
        placement_cfg["offset"] = placement_offset
    else:
        input_ports = []
        output_ports = []
        device_names = []
        port_num = int(N / 4 * 3)
        length_mmi = 36 * port_num / 2
        width_mmi = 10 * port_num / 2
        device = mmi(
            inputs=port_num,
            outputs=port_num,
            length_mmi=length_mmi,
            width_mmi=width_mmi,
            gap_input_tapers=4,
            gap_output_tapers=4,
        )

        mmi_name = f"{prefix}_multiport_{1}"
        se.add_instance(
            mmi_name,
            mmi(
                inputs=port_num,
                outputs=port_num,
                length_mmi=length_mmi,
                width_mmi=width_mmi,
                gap_input_tapers=4,
                gap_output_tapers=4,
            ),
            iloss=0.1,
        )
        device_names.append(mmi_name)
        input_ports.extend(
            [
                (mmi_name, p[0].name)
                for p in sorted(
                    [
                        (port, port.center[1])
                        for port in device.get_ports_list(
                            port_type="optical", orientation=180.0
                        )
                    ],
                    key=lambda x: x[1],
                )
            ]
        )
        output_ports.extend(
            [
                (mmi_name, p[0].name)
                for p in sorted(
                    [
                        (port, port.center[1])
                        for port in device.get_ports_list(
                            port_type="optical", orientation=0.0
                        )
                    ],
                    key=lambda x: x[1],
                )
            ]
        )

        se.schematic.placements[mmi_name].x = (
            placement_cfg["offset"][0] + placement_cfg["pitch"][0]
        )
        se.schematic.placements[mmi_name].y = (
            placement_cfg["offset"][1] - placement_cfg["pitch"][1] * (N - port_num) / 2
        )
        se.schematic.instances[mmi_name].settings["placement"][1] = [
            se.schematic.placements[mmi_name].x,
            se.schematic.placements[mmi_name].y,
        ]
        se.schematic.instances[mmi_name].settings["placement"][3] = placement_cfg[
            "placement_halo"
        ]

        port_num = int(N / 4)
        length_mmi = 36 * port_num / 2
        width_mmi = 10 * port_num / 2
        device = mmi(
            inputs=port_num,
            outputs=port_num,
            length_mmi=length_mmi,
            width_mmi=width_mmi,
            gap_input_tapers=4,
            gap_output_tapers=4,
        )

        mmi_name = f"{prefix}_multiport_{0}"
        se.add_instance(
            mmi_name,
            mmi(
                inputs=port_num,
                outputs=port_num,
                length_mmi=length_mmi,
                width_mmi=width_mmi,
                gap_input_tapers=4,
                gap_output_tapers=4,
            ),
            iloss=0.1,
        )
        device_names.append(mmi_name)
        input_ports.extend(
            [
                (mmi_name, p[0].name)
                for p in sorted(
                    [
                        (port, port.center[1])
                        for port in device.get_ports_list(
                            port_type="optical", orientation=180.0
                        )
                    ],
                    key=lambda x: x[1],
                )
            ]
        )
        output_ports.extend(
            [
                (mmi_name, p[0].name)
                for p in sorted(
                    [
                        (port, port.center[1])
                        for port in device.get_ports_list(
                            port_type="optical", orientation=0.0
                        )
                    ],
                    key=lambda x: x[1],
                )
            ]
        )

        se.schematic.placements[mmi_name].x = (
            placement_cfg["offset"][0] + placement_cfg["pitch"][0]
        )
        se.schematic.placements[mmi_name].y = placement_cfg["offset"][
            1
        ] + placement_cfg["pitch"][1] * (port_num * 1.5)
        se.schematic.instances[mmi_name].settings["placement"][1] = [
            se.schematic.placements[mmi_name].x,
            se.schematic.placements[mmi_name].y,
        ]
        se.schematic.instances[mmi_name].settings["placement"][3] = placement_cfg[
            "placement_halo"
        ]

        placement_offset = [
            float(se.schematic.placements[mmi_name].x + device.size[0]),
            placement_cfg["offset"][1],
        ]
        placement_cfg["offset"] = placement_offset

    return device_names, input_ports, output_ports, placement_cfg


def add_N_devices(
    se,
    prefix: str = "device_array",
    devices: List = [gf.components.mzi2x2_2x2_phase_shifter()] * 2,
    placement_cfg=dict(
        offset=(0, 0),
        mirror=None,
        pitch=(30, 100),
        orient="N",
        placement_halo=[0, 0, 0, 0],
    ),
):
    input_ports = []
    output_ports = []
    electrical_ports = []
    device_names = []
    # y_offset = float(devices[0].bbox[0][1])
    y_offset = devices[0].ymin
    for i, device in enumerate(devices):
        if "mzi" in device.name:
            device_name, iloss = f"{prefix}_mzi_{i}", 1.2
        elif "heater" in device.name:
            device_name, iloss = f"{prefix}_heater_{i}", 0.05
        elif "straight" in device.name:
            device_name, iloss = f"{prefix}_straight_{i}", 0.03
        elif "grating_coupler" in device.name:
            device_name, iloss = f"{prefix}_gc_{i}", 2.0
        elif "ring" in device.name:
            device_name, iloss = f"{prefix}_mrr_{i}", 1

        else:
            raise ValueError(f"Unkonwn device type: {device.name}")
        se.add_instance(device_name, device, iloss)
        device_names.append(device_name)
        input_ports.extend(
            [
                (device_name, p[0].name)
                for p in sorted(
                    [
                        (port, port.center[1])
                        for port in device.get_ports_list(
                            port_type="optical", orientation=180.0
                        )
                    ],
                    key=lambda x: x[1],
                )
            ]
        )
        output_ports.extend(
            [
                (device_name, p[0].name)
                for p in sorted(
                    [
                        (port, port.center[1])
                        for port in device.get_ports_list(
                            port_type="optical", orientation=0.0
                        )
                    ],
                    key=lambda x: x[1],
                )
            ]
        )

        electrical_ports.extend(
            [
                (device_name, port.name)
                for port in device.get_ports_list(
                    port_type="electrical", orientation=180.0
                )
            ]
        )

        se.schematic.placements[device_name].x = (
            placement_cfg["offset"][0] + placement_cfg["pitch"][0]
        )
        se.schematic.placements[device_name].y = (
            placement_cfg["offset"][1]
            + placement_cfg["pitch"][1] * (i - len(devices) / 2 + 0.5)
            + y_offset
        )
        se.schematic.instances[device_name].settings["placement"][1] = [
            se.schematic.placements[device_name].x,
            se.schematic.placements[device_name].y,
        ]
        se.schematic.instances[device_name].settings["placement"][2] = placement_cfg[
            "orient"
        ]
        se.schematic.instances[device_name].settings["placement"][3] = placement_cfg[
            "placement_halo"
        ]

    placement_offset = [
        float(se.schematic.placements[device_name].x + device.xsize),
        placement_cfg["offset"][1],
    ]
    placement_cfg["offset"] = placement_offset
    return device_names, input_ports, output_ports, placement_cfg, electrical_ports


def add_IO_pads(
    se,
    devices: List = [gf.components.pad()] * 2,
    prefix: str = "pad",
    placement_cfg=dict(offset=(0, 0), mirror=None, pitch=(30, 100)),
):
    device_names = []
    electrical_ports = []
    for i, device in enumerate(devices):
        device_name = f"{prefix}_{i}"
        device_names.append(device_name)
        se.add_instance(device_name, device)
        ports = device.get_ports_list(port_type="electrical", orientation=180.0)
        electrical_ports.append((device_name, ports[0].name))

        se.schematic.placements[device_name].x = (
            placement_cfg["offset"][0] + placement_cfg["pitch"][0]
        )
        se.schematic.placements[device_name].y = (
            placement_cfg["offset"][1] + placement_cfg["pitch"][1]
        )
        se.schematic.instances[device_name].settings["placement"][1] = [
            se.schematic.placements[device_name].x,
            se.schematic.placements[device_name].y,
        ]
        se.schematic.instances[device_name].settings["placement"][2] = placement_cfg[
            "orient"
        ]
        se.schematic.instances[device_name].settings["placement"][3] = placement_cfg[
            "placement_halo"
        ]

        placement_offset = [
            float(se.schematic.placements[device_name].x + device.xsize),
            placement_cfg["offset"][1],
        ]
        placement_cfg["offset"] = placement_offset

    return device_names, placement_cfg, electrical_ports


def connect_ports(se, ports1, ports2):
    assert len(ports1) == len(ports2), (
        f"Only can connect the same number of ports, but got #ports1={len(ports1)}, #ports2={len(ports2)}"
    )
    unconnected_ports1 = deepcopy(ports1)
    unconnected_ports2 = deepcopy(ports2)
    for i, (p1, p2) in enumerate(zip(ports1, ports2)):
        if p1 is not None and p2 is not None:
            se.add_net(inst1=p1[0], port1=p1[1], inst2=p2[0], port2=p2[1])
            unconnected_ports1[i] = None
            unconnected_ports2[i] = None
    return unconnected_ports1, unconnected_ports2


def connect_crossing_ports(se, ports1, ports2, target_inversions, seed):
    assert len(ports1) == len(ports2), (
        f"Only can connect the same number of ports, but got #ports1={len(ports1)}, #ports2={len(ports2)}"
    )
    unconnected_ports1 = deepcopy(ports1)
    unconnected_ports2 = deepcopy(ports2)

    length = len(ports1)
    arr = generate_crossing_sequence3(length, target_inversions, seed)

    for i1, i2 in enumerate(arr):
        p1 = ports1[i1]
        p2 = ports2[i2]
        se.add_net(inst1=p1[0], port1=p1[1], inst2=p2[0], port2=p2[1])

    # for i, (p1, p2) in enumerate(zip(ports1, ports2)):
    #     if p1 is not None and p2 is not None:
    #         se.add_net(inst1=p1[0], port1=p1[1], inst2=p2[0], port2=p2[1])
    #         unconnected_ports1[i] = None
    #         unconnected_ports2[i] = None

    return unconnected_ports1, unconnected_ports2


def generate_crossing_sequence3(N, target_inversions, seed):
    def count_inversions(arr):
        inversions = 0
        N = len(arr)
        for i in range(N):
            for j in range(i + 1, N):
                if arr[i] > arr[j]:
                    inversions += 1
        return inversions

    def random_swap_limited_inversions(N, target_inversions):
        arr = np.linspace(start=0, stop=N - 1, num=N, dtype=np.int32).tolist()
        N = len(arr)
        shuffled_arr = arr[:]

        while True:
            # 随机选择两个不同的索引进行交换
            i, j = random.sample(range(N), 2)
            shuffled_arr[i], shuffled_arr[j] = shuffled_arr[j], shuffled_arr[i]

            # 计算当前乱序数
            inversion_count = count_inversions(shuffled_arr)

            # 如果乱序数不超过设定的上限，停止交换
            if inversion_count >= target_inversions and inversion_count < N:
                break
            elif inversion_count >= N:
                shuffled_arr = list(range(N))

        return shuffled_arr

    random.seed(seed)
    arr_limited_inversions = random_swap_limited_inversions(N, target_inversions)
    return arr_limited_inversions


def generate_crossing_sequence2(N, target_inversions, seed):
    def count_inversions(arr):
        inversions = 0
        N = len(arr)
        for i in range(N):
            for j in range(i + 1, N):
                if arr[i] > arr[j]:
                    inversions += 1
        return inversions

    def shuffle_with_limited_inversions(N, target_inversions):
        arr = np.linspace(start=0, stop=N - 1, num=N, dtype=np.int32).tolist()
        N = len(arr)
        shuffled_arr = arr[:]

        while True:
            random.shuffle(shuffled_arr)
            inversion_count = count_inversions(shuffled_arr)
            if inversion_count <= target_inversions:  # 3 * N // 4:
                break
        return shuffled_arr

    random.seed(seed)
    arr_limited_inversions = shuffle_with_limited_inversions(N, target_inversions)
    return arr_limited_inversions


def generate_crossing_sequence(length, target_inversions, seed):
    def calculate_inversions(arr, length):
        current_inversions = 0
        for i in range(length):
            for j in range(i + 1, length):
                if arr[i] > arr[j]:
                    current_inversions += 1
        return current_inversions

    import random

    while 1:
        arr = list(range(0, length))
        seed += 42
        random.seed(seed)
        random.shuffle(arr)
        current_inversions = calculate_inversions(arr, length)
        if current_inversions > target_inversions:
            break

    for i in range(length - 1):
        for j in range(length - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                current_inversions -= 1
                if current_inversions <= target_inversions:
                    return arr


def generate_matrix(
    se,
    ports_list,
    count,
    crossing_layer,
    N=8,
    seed=4321,
    prefix: str = "device_array",
    placement_cfg=dict(
        offset=(0, 0),
        mirror=None,
        pitch=(30, 100),
        orient="N",
        placement_halo=[0, 0, 0, 0],
    ),
):
    device_names = []
    random.seed(seed)
    ps_columns = random.randint(2, 4)
    phase_shifter = gf.components.straight_heater_metal_undercut
    wg_type = gf.components.straight
    for i in range(ps_columns):
        devices = [phase_shifter(length=200, with_undercut=False)] * N
        ps_names, ps_array_in_ports, ps_array_out_ports, placement_cfg, _ = (
            add_N_devices(
                se,
                prefix=f"{prefix}_ps_array_{i}",
                devices=devices,
                placement_cfg=placement_cfg,
            )
        )
        device_names.append(ps_names)
        ports_list.append(ps_array_in_ports)
        ports_list.append(ps_array_out_ports)
        count += 2
        se.add_constraints(
            constraints_cfg=dict(
                type="alignment", settings={"anchor": "left"}, objects=ps_names
            )
        )

        seq = generate_seq_sum_to_N(N, seed * i + 42)
        input_ports = []
        output_ports = []
        j = 0
        cumulative_port = 0
        max_size = 0
        sub_device_name = []
        for port_num in seq:
            if port_num == 1:
                device = wg_type(length=200)
                device_offset = float(device.dymin)
                x_size = float(device.dxmax)
                max_size = max(max_size, x_size)
                mmi_name = f"{prefix}_multiport_{i}_{j}"
                se.add_instance(mmi_name, wg_type(length=200), iloss=0.03)
                device_names.append(mmi_name)
                sub_device_name.append(mmi_name)
                input_ports.append(
                    (
                        mmi_name,
                        device.get_ports_list(port_type="optical", orientation=180.0)[
                            0
                        ].name,
                    )
                )
                output_ports.append(
                    (
                        mmi_name,
                        device.get_ports_list(port_type="optical", orientation=0.0)[
                            0
                        ].name,
                    )
                )

                se.schematic.placements[mmi_name].x = (
                    placement_cfg["offset"][0] + placement_cfg["pitch"][0]
                )
                se.schematic.placements[mmi_name].y = (
                    placement_cfg["offset"][1]
                    + placement_cfg["pitch"][1] * (-N / 2)
                    + placement_cfg["pitch"][1] * (cumulative_port + 0.5)
                    + device_offset
                )
                se.schematic.instances[mmi_name].settings["placement"][1] = [
                    se.schematic.placements[mmi_name].x,
                    se.schematic.placements[mmi_name].y,
                ]
                se.schematic.instances[mmi_name].settings["placement"][3] = (
                    placement_cfg["placement_halo"]
                )
                cumulative_port += port_num
                j += 1
            else:
                length_mmi = 36 * port_num / 2
                width_mmi = 10 * port_num / 2
                device = mmi(
                    inputs=port_num,
                    outputs=port_num,
                    length_mmi=length_mmi,
                    width_mmi=width_mmi,
                    gap_input_tapers=4,
                    gap_output_tapers=4,
                )
                # device_offset = float(device.bbox[0][1])
                device_offset = float(device.dymin)
                # x_size = float(device.bbox[1][0])
                x_size = float(device.dxmax)
                max_size = max(max_size, x_size)
                mmi_name = f"{prefix}_multiport_{i}_{j}"
                se.add_instance(
                    mmi_name,
                    mmi(
                        inputs=port_num,
                        outputs=port_num,
                        length_mmi=length_mmi,
                        width_mmi=width_mmi,
                        gap_input_tapers=4,
                        gap_output_tapers=4,
                    ),
                    iloss=0.1,
                )
                device_names.append(mmi_name)
                sub_device_name.append(mmi_name)
                input_ports.extend(
                    [
                        (mmi_name, p[0].name)
                        for p in sorted(
                            [
                                (port, port.center[1])
                                for port in device.get_ports_list(
                                    port_type="optical", orientation=180.0
                                )
                            ],
                            key=lambda x: x[1],
                        )
                    ]
                )
                output_ports.extend(
                    [
                        (mmi_name, p[0].name)
                        for p in sorted(
                            [
                                (port, port.center[1])
                                for port in device.get_ports_list(
                                    port_type="optical", orientation=0.0
                                )
                            ],
                            key=lambda x: x[1],
                        )
                    ]
                )

                se.schematic.placements[mmi_name].x = (
                    placement_cfg["offset"][0] + placement_cfg["pitch"][0]
                )
                se.schematic.placements[mmi_name].y = (
                    placement_cfg["offset"][1]
                    + placement_cfg["pitch"][1] * (-N / 2)
                    + placement_cfg["pitch"][1] * (cumulative_port + port_num / 2)
                    + device_offset
                )
                se.schematic.instances[mmi_name].settings["placement"][1] = [
                    se.schematic.placements[mmi_name].x,
                    se.schematic.placements[mmi_name].y,
                ]
                se.schematic.instances[mmi_name].settings["placement"][3] = (
                    placement_cfg["placement_halo"]
                )
                cumulative_port += port_num
                j += 1

        placement_offset = [
            float(se.schematic.placements[mmi_name].x + max_size),
            placement_cfg["offset"][1],
        ]
        placement_cfg["offset"] = placement_offset
        count += 2
        crossing_layer.add(count - 1)

        ports_list.append(input_ports)
        ports_list.append(output_ports)
        se.add_constraints(
            constraints_cfg=dict(
                type="alignment", settings={"anchor": "left"}, objects=sub_device_name
            )
        )

    return device_names, count, placement_cfg


def generate_seq_sum_to_N(N, seed=None):
    if seed is not None:
        random.seed(seed)

    if random.random() < 0.3:
        return [int(N / 2), int(N / 2)]

    array = []
    remaining = N
    while remaining >= 2:
        pos = random.random()
        if pos < 0.5:
            value = random.choice([2 * i for i in range(1, remaining // 2 + 1)])
        elif pos < 0.8:
            value = 1
        else:
            value = random.choice([i for i in range(1, remaining // 2 + 1)])

        array.append(value)
        remaining -= value

    if remaining != 0:
        array.append(remaining)

    return array
