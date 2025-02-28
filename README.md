# LiDAR: Automated Curvy Waveguide Detailed Routing for Large-Scale Photonic Integrated Circuits

By [Hongjian Zhou](https://scopex-asu.github.io/index.html), [Keren Zhu](https://krz.engineer/) and [Jiaqi Gu<sup>†</sup>](https://scopex-asu.github.io/index.html).

This repo is the official implementation of ["LiDAR:Automated Curvy Waveguide Detailed Routing for Large-Scale Photonic Integrated Circuits"](https://arxiv.org/abs/2410.01260).
<p align="center">
  <img src="figs/LiDAR_logo.png" width="350" height="400"/>
</p>

## Table of Contents

- [Introduction](#introduction)
- [Folder Structure](#folder-structure)
- [Installation](#installation)

## Introduction

**LiDAR** is developed for automatically generating large-scale real-world PIC routing solutions while considering specific photonic design rules. LiDAR features a grid-based **curvy-aware A\*** engine with adaptive **crossing insertion**, **congestion-aware net ordering and objective**, and **crossing-waveguide optimization scheme**, all tailored to the unique property of PIC. 

<p align="center">
  <img src="figs/curvy aware neighbors.jpg" width="400" height="240"/>
</p>

To efficiently enable curvy-aware A\* search, we propose parametric curvy-aware methods to generate neighbor candidates based on parametric bending geometry and perform a comprehensive DRC check to select legal neighbors for exploration. Each routing node is defined by its spatial location and orientation *(x, y, orientation)*, and we have 45° neighbors to enable 45° routing.

<p align="center">
  <img src="figs/grid map.jpg" width="400" height="260"/>
</p>

To ensure that only feasible neighbors are considered for exploration, a GridMap-based legality check is necessary. Each routed grids are assigned with the waveguide orientation for further crossing insertion.

<p align="center">
  <img src="figs/crossing insertion.jpg" width="600" height="150"/>
</p>

If a neighboring candidate hits a previously routed waveguide (marked as an obstacle), we need to check whether it is feasible to insert a waveguide crossing to pass through it.

<p align="center">
  <img src="figs/port access.jpg" width="500" height="440"/>
</p>
To further enhance overall port accessibility, several port access assignment techniques are proposed.   

- **Port Propagation**: Propagates the ports that are inside the component's bounding box.
- **Bending-Aware Port Access Region Reservation**: To prevent other waveguides from blocking port regions, grids in front of each port are reserved for the corresponding net, ensuring they cannot be crossed by other nets.
- **Congested Port Spreading**: Spreads the access ports with a predefned extension length and spacing for ports in the same grid.
- **Channel Planning via Staggered Access Point Oﬀsets**: For densely placed ports, we progressively extend the access region length and use staggered access ports to facilitate the placement of consecutive crossings.

## Folder Structure
| File             | Description |
| ---------------- | ----------- |
| benchmarks/      | Netlists and its corresponding scripts |
| config/          | Parser and router config |
| database/        | Script for netlist loading and LiDAR database construction |
| drc/             | Scripts for design rule checking |
| main/            | Main function |
| queue/           | Queue data structure for A star search |
| routing/         | Implementation of routing algorithm|
| scripts/         | Script for batch execution of benchmarks |
| utils/           | Utilities for post-processing and logging|



## Installation

### Prerequisites
- Python >=3.11
- [GDSFactory](https://github.com/gdsfactory/gdsfactory) ==8.26.1
- Other required Python packages (listed in requirements.txt)
- klayout and klive (Optional)

### Get the LiDAR Source
```git clone https://github.com/ScopeX-ASU/PICRoute.git```

### Usage
#### 1. How to get benchmarks
```bash
cd src/picroute
python benchmarks/clements.py
python benchmarks/MMIports.py
```
Benchmark currently provides photonic computing circuits. Users can generate different sizes and configurations of benchmarks based on these scripts. 

#### 2. How to run
```bash
cd src/picroute
python scripts/lidar/run_route.py
```
The output layout gdsii files: result/LiDAR/main_results  
The ouput log files: log/LiDAR/main_results

or
```bash
cd src/picroute
python main/picroute.py --benchmark path-to-benchmark --config path to router config
```
The input to LiDAR is formatted as a netlist, and the tool's configuration is as yaml, similar to LEF/DEF.

### LiDAR's Configurations
| YAML Parameter      | Description                                                                               |
| --------------      | ----------------------------------------------------------------------------------------- |
| maxIteration        |  maximum routing iteration                                                                |
| net_order           |  net routing order                                                  |
| group               |  enable group routing order or not                |
| enable_45_neighbor  |  enable diagonal neighbors or not             |
| historyCost         |   history cost of routing grid                         |
| ripup_times         |   maximum ripup and reroute times                     |
| grid_resolution     |    resolution of routing grid >=2             |
| bend_radius         |    radius of bend                                      |
| net_default_bound   |    minimum routing boundary (um)                 |
| net_bound_scale_factor   |   scale factor of the routing boundary                                    |
| loss        |       loss for the propagation, bending, and crossing                     |
| show_temp   |       show the intermediate routing result or not (needs klayout and klive)           |

## Citing LiDAR
```
@inproceedings{hzhou2025lidar,
  title={LiDAR: Automated Curvy Waveguide Detailed Routing for Large-Scale Photonic Integrated Circuits},
  author={Hongjian Zhou and Keren Zhu and Jiaqi Gu},
  booktitile={International Symposium on Physical Design (ISPD)},
  year={2025}
}
```
```
Hongjian Zhou, Keren Zhu and Jiaqi Gu, "LiDAR: Automated Curvy Waveguide Detailed Routing for Large-Scale Photonic Integrated Circuits," International Symposium on Physical Design (ISPD), 2025.
```
