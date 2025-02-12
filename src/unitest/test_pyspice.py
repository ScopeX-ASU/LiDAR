"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-07-16 11:16:59
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-07-16 11:20:46
"""

from PySpice.Spice.Parser import SpiceParser

# from src.Parser import Netlist_parser
p = SpiceParser(
    path="./unitest/netlist_mzi.sp",
    source=None,
    end_of_line_comment="$",
)
c = p.build_circuit()
print(c)

# p = Netlist_parser()
