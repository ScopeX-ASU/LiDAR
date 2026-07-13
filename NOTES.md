This repository implements LiDAR, a detailed router for photonic integrated circuits. Conceptually, it is a sequential, grid-based, negotiated-congestion router whose A* search understands physical waveguide bends and crossings.

  Benchmark YAML
      ↓
  Placed GDSFactory circuit + circuit graph
      ↓
  Directional occupancy bitmap
      ↓
  Port-access planning and net ordering
      ↓
  Curvy-aware A* for each net
      ↓
  Conflict-driven rip-up and reroute
      ↓
  Euler waveguides + physical crossings
      ↓
  Connectivity/loss evaluation + GDS

  ## Repository structure

  The Python package lives under src/picroute:

  - src/picroute/main contains the single-run entry point. src/picroute/main/picroute.py:12 constructs the circuit database, DRC model, and router, then writes the resulting
    GDS.

  - src/picroute/config defines and validates routing parameters. src/picroute/config/config.py:28 covers grid resolution, bend radius, routing order, losses, congestion, and
    iteration limits.

  - src/picroute/database converts YAML into placed GDSFactory objects and LiDAR’s internal net/block/port state. The main loader is src/picroute/database/schematic.py:670.
  - src/picroute/drc holds the occupancy bitmap and incremental design-rule checks. src/picroute/drc/bitmap.py:137 stores obstacles and directional waveguide occupancy; src/
    picroute/drc/drcmanager.py:31 validates candidate A* moves.

  - src/picroute/routing contains the core algorithms:
      - src/picroute/routing/drgridroute.py:16: multi-net routing, ordering, rip-up/reroute, crossing post-processing, evaluation.
      - src/picroute/routing/astarsearch.py:104: orientation-aware A*.
      - src/picroute/routing/geometry.py:13: converts abstract paths into Euler-bend GDS geometry.

  - src/picroute/benchmarks contains YAML netlists and generators for Clements meshes, MRR weight banks, and multi-port MMIs.
  - src/picroute/scripts contains the parallel batch launcher.
  - src/picroute/utils contains smoothing, logging, geometry, and timing helpers.
  - src/unitest tests configuration, bitmap behavior, geometry, schematic loading, routing, and end-to-end GDS generation.
  - figs contains paper/README illustrations.
  - pyproject.toml defines dependencies and the picroute and picroute-batch commands.

  ## Input and internal database

  A benchmark YAML describes:

  - instances: GDSFactory component factories, settings, and placements.
  - library: macro dimensions, ports, and device insertion loss.
  - nets: named two-terminal connections such as instance,o1.
  - settings: die bounds, design name, number of components, and units.
  - constraints: mainly benchmark-generation metadata.

  During loading, LiDAR:

  1. Builds a native GDSFactory schematic.
  2. Places and transforms every component.
  3. Extracts the transformed optical ports and bounding boxes.
  4. Wraps each port in an EndpointState.
  5. Wraps every two-terminal connection in a RouteState.
  6. Constructs a directed Rustworkx circuit graph.

  This logic is in src/picroute/database/schematic.py:610. GDSFactory is deliberately treated as the source of truth for rotated/mirrored port coordinates.

  The circuit graph serves two purposes:

  - It generates breadth-first topological net levels for routing order.
  - After routing, its edge weights are replaced by insertion losses so the router can find the maximum-loss DAG path.

  See src/picroute/database/netlist_graph.py:66.

  ## The occupancy model

  The physical die is discretized using:

  [
  (x_g,y_g)=\left\lfloor\frac{(x,y)-\text{die origin}}
  {\text{grid resolution}}\right\rfloor
  ]

  Each grid cell can be:

  - Empty
  - A component blockage
  - Reserved for a particular net’s port access
  - Occupied by one waveguide
  - Occupied by multiple waveguides at a crossing

  Waveguide cells additionally retain their direction: horizontal, vertical, 45°, 135°, or bend. That orientation metadata is what lets DRC distinguish an illegal overlap
  from a potentially legal perpendicular crossing.

  The representation is compact: dense integer arrays store cell types, while sparse dictionaries store owners and routed-net directions. See src/picroute/drc/bitmap.py:24.

  ## Port-access planning

  Before routing ordinary waveguides, all component bounding boxes are blocked and access corridors are reserved in front of ports.

  When grouped routing is enabled, src/picroute/drc/drcmanager.py:100 handles dense multi-port sides by:

  - Ordering ports along the component edge.
  - Extending access corridors outside the component.
  - Spreading ports that fall into the same or nearby grid cells.
  - Giving different ports staggered extension lengths.
  - Grouping nets attached to the same component side.
  - Sometimes reversing a net so routing starts from the more congested endpoint.

  These reserved cells belong to the associated net. Other nets see them as obstacles, preventing an early route from sealing off a later port.

  ## Net ordering

  src/picroute/routing/drgridroute.py:77 supports:

  - topo: route nets in breadth-first circuit-topology levels.
  - naive: primarily order them by direct endpoint distance.

  Within a level or port group, shorter Euclidean-distance nets normally receive earlier priority. The router also computes how many straight source-to-destination lines
  intersect, although topology_crossing is currently only recorded as a diagnostic.

  This is sequential routing: once one net is accepted, its occupied cells affect every subsequent net.

  ## Curvy-aware A*

  The central difference from ordinary grid A* is its state:

  [
  n=(x,y,\theta)
  ]

  where (\theta) is one of:

  [
  0°,45°,90°,135°,180°,225°,270°,315°
  ]

  Two nodes in the same cell with different headings are therefore different states. That matters because a waveguide cannot turn arbitrarily inside one cell.

  ### Physical transitions

  The transition table in src/picroute/routing/astarsearch.py:30 contains:

  - One-cell cardinal straights.
  - One-cell diagonal straights.
  - Multi-cell 45° bends.
  - Multi-cell 90° bends.

  Bend displacement is derived from:

  [
  r_g=\left\lceil\frac{\text{bend radius}}
  {\text{grid resolution}}\right\rceil
  ]

  A 90° turn therefore jumps across approximately a radius-sized square instead of merely changing heading in place. The 45° transitions use empirically parameterized pieces
  of approximately (0.415r) and (0.3r). This means the path A* explores is already close to something that can be realized with an Euler bend.

  ### Search cost

  For ordinary moves:

  [
  g(n')=g(n)+
  L_{\text{move}}C_{\text{propagation}}+
  C_{\text{bend}}+
  C_{\text{crossing}}+
  C_{\text{congestion}}
  ]

  The exact step costs are initialized in src/picroute/routing/astarsearch.py:212:

  - Straight: physical grid length × propagation penalty.
  - 45° straight: (\sqrt{2}) × grid length × propagation penalty.
  - Bend: smoothed bend length × propagation penalty plus a bend penalty.
  - Crossing: configured crossing penalty.
  - Congestion: nearby obstacles/routes × congestion penalty.
  - History: a penalty associated with previously conflicting (cell, orientation) states.

  The 8-direction heuristic approximates the cheapest mixture of diagonal and cardinal travel and adds a partial bend charge when both axes differ.

  A* is normally restricted to a box around the two endpoints. Every two failures expand that box, allowing progressively larger detours.

  ### Neighbor legality

  For every transition, src/picroute/drc/drcmanager.py:400 examines every grid cell traversed by the full straight or bend primitive.

  It rejects moves that:

  - Leave the die or routing bounds.
  - Enter a component.
  - Enter another net’s reserved access region.
  - Encounter an incompatible or compound waveguide.
  - Lack enough clear cells to form the complete bend.
  - Would immediately run into another obstacle after a bend.

  This is significant: DRC is performed during search, not merely after producing a polyline.

  ## Adaptive crossing insertion

  When a straight move encounters an existing waveguide, the DRC manager tests whether the overlap can become a crossing:

  1. The two routes must be perpendicular: horizontal/vertical or the corresponding diagonal pair.
  2. Both nets must have compatible width, wavelength, and material.
  3. Both must have crossing budget available.
  4. The approaching route must have enough straight lead-in.
  5. The existing route must remain straight and unobstructed across the crossing footprint.
  6. The cells immediately after the crossing must be clear.

  If legal, A* receives a special transition that jumps across the occupied footprint and records the crossed net. This logic starts around src/picroute/drc/
  drcmanager.py:438.

  Crossings are only logical during search. After all nets are settled, src/picroute/routing/drgridroute.py:457:

  - Finds the exact geometric intersection.
  - Splits both waveguide paths.
  - Removes a short section from each.
  - Places a GDSFactory crossing component.
  - Reconnects the four resulting waveguide ends to its ports.

  ## Rip-up and reroute

  A net is first searched under strict DRC. If that fails, LiDAR runs a relaxed search:

  - Components and port regions remain hard obstacles.
  - Existing waveguides may be traversed as violations.
  - The relaxed route records which nets caused conflicts.

  The current net and conflicting nets are then marked for rip-up. At the end of the iteration:

  1. Their bitmap occupancy is deleted.
  2. Their route state is cleared.
  3. Failed-path states contribute to the directional history-cost map.
  4. The nets return to the routing priority queue.
  5. A* tries again with altered costs and, eventually, wider bounds.

  After repeated failures, 45° moves are disabled for that net, reducing the state space and forcing simpler topology. The outer iteration is implemented in src/picroute/
  routing/drgridroute.py:121.

  This resembles negotiated-congestion FPGA routing: collisions are temporarily exposed, converted into penalties, and then reconsidered globally through selective rerouting.

  ## From grid path to GDS

  Once A* reaches one of the destination access cells with the required opposite heading:

  1. Parent pointers reconstruct the (x,y,heading) path.
  2. Explicit bend transition points are inserted.
  3. Collinear points are removed.
  4. GDSFactory checks that the waypoints can be smoothed with the requested radius.
  5. The smoothed geometry is rasterized back onto the bitmap for later nets.

  After routing is complete, src/picroute/routing/geometry.py:212 creates the final geometry:

  - Euler-bend/all-angle backbone.
  - Access waveguides from the exact physical ports to the grid-derived backbone.
  - S-bend fallback where ordinary all-angle access placement fails.
  - Configured PDK cross-section and layer.

  The result is added to the component layout, native optical connectivity checks are run, and the final layout is written as GDS.

  ## Evaluation

  For eval: comp, each routed net is assigned approximately:

  [
  \text{loss}=
  L_{\mathrm{wg}}C_{\mathrm{prop}}+
  \theta_{\mathrm{bend}}C_{\mathrm{bend}}+
  N_{\mathrm{cross}}C_{\mathrm{cross}}+
  \text{device losses}
  ]

  Those losses become circuit-graph edge weights. LiDAR then computes the weighted longest path, representing the critical optical path. It also reports total waveguide
  length, crossings, bends, and design-rule violations.

  The noc evaluation path is more benchmark-specific and compares two fixed groups of routes rather than using the general DAG critical-path calculation.

  ## Important implementation constraints

  A few details matter when extending or using the current checkout:

  - Nets are two-terminal; multi-terminal routing is not implemented.
  - Transformed optical ports must be cardinal when loaded—0°, 90°, 180°, or 270°.
  - Topological ordering and critical-path evaluation assume an acyclic circuit graph.
  - The bitmap uses component bounding boxes, so component blockage modeling is conservative.
  - ripup_times is present in the validated configuration but is not actually used to cap rip-ups; maxIteration currently controls the outer loop.
  - In src/picroute/routing/astarsearch.py:500, the special case for nets shorter than 10 µm accesses startNode before findAcsPts() initializes it. Ordinary longer nets
    short-circuit that branch, but a sufficiently short net appears capable of raising an AttributeError.

  - PICRoute.run() writes a GDS even if the router reports failure; callers should inspect routing state/connectivity rather than treating file existence alone as success.
