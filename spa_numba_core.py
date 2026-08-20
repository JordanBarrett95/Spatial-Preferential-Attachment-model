"""Compiled numerical core for the SPA sampler.

This module has no graph-library dependencies.  All mutable state is represented
by NumPy arrays and scalar metadata so it can be passed through Numba's nopython
mode.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

_SM64_A = np.uint64(0x9E3779B97F4A7C15)
_SM64_B = np.uint64(0xBF58476D1CE4E5B9)
_SM64_C = np.uint64(0x94D049BB133111EB)
_TWO_NEG_53 = 1.0 / 9007199254740992.0
_FLOAT_EPS = np.finfo(np.float64).eps
# Cell lower bounds are deliberately rounded downward.  A looser bound can
# only reduce pruning; an overestimate could incorrectly discard a candidate.
_BOUND_ABS_SLOP = 32.0 * _FLOAT_EPS
_BOUND_RELAX = 1.0 - 64.0 * _FLOAT_EPS


@njit(cache=True, inline="always")
def _splitmix64(value: np.uint64) -> np.uint64:
    value = np.uint64(value + _SM64_A)
    value = np.uint64((value ^ (value >> np.uint64(30))) * _SM64_B)
    value = np.uint64((value ^ (value >> np.uint64(27))) * _SM64_C)
    return np.uint64(value ^ (value >> np.uint64(31)))


@njit(cache=True, inline="always")
def _edge_is_selected(
    seed: np.uint64,
    source: int,
    target: int,
    probability: float,
) -> bool:
    """Counter-based Bernoulli draw keyed by a unique 32-bit vertex pair."""

    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False
    pair_key = (np.uint64(source) << np.uint64(32)) | np.uint64(target)
    bits = _splitmix64(np.uint64(seed ^ pair_key))
    return float(bits >> np.uint64(11)) * _TWO_NEG_53 < probability


@njit(cache=True, inline="always")
def _periodic_difference(a: float, b: float) -> float:
    difference = abs(a - b)
    if difference > 0.5:
        difference = 1.0 - difference
    return difference


@njit(cache=True, inline="always")
def _is_close(
    positions: np.ndarray,
    influence: np.ndarray,
    older: int,
    newer: int,
    time_volume: float,
    dimension: int,
    norm: float,
    infinite_norm: bool,
    dimension_over_norm: float,
) -> bool:
    """Evaluate the sphere condition without constructing a radius array."""

    if infinite_norm:
        distance = 0.0
        for coordinate in range(dimension):
            difference = _periodic_difference(
                positions[older, coordinate], positions[newer, coordinate]
            )
            if difference > distance:
                distance = difference

        distance_power = 1.0
        for _ in range(dimension):
            distance_power *= distance
        return distance_power * time_volume < influence[older]

    norm_sum = 0.0
    for coordinate in range(dimension):
        difference = _periodic_difference(
            positions[older, coordinate], positions[newer, coordinate]
        )
        norm_sum += difference**norm

    # d**dimension = (sum(delta**norm))**(dimension/norm)
    return norm_sum**dimension_over_norm * time_volume < influence[older]


@njit(cache=True, inline="always")
def _is_close_2d_linf(
    positions: np.ndarray,
    influence: np.ndarray,
    older: int,
    newer: int,
    time_volume: float,
) -> bool:
    """Specialized two-dimensional L-infinity sphere test."""

    difference_x = _periodic_difference(positions[older, 0], positions[newer, 0])
    difference_y = _periodic_difference(positions[older, 1], positions[newer, 1])
    distance = difference_x if difference_x >= difference_y else difference_y
    return distance * distance * time_volume < influence[older]


@njit(cache=True, inline="always")
def _cell_id(position: np.ndarray, resolution: int, dimension: int) -> int:
    identifier = 0
    for coordinate in range(dimension):
        cell_coordinate = int(position[coordinate] * resolution)
        if cell_coordinate >= resolution:  # defensive: random positions are < 1
            cell_coordinate = resolution - 1
        identifier = identifier * resolution + cell_coordinate
    return identifier


@njit(cache=True, inline="always")
def _neighbour_cell_id(
    position: np.ndarray,
    offset: np.ndarray,
    resolution: int,
    dimension: int,
) -> int:
    identifier = 0
    for coordinate in range(dimension):
        cell_coordinate = int(position[coordinate] * resolution)
        if cell_coordinate >= resolution:
            cell_coordinate = resolution - 1
        cell_coordinate += int(offset[coordinate])
        if cell_coordinate < 0:
            cell_coordinate += resolution
        elif cell_coordinate >= resolution:
            cell_coordinate -= resolution
        identifier = identifier * resolution + cell_coordinate
    return identifier


@njit(cache=True, inline="always")
def _axis_distance_to_neighbour_cell(
    fractional_cell_coordinate: float, offset: int, inverse_resolution: float
) -> float:
    """Conservative distance from a point to one adjacent cell along one axis."""

    if offset == 0:
        return 0.0
    if offset < 0:
        distance = fractional_cell_coordinate * inverse_resolution
    else:
        distance = (1.0 - fractional_cell_coordinate) * inverse_resolution
    distance -= _BOUND_ABS_SLOP
    return distance if distance > 0.0 else 0.0


@njit(cache=True, inline="always")
def _cell_lower_measure(
    position: np.ndarray,
    offset: np.ndarray,
    resolution: int,
    dimension: int,
    norm: float,
    infinite_norm: bool,
    dimension_over_norm: float,
    time_volume: float,
) -> float:
    """Lower bound for ``distance**dimension * time_volume`` to a cell."""

    inverse_resolution = 1.0 / float(resolution)
    if infinite_norm:
        distance = 0.0
        for coordinate in range(dimension):
            scaled = position[coordinate] * resolution
            base = int(scaled)
            if base >= resolution:
                base = resolution - 1
            fractional = scaled - float(base)
            axis_distance = _axis_distance_to_neighbour_cell(
                fractional, int(offset[coordinate]), inverse_resolution
            )
            if axis_distance > distance:
                distance = axis_distance
        measure = 1.0
        for _ in range(dimension):
            measure *= distance
    else:
        norm_sum = 0.0
        for coordinate in range(dimension):
            scaled = position[coordinate] * resolution
            base = int(scaled)
            if base >= resolution:
                base = resolution - 1
            fractional = scaled - float(base)
            axis_distance = _axis_distance_to_neighbour_cell(
                fractional, int(offset[coordinate]), inverse_resolution
            )
            norm_sum += axis_distance**norm
        measure = norm_sum**dimension_over_norm

    bound = measure * time_volume
    return bound * _BOUND_RELAX if bound > 0.0 else 0.0


@njit(cache=True, inline="always")
def _cell_lower_measure_2d_linf(
    fractional_x: float,
    fractional_y: float,
    offset_x: int,
    offset_y: int,
    inverse_resolution: float,
    time_volume: float,
) -> float:
    distance_x = _axis_distance_to_neighbour_cell(
        fractional_x, offset_x, inverse_resolution
    )
    distance_y = _axis_distance_to_neighbour_cell(
        fractional_y, offset_y, inverse_resolution
    )
    distance = distance_x if distance_x >= distance_y else distance_y
    bound = distance * distance * time_volume
    return bound * _BOUND_RELAX if bound > 0.0 else 0.0


@njit(cache=True)
def _grow_edges(
    sources: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    old_capacity = sources.shape[0]
    new_capacity = 16 if old_capacity == 0 else old_capacity * 2
    new_sources = np.empty(new_capacity, dtype=np.int32)
    new_targets = np.empty(new_capacity, dtype=np.int32)
    if old_capacity:
        new_sources[:old_capacity] = sources
        new_targets[:old_capacity] = targets
    return new_sources, new_targets


@njit(cache=True, inline="always")
def _append_edge(
    sources: np.ndarray,
    targets: np.ndarray,
    edge_count: int,
    source: int,
    target: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if edge_count == sources.shape[0]:
        sources, targets = _grow_edges(sources, targets)
    sources[edge_count] = source
    targets[edge_count] = target
    return sources, targets, edge_count + 1


@njit(cache=True, inline="always")
def _cell_count(resolution: int, dimension: int, limit: int) -> int:
    count = 1
    for _ in range(dimension):
        if count > limit // resolution:
            return limit + 1
        count *= resolution
    return count


@njit(cache=True)
def _cap_resolution(
    desired: int, dimension: int, budget: int
) -> int:
    """Largest resolution at most ``desired`` whose dense grid fits budget."""

    if desired < 3:
        desired = 3
    if _cell_count(3, dimension, budget) > budget:
        return 0
    if _cell_count(desired, dimension, budget) <= budget:
        return desired

    estimate = int(budget ** (1.0 / dimension))
    if estimate > desired:
        estimate = desired
    if estimate < 3:
        return 0
    while estimate > 3 and _cell_count(estimate, dimension, budget) > budget:
        estimate -= 1
    while estimate < desired:
        candidate = estimate + 1
        if _cell_count(candidate, dimension, budget) > budget:
            break
        estimate = candidate
    return estimate


@njit(cache=True, inline="always")
def _upper_bound(sorted_values: np.ndarray, threshold: float) -> int:
    """Index after the final value <= threshold."""

    low = 0
    high = sorted_values.shape[0]
    while low < high:
        middle = (low + high) // 2
        if sorted_values[middle] <= threshold:
            low = middle + 1
        else:
            high = middle
    return low


@njit(cache=True)
def _choose_resolution(
    influence: np.ndarray,
    start: int,
    stop: int,
    time: int,
    unit_ball_volume: float,
    dimension: int,
    grid_cells_per_vertex: float,
    max_grid_cells: int,
    neighbour_count: int,
    max_static_overflow: int,
) -> int:
    """Choose a cost-aware exact grid, allowing influential outliers to overflow."""

    count = stop - start
    if count <= neighbour_count or neighbour_count == 0:
        return 0

    minimum_cells = neighbour_count  # 3**dimension
    budget = int(math.ceil(grid_cells_per_vertex * count))
    if budget < minimum_cells:
        budget = minimum_cells
    if budget > max_grid_cells:
        budget = max_grid_cells
    if budget < minimum_cells:
        return 0

    ordered = np.empty(count, dtype=np.float64)
    for index in range(count):
        ordered[index] = influence[start + index]
    ordered.sort()

    allowed_overflow = max_static_overflow
    if allowed_overflow > count - 1:
        allowed_overflow = count - 1
    if allowed_overflow < 0:
        allowed_overflow = 0

    best_resolution = 0
    best_cost = float(count)  # compiled contiguous brute-force scan
    time_volume = float(time) * unit_ball_volume

    overflow_target = 0
    while True:
        retained_maximum = ordered[count - 1 - overflow_target]
        if retained_maximum > 0.0:
            desired = int(
                math.floor((time_volume / retained_maximum) ** (1.0 / dimension))
            )
            resolution = _cap_resolution(desired, dimension, budget)
            if resolution > 3:
                cells = _cell_count(resolution, dimension, budget)
                safe_influence = time_volume / float(cells)
                first_overflow = _upper_bound(ordered, safe_influence)
                actual_overflow = count - first_overflow
                retained = count - actual_overflow

                expected_grid_candidates = (
                    float(retained) * float(neighbour_count) / float(cells)
                )
                if expected_grid_candidates > retained:
                    expected_grid_candidates = float(retained)
                estimated_cost = (
                    float(neighbour_count)
                    + float(actual_overflow)
                    + expected_grid_candidates
                )
                if estimated_cost < best_cost:
                    best_cost = estimated_cost
                    best_resolution = resolution

        if overflow_target >= allowed_overflow:
            break
        if overflow_target == 0:
            next_target = 1
        else:
            next_target = overflow_target * 2
        if next_target > allowed_overflow:
            next_target = allowed_overflow
        if next_target == overflow_target:
            break
        overflow_target = next_target

    return best_resolution


@njit(cache=True)
def _build_one_grid(
    positions: np.ndarray,
    influence: np.ndarray,
    next_vertex: np.ndarray,
    overflow: np.ndarray,
    overflow_vertices: np.ndarray,
    overflow_count: int,
    start: int,
    stop: int,
    resolution: int,
    safe_influence: float,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if resolution == 0:
        for vertex in range(start, stop):
            next_vertex[vertex] = -1
        return (
            np.full(1, -1, dtype=np.int32),
            np.zeros(1, dtype=np.float64),
            overflow_count,
        )

    cells = 1
    for _ in range(dimension):
        cells *= resolution
    heads = np.full(cells, -1, dtype=np.int32)
    cell_maximum = np.zeros(cells, dtype=np.float64)

    for vertex in range(start, stop):
        if influence[vertex] > safe_influence:
            overflow[vertex] = 1
            overflow_vertices[overflow_count] = vertex
            overflow_count += 1
            next_vertex[vertex] = -1
            continue
        identifier = _cell_id(positions[vertex], resolution, dimension)
        next_vertex[vertex] = heads[identifier]
        heads[identifier] = vertex
        if influence[vertex] > cell_maximum[identifier]:
            cell_maximum[identifier] = influence[vertex]

    return heads, cell_maximum, overflow_count


@njit(cache=True)
def _rebuild_grids(
    positions: np.ndarray,
    influence: np.ndarray,
    next_vertex: np.ndarray,
    overflow: np.ndarray,
    overflow_vertices: np.ndarray,
    time: int,
    alpha: float,
    unit_ball_volume: float,
    dimension: int,
    grid_cells_per_vertex: float,
    max_grid_cells: int,
    neighbour_count: int,
    max_static_overflow: int,
) -> tuple:
    split = int(float(time) ** alpha)
    if split < 0:
        split = 0
    elif split > time:
        split = time

    for vertex in range(time):
        overflow[vertex] = 0

    old_resolution = _choose_resolution(
        influence,
        0,
        split,
        time,
        unit_ball_volume,
        dimension,
        grid_cells_per_vertex,
        max_grid_cells,
        neighbour_count,
        max_static_overflow,
    )
    young_resolution = _choose_resolution(
        influence,
        split,
        time,
        time,
        unit_ball_volume,
        dimension,
        grid_cells_per_vertex,
        max_grid_cells,
        neighbour_count,
        max_static_overflow,
    )

    old_power = 0.0
    old_safe = 0.0
    if old_resolution:
        old_power = (1.0 / old_resolution) ** dimension
        old_safe = float(time) * unit_ball_volume * old_power

    young_power = 0.0
    young_safe = 0.0
    if young_resolution:
        young_power = (1.0 / young_resolution) ** dimension
        young_safe = float(time) * unit_ball_volume * young_power

    overflow_count = 0
    old_heads, old_cell_maximum, overflow_count = _build_one_grid(
        positions,
        influence,
        next_vertex,
        overflow,
        overflow_vertices,
        overflow_count,
        0,
        split,
        old_resolution,
        old_safe,
        dimension,
    )
    young_heads, young_cell_maximum, overflow_count = _build_one_grid(
        positions,
        influence,
        next_vertex,
        overflow,
        overflow_vertices,
        overflow_count,
        split,
        time,
        young_resolution,
        young_safe,
        dimension,
    )

    return (
        old_heads,
        young_heads,
        old_cell_maximum,
        young_cell_maximum,
        old_resolution,
        young_resolution,
        old_power,
        young_power,
        split,
        overflow_count,
        overflow_count,  # baseline: intentional static overflow at rebuild
    )


@njit(cache=True, inline="always")
def _mark_dynamic_overflow(
    vertex: int,
    next_time: int,
    influence: np.ndarray,
    overflow: np.ndarray,
    overflow_vertices: np.ndarray,
    overflow_count: int,
    resolution: int,
    cell_power: float,
    unit_ball_volume: float,
) -> int:
    if resolution <= 3 or overflow[vertex] != 0:
        return overflow_count
    safe_influence = float(next_time) * unit_ball_volume * cell_power
    if influence[vertex] > safe_influence:
        overflow[vertex] = 1
        overflow_vertices[overflow_count] = vertex
        overflow_count += 1
    return overflow_count


@njit(cache=True)
def _scan_grid(
    positions: np.ndarray,
    degrees: np.ndarray,
    influence: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    edge_count: int,
    next_vertex: np.ndarray,
    overflow: np.ndarray,
    overflow_vertices: np.ndarray,
    overflow_count: int,
    heads: np.ndarray,
    cell_maximum: np.ndarray,
    resolution: int,
    cell_power: float,
    offsets: np.ndarray,
    newer: int,
    probability: float,
    A1: float,
    unit_ball_volume: float,
    dimension: int,
    norm: float,
    infinite_norm: bool,
    dimension_over_norm: float,
    edge_seed: np.uint64,
    cell_pruning: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if resolution == 0:
        return sources, targets, edge_count, overflow_count

    time_volume = float(newer) * unit_ball_volume
    for offset_index in range(offsets.shape[0]):
        offset = offsets[offset_index]
        identifier = _neighbour_cell_id(
            positions[newer], offset, resolution, dimension
        )
        maximum = cell_maximum[identifier]
        if maximum <= 0.0:
            continue
        if cell_pruning:
            lower_measure = _cell_lower_measure(
                positions[newer],
                offset,
                resolution,
                dimension,
                norm,
                infinite_norm,
                dimension_over_norm,
                time_volume,
            )
            if lower_measure >= maximum:
                continue

        older = int(heads[identifier])
        while older != -1:
            following = int(next_vertex[older])
            if overflow[older] == 0:
                if _is_close(
                    positions,
                    influence,
                    older,
                    newer,
                    time_volume,
                    dimension,
                    norm,
                    infinite_norm,
                    dimension_over_norm,
                ) and _edge_is_selected(edge_seed, newer, older, probability):
                    sources, targets, edge_count = _append_edge(
                        sources, targets, edge_count, newer, older
                    )
                    degrees[older] += 1
                    influence[older] += A1
                    if influence[older] > cell_maximum[identifier]:
                        cell_maximum[identifier] = influence[older]
                    overflow_count = _mark_dynamic_overflow(
                        older,
                        newer + 1,
                        influence,
                        overflow,
                        overflow_vertices,
                        overflow_count,
                        resolution,
                        cell_power,
                        unit_ball_volume,
                    )
            older = following

    return sources, targets, edge_count, overflow_count


@njit(cache=True)
def _scan_grid_2d_linf(
    positions: np.ndarray,
    degrees: np.ndarray,
    influence: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    edge_count: int,
    next_vertex: np.ndarray,
    overflow: np.ndarray,
    overflow_vertices: np.ndarray,
    overflow_count: int,
    heads: np.ndarray,
    cell_maximum: np.ndarray,
    resolution: int,
    cell_power: float,
    newer: int,
    probability: float,
    A1: float,
    unit_ball_volume: float,
    edge_seed: np.uint64,
    cell_pruning: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Unrolled 3x3 periodic grid query for the common 2D L-infinity case."""

    if resolution == 0:
        return sources, targets, edge_count, overflow_count

    time_volume = float(newer) * unit_ball_volume
    scaled_x = positions[newer, 0] * resolution
    scaled_y = positions[newer, 1] * resolution
    base_x = int(scaled_x)
    base_y = int(scaled_y)
    if base_x >= resolution:
        base_x = resolution - 1
    if base_y >= resolution:
        base_y = resolution - 1
    fractional_x = scaled_x - float(base_x)
    fractional_y = scaled_y - float(base_y)
    inverse_resolution = 1.0 / float(resolution)

    for offset_x in range(-1, 2):
        cell_x = base_x + offset_x
        if cell_x < 0:
            cell_x += resolution
        elif cell_x >= resolution:
            cell_x -= resolution

        for offset_y in range(-1, 2):
            cell_y = base_y + offset_y
            if cell_y < 0:
                cell_y += resolution
            elif cell_y >= resolution:
                cell_y -= resolution

            identifier = cell_x * resolution + cell_y
            maximum = cell_maximum[identifier]
            if maximum <= 0.0:
                continue
            if cell_pruning:
                lower_measure = _cell_lower_measure_2d_linf(
                    fractional_x,
                    fractional_y,
                    offset_x,
                    offset_y,
                    inverse_resolution,
                    time_volume,
                )
                if lower_measure >= maximum:
                    continue

            older = int(heads[identifier])
            while older != -1:
                following = int(next_vertex[older])
                if overflow[older] == 0:
                    if _is_close_2d_linf(
                        positions, influence, older, newer, time_volume
                    ) and _edge_is_selected(
                        edge_seed, newer, older, probability
                    ):
                        sources, targets, edge_count = _append_edge(
                            sources, targets, edge_count, newer, older
                        )
                        degrees[older] += 1
                        influence[older] += A1
                        if influence[older] > cell_maximum[identifier]:
                            cell_maximum[identifier] = influence[older]
                        overflow_count = _mark_dynamic_overflow(
                            older,
                            newer + 1,
                            influence,
                            overflow,
                            overflow_vertices,
                            overflow_count,
                            resolution,
                            cell_power,
                            unit_ball_volume,
                        )
                older = following

    return sources, targets, edge_count, overflow_count


@njit(cache=True)
def _scan_range(
    positions: np.ndarray,
    degrees: np.ndarray,
    influence: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    edge_count: int,
    start: int,
    stop: int,
    newer: int,
    probability: float,
    A1: float,
    unit_ball_volume: float,
    dimension: int,
    norm: float,
    infinite_norm: bool,
    dimension_over_norm: float,
    edge_seed: np.uint64,
) -> tuple[np.ndarray, np.ndarray, int]:
    time_volume = float(newer) * unit_ball_volume
    for older in range(start, stop):
        if _is_close(
            positions,
            influence,
            older,
            newer,
            time_volume,
            dimension,
            norm,
            infinite_norm,
            dimension_over_norm,
        ) and _edge_is_selected(edge_seed, newer, older, probability):
            sources, targets, edge_count = _append_edge(
                sources, targets, edge_count, newer, older
            )
            degrees[older] += 1
            influence[older] += A1
    return sources, targets, edge_count


@njit(cache=True)
def _scan_range_2d_linf(
    positions: np.ndarray,
    degrees: np.ndarray,
    influence: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    edge_count: int,
    start: int,
    stop: int,
    newer: int,
    probability: float,
    A1: float,
    unit_ball_volume: float,
    edge_seed: np.uint64,
) -> tuple[np.ndarray, np.ndarray, int]:
    time_volume = float(newer) * unit_ball_volume
    for older in range(start, stop):
        if _is_close_2d_linf(
            positions, influence, older, newer, time_volume
        ) and _edge_is_selected(edge_seed, newer, older, probability):
            sources, targets, edge_count = _append_edge(
                sources, targets, edge_count, newer, older
            )
            degrees[older] += 1
            influence[older] += A1
    return sources, targets, edge_count


@njit(cache=True)
def _scan_overflow(
    positions: np.ndarray,
    degrees: np.ndarray,
    influence: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    edge_count: int,
    overflow_vertices: np.ndarray,
    query_count: int,
    newer: int,
    probability: float,
    A1: float,
    unit_ball_volume: float,
    dimension: int,
    norm: float,
    infinite_norm: bool,
    dimension_over_norm: float,
    edge_seed: np.uint64,
) -> tuple[np.ndarray, np.ndarray, int]:
    time_volume = float(newer) * unit_ball_volume
    for index in range(query_count):
        older = int(overflow_vertices[index])
        if _is_close(
            positions,
            influence,
            older,
            newer,
            time_volume,
            dimension,
            norm,
            infinite_norm,
            dimension_over_norm,
        ) and _edge_is_selected(edge_seed, newer, older, probability):
            sources, targets, edge_count = _append_edge(
                sources, targets, edge_count, newer, older
            )
            degrees[older] += 1
            influence[older] += A1
    return sources, targets, edge_count


@njit(cache=True)
def _scan_overflow_2d_linf(
    positions: np.ndarray,
    degrees: np.ndarray,
    influence: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    edge_count: int,
    overflow_vertices: np.ndarray,
    query_count: int,
    newer: int,
    probability: float,
    A1: float,
    unit_ball_volume: float,
    edge_seed: np.uint64,
) -> tuple[np.ndarray, np.ndarray, int]:
    time_volume = float(newer) * unit_ball_volume
    for index in range(query_count):
        older = int(overflow_vertices[index])
        if _is_close_2d_linf(
            positions, influence, older, newer, time_volume
        ) and _edge_is_selected(edge_seed, newer, older, probability):
            sources, targets, edge_count = _append_edge(
                sources, targets, edge_count, newer, older
            )
            degrees[older] += 1
            influence[older] += A1
    return sources, targets, edge_count



@njit(cache=True)
def _insert_new_vertex(
    positions: np.ndarray,
    influence: np.ndarray,
    next_vertex: np.ndarray,
    overflow: np.ndarray,
    overflow_vertices: np.ndarray,
    overflow_count: int,
    young_heads: np.ndarray,
    young_cell_maximum: np.ndarray,
    young_resolution: int,
    young_cell_power: float,
    newer: int,
    unit_ball_volume: float,
    dimension: int,
) -> int:
    overflow[newer] = 0
    if young_resolution == 0:
        next_vertex[newer] = -1
        return overflow_count

    safe_influence = float(newer + 1) * unit_ball_volume * young_cell_power
    if influence[newer] > safe_influence:
        overflow[newer] = 1
        overflow_vertices[overflow_count] = newer
        overflow_count += 1
        next_vertex[newer] = -1
        return overflow_count

    identifier = _cell_id(positions[newer], young_resolution, dimension)
    next_vertex[newer] = young_heads[identifier]
    young_heads[identifier] = newer
    if influence[newer] > young_cell_maximum[identifier]:
        young_cell_maximum[identifier] = influence[newer]
    return overflow_count


@njit(cache=True, inline="always")
def _power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@njit(cache=True)
def advance(
    positions: np.ndarray,
    degrees: np.ndarray,
    influence: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    edge_count: int,
    next_vertex: np.ndarray,
    overflow: np.ndarray,
    overflow_vertices: np.ndarray,
    overflow_count: int,
    overflow_baseline: int,
    old_heads: np.ndarray,
    young_heads: np.ndarray,
    old_cell_maximum: np.ndarray,
    young_cell_maximum: np.ndarray,
    old_resolution: int,
    young_resolution: int,
    old_cell_power: float,
    young_cell_power: float,
    split: int,
    grid_initialized: bool,
    start: int,
    stop: int,
    probability: float,
    A1: float,
    unit_ball_volume: float,
    dimension: int,
    norm: float,
    infinite_norm: bool,
    dimension_over_norm: float,
    alpha: float,
    grid_start: int,
    offsets: np.ndarray,
    edge_seed: np.uint64,
    grid_cells_per_vertex: float,
    max_grid_cells: int,
    max_static_overflow: int,
    cell_pruning: bool,
) -> tuple:
    """Advance the model from vertex ``start`` up to but excluding ``stop``."""

    neighbour_count = offsets.shape[0]
    specialized_2d_linf = dimension == 2 and infinite_norm

    for newer in range(start, stop):
        if newer < grid_start or neighbour_count == 0:
            if specialized_2d_linf:
                sources, targets, edge_count = _scan_range_2d_linf(
                    positions,
                    degrees,
                    influence,
                    sources,
                    targets,
                    edge_count,
                    0,
                    newer,
                    newer,
                    probability,
                    A1,
                    unit_ball_volume,
                    edge_seed,
                )
            else:
                sources, targets, edge_count = _scan_range(
                    positions,
                    degrees,
                    influence,
                    sources,
                    targets,
                    edge_count,
                    0,
                    newer,
                    newer,
                    probability,
                    A1,
                    unit_ball_volume,
                    dimension,
                    norm,
                    infinite_norm,
                    dimension_over_norm,
                    edge_seed,
                )
            continue

        dynamic_overflow = overflow_count - overflow_baseline
        rebuild_threshold = int(math.sqrt(float(newer)))
        if rebuild_threshold < 64:
            rebuild_threshold = 64

        if (
            not grid_initialized
            or _power_of_two(newer)
            or dynamic_overflow > rebuild_threshold
        ):
            (
                old_heads,
                young_heads,
                old_cell_maximum,
                young_cell_maximum,
                old_resolution,
                young_resolution,
                old_cell_power,
                young_cell_power,
                split,
                overflow_count,
                overflow_baseline,
            ) = _rebuild_grids(
                positions,
                influence,
                next_vertex,
                overflow,
                overflow_vertices,
                newer,
                alpha,
                unit_ball_volume,
                dimension,
                grid_cells_per_vertex,
                max_grid_cells,
                neighbour_count,
                max_static_overflow,
            )
            grid_initialized = True

        # Vertices that become overflow during this query were already tested in
        # their grid cell and must not be tested a second time for the same edge.
        overflow_query_count = overflow_count

        if old_resolution == 0:
            if specialized_2d_linf:
                sources, targets, edge_count = _scan_range_2d_linf(
                    positions,
                    degrees,
                    influence,
                    sources,
                    targets,
                    edge_count,
                    0,
                    split,
                    newer,
                    probability,
                    A1,
                    unit_ball_volume,
                    edge_seed,
                )
            else:
                sources, targets, edge_count = _scan_range(
                    positions,
                    degrees,
                    influence,
                    sources,
                    targets,
                    edge_count,
                    0,
                    split,
                    newer,
                    probability,
                    A1,
                    unit_ball_volume,
                    dimension,
                    norm,
                    infinite_norm,
                    dimension_over_norm,
                    edge_seed,
                )
        elif specialized_2d_linf:
            sources, targets, edge_count, overflow_count = _scan_grid_2d_linf(
                positions,
                degrees,
                influence,
                sources,
                targets,
                edge_count,
                next_vertex,
                overflow,
                overflow_vertices,
                overflow_count,
                old_heads,
                old_cell_maximum,
                old_resolution,
                old_cell_power,
                newer,
                probability,
                A1,
                unit_ball_volume,
                edge_seed,
                cell_pruning,
            )
        else:
            sources, targets, edge_count, overflow_count = _scan_grid(
                positions,
                degrees,
                influence,
                sources,
                targets,
                edge_count,
                next_vertex,
                overflow,
                overflow_vertices,
                overflow_count,
                old_heads,
                old_cell_maximum,
                old_resolution,
                old_cell_power,
                offsets,
                newer,
                probability,
                A1,
                unit_ball_volume,
                dimension,
                norm,
                infinite_norm,
                dimension_over_norm,
                edge_seed,
                cell_pruning,
            )

        if young_resolution == 0:
            if specialized_2d_linf:
                sources, targets, edge_count = _scan_range_2d_linf(
                    positions,
                    degrees,
                    influence,
                    sources,
                    targets,
                    edge_count,
                    split,
                    newer,
                    newer,
                    probability,
                    A1,
                    unit_ball_volume,
                    edge_seed,
                )
            else:
                sources, targets, edge_count = _scan_range(
                    positions,
                    degrees,
                    influence,
                    sources,
                    targets,
                    edge_count,
                    split,
                    newer,
                    newer,
                    probability,
                    A1,
                    unit_ball_volume,
                    dimension,
                    norm,
                    infinite_norm,
                    dimension_over_norm,
                    edge_seed,
                )
        elif specialized_2d_linf:
            sources, targets, edge_count, overflow_count = _scan_grid_2d_linf(
                positions,
                degrees,
                influence,
                sources,
                targets,
                edge_count,
                next_vertex,
                overflow,
                overflow_vertices,
                overflow_count,
                young_heads,
                young_cell_maximum,
                young_resolution,
                young_cell_power,
                newer,
                probability,
                A1,
                unit_ball_volume,
                edge_seed,
                cell_pruning,
            )
        else:
            sources, targets, edge_count, overflow_count = _scan_grid(
                positions,
                degrees,
                influence,
                sources,
                targets,
                edge_count,
                next_vertex,
                overflow,
                overflow_vertices,
                overflow_count,
                young_heads,
                young_cell_maximum,
                young_resolution,
                young_cell_power,
                offsets,
                newer,
                probability,
                A1,
                unit_ball_volume,
                dimension,
                norm,
                infinite_norm,
                dimension_over_norm,
                edge_seed,
                cell_pruning,
            )

        if overflow_query_count:
            if specialized_2d_linf:
                sources, targets, edge_count = _scan_overflow_2d_linf(
                    positions,
                    degrees,
                    influence,
                    sources,
                    targets,
                    edge_count,
                    overflow_vertices,
                    overflow_query_count,
                    newer,
                    probability,
                    A1,
                    unit_ball_volume,
                    edge_seed,
                )
            else:
                sources, targets, edge_count = _scan_overflow(
                    positions,
                    degrees,
                    influence,
                    sources,
                    targets,
                    edge_count,
                    overflow_vertices,
                    overflow_query_count,
                    newer,
                    probability,
                    A1,
                    unit_ball_volume,
                    dimension,
                    norm,
                    infinite_norm,
                    dimension_over_norm,
                    edge_seed,
                )

        overflow_count = _insert_new_vertex(
            positions,
            influence,
            next_vertex,
            overflow,
            overflow_vertices,
            overflow_count,
            young_heads,
            young_cell_maximum,
            young_resolution,
            young_cell_power,
            newer,
            unit_ball_volume,
            dimension,
        )

    return (
        sources,
        targets,
        edge_count,
        old_heads,
        young_heads,
        old_cell_maximum,
        young_cell_maximum,
        old_resolution,
        young_resolution,
        old_cell_power,
        young_cell_power,
        split,
        grid_initialized,
        overflow_count,
        overflow_baseline,
    )

