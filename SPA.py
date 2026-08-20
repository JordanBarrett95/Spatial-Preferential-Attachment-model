"""Numba-accelerated spatial preferential attachment sampler.

The hot path uses contiguous NumPy arrays and exact periodic linked-cell grids.
Edges are directed from the new vertex to the older endpoint, as in the attached
implementation.  ``degrees`` is the in-degree array used by the SPA rule.
"""

from __future__ import annotations

import math
import pickle
from itertools import product
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np

try:
    from spa_numba_core import advance
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The accelerated SPA sampler requires numba and its compiled core. "
        "Install the package with `python -m pip install .`."
    ) from exc

__all__ = [
    "SPA",
    "ArrayGraphView",
    "save_spa_graphs",
    "load_spa_graph",
    "draw_spa_graph",
]
__version__ = "0.3.0"

_INT32_MAX = np.iinfo(np.int32).max


class ArrayGraphView:
    """Read-only graph facade over the sampler's edge arrays.

    This intentionally exposes only cheap operations.  Convert explicitly for
    general graph algorithms.
    """

    def __init__(self, spa: "SPA") -> None:
        self._spa = spa

    def vcount(self) -> int:
        return self._spa.num_vertices

    def ecount(self) -> int:
        return self._spa.num_edges

    def summary(self) -> str:
        return (
            f"Directed SPA edge-array graph: {self.vcount()} vertices, "
            f"{self.ecount()} edges"
        )

    def iter_edges(self) -> Iterator[tuple[int, int]]:
        return self._spa.iter_edges()

    def get_edgelist(self) -> list[tuple[int, int]]:
        return list(self.iter_edges())

    def degree(
        self, mode: Literal["all", "in", "out"] = "all"
    ) -> list[int]:
        return self._spa.degree(mode=mode).tolist()

    def to_igraph(self):
        return self._spa.to_igraph()

    def to_networkx(self):
        return self._spa.to_networkx()

    def __repr__(self) -> str:
        return self.summary()


class SPA:
    """Sample the spatial preferential attachment model.

    Parameters
    ----------
    n:
        Initial number of vertices.  At least one.
    p, A1, A2, dimension, norm, alpha:
        SPA model parameters with the same meaning as in the attached code.
    progress:
        Show a chunk-level tqdm progress bar.  tqdm is optional.
    seed:
        Reproducible position and edge seed.  Results do not depend on how
        construction is divided among ``update`` calls.
    grid_start:
        Use a compiled brute-force scan below this vertex count, then build
        periodic grids.
    max_grid_cells, grid_cells_per_vertex:
        Memory limits for each dense cell-head array.  Limits make a grid
        coarser, never approximate.
    max_neighbour_cells:
        Above this value of ``3**dimension``, use exact compiled brute force.
    max_static_overflow:
        Maximum number of high-influence outliers considered for an exact
        overflow scan when choosing each grid's cell size.
    cell_pruning:
        Skip complete cells when their maximum influence is too small to reach
        the new vertex.  This is exact and can be disabled for validation.

    Notes
    -----
    The grid search is exact relative to the sampled positions and Bernoulli
    values.  New vertices are inserted immediately.  Vertices whose radii grow
    beyond their cell are skipped in the stale grid and scanned through an
    overflow list until rebuilding.
    """

    def __init__(
        self,
        n: int,
        p: float = 0.75,
        A1: float = 1.0,
        A2: float = 1.0,
        dimension: int = 2,
        norm: float = np.inf,
        alpha: float = 0.65,
        progress: bool = False,
        *,
        seed: int | None = None,
        grid_start: int = 1024,
        max_grid_cells: int = 8_000_000,
        grid_cells_per_vertex: float = 8.0,
        max_neighbour_cells: int = 729,
        max_static_overflow: int = 256,
        cell_pruning: bool = True,
    ) -> None:
        self._validate(
            n=n,
            p=p,
            A1=A1,
            A2=A2,
            dimension=dimension,
            norm=norm,
            alpha=alpha,
            grid_start=grid_start,
            max_grid_cells=max_grid_cells,
            grid_cells_per_vertex=grid_cells_per_vertex,
            max_neighbour_cells=max_neighbour_cells,
            max_static_overflow=max_static_overflow,
        )

        self.p = float(p)
        self.A1 = float(A1)
        self.A2 = float(A2)
        self.dimension = int(dimension)
        self.norm = float(norm)
        self.alpha = float(alpha)
        self.progress = bool(progress)
        self.seed = seed

        self.grid_start = int(grid_start)
        self.max_grid_cells = int(max_grid_cells)
        self.grid_cells_per_vertex = float(grid_cells_per_vertex)
        self.max_neighbour_cells = int(max_neighbour_cells)
        self.max_static_overflow = int(max_static_overflow)
        self.cell_pruning = bool(cell_pruning)

        self.is_infinite_norm = bool(np.isinf(self.norm))
        self.dimension_inverse = 1.0 / self.dimension
        self.dimension_over_norm = (
            0.0 if self.is_infinite_norm else self.dimension / self.norm
        )
        if self.is_infinite_norm:
            self.unit_ball_volume = float(2**self.dimension)
        else:
            inverse_norm = 1.0 / self.norm
            self.unit_ball_volume = float(
                (2.0 * math.gamma(1.0 + inverse_norm)) ** self.dimension
                / math.gamma(1.0 + self.dimension / self.norm)
            )

        seed_sequence = np.random.SeedSequence(seed)
        position_sequence, edge_sequence = seed_sequence.spawn(2)
        self._position_rng = np.random.default_rng(position_sequence)
        self._edge_seed = edge_sequence.generate_state(1, dtype=np.uint64)[0]

        capacity = max(1, int(n))
        self._positions = np.empty((capacity, self.dimension), dtype=np.float64)
        self._degrees = np.zeros(capacity, dtype=np.int64)
        self._influence = np.full(capacity, self.A2, dtype=np.float64)
        self._next_vertex = np.full(capacity, -1, dtype=np.int32)
        self._overflow = np.zeros(capacity, dtype=np.uint8)
        self._overflow_vertices = np.empty(capacity, dtype=np.int32)

        edge_capacity = max(16, 4 * int(n))
        self._edge_sources = np.empty(edge_capacity, dtype=np.int32)
        self._edge_targets = np.empty(edge_capacity, dtype=np.int32)
        self._edge_count = 0

        self._old_heads = np.full(1, -1, dtype=np.int32)
        self._young_heads = np.full(1, -1, dtype=np.int32)
        self._old_cell_maximum = np.zeros(1, dtype=np.float64)
        self._young_cell_maximum = np.zeros(1, dtype=np.float64)
        self._old_resolution = 0
        self._young_resolution = 0
        self._old_cell_power = 0.0
        self._young_cell_power = 0.0
        self._split = 0
        self._grid_initialized = False
        self._overflow_count = 0
        self._overflow_baseline = 0

        self._offsets = self._make_offsets(
            self.dimension, self.max_neighbour_cells
        )
        self._size = 1
        self._positions[0] = self._position_rng.random(self.dimension)
        self._igraph_cache = None

        if n > 1:
            self.update(n - 1)

    @staticmethod
    def _validate(
        *,
        n: int,
        p: float,
        A1: float,
        A2: float,
        dimension: int,
        norm: float,
        alpha: float,
        grid_start: int,
        max_grid_cells: int,
        grid_cells_per_vertex: float,
        max_neighbour_cells: int,
        max_static_overflow: int,
    ) -> None:
        if not isinstance(n, (int, np.integer)) or int(n) < 1:
            raise ValueError("n must be a positive integer")
        if int(n) > _INT32_MAX:
            raise ValueError("n must fit in a signed 32-bit vertex identifier")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between 0 and 1")
        if A1 <= 0.0:
            raise ValueError("A1 must be positive")
        if A2 <= 0.0:
            raise ValueError("A2 must be positive")
        if p * A1 < 0.5 or p * A1 > 1.0:
            raise ValueError("p*A1 must be between 1/2 and 1")
        if not isinstance(dimension, (int, np.integer)) or int(dimension) < 1:
            raise ValueError("dimension must be a positive integer")
        if not (np.isinf(norm) or (np.isfinite(norm) and norm > 0.0)):
            raise ValueError("norm must be positive or numpy.inf")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not isinstance(grid_start, (int, np.integer)) or int(grid_start) < 2:
            raise ValueError("grid_start must be an integer of at least 2")
        if max_grid_cells < 1:
            raise ValueError("max_grid_cells must be positive")
        if grid_cells_per_vertex <= 0.0:
            raise ValueError("grid_cells_per_vertex must be positive")
        if max_neighbour_cells < 1:
            raise ValueError("max_neighbour_cells must be positive")
        if max_static_overflow < 0:
            raise ValueError("max_static_overflow must be non-negative")

    @staticmethod
    def _make_offsets(dimension: int, maximum: int) -> np.ndarray:
        count = 3**dimension
        if count > maximum:
            return np.empty((0, dimension), dtype=np.int8)
        return np.asarray(
            list(product((-1, 0, 1), repeat=dimension)), dtype=np.int8
        )

    @staticmethod
    def _readonly(array: np.ndarray) -> np.ndarray:
        view = array.view()
        view.flags.writeable = False
        return view

    @property
    def num_vertices(self) -> int:
        return self._size

    @property
    def num_edges(self) -> int:
        return self._edge_count

    @property
    def positions(self) -> np.ndarray:
        """Read-only view of vertex positions."""

        return self._readonly(self._positions[: self._size])

    @property
    def degrees(self) -> np.ndarray:
        """Read-only in-degree view used by the attachment rule."""

        return self._readonly(self._degrees[: self._size])

    @property
    def edge_sources(self) -> np.ndarray:
        """Read-only newer endpoint for every directed edge."""

        return self._readonly(self._edge_sources[: self._edge_count])

    @property
    def edge_targets(self) -> np.ndarray:
        """Read-only older endpoint for every directed edge."""

        return self._readonly(self._edge_targets[: self._edge_count])

    @property
    def edges(self) -> np.ndarray:
        """An ``(m, 2)`` copy of directed ``(source, target)`` pairs."""

        return np.column_stack((self.edge_sources, self.edge_targets))

    @property
    def G(self) -> ArrayGraphView:
        """Lightweight compatibility facade; this is not an igraph object."""

        return ArrayGraphView(self)

    @property
    def adj(self) -> list[list[int]]:
        """Materialize the legacy target-to-newer-neighbours adjacency lists."""

        adjacency: list[list[int]] = [[] for _ in range(self._size)]
        for source, target in self.iter_edges():
            adjacency[target].append(source)
        return adjacency

    @property
    def grid_diagnostics(self) -> dict[str, int | bool]:
        """Current spatial-index metadata useful when tuning large samples."""

        return {
            "enabled": bool(self._offsets.shape[0]),
            "old_resolution": self._old_resolution,
            "young_resolution": self._young_resolution,
            "old_cells": int(self._old_heads.shape[0]),
            "young_cells": int(self._young_heads.shape[0]),
            "split": self._split,
            "overflow_vertices": self._overflow_count,
            "static_overflow_vertices": self._overflow_baseline,
            "cell_pruning": self.cell_pruning,
            "specialized_2d_linf": self.dimension == 2 and self.is_infinite_norm,
        }

    def __repr__(self) -> str:
        backend = "periodic grids" if self._offsets.shape[0] else "brute force"
        return (
            f"SPA(n={self.num_vertices}, m={self.num_edges}, "
            f"dimension={self.dimension}, norm={self.norm}, backend={backend!r})"
        )

    def _ensure_vertex_capacity(self, required: int) -> None:
        if required <= self._positions.shape[0]:
            return
        if required > _INT32_MAX:
            raise ValueError("vertex identifiers must fit in signed int32")

        old_capacity = self._positions.shape[0]
        new_capacity = max(required, old_capacity * 2)
        if new_capacity > _INT32_MAX:
            new_capacity = required

        positions = np.empty((new_capacity, self.dimension), dtype=np.float64)
        positions[: self._size] = self._positions[: self._size]
        self._positions = positions

        degrees = np.zeros(new_capacity, dtype=np.int64)
        degrees[: self._size] = self._degrees[: self._size]
        self._degrees = degrees

        influence = np.full(new_capacity, self.A2, dtype=np.float64)
        influence[: self._size] = self._influence[: self._size]
        self._influence = influence

        next_vertex = np.full(new_capacity, -1, dtype=np.int32)
        next_vertex[: self._size] = self._next_vertex[: self._size]
        self._next_vertex = next_vertex

        overflow = np.zeros(new_capacity, dtype=np.uint8)
        overflow[: self._size] = self._overflow[: self._size]
        self._overflow = overflow

        overflow_vertices = np.empty(new_capacity, dtype=np.int32)
        if self._overflow_count:
            overflow_vertices[: self._overflow_count] = self._overflow_vertices[
                : self._overflow_count
            ]
        self._overflow_vertices = overflow_vertices

    def update(self, num_new: int = 1) -> None:
        """Append vertices and sample their directed edges to older vertices."""

        if not isinstance(num_new, (int, np.integer)) or int(num_new) < 0:
            raise ValueError("num_new must be a non-negative integer")
        num_new = int(num_new)
        if num_new == 0:
            return

        start = self._size
        stop = start + num_new
        self._ensure_vertex_capacity(stop)
        self._positions[start:stop] = self._position_rng.random(
            (num_new, self.dimension)
        )
        self._degrees[start:stop] = 0
        self._influence[start:stop] = self.A2
        self._next_vertex[start:stop] = -1
        self._overflow[start:stop] = 0

        if self.progress:
            try:
                from tqdm.auto import tqdm
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "progress=True requires tqdm; install the 'progress' extra."
                ) from exc
            chunk_size = max(1, min(8192, num_new))
            for chunk_start in tqdm(
                range(start, stop, chunk_size), desc="Building SPA"
            ):
                chunk_stop = min(stop, chunk_start + chunk_size)
                self._advance(chunk_start, chunk_stop)
                self._size = chunk_stop
        else:
            self._advance(start, stop)
            self._size = stop

        self._igraph_cache = None

    def _advance(self, start: int, stop: int) -> None:
        (
            self._edge_sources,
            self._edge_targets,
            self._edge_count,
            self._old_heads,
            self._young_heads,
            self._old_cell_maximum,
            self._young_cell_maximum,
            self._old_resolution,
            self._young_resolution,
            self._old_cell_power,
            self._young_cell_power,
            self._split,
            self._grid_initialized,
            self._overflow_count,
            self._overflow_baseline,
        ) = advance(
            self._positions,
            self._degrees,
            self._influence,
            self._edge_sources,
            self._edge_targets,
            self._edge_count,
            self._next_vertex,
            self._overflow,
            self._overflow_vertices,
            self._overflow_count,
            self._overflow_baseline,
            self._old_heads,
            self._young_heads,
            self._old_cell_maximum,
            self._young_cell_maximum,
            self._old_resolution,
            self._young_resolution,
            self._old_cell_power,
            self._young_cell_power,
            self._split,
            self._grid_initialized,
            start,
            stop,
            self.p,
            self.A1,
            self.unit_ball_volume,
            self.dimension,
            self.norm,
            self.is_infinite_norm,
            self.dimension_over_norm,
            self.alpha,
            self.grid_start,
            self._offsets,
            self._edge_seed,
            self.grid_cells_per_vertex,
            self.max_grid_cells,
            self.max_static_overflow,
            self.cell_pruning,
        )

    def degree(
        self, mode: Literal["all", "in", "out"] = "all"
    ) -> np.ndarray:
        normalized = mode.lower()
        if normalized not in {"all", "in", "out"}:
            raise ValueError("mode must be 'all', 'in', or 'out'")
        in_degree = self.degrees.copy()
        if normalized == "in":
            return in_degree
        out_degree = np.bincount(
            self.edge_sources, minlength=self._size
        ).astype(np.int64, copy=False)
        if normalized == "out":
            return out_degree
        return in_degree + out_degree

    def iter_edges(self) -> Iterator[tuple[int, int]]:
        for source, target in zip(self.edge_sources, self.edge_targets):
            yield int(source), int(target)

    def to_igraph(self):
        """Materialize and cache an optional :mod:`igraph` graph."""

        if self._igraph_cache is not None:
            return self._igraph_cache
        try:
            import igraph as ig
        except ImportError as exc:
            raise ImportError(
                "to_igraph() requires python-igraph; install the 'igraph' extra."
            ) from exc
        graph = ig.Graph(
            n=self._size,
            edges=self.G.get_edgelist(),
            directed=True,
        )
        graph.vs["pos"] = self.positions.tolist()
        self._igraph_cache = graph
        return graph

    def to_networkx(self):
        """Materialize a :class:`networkx.DiGraph` with position attributes."""

        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "to_networkx() requires networkx; install the 'draw' extra."
            ) from exc
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self._size))
        graph.add_edges_from(self.iter_edges())
        nx.set_node_attributes(
            graph,
            {vertex: self.positions[vertex].copy() for vertex in range(self._size)},
            "pos",
        )
        return graph

    def draw(
        self,
        node_size: float = 1.0,
        show_torus_edges: bool = False,
        filename: str | Path | None = None,
    ) -> None:
        draw_spa_graph(
            self,
            node_size=node_size,
            show_torus_edges=show_torus_edges,
            filename=filename,
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_igraph_cache"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore current and v0.2.x pickles safely.

        Older accelerated pickles did not contain per-cell influence maxima.
        They are initialized here and the grids are rebuilt before the next
        insertion, preserving exact continuation.
        """

        self.__dict__.update(state)
        self._igraph_cache = None
        if not hasattr(self, "cell_pruning"):
            self.cell_pruning = True
        if not hasattr(self, "_old_cell_maximum"):
            self._old_cell_maximum = np.zeros(
                self._old_heads.shape[0], dtype=np.float64
            )
            self._grid_initialized = False
        if not hasattr(self, "_young_cell_maximum"):
            self._young_cell_maximum = np.zeros(
                self._young_heads.shape[0], dtype=np.float64
            )
            self._grid_initialized = False


def save_spa_graphs(
    filename: str | Path,
    n_values: Sequence[int],
    p_values: Sequence[float] = (0.75,),
    A1_values: Sequence[float] = (1.0,),
    A2_values: Sequence[float] = (1.0,),
    dimension_values: Sequence[int] = (2,),
    norm_values: Sequence[float] = (np.inf,),
    alpha_values: Sequence[float] = (0.65,),
    *,
    seed: int | None = None,
) -> None:
    """Sample and pickle a parameter-indexed collection of SPA objects."""

    parameter_sets = [
        (n, p, A1, A2, dimension, norm, alpha)
        for n in n_values
        for p in p_values
        for A1 in A1_values
        for A2 in A2_values
        for dimension in dimension_values
        for norm in norm_values
        for alpha in alpha_values
    ]
    child_sequences = np.random.SeedSequence(seed).spawn(len(parameter_sets))
    graphs: dict[tuple, SPA] = {}

    for parameters, child_sequence in zip(parameter_sets, child_sequences):
        n, p, A1, A2, dimension, norm, alpha = parameters
        print(
            f"n={n}, p={p}, A1={A1}, A2={A2}, "
            f"dimension={dimension}, norm={norm}, alpha={alpha}"
        )
        graph_seed = int(child_sequence.generate_state(1, dtype=np.uint64)[0])
        graphs[parameters] = SPA(
            n=n,
            p=p,
            A1=A1,
            A2=A2,
            dimension=dimension,
            norm=norm,
            alpha=alpha,
            seed=graph_seed,
        )

    with Path(filename).open("wb") as file:
        pickle.dump(graphs, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_spa_graph(
    filename: str | Path,
    n: int,
    p: float = 0.75,
    A1: float = 1.0,
    A2: float = 1.0,
    dimension: int = 2,
    norm: float = np.inf,
    alpha: float = 0.65,
) -> SPA:
    with Path(filename).open("rb") as file:
        graphs = pickle.load(file)
    key = (n, p, A1, A2, dimension, norm, alpha)
    try:
        return graphs[key]
    except KeyError as exc:
        raise KeyError(f"No SPA graph stored for parameters {key!r}") from exc


def draw_spa_graph(
    graph: SPA | ArrayGraphView,
    node_size: float = 1.0,
    show_torus_edges: bool = False,
    filename: str | Path | None = None,
) -> None:
    """Draw a two-dimensional SPA sample without depending on igraph."""

    if isinstance(graph, ArrayGraphView):
        spa = graph._spa
    elif isinstance(graph, SPA):
        spa = graph
    else:
        raise TypeError("graph must be an SPA instance or ArrayGraphView")
    if spa.dimension != 2:
        raise ValueError("draw_spa_graph supports only two-dimensional positions")

    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "Drawing requires matplotlib and networkx; install the 'draw' extra."
        ) from exc

    undirected = nx.Graph()
    undirected.add_nodes_from(range(spa.num_vertices))
    shift = np.array([0.5, 0.5]) - spa.positions[0]
    shifted = (spa.positions + shift) % 1.0

    for source, target in spa.iter_edges():
        if not show_torus_edges and np.any(
            np.abs(shifted[source] - shifted[target]) > 0.5
        ):
            continue
        undirected.add_edge(source, target)

    components = list(nx.connected_components(undirected))
    component_id = np.zeros(spa.num_vertices, dtype=np.int64)
    for identifier, component in enumerate(components):
        for vertex in component:
            component_id[vertex] = identifier

    positions = {v: shifted[v] for v in range(spa.num_vertices)}
    colour_map = plt.get_cmap("tab20", max(1, len(components)))
    colours = [colour_map(component_id[v]) for v in undirected.nodes()]

    plt.figure(figsize=(6, 6))
    nx.draw(
        undirected,
        pos=positions,
        node_size=node_size,
        with_labels=False,
        edge_color="black",
        node_color=colours,
        width=0.6,
    )
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
