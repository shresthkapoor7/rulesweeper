from __future__ import annotations
from abc import ABC, abstractmethod


class Neighborhood(ABC):
    """
    Defines what cells are 'adjacent' for adjacency counts, cascade reveals,
    safe-first-click zones, and any MineBehavior that walks neighbors
    (drifting, chain-reaction). The single chokepoint is Board.neighbors,
    which filters these offsets against the board bounds.
    """

    @abstractmethod
    def offsets(self) -> list[tuple[int, int]]:
        """Return (dr, dc) pairs. Must not include (0, 0)."""
        ...

    def summary(self) -> dict:
        """
        Default summary lists the offsets so an agent can read the topology
        without poking at the private class-level constant.
        """
        return {"offsets": list(self.offsets())}


class MooreNeighborhood(Neighborhood):
    """Standard 8-connected neighborhood — the canonical minesweeper seed."""

    _OFFSETS = [
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    ]

    def offsets(self) -> list[tuple[int, int]]:
        return self._OFFSETS


class VonNeumannNeighborhood(Neighborhood):
    """4 orthogonal neighbors only. Less information per cell — strictly harder."""

    _OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def offsets(self) -> list[tuple[int, int]]:
        return self._OFFSETS


class DiagonalNeighborhood(Neighborhood):
    """4 diagonal neighbors only. Numbers measure the X-pattern around a cell."""

    _OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def offsets(self) -> list[tuple[int, int]]:
        return self._OFFSETS


class KnightNeighborhood(Neighborhood):
    """Chess knight moves. Cascade jumps non-locally — surreal but playable."""

    _OFFSETS = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        ( 1, -2), ( 1, 2), ( 2, -1), ( 2, 1),
    ]

    def offsets(self) -> list[tuple[int, int]]:
        return self._OFFSETS


class Radius2MooreNeighborhood(Neighborhood):
    """24 cells in a 5x5 box. Numbers cover much larger area; far chattier."""

    _OFFSETS = [
        (dr, dc)
        for dr in range(-2, 3)
        for dc in range(-2, 3)
        if (dr, dc) != (0, 0)
    ]

    def offsets(self) -> list[tuple[int, int]]:
        return self._OFFSETS


NEIGHBORHOODS: dict[str, type[Neighborhood]] = {
    "moore":          MooreNeighborhood,
    "von-neumann":    VonNeumannNeighborhood,
    "diagonal":       DiagonalNeighborhood,
    "knight":         KnightNeighborhood,
    "radius-2-moore": Radius2MooreNeighborhood,
}
