"""The five collection strategies the benchmark compares.

Each strategy answers one question per day: **which bins do we send a truck to?** It is given the
day number and that day's sensor readings, and nothing else. It cannot see the true fill, the true
weight or any other strategy's decisions -- a strategy only ever receives
:class:`~smartwaste.noise.SensorReading` objects, which structurally carry no ground truth.

| id | label | what it reads |
|---|---|---|
| a | fixed-route, every bin daily | nothing (the paper's baseline) |
| b | fixed schedule, every N days | nothing |
| c | single ultrasonic threshold | centre sensor only, no load cell |
| d | paper's sensor fusion | 3-sensor MEAN + load cell |
| e | median fusion | 3-sensor MEDIAN + load cell -- **extension, not in the paper** |

(a) and (b) are the fair baselines SPEC section 3 asks for: a schedule can be run with no sensors
at all, so if a schedule matches the sensor strategies then the sensors are not earning their
keep. (c) isolates what the extra two ultrasonic sensors and the load cell actually add.

(d) is the paper's rule, unchanged. (e) is ours: Phase 1 measured the mean collapsing when one
sensor sticks (20.85 pp worse than a single sensor), which a median would survive. (e) is expected
to lose slightly to (d) when no sensor is broken, because a median of three is a noisier estimator
than a mean -- see :func:`smartwaste.fusion.median_distance`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from smartwaste.config import BinConfig
from smartwaste.fusion import BinState, fuse, mean_distance, median_distance
from smartwaste.noise import CENTER_INDEX, SensorReading


class Strategy(ABC):
    """A collection policy: given a day's readings, choose bins to visit."""

    #: Stable short key used in CSV output and cache keys.
    key: str
    #: Human-readable label, printed in the results tables exactly as written.
    label: str
    #: True when the strategy consults sensors at all.
    uses_sensors: bool

    @abstractmethod
    def select(self, day: int, readings: Sequence[SensorReading]) -> list[int]:
        """Indices of the bins to collect today, ascending."""


class FixedRoute(Strategy):
    """(a) Visit every bin every day -- the paper's baseline.

    Reads no sensors, so it never misses a full bin and never overflows. Its cost is the price of
    that guarantee, and it is the number every saving in this project is measured against.
    """

    key = "a_fixed_route"
    label = "(a) fixed-route: all bins daily [paper's baseline]"
    uses_sensors = False

    def select(self, day: int, readings: Sequence[SensorReading]) -> list[int]:
        return list(range(len(readings)))


class FixedSchedule(Strategy):
    """(b) Visit every bin every N days.

    The cheapest honest baseline: no sensors, no telemetry, no capital cost. If this matches the
    sensor strategies on overflow and beats them on cost, the sensors are not paying for
    themselves -- and the results doc says so rather than burying it.
    """

    key = "b_fixed_schedule"
    label = "(b) fixed schedule: all bins every N days"
    uses_sensors = False

    def __init__(self, every_n_days: int) -> None:
        if every_n_days < 1:
            raise ValueError(f"every_n_days must be >= 1, got {every_n_days}")
        self.every_n_days = every_n_days
        self.label = f"(b) fixed schedule: all bins every {every_n_days} days"

    def select(self, day: int, readings: Sequence[SensorReading]) -> list[int]:
        if day % self.every_n_days != 0:
            return []
        return list(range(len(readings)))


class ThresholdStrategy(Strategy):
    """Collect exactly the bins this sensor configuration classifies as RED.

    All three sensor strategies share this: they differ only in which distances they read, whether
    they use the load cell, and how they reduce the distances to ``d_bar``. YELLOW bins are never
    collected, which is the paper's rule and its known weakness (Phase 3 addresses it).
    """

    uses_sensors = True

    def __init__(
        self,
        key: str,
        label: str,
        bin_cfg: BinConfig,
        *,
        use_load_cell: bool,
        sensor_indices: tuple[int, ...] | None,
        reducer: Callable[[Sequence[float]], float] | None = None,
    ) -> None:
        self.key = key
        self.label = label
        self._bin = bin_cfg
        self._use_load_cell = use_load_cell
        self._sensor_indices = sensor_indices
        self._reducer = mean_distance if reducer is None else reducer

    def select(self, day: int, readings: Sequence[SensorReading]) -> list[int]:
        selected: list[int] = []
        for index, reading in enumerate(readings):
            result = fuse(
                reading,
                self._bin,
                use_load_cell=self._use_load_cell,
                sensor_indices=self._sensor_indices,
                reducer=self._reducer,
            )
            if result.state is BinState.RED:
                selected.append(index)
        return selected


def build_strategies(bin_cfg: BinConfig, fixed_schedule_days: int) -> tuple[Strategy, ...]:
    """The five strategies, in the order they appear in every results table."""
    return (
        FixedRoute(),
        FixedSchedule(fixed_schedule_days),
        ThresholdStrategy(
            "c_single_ultrasonic",
            "(c) single-ultrasonic threshold: center sensor only, no load cell",
            bin_cfg,
            use_load_cell=False,
            sensor_indices=(CENTER_INDEX,),
        ),
        ThresholdStrategy(
            "d_paper_fusion",
            "(d) paper's sensor fusion: 3-sensor MEAN + load cell [the paper's method]",
            bin_cfg,
            use_load_cell=True,
            sensor_indices=None,
            reducer=mean_distance,
        ),
        ThresholdStrategy(
            "e_median_fusion",
            "(e) median fusion: 3-sensor MEDIAN + load cell [extension, not in the paper]",
            bin_cfg,
            use_load_cell=True,
            sensor_indices=None,
            reducer=median_distance,
        ),
    )
