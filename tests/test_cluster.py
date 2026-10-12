"""KMeans clustering: determinism above all, because labels set every downstream distance."""

import random

import pytest

from smartwaste.routing.cluster import ClusterResult, kmeans

pytestmark = pytest.mark.routing


def _random_points(n: int, seed: int, span: float = 100.0) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(rng.uniform(0, span), rng.uniform(0, span)) for _ in range(n)]


def test_same_seed_gives_identical_labels() -> None:
    """The headline determinism property: same points, same k, same seed, same labels."""
    points = _random_points(300, seed=1)

    first = kmeans(points, 8, random.Random(42))
    second = kmeans(points, 8, random.Random(42))

    assert first.labels == second.labels
    assert first.centroids == second.centroids


def test_different_seeds_can_give_different_labels() -> None:
    """Determinism must come from the seed, not from the algorithm ignoring it."""
    points = _random_points(300, seed=2)

    a = kmeans(points, 8, random.Random(1))
    b = kmeans(points, 8, random.Random(2))

    # Not guaranteed different in principle, but with 300 points and k=8 it would be remarkable.
    assert a.labels != b.labels


def test_every_point_gets_exactly_one_valid_label() -> None:
    points = _random_points(250, seed=3)
    result = kmeans(points, 20, random.Random(42))

    assert len(result.labels) == len(points)
    assert all(0 <= label < result.n_clusters for label in result.labels)


def test_clusters_partition_the_bins() -> None:
    """Membership lists must together contain every bin exactly once."""
    points = _random_points(200, seed=4)
    result = kmeans(points, 10, random.Random(42))

    everything: list[int] = []
    for cluster_id in range(result.n_clusters):
        everything.extend(result.members(cluster_id))

    assert sorted(everything) == list(range(len(points)))


def test_members_are_ascending() -> None:
    points = _random_points(100, seed=5)
    result = kmeans(points, 5, random.Random(42))
    for cluster_id in range(result.n_clusters):
        members = result.members(cluster_id)
        assert members == sorted(members)


def test_k_is_produced_when_there_are_enough_points() -> None:
    points = _random_points(100, seed=6)
    assert kmeans(points, 20, random.Random(42)).n_clusters == 20


def test_k_is_clamped_to_the_point_count() -> None:
    """Asking for more territories than bins yields one per bin, not an empty cluster."""
    points = _random_points(3, seed=7)
    result = kmeans(points, 10, random.Random(42))
    assert result.n_clusters == 3
    assert sorted(result.labels) == [0, 1, 2]


def test_well_separated_blobs_are_recovered() -> None:
    """Sanity: three far-apart blobs must land in three different clusters."""
    blobs = [(0.0, 0.0), (0.5, 0.5), (100.0, 100.0), (100.5, 100.5), (0.0, 100.0), (0.5, 100.5)]

    result = kmeans(blobs, 3, random.Random(42))

    # Points within a blob share a label; points in different blobs do not.
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] == result.labels[3]
    assert result.labels[4] == result.labels[5]
    assert len({result.labels[0], result.labels[2], result.labels[4]}) == 3


def test_identical_points_are_handled() -> None:
    """A degenerate layout (every bin stacked) must not loop forever or crash."""
    points = [(5.0, 5.0)] * 10
    result = kmeans(points, 3, random.Random(42))
    assert len(result.labels) == 10


def test_single_point_single_cluster() -> None:
    result = kmeans([(1.0, 2.0)], 1, random.Random(42))
    assert result.labels == (0,)
    assert result.centroids == ((1.0, 2.0),)


def test_assignment_ties_go_to_the_lowest_label() -> None:
    """A point equidistant from two centroids must break deterministically, not arbitrarily."""
    # Two points, k=2, so each becomes its own centroid; then a third equidistant point.
    points = [(0.0, 0.0), (10.0, 0.0), (5.0, 0.0)]
    first = kmeans(points, 2, random.Random(42))
    second = kmeans(points, 2, random.Random(42))
    assert first.labels == second.labels


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        kmeans([], 3, random.Random(42))


def test_non_positive_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        kmeans([(0.0, 0.0)], 0, random.Random(42))


def test_result_is_frozen() -> None:
    result = kmeans(_random_points(10, seed=8), 2, random.Random(42))
    assert isinstance(result, ClusterResult)
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
        result.labels = ()  # type: ignore[misc]
