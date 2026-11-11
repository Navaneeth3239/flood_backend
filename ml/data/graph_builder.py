"""Geographic and directed hydrological edges for PyTorch Geometric graphs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch


def build_graph(
    nodes: pd.DataFrame | list[dict],
    *,
    k_neighbors: int = 4,
    max_distance_km: float | None = 5.0,
    hydrological_edges: list[tuple[str, str]] | None = None,
    river_edges: list[tuple[str, str]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``edge_index[2,E]`` and ``edge_attr[E,4]`` in input-node order.

    Proximity edges connect locations within ``max_distance_km``. KNN edges also
    connect the nearest ``k_neighbors`` sites, including beyond that radius.
    Known downstream links are directed; river links connect successive sites
    along descending elevation within a ``river_id``. No hydrological connection
    is inferred just from geographic proximity. Distances are kilometres; elevation
    and slope differences are destination minus source in input units. Isolated
    nodes receive self edges. Missing terrain is neutral in edge differences and
    remains explicitly masked in the node's static inputs.
    """
    nodes = pd.DataFrame(nodes).reset_index(drop=True)
    if nodes.empty:
        raise ValueError("A graph needs at least one node")
    if not {"node_id", "latitude", "longitude"}.issubset(nodes.columns):
        raise ValueError("Graph nodes require node_id, latitude and longitude")
    if nodes["node_id"].astype(str).duplicated().any():
        raise ValueError("Graph node_id values must be unique")
    coordinates = nodes[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all() or (np.abs(coordinates[:, 0]) > 90).any() or (np.abs(coordinates[:, 1]) > 180).any():
        raise ValueError("Graph coordinates must be finite WGS84 latitude/longitude")
    if k_neighbors < 0 or (max_distance_km is not None and (not math.isfinite(max_distance_km) or max_distance_km < 0)):
        raise ValueError("Graph neighbor count and distance must be nonnegative")
    radians = np.radians(coordinates)
    lat = radians[:, 0]
    lon = radians[:, 1]
    hav = np.sin((lat[:, None] - lat[None, :]) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2
    distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    count = len(nodes)
    edges: dict[tuple[int, int], float] = {}
    for source in range(count):
        closest = np.argsort(distances[source], kind="stable")
        closest = closest[closest != source][:min(k_neighbors, count - 1)]
        destinations = set(closest.tolist())
        if max_distance_km is not None:
            destinations.update(np.flatnonzero(distances[source] <= max_distance_km).tolist())
        destinations.discard(source)
        for destination in destinations:
            edges[(source, destination)] = 0.0
    node_index = {str(node_id): index for index, node_id in enumerate(nodes["node_id"])}

    def connect(source_id: str, destination_id: str) -> None:
        if str(source_id) in node_index and str(destination_id) in node_index:
            source, destination = node_index[str(source_id)], node_index[str(destination_id)]
            if source != destination:
                edges[(source, destination)] = 1.0

    for source_id, destination_id in (hydrological_edges or []) + (river_edges or []):
        connect(source_id, destination_id)
    if "downstream_node_id" in nodes:
        for _, row in nodes.dropna(subset=["downstream_node_id"]).iterrows():
            connect(row["node_id"], row["downstream_node_id"])
    if {"river_id", "elevation"}.issubset(nodes.columns):
        river_nodes = nodes.dropna(subset=["river_id", "elevation"])
        river_nodes = river_nodes[river_nodes["river_id"].astype(str).str.strip().ne("")]
        for _, group in river_nodes.groupby("river_id", sort=False):
            ordered = group.sort_values("elevation", ascending=False, kind="stable")["node_id"].tolist()
            for source_id, destination_id in zip(ordered, ordered[1:]):
                connect(source_id, destination_id)
    for node in range(count):
        if not any(node in edge for edge in edges):
            edges[(node, node)] = 0.0
    ordered_edges = sorted(edges)
    elevation = pd.to_numeric(nodes.get("elevation", pd.Series(np.nan, index=nodes.index)), errors="coerce").to_numpy()
    slope = pd.to_numeric(nodes.get("slope", pd.Series(np.nan, index=nodes.index)), errors="coerce").to_numpy()
    attributes = []
    for source, destination in ordered_edges:
        elevation_difference = elevation[destination] - elevation[source]
        slope_difference = slope[destination] - slope[source]
        attributes.append([
            distances[source, destination],
            elevation_difference if np.isfinite(elevation_difference) else 0.0,
            slope_difference if np.isfinite(slope_difference) else 0.0,
            edges[(source, destination)],
        ])
    return torch.tensor(ordered_edges, dtype=torch.long).t().contiguous(), torch.tensor(attributes, dtype=torch.float32)


build_spatial_graph = build_graph
