"""Bike-facility type vocabulary shared by the sign, join and gap stages."""
from __future__ import annotations

# YOLO classes. OnlyBikeBus is detected but dropped by default (signs.conf_keep).
CLASSES = ("BikeOnly", "Sharrow", "OnlyBikeBus")

# tie-break when a line carries equal counts of two classes
TYPE_PRIORITY = {"BikeOnly": 0, "Sharrow": 1, "OnlyBikeBus": 2}

# `type` of a connector edge in the network layer (facility edges carry a CLASSES value)
INTERSECTION = "Intersection"


def majority_type(classes: dict) -> str | None:
    if not classes:
        return None
    return sorted(classes.items(), key=lambda kv: (-kv[1], TYPE_PRIORITY.get(kv[0], 9)))[0][0]
