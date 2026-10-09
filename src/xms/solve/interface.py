"""Every solver publishes the same AnimationBundle through its sole writer."""
from typing import Protocol
from xms.animation.bundle import AnimationBundle


class Solver(Protocol):
    def __call__(self, observations: tuple, timeline: dict, profile: dict, rig: dict) -> AnimationBundle: ...
