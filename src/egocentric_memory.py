"""
Backward-compatibility shim: imports GeometricMemory as EgocentricMemory.
New code should import from src.geometric_memory or src.semantic_memory directly.
"""
from src.geometric_memory import GeometricMemory as EgocentricMemory, StuckTracker, SearchTrail
from src.memory_interface import MemoryInterface
