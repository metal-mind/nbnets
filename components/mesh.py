"""
Neural graph components that might need to be imported by different modules. Separated to minimize dependency
requirements across different processes importing this module
"""
import re

from dataclasses import dataclass, field, is_dataclass, fields
from typing import List, Tuple, Type, TypeVar
from enum import IntEnum

from components.common import NeuralIODefinition

T = TypeVar("T")


__camel_to_snake_pattern = re.compile(r"(?<!^)(?=[A-Z])")
def camel_to_snake(input: str) -> str:
    return __camel_to_snake_pattern.sub("_", input).lower()


class MeshType(IntEnum):
    BASE = 1
    OUTPUT = 2
    BASAL_GANGLIA = 3


@dataclass
class NeuroMeshDefinition:
    """
    Dataclass that defines an independent segment of the network. While the greater network acts as a whole, individual
    parts of the network can be assigned various parameters to control it on an individual basis.

    An example of this would be the neuromesh for a given sensory input. Depending on the temporal resolution desired
    for a given input you might want to independently control parameters.
    """
    n_mesh_id: str = ""  # Currently maps to mesh ids in graph-tool to map vertex to geometry
    output_mesh_ids: Tuple[str] = field(default_factory=tuple)
    max_n_count: int = 0  # Defines how many neurons can be inserted into the plane, zero is uncapped

    mesh_type: MeshType = MeshType.BASE
    static: bool = False
    do_consolidation: bool = True
    duplicate_count_threshold: int = 50

    # Physical Parameters
    number_dimensions: int = 2
    n_mesh_upper_bounds: tuple = (100.0, 100.0)
    n_mesh_lower_bounds: tuple = (0.0, 0.0)
    starting_rank: float = 2.0
    max_rank: float = 20.0
    max_rank_distance: int = 1  # Defines physical proximity across layers/ranks
    abstraction_limit: int = 16

    # Temporal Parameters
    max_ta: int = -1  # Peak temporal aperture (largest time slice to consider a pattern) expressed as number of steps Zero for unlimited; Negative one (default) for computed TA
    starting_ta: int = 1  # Climbing temporal aperture, grows up to peak_ta

    # Basal Parameters
    # Definitions for I/O
    n_io_defs: List[NeuralIODefinition] = field(default_factory=list)

    def update_mesh_def(self, config_dict):
        attributes = list(self.__dict__.keys())
        for k, v in config_dict.items():
            if k in attributes:
                self.__dict__[k] = v

    @classmethod
    def from_json(cls: Type[T], json: dict) -> T:
        """
        This allows us to instantiate a mesh def from a json
        Args:
            json (dict): Json dictionary

        Raises:
            ValueError: When `this` isn't a dataclass

        Returns:
            T: New instance
        """
        if not is_dataclass(cls):
            raise ValueError(f"{cls.__name__} must be a dataclass")
        field_names = {field.name for field in fields(cls)}
        kwargs = {
            camel_to_snake(key): value
            for key, value in json.items()
            if camel_to_snake(key) in field_names
        }
        return cls(**kwargs)