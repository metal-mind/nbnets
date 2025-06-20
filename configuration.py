import os
import pickle

import yaml

from sensory import ImageInterface, CharacterInterface, BERTInterface, XORInterface, GymInterface
from components.mesh import NeuroMeshDefinition


INTERFACE_MAP = {
    "dog": ImageInterface,
    "character": CharacterInterface,
    "bert": BERTInterface,
    "image": ImageInterface,
    "xor": XORInterface,
    "gym": GymInterface
}


def mesh_from_template(mesh_id, mesh_config):
    mesh = None
    mesh_def = None
    mesh_source_file = mesh_config.get("snapshot", "")
    if mesh_source_file and os.path.exists(mesh_source_file):
        with open(mesh_source_file, 'rb') as f:
            mesh = pickle.load(f)
    else:
        config_data = mesh_config.get("config", False)
        # Check if we have config data and minimal configuration
        if config_data:
            if not config_data.get("n_mesh_id", False):
                config_data["n_mesh_id"] = mesh_id
            mesh_def = NeuroMeshDefinition.from_json(config_data)
    return mesh, mesh_def


def setup_from_config(config_path):
    """
    Pulls from yaml mesh definitions, mesh instances, and interfaces
    """
    meshes = []  # List of mesh instances if loading snapshot
    mesh_defs = []
    interfaces = []

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Start with meshes as interfaces can change mesh def details
    m_configs = config.get("meshes")
    if m_configs:  # Can sometimes happen in tests where we define all meshes under interfaces
        for mesh_id, mesh_config in m_configs.items():
            mesh, mesh_def = mesh_from_template(mesh_id, mesh_config)
            if mesh is not None:
                meshes.append(mesh)
            elif mesh_def is not None:
                mesh_def_dict = mesh_config.get("config", False)
                mesh_defs.append(NeuroMeshDefinition.from_json(mesh_def_dict))

    # Start by processing interfaces
    i_configs = config.get("interfaces")  # Get a dictionary of interfaces
    for interface_name, interface_config in i_configs.items():
        i_type = interface_config.get("type", "")
        interface_class = INTERFACE_MAP.get(i_type, False)
        assert interface_class
        source = interface_config.get("source", "")
        if source:
            source_string = source
        else:
            source_string = None
        interface_instance = interface_class(interface_name, source_string)

        # Update any mesh parameters with those from config
        mesh_defs_dict = interface_config.get("mesh_defs", "")
        for n_mesh_id, mesh_def_dict in mesh_defs_dict.items():
            mesh_matched = False
            for mesh_def in interface_instance.n_mesh_defs:
                if mesh_def.n_mesh_id == n_mesh_id:
                    mesh_def.update_mesh_def(mesh_def_dict)
                    mesh_matched = True
                    break
            assert mesh_matched
        # Finally store off our interface instance
        interfaces.append(interface_instance)

    return meshes, mesh_defs, interfaces



