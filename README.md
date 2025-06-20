# NBNets

Framework for growing biologically plausible neural networks. This framework uses a simulation like time step model that allows for temporal interactions between neurons for an organic capture of information.


# Setup
This project utilizes UV for python and package management. Below is a quick start with or without UV.

With uv:
```bash
cd checkout_folder
# Install packages
uv sync
# Create venv
uv venv
# Activate venv
source .venv/bin/activate
```

Without uv
```bash
cd checkout_folder
# Install from pyproject.toml with pip
python -m pip install .
```

# Running Simulations
After installing dependencies, you can call `python main.py --help` to get some details. Most of the simulations are invoked through main.
This nbnets project is declarative configuration driven, meaning that configuration files define information about meshes, their connections, interfaces and their sources. These can be invoked with the --config option: `python main.py --config configs/mNIST.yml`.
