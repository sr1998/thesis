#!/bin/bash
source params.sh

# Test if the container loads the conda environment by default
apptainer exec ${NAME}.sif which python

# Show all libraries including version numbers installed
apptainer exec ${NAME}.sif conda env export --no-builds

# Find jupyterlab
apptainer exec ${NAME}.sif which jupyter-lab

