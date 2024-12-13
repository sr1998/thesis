#!/bin/bash

# Your command(s) to deploy the container to the right location(s).
# $USER assumes you have the same username on the local and remote system
source params.sh
rsync -avc --progress ${NAME}.sif dblue:$HOME/ondemand/jupyter/${NAME}.sif

