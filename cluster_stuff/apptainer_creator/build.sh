#!/bin/bash
source params.sh
apptainer build ${NAME}.sif Apptainer.def | tee build.log