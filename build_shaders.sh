#!/bin/sh
glslangValidator -V ./shaders/geometry.vert -o ./shaders/geometry.vert.spv
glslangValidator -V ./shaders/geometry.frag -o ./shaders/geometry.frag.spv
glslangValidator -V ./shaders/background.vert -o ./shaders/background.vert.spv
glslangValidator -V ./shaders/background.frag -o ./shaders/background.frag.spv

