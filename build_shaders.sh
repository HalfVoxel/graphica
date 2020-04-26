#!/bin/sh
glslangValidator -V  ./shaders/geometry.vert -o ./shaders/geometry.vert.spv
glslangValidator -V  ./shaders/geometry.frag -o ./shaders/geometry.frag.spv
glslangValidator -V  ./shaders/background.vert -o ./shaders/background.vert.spv
glslangValidator -V  ./shaders/background.frag -o ./shaders/background.frag.spv
glslangValidator -V  ./shaders/brush.vert -o ./shaders/brush.vert.spv
glslangValidator -V  ./shaders/brush.frag -o ./shaders/brush.frag.spv
glslangValidator -V  ./shaders/clone_brush.vert -o ./shaders/clone_brush.vert.spv
glslangValidator -V  ./shaders/clone_brush.frag -o ./shaders/clone_brush.frag.spv
glslangValidator -V  ./shaders/blit.vert -o ./shaders/blit.vert.spv
glslangValidator -V  ./shaders/blit.frag -o ./shaders/blit.frag.spv

glslangValidator -V  ./shaders/blur.comp -o ./shaders/blur.comp.spv
glslangValidator -V  ./shaders/downsample_2x2_box.comp -o ./shaders/downsample_2x2_box.comp.spv
glslangValidator -V  ./shaders/clone_brush_batch.comp -o ./shaders/clone_brush_batch.comp.spv