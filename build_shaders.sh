#!/bin/sh
glslangValidator -V1.3 --target-env spirv1.5 ./shaders/geometry.vert -o ./shaders/geometry.vert.spv
glslangValidator -V1.3 --target-env spirv1.5 ./shaders/geometry.frag -o ./shaders/geometry.frag.spv
glslangValidator -V1.3 --target-env spirv1.5 ./shaders/background.vert -o ./shaders/background.vert.spv
glslangValidator -V1.3 --target-env spirv1.5 ./shaders/background.frag -o ./shaders/background.frag.spv
glslangValidator -V1.3 --target-env spirv1.5 ./shaders/brush.vert -o ./shaders/brush.vert.spv
glslangValidator -V1.3 --target-env spirv1.5 ./shaders/brush.frag -o ./shaders/brush.frag.spv
glslangValidator -V1.3 --target-env spirv1.5 ./shaders/clone_brush.vert -o ./shaders/clone_brush.vert.spv
glslangValidator -V1.3 --target-env spirv1.5 ./shaders/clone_brush.frag -o ./shaders/clone_brush.frag.spv
glslangValidator -V1.3 --target-env spirv1.5 ./shaders/blit.vert -o ./shaders/blit.vert.spv
glslangValidator -V1.3 --target-env spirv1.5 ./shaders/blit.frag -o ./shaders/blit.frag.spv

glslangValidator -V1.3 --target-env spirv1.5 ./shaders/blur.comp -o ./shaders/blur.comp.spv
glslangValidator -V1.3 --target-env spirv1.5 ./shaders/downsample_2x2_box.comp -o ./shaders/downsample_2x2_box.comp.spv