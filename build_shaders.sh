#!/bin/sh
ARGS="-O -Werror -g"
glslc --target-env=vulkan1.0 $ARGS ./shaders/geometry.vert -o ./shaders/geometry.vert.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/geometry.frag -o ./shaders/geometry.frag.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/background.vert -o ./shaders/background.vert.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/background.frag -o ./shaders/background.frag.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/brush.vert -o ./shaders/brush.vert.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/brush.frag -o ./shaders/brush.frag.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/clone_brush.vert -o ./shaders/clone_brush.vert.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/clone_brush.frag -o ./shaders/clone_brush.frag.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/blit.vert -o ./shaders/blit.vert.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/blit.frag -o ./shaders/blit.frag.spv

glslc --target-env=vulkan1.0 $ARGS ./shaders/blur.comp -o ./shaders/blur.comp.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/downsample_2x2_box.comp -o ./shaders/downsample_2x2_box.comp.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/clone_brush_batch.comp -o ./shaders/clone_brush_batch.comp.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/blend_over.comp -o ./shaders/blend_over.comp.spv
glslc --target-env=vulkan1.0 $ARGS ./shaders/rgb_to_srgb.comp -o ./shaders/rgb_to_srgb.comp.spv