use std::sync::{Arc, Mutex};

use wgpu::{util::StagingBelt, CommandEncoder, Device};

use crate::{
    geometry_utilities::types::CanvasRect,
    persistent_graph::{PersistentGraph, RenderNode},
    render_graph::GraphNode,
};

pub struct BlitNode {
    pub source: Arc<Mutex<dyn RenderNode>>,
    pub target: Arc<Mutex<dyn RenderNode>>,
    pub source_rect: CanvasRect,
    pub target_rect: CanvasRect,
}

impl RenderNode for BlitNode {
    fn render_node(&self, graph: &mut PersistentGraph) -> GraphNode {
        let mut source = graph.render(&self.source);
        let target = graph.render(&self.target);
        if self.source_rect.size != self.target_rect.size {
            source = graph.render_graph.generate_mipmaps(source);
        }
        graph
            .render_graph
            .blit(source, target, self.source_rect, self.target_rect)
    }

    fn update(&mut self, device: &Device, encoder: &mut CommandEncoder, staging_belt: &mut StagingBelt) {}
}
