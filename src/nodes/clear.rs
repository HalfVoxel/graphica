use euclid::{Size2D, UnknownUnit};
use wgpu::{util::StagingBelt, CommandEncoder, Device};

use crate::{
    persistent_graph::{PersistentGraph, RenderNode},
    render_graph::GraphNode,
};

pub struct Clear {
    pub size: Size2D<u32, UnknownUnit>,
    pub color: wgpu::Color,
}

impl RenderNode for Clear {
    fn render_node(&self, graph: &mut PersistentGraph) -> GraphNode {
        graph.render_graph.clear(self.size, self.color)
    }

    fn update(&mut self, device: &Device, encoder: &mut CommandEncoder, staging_belt: &mut StagingBelt) {}
}
