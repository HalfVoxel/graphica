use std::sync::{Arc, Mutex};

use wgpu::{util::StagingBelt, CommandEncoder, Device};

use crate::{
    persistent_graph::{PersistentGraph, RenderNode},
    render_graph::GraphNode,
    texture::RenderTexture,
};

pub struct Cache {
    source: Arc<Mutex<dyn RenderNode>>,
    cache: Arc<Mutex<Option<RenderTexture>>>,
}

impl Cache {
    pub fn new(source: Arc<Mutex<dyn RenderNode>>) -> Self {
        Self {
            source,
            cache: Default::default(),
        }
    }
}

impl RenderNode for Cache {
    fn render_node(&self, graph: &mut PersistentGraph) -> GraphNode {
        let guard = self.cache.lock().unwrap();
        if let Some(tex) = &*guard {
            graph.render_graph.texture(tex.clone())
        } else {
            let source = graph.render(&self.source);
            graph.render_graph.snapshot_texture(source.clone(), self.cache.clone());
            source
        }
    }

    fn update(&mut self, _device: &Device, _encoder: &mut CommandEncoder, _staging_belt: &mut StagingBelt) {}
}
