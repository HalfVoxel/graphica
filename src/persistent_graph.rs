use crate::render_graph::{GraphNode, RenderGraph};
use by_address::ByAddress;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use wgpu::{util::StagingBelt, CommandEncoder, Device};

pub trait RenderNode {
    fn update(&mut self, device: &Device, encoder: &mut CommandEncoder, staging_belt: &mut StagingBelt);
    fn render_node(&self, graph: &mut PersistentGraph) -> GraphNode;
}

pub struct PersistentGraph<'a> {
    pub render_graph: &'a mut RenderGraph,
    rendered: HashMap<ByAddress<Arc<Mutex<dyn RenderNode>>>, GraphNode>,
}

impl<'a> PersistentGraph<'a> {
    pub fn new(render_graph: &'a mut RenderGraph) -> Self {
        Self {
            render_graph,
            rendered: Default::default(),
        }
    }

    pub fn render(&mut self, node: &Arc<Mutex<dyn RenderNode>>) -> GraphNode {
        let node = ByAddress::from(node.clone());
        if let Some(res) = self.rendered.get(&node) {
            res.clone()
        } else {
            let res = node.lock().unwrap().render_node(self);
            self.rendered.insert(node, res.clone());
            res
        }
    }
}
