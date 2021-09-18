use std::{collections::HashMap, sync::Arc};

use by_address::ByAddress;
use wgpu::{
    BlendState, ColorWrites, DepthStencilState, Device, MultisampleState, PipelineLayout, PrimitiveState,
    RenderPipeline, RenderPipelineDescriptor, ShaderModule, TextureFormat,
};

#[derive(Default)]
pub struct RenderPipelineCache {
    cache: HashMap<RenderPipelineKey, CachedRenderPipeline>,
}

impl RenderPipelineCache {
    pub fn get<'a>(&'a mut self, device: &Device, key: RenderPipelineKey) -> &'a CachedRenderPipeline {
        puffin::profile_function!();
        self.cache.entry(key).or_insert_with_key(|key| {
            let pipeline = key.base.to_wgpu_pipeline(
                device,
                &[wgpu::ColorTargetState {
                    format: key.target_format,
                    blend: Some(key.blend_state),
                    write_mask: ColorWrites::all(),
                }],
                key.depth_format.map(|format| DepthStencilState {
                    format,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                MultisampleState {
                    count: key.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
            );

            Arc::new(CachedRenderPipelineImpl { pipeline })
        })
    }
}

pub type CachedRenderPipeline = Arc<CachedRenderPipelineImpl>;

impl std::fmt::Debug for CachedRenderPipelineImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("render pipeline")
    }
}

pub struct CachedRenderPipelineImpl {
    pub pipeline: RenderPipeline,
}

#[derive(Clone)]
pub struct RenderPipelineBase {
    pub label: String,
    pub layout: Arc<PipelineLayout>,
    pub module: Arc<ShaderModule>,
    pub vertex_buffer_layout: wgpu::VertexBufferLayout<'static>,
    pub vertex_entry: String,
    pub fragment_entry: String,
    pub primitive: PrimitiveState,
    pub target_count: usize,
}

impl std::fmt::Debug for RenderPipelineBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RenderPipelineBase({})", self.label)
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct RenderPipelineKey {
    pub base: ByAddress<Arc<RenderPipelineBase>>,
    pub sample_count: u32,
    pub depth_format: Option<TextureFormat>,
    pub target_format: TextureFormat,
    pub blend_state: BlendState,
}

impl RenderPipelineBase {
    fn to_wgpu_pipeline(
        &self,
        device: &Device,
        targets: &[wgpu::ColorTargetState],
        depth_stencil: Option<DepthStencilState>,
        multisample: MultisampleState,
    ) -> RenderPipeline {
        assert_eq!(self.target_count, targets.len());

        let render_pipeline_descriptor = RenderPipelineDescriptor {
            label: Some(&self.label),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState {
                module: &self.module,
                entry_point: &self.vertex_entry,
                buffers: &[self.vertex_buffer_layout.clone()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.module,
                entry_point: &self.fragment_entry,
                targets,
            }),
            primitive: self.primitive,
            depth_stencil,
            multisample,
        };

        device.create_render_pipeline(&render_pipeline_descriptor)
    }
}
