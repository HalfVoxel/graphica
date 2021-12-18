use std::{collections::HashMap, hash::Hasher, rc::Rc, sync::Arc};

use by_address::ByAddress;
use once_cell::unsync::OnceCell;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayout, BlendState, Device, Sampler};

use crate::{
    render_graph::GraphNode,
    texture::{RenderTexture, Texture},
};

use super::ephermal_buffer_cache::BufferRange;

pub struct Material {
    bind_group_layout: ByAddress<Arc<BindGroupLayout>>,
    label: String,
    bindings: Vec<BindGroupEntryArc>,
    bind_group: OnceCell<Option<Rc<BindGroup>>>,
    pub blend: BlendState,
}

impl std::fmt::Debug for Material {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Material({})", self.label)
    }
}

impl Material {
    pub fn bind_group(&self, device: &Device) -> &Rc<BindGroup> {
        let bind_group = self.bind_group.get_or_init(|| {
            Self::bind_group_from_entries(device, &self.label, &self.bind_group_layout, &self.bindings).map(Rc::new)
        });

        if let Some(b) = &bind_group {
            b
        } else {
            panic!("A pass tried to use the material {}, but that material does not have all bindings specified.\nEntries: {:#?}", self.label, self.bindings)
        }
    }

    pub fn new(
        label: String,
        blend: BlendState,
        bind_group_layout: Arc<BindGroupLayout>,
        bindings: Vec<BindGroupEntryArc>,
    ) -> Material {
        Material {
            bind_group_layout: bind_group_layout.into(),
            label,
            bindings,
            bind_group: OnceCell::default(),
            blend,
        }
    }

    pub fn from_consecutive_entries(
        label: &str,
        blend: BlendState,
        bind_group_layout: Arc<BindGroupLayout>,
        bindings: Vec<BindingResourceArc>,
    ) -> Material {
        let entries = bindings
            .into_iter()
            .enumerate()
            .map(|(index, resource)| BindGroupEntryArc {
                binding: index as u32,
                resource,
            })
            .collect::<Vec<_>>();
        Self::new(label.to_string(), blend, bind_group_layout, entries)
    }

    fn bind_group_from_entries(
        device: &Device,
        label: &str,
        layout: &BindGroupLayout,
        entries: &[BindGroupEntryArc],
    ) -> Option<BindGroup> {
        let bind_entries = entries.iter().map(|b| b.to_wgpu()).collect::<Option<Vec<_>>>();
        bind_entries.map(|entries| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &entries,
            })
        })
    }

    pub fn modified(&self, overrides: &[BindGroupEntryArc]) -> Material {
        puffin::profile_function!();
        let mut new_bindings = self.bindings.clone();
        for change in overrides {
            new_bindings[change.binding as usize] = change.to_owned();
        }

        Material {
            bind_group_layout: self.bind_group_layout.clone(),
            label: self.label.clone(),
            bindings: new_bindings,
            bind_group: OnceCell::default(),
            blend: self.blend,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DynamicMaterial {
    pub material: Arc<Material>,
    pub overrides: Vec<BindGroupEntryArc>,
}

impl DynamicMaterial {
    pub fn new(material: Arc<Material>, overrides: Vec<BindGroupEntryArc>) -> Self {
        Self { material, overrides }
    }
}

#[derive(Clone, Debug)]
pub struct BindGroupEntryArc {
    pub binding: u32,
    pub resource: BindingResourceArc,
}

impl BindGroupEntryArc {
    fn to_wgpu(&self) -> Option<wgpu::BindGroupEntry> {
        self.resource.to_wgpu().map(|r| wgpu::BindGroupEntry {
            binding: self.binding,
            resource: r,
        })
    }
}

#[derive(Clone, Debug)]
pub enum BindingResourceArc {
    // TODO: This box is completely unnecessary.
    // It is there to work around a rustc ICE
    Sampler(Option<ByAddress<Arc<Sampler>>>),
    Texture(Option<RenderTexture>),
    // Will be resolved to a `Texture`.
    GraphNode(GraphNode),
    Mipmap(Option<(RenderTexture, u32)>),
    Buffer(Option<BufferRange>),
}

impl BindingResourceArc {
    fn to_wgpu(&self) -> Option<wgpu::BindingResource> {
        match self {
            BindingResourceArc::Sampler(Some(sampler)) => Some(wgpu::BindingResource::Sampler(sampler)),
            BindingResourceArc::Texture(Some(tex)) => Some(wgpu::BindingResource::TextureView(tex.default_view().view)),
            BindingResourceArc::Mipmap(Some((tex, mip))) => Some(wgpu::BindingResource::TextureView(
                tex.get_mip_level_view(*mip).unwrap().view,
            )),
            BindingResourceArc::Buffer(Some(buffer)) => Some(wgpu::BindingResource::Buffer(buffer.as_binding())),
            _ => None,
        }
    }

    pub fn buffer(buffer: Option<BufferRange>) -> Self {
        Self::Buffer(buffer)
    }

    pub fn sampler(sampler: Option<Arc<Sampler>>) -> Self {
        Self::Sampler(sampler.map(ByAddress::from))
        // Self::texture(None)
    }

    pub fn mipmap(texture: Option<(RenderTexture, u32)>) -> Self {
        Self::Mipmap(texture)
    }

    pub fn texture(texture: Option<Arc<Texture>>) -> Self {
        Self::render_texture(texture.map(RenderTexture::from))
    }

    pub fn render_texture(texture: Option<RenderTexture>) -> Self {
        Self::Texture(texture)
    }

    pub fn graph_node(texture: GraphNode) -> Self {
        Self::GraphNode(texture)
    }

    pub fn hashcode(&self) -> u64 {
        match self {
            BindingResourceArc::Sampler(Some(o)) => Arc::as_ptr(&o.0) as u64,
            BindingResourceArc::Texture(Some(RenderTexture::Texture(t))) => Arc::as_ptr(&t.0) as u64,
            BindingResourceArc::Texture(Some(RenderTexture::SwapchainImage(t))) => Arc::as_ptr(&t.0) as u64,
            BindingResourceArc::Buffer(Some(b)) => {
                Arc::as_ptr(&b.buffer) as u64 ^ (31 * (b.range.start ^ (31 * b.range.end)))
            }
            BindingResourceArc::Mipmap(Some((RenderTexture::Texture(t), index))) => {
                Arc::as_ptr(&t.0) as u64 ^ (31 * (*index as u64))
            }
            BindingResourceArc::Mipmap(Some((RenderTexture::SwapchainImage(t), index))) => {
                Arc::as_ptr(&t.0) as u64 ^ (31 * (*index as u64))
            }
            BindingResourceArc::GraphNode(_node) => {
                panic!("GraphNode hasn't been replaced")
            }
            _ => 0,
        }
    }
}

#[derive(Default)]
pub struct MaterialCache {
    cache: HashMap<u64, Arc<Material>>,
}

impl MaterialCache {
    fn hash_material(material: &Material, changes: &[BindGroupEntryArc]) -> u64 {
        puffin::profile_function!();
        let mut hasher = std::collections::hash_map::DefaultHasher::default();
        // Manual hashing to work around rustc ICE
        // https://github.com/rust-lang/rust/issues/86469
        hasher.write_u64(Arc::as_ptr(&material.bind_group_layout.0) as u64);
        let mut j = 0;
        for i in 0..material.bindings.len() {
            if j < changes.len() && changes[j].binding as usize == i {
                // changes[j].hash(&mut hasher);
                // Manual hashing to work around rustc ICE
                hasher.write_u32(changes[j].binding);
                hasher.write_u64(changes[j].resource.hashcode());
                j += 1;
            } else {
                // material.bindings[i].hash(&mut hasher);
                // Manual hashing to work around rustc ICE
                hasher.write_u32(material.bindings[i].binding);
                hasher.write_u64(material.bindings[i].resource.hashcode());
            }
        }
        assert_eq!(j, changes.len());
        hasher.finish()
        // let mut new_hash = material.hash;
        // for change in changes {
        //     assert_eq!(material.bindings[change.binding as usize].binding, change.binding);
        //     let mut hasher = DefaultHasher::new();
        //     change.resource.hash(&mut hasher);
        //     let m = 31u64.pow(change.binding);
        //     let new_binding_hash = m * hasher.finish();

        //     let mut hasher = DefaultHasher::new();
        //     material.bindings[change.binding as usize].resource.hash(&mut hasher);
        //     let old_binding_hash = m * hasher.finish();
        //     // TODO: Should be taking the hashes modulo a prime

        //     new_hash += new_binding_hash - old_binding_hash;
        // }
    }

    pub fn override_material(&mut self, material: &Material, overrides: &[BindGroupEntryArc]) -> &Arc<Material> {
        let hash = MaterialCache::hash_material(material, overrides);
        self.cache
            .entry(hash)
            .or_insert_with(|| Arc::new(material.modified(overrides)))
    }
}
