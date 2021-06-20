use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    rc::Rc,
    sync::Arc,
};

use by_address::ByAddress;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayout, Device, Sampler};

use crate::texture::{RenderTexture, Texture};

pub struct Material {
    bind_group_layout: ByAddress<Arc<BindGroupLayout>>,
    label: String,
    bindings: Vec<BindGroupEntryArc>,
    bind_group: Option<Rc<BindGroup>>,
}

impl Material {
    pub fn bind_group(&self) -> &Rc<BindGroup> {
        if let Some(b) = &self.bind_group {
            b
        } else {
            panic!("A pass tried to use the material {}, but that material does not have all bindings specified.\nEntries: {:#?}", self.label, self.bindings)
        }
    }

    pub fn new(
        device: &Device,
        label: String,
        bind_group_layout: Arc<BindGroupLayout>,
        bindings: Vec<BindGroupEntryArc>,
    ) -> Material {
        let bind_group = Self::bind_group_from_entries(device, &label, &bind_group_layout, &bindings).map(Rc::new);
        Material {
            bind_group_layout: bind_group_layout.into(),
            label,
            bindings,
            bind_group,
        }
    }

    pub fn from_consecutive_entries(
        device: &Device,
        label: &str,
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
        Self::new(device, label.to_string(), bind_group_layout, entries)
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

    pub fn modified(&self, device: &Device, overrides: &[BindGroupEntryArc]) -> Material {
        puffin::profile_function!();
        let mut new_bindings = self.bindings.clone();
        for change in overrides {
            new_bindings[change.binding as usize] = change.to_owned();
        }

        let bind_group =
            Self::bind_group_from_entries(device, &self.label, &self.bind_group_layout, &new_bindings).map(Rc::new);

        Material {
            bind_group_layout: self.bind_group_layout.clone(),
            label: self.label.clone(),
            bindings: new_bindings,
            bind_group,
        }
    }
}

#[derive(Hash, Clone, Debug)]
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

#[derive(Hash, Clone, Debug)]
pub enum BindingResourceArc {
    // TODO: This box is completely unnecessary.
    // It is there to work around a rustc ICE
    Sampler(Option<ByAddress<Arc<Sampler>>>),
    Texture(Option<RenderTexture>),
}

impl BindingResourceArc {
    fn to_wgpu(&self) -> Option<wgpu::BindingResource> {
        match self {
            BindingResourceArc::Sampler(Some(sampler)) => Some(wgpu::BindingResource::Sampler(sampler)),
            BindingResourceArc::Texture(Some(tex)) => Some(wgpu::BindingResource::TextureView(tex.default_view().view)),
            _ => None,
        }
    }

    pub fn sampler(sampler: Option<Arc<Sampler>>) -> Self {
        Self::Sampler(sampler.map(ByAddress::from))
        // Self::texture(None)
    }

    pub fn texture(texture: Option<Rc<Texture>>) -> Self {
        Self::render_texture(texture.map(RenderTexture::from))
    }

    pub fn render_texture(texture: Option<RenderTexture>) -> Self {
        Self::Texture(texture)
    }

    pub fn hashcode(&self) -> usize {
        match self {
            BindingResourceArc::Sampler(Some(o)) => Arc::as_ptr(&o.0) as usize,
            BindingResourceArc::Texture(Some(RenderTexture::Texture(t))) => Rc::as_ptr(&t.0) as usize,
            BindingResourceArc::Texture(Some(RenderTexture::SwapchainImage(t))) => Rc::as_ptr(&t.0) as usize,
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
        hasher.write_usize(Arc::as_ptr(&material.bind_group_layout.0) as usize);
        let mut j = 0;
        for i in 0..material.bindings.len() {
            if j < changes.len() && changes[j].binding as usize == i {
                // changes[j].hash(&mut hasher);
                // Manual hashing to work around rustc ICE
                hasher.write_u32(changes[j].binding);
                hasher.write_usize(changes[j].resource.hashcode());
                j += 1;
            } else {
                // material.bindings[i].hash(&mut hasher);
                // Manual hashing to work around rustc ICE
                hasher.write_u32(material.bindings[i].binding);
                hasher.write_usize(material.bindings[i].resource.hashcode());
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

    pub fn override_material(
        &mut self,
        device: &Device,
        material: &Material,
        overrides: &[BindGroupEntryArc],
    ) -> &Arc<Material> {
        let hash = MaterialCache::hash_material(material, overrides);
        self.cache
            .entry(hash)
            .or_insert_with(|| Arc::new(material.modified(device, overrides)))
    }
}
