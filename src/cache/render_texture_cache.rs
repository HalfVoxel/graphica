use std::rc::Rc;

use euclid::default::Size2D;
use wgpu::{Device, Extent3d, TextureFormat, TextureUsage};

use crate::texture::{RenderTexture, Texture};

#[derive(Default)]
pub struct RenderTextureCache {
    render_textures: Vec<RenderTexture>,
}

impl RenderTextureCache {
    pub fn push(&mut self, rt: RenderTexture) {
        self.render_textures.push(rt);
    }

    pub fn temporary_render_texture(
        &mut self,
        device: &Device,
        size: Size2D<u32>,
        format: TextureFormat,
        mipmaps: bool,
        usage: TextureUsage,
    ) -> RenderTexture {
        puffin::profile_function!();
        let best_tex = self
            .render_textures
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                if t.format() != format {
                    return false;
                }
                // TODO: Handle tiny textures
                if mipmaps != (t.mip_level_count() > 1) {
                    return false;
                }

                // A bit too strict maybe
                if usage != t.usage() {
                    return false;
                }

                let tsize = t.size();
                tsize.width >= size.width
                    && tsize.height >= size.height
                    && tsize.width * tsize.height <= size.area() * 4
            })
            .min_by_key(|(_i, t)| t.size().width * t.size().height)
            .map(|(i, _t)| i);

        if let Some(index) = best_tex {
            self.render_textures.swap_remove(index)
        } else {
            let extent = Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            };
            let tex = Texture::new(
                device,
                wgpu::TextureDescriptor {
                    label: Some("Temp texture"),
                    size: extent,
                    mip_level_count: crate::mipmap::max_mipmaps(extent),
                    sample_count: 1,
                    // array_layer_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: crate::config::TEXTURE_FORMAT,
                    usage,
                },
            );
            RenderTexture::from(Rc::new(tex))
        }
    }
}
