use by_address::ByAddress;
use euclid::rect;
use lazy_init::Lazy;
use std::{num::NonZeroU32, rc::Rc, sync::Arc};
use wgpu::{util::DeviceExt, CommandEncoder, Device, Extent3d, TextureFormat, TextureUsage, TextureView};

pub struct Texture {
    pub descriptor: wgpu::TextureDescriptor<'static>,
    pub buffer: wgpu::Texture,
    pub view: wgpu::TextureView,
    mipmap_views: Vec<Lazy<wgpu::TextureView>>,
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Texture({}x{})",
            self.descriptor.size.width, self.descriptor.size.height
        )
    }
}

pub struct SwapchainImageWrapper {
    descriptor: wgpu::SwapChainDescriptor,
    image: wgpu::SwapChainFrame,
}

impl std::fmt::Debug for SwapchainImageWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SwapchainImage({}x{})",
            self.descriptor.width, self.descriptor.height
        )
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum RenderTexture {
    Texture(ByAddress<Arc<Texture>>),
    SwapchainImage(ByAddress<Arc<SwapchainImageWrapper>>),
}

impl std::fmt::Debug for RenderTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Get rid of some noisy indirections in the output
        match self {
            RenderTexture::Texture(x) => write!(f, "RT({:x},{:?})", Arc::as_ptr(x) as u64, **x),
            RenderTexture::SwapchainImage(x) => write!(f, "RT({:x},{:?})", Arc::as_ptr(x) as u64, **x),
        }
    }
}

pub struct RenderTextureView<'a> {
    pub texture: &'a RenderTexture,
    pub view: &'a TextureView,
}

impl<'a> RenderTextureView<'a> {
    pub fn new(texture: &'a RenderTexture, view: &'a TextureView) -> RenderTextureView<'a> {
        RenderTextureView { texture, view }
    }
}

impl SwapchainImageWrapper {
    pub fn from_swapchain_image(swapchain_image: wgpu::SwapChainFrame, descriptor: &wgpu::SwapChainDescriptor) -> Self {
        Self {
            descriptor: descriptor.clone(),
            image: swapchain_image,
        }
    }
}

impl From<Arc<Texture>> for RenderTexture {
    fn from(tex: Arc<Texture>) -> RenderTexture {
        RenderTexture::Texture(tex.into())
    }
}

impl From<SwapchainImageWrapper> for RenderTexture {
    fn from(tex: SwapchainImageWrapper) -> RenderTexture {
        RenderTexture::SwapchainImage(Arc::new(tex).into())
    }
}

impl RenderTexture {
    pub fn get_mip_level_view(&self, miplevel: u32) -> Result<RenderTextureView, &str> {
        match self {
            RenderTexture::Texture(tex) => Ok(RenderTextureView::new(self, tex.get_mip_level_view(miplevel))),
            RenderTexture::SwapchainImage(tex) => {
                if miplevel == 0 {
                    Ok(RenderTextureView::new(self, &tex.image.output.view))
                } else {
                    Err("Cannot get mip levels other than level 0 from a swapchain image")
                }
            }
        }
    }

    /// View which includes all mip levels
    pub fn default_view(&self) -> RenderTextureView {
        match self {
            RenderTexture::Texture(tex) => RenderTextureView::new(self, &tex.view),
            RenderTexture::SwapchainImage(tex) => RenderTextureView::new(self, &tex.image.output.view),
        }
    }

    pub fn sample_count(&self) -> u32 {
        match self {
            RenderTexture::Texture(tex) => tex.descriptor.sample_count,
            // Swapchain images are never multi-sampled
            RenderTexture::SwapchainImage(_tex) => 1,
        }
    }

    pub fn format(&self) -> TextureFormat {
        match self {
            RenderTexture::Texture(tex) => tex.descriptor.format,
            RenderTexture::SwapchainImage(tex) => tex.descriptor.format,
        }
    }

    pub fn size(&self) -> Extent3d {
        match self {
            RenderTexture::Texture(tex) => tex.descriptor.size,
            RenderTexture::SwapchainImage(tex) => Extent3d {
                width: tex.descriptor.width,
                height: tex.descriptor.height,
                depth_or_array_layers: 1,
            },
        }
    }

    pub fn usage(&self) -> TextureUsage {
        match self {
            RenderTexture::Texture(tex) => tex.descriptor.usage,
            RenderTexture::SwapchainImage(t) => t.descriptor.usage,
        }
    }

    pub fn mip_level_count(&self) -> u32 {
        match self {
            RenderTexture::Texture(tex) => tex.descriptor.mip_level_count,
            &RenderTexture::SwapchainImage(_) => 1,
        }
    }
}

pub fn partition_into_squares(size: Extent3d, max_tile_size: u32) -> Vec<euclid::default::Rect<u32>> {
    let sx = (size.width + max_tile_size - 1) / max_tile_size;
    let sy = (size.height + max_tile_size - 1) / max_tile_size;
    let mut result = vec![];
    for y in 0..sy {
        for x in 0..sx {
            result.push(rect(
                x * max_tile_size,
                y * max_tile_size,
                size.width.min((x + 1) * max_tile_size) - x * max_tile_size,
                size.height.min((y + 1) * max_tile_size) - y * max_tile_size,
            ));
        }
    }

    result
}

impl Texture {
    pub fn get_mip_level_view(&self, miplevel: u32) -> &wgpu::TextureView {
        assert!(
            (miplevel as usize) < self.mipmap_views.len(),
            "Trying to access a mip level that does not exist. {} >= {}",
            miplevel,
            self.mipmap_views.len()
        );
        self.mipmap_views[miplevel as usize].get_or_create(|| {
            self.buffer.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: Some(self.descriptor.format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: miplevel,
                mip_level_count: Some(NonZeroU32::new(1).unwrap()),
                base_array_layer: 0,
                array_layer_count: None,
            })
        })
    }

    pub fn new(device: &Device, descriptor: wgpu::TextureDescriptor) -> Texture {
        let tex = device.create_texture(&descriptor);

        // Remove the label which we do not have a static lifetime for
        let descriptor = wgpu::TextureDescriptor::<'static> {
            label: None,
            size: descriptor.size,
            // array_layer_count: descriptor.array_layer_count,
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
            dimension: descriptor.dimension,
            format: descriptor.format,
            usage: descriptor.usage,
        };

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut views = vec![];
        for _ in 0..descriptor.mip_level_count {
            views.push(Lazy::new());
        }

        Texture {
            descriptor,
            buffer: tex,
            view,
            mipmap_views: views,
        }
    }

    pub fn load_from_file(
        path: &std::path::Path,
        device: &Device,
        encoder: &mut CommandEncoder,
    ) -> Result<Texture, image::ImageError> {
        let loaded_image = image::open(path)?;

        let rgba = loaded_image.into_rgba8();
        let width = rgba.width();
        let height = rgba.height();

        let texture_extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let descriptor = wgpu::TextureDescriptor {
            label: Some(path.to_str().unwrap()),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            // array_layer_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::config::TEXTURE_FORMAT,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        };
        let texture = Self::new(device, descriptor);

        let raw = rgba.into_raw();

        let transfer_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &raw,
            usage: wgpu::BufferUsage::COPY_SRC,
        });

        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &transfer_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        NonZeroU32::new(4 * width * std::mem::size_of::<u8>() as u32).expect("image width was zero"),
                    ),
                    rows_per_image: Some(NonZeroU32::new(height).expect("image height was zero")),
                },
            },
            wgpu::ImageCopyTexture {
                texture: &texture.buffer,
                mip_level: 0,
                // array_layer: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            },
            texture_extent,
        );

        Ok(texture)
    }
}
