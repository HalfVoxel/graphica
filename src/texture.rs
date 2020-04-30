use euclid::rect;
use lazy_init::Lazy;
use std::rc::Rc;
use wgpu::{CommandEncoder, Device, Extent3d, TextureFormat, TextureView};

pub struct Texture {
    pub descriptor: wgpu::TextureDescriptor<'static>,
    pub buffer: wgpu::Texture,
    pub view: wgpu::TextureView,
    mipmap_views: Vec<Lazy<wgpu::TextureView>>,
}

pub struct SwapchainImageWrapper {
    descriptor: wgpu::SwapChainDescriptor,
    image: wgpu::SwapChainOutput,
}

#[derive(Clone)]
pub enum RenderTexture {
    Texture(Rc<Texture>),
    SwapchainImage(Rc<SwapchainImageWrapper>),
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
    pub fn from_swapchain_image(
        swapchain_image: wgpu::SwapChainOutput,
        descriptor: &wgpu::SwapChainDescriptor,
    ) -> Self {
        Self {
            descriptor: descriptor.clone(),
            image: swapchain_image,
        }
    }
}

impl From<Rc<Texture>> for RenderTexture {
    fn from(tex: Rc<Texture>) -> RenderTexture {
        RenderTexture::Texture(tex)
    }
}

impl From<SwapchainImageWrapper> for RenderTexture {
    fn from(tex: SwapchainImageWrapper) -> RenderTexture {
        RenderTexture::SwapchainImage(Rc::new(tex))
    }
}

impl RenderTexture {
    pub fn get_mip_level_view(&self, miplevel: u32) -> Result<RenderTextureView, &str> {
        match self {
            RenderTexture::Texture(tex) => Ok(RenderTextureView::new(self, &tex.get_mip_level_view(miplevel))),
            RenderTexture::SwapchainImage(tex) => Err("Cannot get mip levels from a swapchain image"),
        }
    }

    pub fn default_view(&self) -> RenderTextureView {
        match self {
            RenderTexture::Texture(tex) => RenderTextureView::new(self, &tex.view),
            RenderTexture::SwapchainImage(tex) => RenderTextureView::new(self, &tex.image.view),
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
                height: tex.descriptor.width,
                depth: 1,
            },
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
                format: self.descriptor.format,
                dimension: wgpu::TextureViewDimension::D2,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: miplevel,
                level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
            })
        })
    }

    pub fn new(device: &Device, descriptor: wgpu::TextureDescriptor) -> Texture {
        let tex = device.create_texture(&descriptor);

        // Remove the label which we do not have a static lifetime for
        let descriptor = wgpu::TextureDescriptor::<'static> {
            label: None,
            size: descriptor.size,
            array_layer_count: descriptor.array_layer_count,
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
            dimension: descriptor.dimension,
            format: descriptor.format,
            usage: descriptor.usage,
        };

        let view = tex.create_default_view();
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

        let rgba = loaded_image.into_rgba();
        let width = rgba.width();
        let height = rgba.height();

        let texture_extent = wgpu::Extent3d {
            width: width,
            height: height,
            depth: 1,
        };
        let descriptor = wgpu::TextureDescriptor {
            label: Some(path.to_str().unwrap()),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            array_layer_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::config::TEXTURE_FORMAT,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        };
        let texture = Self::new(device, descriptor);

        let raw = rgba.into_raw();

        let transfer_buffer = device.create_buffer_with_data(&raw, wgpu::BufferUsage::COPY_SRC);

        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &transfer_buffer,
                offset: 0,
                bytes_per_row: 4 * width * std::mem::size_of::<u8>() as u32,
                rows_per_image: height,
            },
            wgpu::TextureCopyView {
                texture: &texture.buffer,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            },
            texture_extent,
        );

        Ok(texture)
    }
}