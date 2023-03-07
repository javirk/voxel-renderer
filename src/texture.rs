use wgpu::util::DeviceExt;

use anyhow::*;

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        width: u32,
        height: u32,
        depth: u32,
        label: Option<&str>,
    ) -> Result<Self> {
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        };

        let texture_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: bytes,
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("temp texture encoder"),
        });

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Unorm,
            // sampled: use in shader
            // copy dst, we want to copy data to this texture
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("my texture"),
        });
        
        queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            // The actual pixel data
            bytes,
            // The layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(width),
                rows_per_image: std::num::NonZeroU32::new(height),
            },
            texture_size,
        );

        queue.submit(std::iter::once(encoder.finish()));
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let address_mode = wgpu::AddressMode::ClampToEdge;
        let filter_mode = wgpu::FilterMode::Nearest;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: address_mode,
            address_mode_v: address_mode,
            address_mode_w: address_mode,
            mag_filter: filter_mode,
            min_filter: filter_mode,
            mipmap_filter: filter_mode,
            ..Default::default()
        });
        Ok(Self {
            texture,
            view,
            sampler,
        })
    }
}