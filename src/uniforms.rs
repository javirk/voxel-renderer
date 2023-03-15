use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::mem;
use std::time::{SystemTime, UNIX_EPOCH};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniform {  // Annoying 16 byte alignment
    pub itime: f32,
    _padding: [u32; 3],
    pub texture_dims: [f32; 3],
    _padding2: u32,
}

impl Uniform {
    pub fn new(texture_dims: [f32; 3]) -> Self {
        Self {
            itime: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f32(),
            _padding: [0; 3],
            texture_dims: texture_dims,
            _padding2: 0,
        }
    }
}

pub struct UniformBuffer {
    pub buffer: wgpu::Buffer,
}

impl UniformBuffer {
    pub fn new(uniform: Uniform, device: &wgpu::Device,) -> Self {
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        println!("{:?}", mem::size_of::<Uniform>() as u32);

        UniformBuffer {
            buffer: uniform_buf,
        }
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource {
        self.buffer.as_entire_binding()
    }

    pub fn binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None // wgpu::BufferSize::new(mem::size_of::<Uniform>() as _),
        }
    }
}