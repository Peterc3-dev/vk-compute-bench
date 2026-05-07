use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;
use prettytable::{row, Table};
use rand::Rng;
use std::ffi::CStr;
use std::mem;
use std::slice;
use std::time::Instant;

// ── Push-constant structs ──────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SaxpyPush {
    a: f32,
    n: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ReducePush {
    n: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MatmulPush {
    m: u32,
    n: u32,
    k: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CopyPush {
    n: u32,
}

// ── Result type ────────────────────────────────────────────────────────

struct BenchResult {
    name: String,
    size_mb: f64,
    time_ms: f64,
    metric: String,
    metric_val: f64,
    peak_pct: Option<f64>,
}

// ── Vulkan context ─────────────────────────────────────────────────────

#[allow(dead_code)]
struct VkContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    command_pool: vk::CommandPool,
    allocator: mem::ManuallyDrop<Allocator>,
    timestamp_period: f64, // nanoseconds per tick
}

struct Buffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
    size: vk::DeviceSize,
}

impl VkContext {
    fn new() -> Result<Self, String> {
        unsafe {
            let entry = ash::Entry::load()
                .map_err(|e| format!("Failed to load Vulkan entry point: {:?}", e))?;

            // Instance
            let app_info = vk::ApplicationInfo::default()
                .application_name(CStr::from_bytes_with_nul(b"vk-compute-bench\0").unwrap())
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(CStr::from_bytes_with_nul(b"none\0").unwrap())
                .api_version(vk::make_api_version(0, 1, 1, 0));

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info);

            let instance = entry
                .create_instance(&create_info, None)
                .map_err(|e| format!("Failed to create Vulkan instance: {:?}", e))?;

            // Physical device
            let phys_devices = instance
                .enumerate_physical_devices()
                .map_err(|e| format!("Failed to enumerate devices: {:?}", e))?;

            if phys_devices.is_empty() {
                return Err("No Vulkan physical devices found".into());
            }

            // Prefer discrete GPU, fall back to integrated
            let physical_device = phys_devices
                .iter()
                .copied()
                .find(|&pd| {
                    let props = instance.get_physical_device_properties(pd);
                    props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .or_else(|| {
                    phys_devices
                        .iter()
                        .copied()
                        .find(|&pd| {
                            let props = instance.get_physical_device_properties(pd);
                            props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
                        })
                })
                .unwrap_or(phys_devices[0]);

            let props = instance.get_physical_device_properties(physical_device);
            let device_name = CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();
            let timestamp_period = props.limits.timestamp_period as f64;

            println!("Device: {}", device_name);
            println!(
                "Driver: {}.{}.{}",
                vk::api_version_major(props.driver_version),
                vk::api_version_minor(props.driver_version),
                vk::api_version_patch(props.driver_version),
            );

            // Queue family — compute
            let queue_families =
                instance.get_physical_device_queue_family_properties(physical_device);

            let queue_family = queue_families
                .iter()
                .enumerate()
                .find(|(_, qf)| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(i, _)| i as u32)
                .ok_or("No compute queue family found")?;

            // Logical device
            let queue_priorities = [1.0_f32];
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&queue_priorities);

            let queue_create_infos = [queue_create_info];
            let device_create_info =
                vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_infos);

            let device = instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(|e| format!("Failed to create device: {:?}", e))?;

            let queue = device.get_device_queue(queue_family, 0);

            // Command pool
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            let command_pool = device
                .create_command_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create command pool: {:?}", e))?;

            // Allocator
            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .map_err(|e| format!("Failed to create allocator: {:?}", e))?;

            // Print memory info
            let mem_props =
                instance.get_physical_device_memory_properties(physical_device);
            let mut device_local_mb = 0u64;
            for i in 0..mem_props.memory_heap_count {
                let heap = mem_props.memory_heaps[i as usize];
                if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                    device_local_mb += heap.size / (1024 * 1024);
                }
            }
            println!("VRAM (device-local heaps): {} MB", device_local_mb);
            println!("Timestamp period: {:.1} ns/tick", timestamp_period);
            println!();

            Ok(VkContext {
                _entry: entry,
                instance,
                device,
                physical_device,
                queue,
                queue_family,
                command_pool,
                allocator: mem::ManuallyDrop::new(allocator),
                timestamp_period,
            })
        }
    }

    fn create_buffer(&mut self, size: vk::DeviceSize, location: MemoryLocation) -> Buffer {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device.create_buffer(&buffer_info, None).unwrap()
        };

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .allocator
            .allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();
        }

        Buffer {
            buffer,
            allocation: Some(allocation),
            size,
        }
    }

    fn destroy_buffer(&mut self, buf: &mut Buffer) {
        unsafe {
            self.device.destroy_buffer(buf.buffer, None);
        }
        if let Some(alloc) = buf.allocation.take() {
            self.allocator.free(alloc).unwrap();
        }
    }

    fn upload_data<T: Pod>(&mut self, buf: &Buffer, data: &[T]) {
        let alloc = buf.allocation.as_ref().unwrap();
        let mapped = alloc.mapped_ptr().expect("Buffer not host-visible");
        let byte_len = data.len() * mem::size_of::<T>();
        assert!(byte_len <= buf.size as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                mapped.as_ptr() as *mut u8,
                byte_len,
            );
        }
    }

    fn download_data<T: Pod>(&self, buf: &Buffer, count: usize) -> Vec<T> {
        let alloc = buf.allocation.as_ref().unwrap();
        let mapped = alloc.mapped_ptr().expect("Buffer not host-visible");
        let mut out = vec![T::zeroed(); count];
        let byte_len = count * mem::size_of::<T>();
        assert!(byte_len <= buf.size as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped.as_ptr() as *const u8,
                out.as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
        out
    }

    fn alloc_cmd(&self) -> vk::CommandBuffer {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .unwrap()[0]
        }
    }

    fn create_timestamp_query_pool(&self, count: u32) -> vk::QueryPool {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);
        unsafe { self.device.create_query_pool(&info, None).unwrap() }
    }

    fn create_compute_pipeline(
        &self,
        spirv: &[u8],
        descriptor_set_layout: vk::DescriptorSetLayout,
        push_constant_size: u32,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        // Ensure alignment
        let spirv_u32: Vec<u32> = {
            assert!(spirv.len() % 4 == 0);
            let mut v = vec![0u32; spirv.len() / 4];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    spirv.as_ptr(),
                    v.as_mut_ptr() as *mut u8,
                    spirv.len(),
                );
            }
            v
        };

        let shader_module_create_info =
            vk::ShaderModuleCreateInfo::default().code(&spirv_u32);

        let shader_module = unsafe {
            self.device
                .create_shader_module(&shader_module_create_info, None)
                .unwrap()
        };

        let entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_name);

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: push_constant_size,
        };

        let layouts = [descriptor_set_layout];
        let push_ranges = [push_constant_range];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_ranges);

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&layout_info, None)
                .unwrap()
        };

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()[0]
        };

        unsafe {
            self.device.destroy_shader_module(shader_module, None);
        }

        (pipeline, pipeline_layout)
    }

    fn create_descriptor_set_layout(
        &self,
        binding_count: u32,
    ) -> vk::DescriptorSetLayout {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..binding_count)
            .map(|i| {
                vk::DescriptorSetLayoutBinding {
                    binding: i,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                }
            })
            .collect();

        let layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        }
    }

    fn create_descriptor_pool(&self, max_sets: u32, max_descriptors: u32) -> vk::DescriptorPool {
        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: max_descriptors,
        };
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);

        unsafe {
            self.device
                .create_descriptor_pool(&pool_info, None)
                .unwrap()
        }
    }

    fn allocate_descriptor_set(
        &self,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .unwrap()[0]
        }
    }

    fn update_descriptor_set(&self, set: vk::DescriptorSet, buffers: &[&Buffer]) {
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|b| vk::DescriptorBufferInfo {
                buffer: b.buffer,
                offset: 0,
                range: b.size,
            })
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(slice::from_ref(info))
            })
            .collect();

        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    fn submit_and_wait(&self, cmd: vk::CommandBuffer) {
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmds);

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device.create_fence(&fence_info, None).unwrap() };

        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], fence)
                .unwrap();
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .unwrap();
            self.device.destroy_fence(fence, None);
        }
    }

    fn get_timestamps(&self, query_pool: vk::QueryPool, first: u32, count: u32) -> Vec<u64> {
        let mut results = vec![0u64; count as usize];
        unsafe {
            self.device
                .get_query_pool_results(
                    query_pool,
                    first,
                    &mut results,
                    vk::QueryResultFlags::TYPE_64,
                )
                .unwrap();
        }
        results
    }

    fn ticks_to_ms(&self, ticks: u64) -> f64 {
        (ticks as f64 * self.timestamp_period) / 1_000_000.0
    }

    fn cleanup_pipeline(&self, pipeline: vk::Pipeline, layout: vk::PipelineLayout) {
        unsafe {
            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_pipeline_layout(layout, None);
        }
    }

    fn cleanup_descriptors(
        &self,
        pool: vk::DescriptorPool,
        set_layout: vk::DescriptorSetLayout,
    ) {
        unsafe {
            self.device.destroy_descriptor_pool(pool, None);
            self.device
                .destroy_descriptor_set_layout(set_layout, None);
        }
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_command_pool(self.command_pool, None);
            // Drop allocator BEFORE destroying device — its destructor calls
            // Vulkan device functions, so the device must still be live.
            mem::ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ── Benchmarks ─────────────────────────────────────────────────────────

fn bench_memory_bandwidth(ctx: &mut VkContext) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let n: usize = 16 * 1024 * 1024; // 16M floats = 64 MB
    let byte_size = (n * 4) as vk::DeviceSize;
    let size_mb = byte_size as f64 / (1024.0 * 1024.0);

    // ── Device-local copy (via compute shader) ──
    {
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/copy.spv"));
        let ds_layout = ctx.create_descriptor_set_layout(2);
        let (pipeline, p_layout) =
            ctx.create_compute_pipeline(spirv, ds_layout, mem::size_of::<CopyPush>() as u32);
        let pool = ctx.create_descriptor_pool(1, 2);
        let ds = ctx.allocate_descriptor_set(pool, ds_layout);

        let mut src = ctx.create_buffer(byte_size, MemoryLocation::CpuToGpu);

        // Try GpuOnly for dst (true device-local VRAM). On UMA drivers that
        // lack a distinct device-local heap, fall back to CpuToGpu.
        let (mut dst, dst_is_device_local) = {
            // gpu-allocator returns Err when no suitable memory type exists.
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST;
            let buf_info = vk::BufferCreateInfo::default()
                .size(byte_size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let vk_buf = unsafe { ctx.device.create_buffer(&buf_info, None).unwrap() };
            let reqs = unsafe { ctx.device.get_buffer_memory_requirements(vk_buf) };
            match ctx.allocator.allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements: reqs,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            }) {
                Ok(alloc) => {
                    unsafe {
                        ctx.device
                            .bind_buffer_memory(vk_buf, alloc.memory(), alloc.offset())
                            .unwrap();
                    }
                    (Buffer { buffer: vk_buf, allocation: Some(alloc), size: byte_size }, true)
                }
                Err(_) => {
                    unsafe { ctx.device.destroy_buffer(vk_buf, None); }
                    (ctx.create_buffer(byte_size, MemoryLocation::CpuToGpu), false)
                }
            }
        };
        if !dst_is_device_local {
            println!("  [note] GpuOnly alloc failed; dst buffer using CpuToGpu (UMA shared memory)");
        }

        // Fill source
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        ctx.upload_data(&src, &data);
        ctx.update_descriptor_set(ds, &[&src, &dst]);

        let cmd = ctx.alloc_cmd();
        let qp = ctx.create_timestamp_query_pool(2);

        // Warmup
        unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            ctx.device.begin_command_buffer(cmd, &begin).unwrap();
            ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            ctx.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                p_layout,
                0,
                &[ds],
                &[],
            );
            let push = CopyPush { n: n as u32 };
            ctx.device.cmd_push_constants(
                cmd,
                p_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            ctx.device.cmd_dispatch(cmd, ((n + 255) / 256) as u32, 1, 1);
            ctx.device.end_command_buffer(cmd).unwrap();
        }
        ctx.submit_and_wait(cmd);

        // Timed run
        unsafe {
            ctx.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            ctx.device.begin_command_buffer(cmd, &begin).unwrap();
            ctx.device.cmd_reset_query_pool(cmd, qp, 0, 2);
            ctx.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
            ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            ctx.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                p_layout,
                0,
                &[ds],
                &[],
            );
            let push = CopyPush { n: n as u32 };
            ctx.device.cmd_push_constants(
                cmd,
                p_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            ctx.device.cmd_dispatch(cmd, ((n + 255) / 256) as u32, 1, 1);
            ctx.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
            ctx.device.end_command_buffer(cmd).unwrap();
        }
        ctx.submit_and_wait(cmd);

        let ts = ctx.get_timestamps(qp, 0, 2);
        let ms = ctx.ticks_to_ms(ts[1].wrapping_sub(ts[0]));
        // Read + write = 2x
        let bw = (2.0 * size_mb) / (ms / 1000.0) / 1024.0; // GB/s

        let copy_name = if dst_is_device_local {
            "Memory: Buffer Copy (compute, device-local)"
        } else {
            "Memory: Buffer Copy (compute, shared/UMA)"
        };
        results.push(BenchResult {
            name: copy_name.into(),
            size_mb: size_mb * 2.0,
            time_ms: ms,
            metric: "GB/s".into(),
            metric_val: bw,
            // Peak% only meaningful when dst is truly device-local VRAM
            peak_pct: if dst_is_device_local {
                Some(bw / 89.6 * 100.0) // 890M theoretical ~89.6 GB/s
            } else {
                None
            },
        });

        unsafe {
            ctx.device.destroy_query_pool(qp, None);
            ctx.device.free_command_buffers(ctx.command_pool, &[cmd]);
        }
        ctx.destroy_buffer(&mut src);
        ctx.destroy_buffer(&mut dst);
        ctx.cleanup_pipeline(pipeline, p_layout);
        ctx.cleanup_descriptors(pool, ds_layout);
    }

    // ── Host-visible write bandwidth (CPU->GPU mapped) ──
    {
        let mut buf = ctx.create_buffer(byte_size, MemoryLocation::CpuToGpu);
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let start = Instant::now();
        for _ in 0..10 {
            ctx.upload_data(&buf, &data);
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / 10.0;
        let bw = size_mb / (elapsed / 1000.0) / 1024.0;

        results.push(BenchResult {
            name: "Memory: Host->Device write".into(),
            size_mb,
            time_ms: elapsed,
            metric: "GB/s".into(),
            metric_val: bw,
            peak_pct: None,
        });

        ctx.destroy_buffer(&mut buf);
    }

    // ── Host-visible read bandwidth (GPU->CPU mapped) ──
    {
        let mut buf = ctx.create_buffer(byte_size, MemoryLocation::GpuToCpu);
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        ctx.upload_data(&buf, &data);

        let start = Instant::now();
        for _ in 0..10 {
            let _v: Vec<f32> = ctx.download_data(&buf, n);
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / 10.0;
        let bw = size_mb / (elapsed / 1000.0) / 1024.0;

        results.push(BenchResult {
            name: "Memory: Device->Host read".into(),
            size_mb,
            time_ms: elapsed,
            metric: "GB/s".into(),
            metric_val: bw,
            peak_pct: None,
        });

        ctx.destroy_buffer(&mut buf);
    }

    results
}

fn bench_saxpy(ctx: &mut VkContext) -> BenchResult {
    let n: usize = 16 * 1024 * 1024; // 16M elements
    let byte_size = (n * 4) as vk::DeviceSize;
    let size_mb = byte_size as f64 / (1024.0 * 1024.0);

    let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/saxpy.spv"));
    let ds_layout = ctx.create_descriptor_set_layout(3);
    let (pipeline, p_layout) =
        ctx.create_compute_pipeline(spirv, ds_layout, mem::size_of::<SaxpyPush>() as u32);
    let pool = ctx.create_descriptor_pool(1, 3);
    let ds = ctx.allocate_descriptor_set(pool, ds_layout);

    let mut buf_x = ctx.create_buffer(byte_size, MemoryLocation::CpuToGpu);
    let mut buf_y = ctx.create_buffer(byte_size, MemoryLocation::CpuToGpu);
    let mut buf_out = ctx.create_buffer(byte_size, MemoryLocation::CpuToGpu);

    let mut rng = rand::thread_rng();
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    ctx.upload_data(&buf_x, &x);
    ctx.upload_data(&buf_y, &y);
    ctx.update_descriptor_set(ds, &[&buf_x, &buf_y, &buf_out]);

    let cmd = ctx.alloc_cmd();
    let qp = ctx.create_timestamp_query_pool(2);
    let workgroups = ((n + 255) / 256) as u32;
    let push = SaxpyPush {
        a: 2.5,
        n: n as u32,
    };

    // Warmup
    unsafe {
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        ctx.device.begin_command_buffer(cmd, &begin).unwrap();
        ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            p_layout,
            0,
            &[ds],
            &[],
        );
        ctx.device.cmd_push_constants(
            cmd,
            p_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push),
        );
        ctx.device.cmd_dispatch(cmd, workgroups, 1, 1);
        ctx.device.end_command_buffer(cmd).unwrap();
    }
    ctx.submit_and_wait(cmd);

    // Timed
    unsafe {
        ctx.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .unwrap();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        ctx.device.begin_command_buffer(cmd, &begin).unwrap();
        ctx.device.cmd_reset_query_pool(cmd, qp, 0, 2);
        ctx.device
            .cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
        ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            p_layout,
            0,
            &[ds],
            &[],
        );
        ctx.device.cmd_push_constants(
            cmd,
            p_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push),
        );
        ctx.device.cmd_dispatch(cmd, workgroups, 1, 1);
        ctx.device
            .cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
        ctx.device.end_command_buffer(cmd).unwrap();
    }
    ctx.submit_and_wait(cmd);

    let ts = ctx.get_timestamps(qp, 0, 2);
    let ms = ctx.ticks_to_ms(ts[1].wrapping_sub(ts[0]));
    // SAXPY: 2 FLOPs per element (multiply + add)
    let gflops = (2.0 * n as f64) / (ms / 1000.0) / 1e9;
    // Radeon 890M theoretical FP32 ~8.6 TFLOPS
    let peak_pct = gflops / 8600.0 * 100.0;

    unsafe {
        ctx.device.destroy_query_pool(qp, None);
        ctx.device.free_command_buffers(ctx.command_pool, &[cmd]);
    }
    ctx.destroy_buffer(&mut buf_x);
    ctx.destroy_buffer(&mut buf_y);
    ctx.destroy_buffer(&mut buf_out);
    ctx.cleanup_pipeline(pipeline, p_layout);
    ctx.cleanup_descriptors(pool, ds_layout);

    BenchResult {
        name: "Compute: SAXPY (a*x+y)".into(),
        size_mb: size_mb * 3.0,
        time_ms: ms,
        metric: "GFLOPS".into(),
        metric_val: gflops,
        peak_pct: Some(peak_pct),
    }
}

fn bench_reduction(ctx: &mut VkContext) -> BenchResult {
    let n: usize = 16 * 1024 * 1024;
    let byte_size = (n * 4) as vk::DeviceSize;
    let size_mb = byte_size as f64 / (1024.0 * 1024.0);

    let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/reduce.spv"));
    let ds_layout = ctx.create_descriptor_set_layout(2);
    let (pipeline, p_layout) =
        ctx.create_compute_pipeline(spirv, ds_layout, mem::size_of::<ReducePush>() as u32);
    let pool = ctx.create_descriptor_pool(1, 2);
    let ds = ctx.allocate_descriptor_set(pool, ds_layout);

    let workgroups = ((n + 255) / 256) as u32;
    let out_size = (workgroups as usize * 4) as vk::DeviceSize;

    let mut buf_in = ctx.create_buffer(byte_size, MemoryLocation::CpuToGpu);
    let mut buf_out = ctx.create_buffer(out_size, MemoryLocation::CpuToGpu);

    let data: Vec<f32> = vec![1.0f32; n];
    ctx.upload_data(&buf_in, &data);
    ctx.update_descriptor_set(ds, &[&buf_in, &buf_out]);

    let cmd = ctx.alloc_cmd();
    let qp = ctx.create_timestamp_query_pool(2);
    let push = ReducePush { n: n as u32 };

    // Warmup
    unsafe {
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        ctx.device.begin_command_buffer(cmd, &begin).unwrap();
        ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            p_layout,
            0,
            &[ds],
            &[],
        );
        ctx.device.cmd_push_constants(
            cmd,
            p_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push),
        );
        ctx.device.cmd_dispatch(cmd, workgroups, 1, 1);
        ctx.device.end_command_buffer(cmd).unwrap();
    }
    ctx.submit_and_wait(cmd);

    // Timed
    unsafe {
        ctx.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .unwrap();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        ctx.device.begin_command_buffer(cmd, &begin).unwrap();
        ctx.device.cmd_reset_query_pool(cmd, qp, 0, 2);
        ctx.device
            .cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
        ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            p_layout,
            0,
            &[ds],
            &[],
        );
        ctx.device.cmd_push_constants(
            cmd,
            p_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push),
        );
        ctx.device.cmd_dispatch(cmd, workgroups, 1, 1);
        ctx.device
            .cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
        ctx.device.end_command_buffer(cmd).unwrap();
    }
    ctx.submit_and_wait(cmd);

    let ts = ctx.get_timestamps(qp, 0, 2);
    let ms = ctx.ticks_to_ms(ts[1].wrapping_sub(ts[0]));
    // Reduction: n-1 additions, effectively n elements processed
    let gelements_s = (n as f64) / (ms / 1000.0) / 1e9;

    unsafe {
        ctx.device.destroy_query_pool(qp, None);
        ctx.device.free_command_buffers(ctx.command_pool, &[cmd]);
    }
    ctx.destroy_buffer(&mut buf_in);
    ctx.destroy_buffer(&mut buf_out);
    ctx.cleanup_pipeline(pipeline, p_layout);
    ctx.cleanup_descriptors(pool, ds_layout);

    BenchResult {
        name: "Compute: Parallel Reduction".into(),
        size_mb,
        time_ms: ms,
        metric: "GElem/s".into(),
        metric_val: gelements_s,
        peak_pct: None,
    }
}

fn bench_matmul(ctx: &mut VkContext) -> BenchResult {
    let m: u32 = 1024;
    let n: u32 = 1024;
    let k: u32 = 1024;
    let a_size = (m * k * 4) as vk::DeviceSize;
    let b_size = (k * n * 4) as vk::DeviceSize;
    let c_size = (m * n * 4) as vk::DeviceSize;
    let total_mb = (a_size + b_size + c_size) as f64 / (1024.0 * 1024.0);

    let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/matmul.spv"));
    let ds_layout = ctx.create_descriptor_set_layout(3);
    let (pipeline, p_layout) =
        ctx.create_compute_pipeline(spirv, ds_layout, mem::size_of::<MatmulPush>() as u32);
    let pool = ctx.create_descriptor_pool(1, 3);
    let ds = ctx.allocate_descriptor_set(pool, ds_layout);

    let mut buf_a = ctx.create_buffer(a_size, MemoryLocation::CpuToGpu);
    let mut buf_b = ctx.create_buffer(b_size, MemoryLocation::CpuToGpu);
    let mut buf_c = ctx.create_buffer(c_size, MemoryLocation::CpuToGpu);

    let mut rng = rand::thread_rng();
    let a_data: Vec<f32> = (0..(m * k) as usize)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let b_data: Vec<f32> = (0..(k * n) as usize)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    ctx.upload_data(&buf_a, &a_data);
    ctx.upload_data(&buf_b, &b_data);
    ctx.update_descriptor_set(ds, &[&buf_a, &buf_b, &buf_c]);

    let cmd = ctx.alloc_cmd();
    let qp = ctx.create_timestamp_query_pool(2);
    let push = MatmulPush { m, n, k };

    let wg_x = (n + 15) / 16;
    let wg_y = (m + 15) / 16;

    // Warmup
    unsafe {
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        ctx.device.begin_command_buffer(cmd, &begin).unwrap();
        ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            p_layout,
            0,
            &[ds],
            &[],
        );
        ctx.device.cmd_push_constants(
            cmd,
            p_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push),
        );
        ctx.device.cmd_dispatch(cmd, wg_x, wg_y, 1);
        ctx.device.end_command_buffer(cmd).unwrap();
    }
    ctx.submit_and_wait(cmd);

    // Timed
    unsafe {
        ctx.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .unwrap();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        ctx.device.begin_command_buffer(cmd, &begin).unwrap();
        ctx.device.cmd_reset_query_pool(cmd, qp, 0, 2);
        ctx.device
            .cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
        ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            p_layout,
            0,
            &[ds],
            &[],
        );
        ctx.device.cmd_push_constants(
            cmd,
            p_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push),
        );
        ctx.device.cmd_dispatch(cmd, wg_x, wg_y, 1);
        ctx.device
            .cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
        ctx.device.end_command_buffer(cmd).unwrap();
    }
    ctx.submit_and_wait(cmd);

    let ts = ctx.get_timestamps(qp, 0, 2);
    let ms = ctx.ticks_to_ms(ts[1].wrapping_sub(ts[0]));
    // Matmul FLOPs: 2*M*N*K (multiply + accumulate per output element)
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = flops / (ms / 1000.0) / 1e9;
    let peak_pct = gflops / 8600.0 * 100.0;

    unsafe {
        ctx.device.destroy_query_pool(qp, None);
        ctx.device.free_command_buffers(ctx.command_pool, &[cmd]);
    }
    ctx.destroy_buffer(&mut buf_a);
    ctx.destroy_buffer(&mut buf_b);
    ctx.destroy_buffer(&mut buf_c);
    ctx.cleanup_pipeline(pipeline, p_layout);
    ctx.cleanup_descriptors(pool, ds_layout);

    BenchResult {
        name: "Compute: MatMul 1024x1024".into(),
        size_mb: total_mb,
        time_ms: ms,
        metric: "GFLOPS".into(),
        metric_val: gflops,
        peak_pct: Some(peak_pct),
    }
}

// ── Main ───────────────────────────────────────────────────────────────

fn main() {
    println!("=== vk-compute-bench ===");
    println!("Vulkan compute benchmark for AMD Radeon 890M (RDNA 3.5)");
    println!("Theoretical peak: ~8.6 TFLOPS FP32, ~89.6 GB/s mem BW");
    println!();

    let mut ctx = match VkContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Make sure Vulkan drivers (RADV) are installed.");
            std::process::exit(1);
        }
    };

    let mut all_results: Vec<BenchResult> = Vec::new();

    // Memory bandwidth
    print!("Running memory bandwidth tests... ");
    let mem_results = bench_memory_bandwidth(&mut ctx);
    println!("done");
    all_results.extend(mem_results);

    // SAXPY
    print!("Running SAXPY compute test... ");
    all_results.push(bench_saxpy(&mut ctx));
    println!("done");

    // Reduction
    print!("Running parallel reduction test... ");
    all_results.push(bench_reduction(&mut ctx));
    println!("done");

    // Matrix multiply
    print!("Running matrix multiply test... ");
    all_results.push(bench_matmul(&mut ctx));
    println!("done");

    println!();

    // ── Results table ──
    let mut table = Table::new();
    table.add_row(row![
        "Benchmark",
        "Data (MB)",
        "Time (ms)",
        "Throughput",
        "Value",
        "% of Peak"
    ]);

    for r in &all_results {
        let peak_str = match r.peak_pct {
            Some(p) => format!("{:.1}%", p),
            None => "-".into(),
        };
        table.add_row(row![
            r.name,
            format!("{:.1}", r.size_mb),
            format!("{:.3}", r.time_ms),
            r.metric,
            format!("{:.2}", r.metric_val),
            peak_str,
        ]);
    }

    table.printstd();
    println!();
    println!("Peak references: FP32 ~8.6 TFLOPS, Memory BW ~89.6 GB/s (LPDDR5X-7500)");
}
