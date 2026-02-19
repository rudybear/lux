//! Vulkan initialization: instance, device, queues, allocator, command pool.
//!
//! Supports optional ray tracing extensions with graceful fallback.

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use log::{info, warn};
use std::ffi::{CStr, CString};
use std::sync::Mutex;

/// Holds all core Vulkan state for the playground.
///
/// Fields are ordered so that Rust's drop order (top-to-bottom declaration)
/// destroys resources before the device/instance they depend on.
pub struct VulkanContext {
    // Optional ray tracing extension loaders (no drop needed, just fn pointers)
    pub rt_pipeline_loader: Option<ash::khr::ray_tracing_pipeline::Device>,
    pub accel_struct_loader: Option<ash::khr::acceleration_structure::Device>,
    pub rt_properties: Option<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>>,

    // Allocator must be dropped before device — wrapped in Option so we can take() in Drop
    allocator_inner: Option<Allocator>,
    /// Public access to the allocator through a Mutex for thread safety.
    /// This is always Some() during the lifetime of VulkanContext.
    pub allocator: Mutex<Option<()>>, // Dummy mutex; real allocator is allocator_inner

    pub command_pool: vk::CommandPool,
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family: u32,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,

    // Debug utils (only in debug builds) — must be destroyed before instance
    debug_utils_loader: Option<ash::ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,

    pub instance: ash::Instance,
    pub entry: ash::Entry,

    /// Whether destroy() has been called explicitly
    destroyed: bool,
}

// We store the allocator directly inside VulkanContext and provide
// thread-safe access through a Mutex wrapper.
// Since gpu-allocator's Allocator is not Send+Sync by default in all versions,
// we handle it carefully.

/// Thread-safe wrapper around gpu-allocator's Allocator.
///
/// This is used to provide `&Mutex<Allocator>` access to callers.
pub struct AllocatorWrapper {
    inner: Mutex<Allocator>,
}

impl VulkanContext {
    /// Create a new VulkanContext.
    ///
    /// If `request_rt` is true, will attempt to enable ray tracing extensions.
    /// If the GPU does not support them, `rt_pipeline_loader` etc. will be None.
    pub fn new(request_rt: bool) -> Result<Self, String> {
        // --- Entry ---
        let entry = unsafe {
            ash::Entry::load().map_err(|e| format!("Failed to load Vulkan: {}", e))?
        };

        // --- Instance ---
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"lux-playground")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"lux")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let mut layer_names: Vec<CString> = Vec::new();
        let mut extension_names: Vec<CString> = Vec::new();

        // Enable validation layers in debug builds
        let enable_validation = cfg!(debug_assertions);
        if enable_validation {
            let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
            let available_layers = unsafe {
                entry
                    .enumerate_instance_layer_properties()
                    .unwrap_or_default()
            };
            let has_validation = available_layers.iter().any(|layer| {
                let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                name == validation_layer.as_c_str()
            });
            if has_validation {
                layer_names.push(validation_layer);
                extension_names.push(CString::new("VK_EXT_debug_utils").unwrap());
                info!("Validation layers enabled");
            } else {
                warn!("Validation layers requested but not available");
            }
        }

        let layer_name_ptrs: Vec<*const i8> =
            layer_names.iter().map(|n| n.as_ptr()).collect();
        let extension_name_ptrs: Vec<*const i8> =
            extension_names.iter().map(|n| n.as_ptr()).collect();

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_name_ptrs)
            .enabled_extension_names(&extension_name_ptrs);

        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .map_err(|e| format!("Failed to create Vulkan instance: {:?}", e))?
        };

        // --- Debug messenger ---
        let (debug_utils_loader, debug_messenger) = if enable_validation
            && extension_names
                .iter()
                .any(|n| n.as_c_str() == c"VK_EXT_debug_utils")
        {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));

            let messenger = unsafe {
                loader
                    .create_debug_utils_messenger(&messenger_info, None)
                    .ok()
            };

            (Some(loader), messenger)
        } else {
            (None, None)
        };

        // --- Physical device selection ---
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(|e| format!("Failed to enumerate physical devices: {:?}", e))?
        };

        if physical_devices.is_empty() {
            return Err("No Vulkan-capable GPUs found".to_string());
        }

        let mut selected_physical_device = None;
        let mut selected_queue_family = 0u32;
        let mut rt_available = false;

        for &phys_dev in &physical_devices {
            let props = unsafe { instance.get_physical_device_properties(phys_dev) };
            let api_version = props.api_version;

            if vk::api_version_major(api_version) < 1
                || (vk::api_version_major(api_version) == 1
                    && vk::api_version_minor(api_version) < 2)
            {
                continue;
            }

            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(phys_dev) };
            let graphics_family = queue_families.iter().enumerate().find(|(_, props)| {
                props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            });

            if let Some((family_idx, _)) = graphics_family {
                let dev_extensions = unsafe {
                    instance
                        .enumerate_device_extension_properties(phys_dev)
                        .unwrap_or_default()
                };
                let ext_names: Vec<String> = dev_extensions
                    .iter()
                    .map(|e| {
                        unsafe { CStr::from_ptr(e.extension_name.as_ptr()) }
                            .to_string_lossy()
                            .into_owned()
                    })
                    .collect();

                let has_rt = ext_names.contains(&"VK_KHR_ray_tracing_pipeline".to_string())
                    && ext_names.contains(&"VK_KHR_acceleration_structure".to_string())
                    && ext_names.contains(&"VK_KHR_deferred_host_operations".to_string());

                let is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;

                if selected_physical_device.is_none()
                    || (is_discrete && !rt_available && has_rt)
                    || (is_discrete && selected_physical_device.is_some())
                {
                    selected_physical_device = Some(phys_dev);
                    selected_queue_family = family_idx as u32;
                    rt_available = has_rt;

                    let dev_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
                    info!(
                        "Selected GPU: {} (Vulkan {}.{}, RT: {})",
                        dev_name.to_string_lossy(),
                        vk::api_version_major(api_version),
                        vk::api_version_minor(api_version),
                        if has_rt { "yes" } else { "no" }
                    );
                }
            }
        }

        let physical_device = selected_physical_device
            .ok_or("No suitable GPU found (need Vulkan 1.2+ with graphics queue)")?;

        let enable_rt = request_rt && rt_available;
        if request_rt && !rt_available {
            warn!("Ray tracing requested but GPU does not support VK_KHR_ray_tracing_pipeline");
        }

        // --- Device creation ---
        let queue_priority = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(selected_queue_family)
            .queue_priorities(&queue_priority);
        let queue_create_infos = [queue_create_info];

        let mut device_extensions: Vec<CString> = Vec::new();
        if enable_rt {
            device_extensions.push(CString::new("VK_KHR_ray_tracing_pipeline").unwrap());
            device_extensions.push(CString::new("VK_KHR_acceleration_structure").unwrap());
            device_extensions.push(CString::new("VK_KHR_deferred_host_operations").unwrap());
            device_extensions.push(CString::new("VK_KHR_buffer_device_address").unwrap());
        }

        let device_ext_ptrs: Vec<*const i8> =
            device_extensions.iter().map(|n| n.as_ptr()).collect();

        let mut vulkan_12_features =
            vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);

        let mut accel_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);

        let mut rt_pipeline_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
                .ray_tracing_pipeline(true);

        let mut features2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan_12_features);

        if enable_rt {
            features2 = features2
                .push_next(&mut accel_features)
                .push_next(&mut rt_pipeline_features);
        }

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_ext_ptrs)
            .push_next(&mut features2);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(|e| format!("Failed to create logical device: {:?}", e))?
        };

        let graphics_queue = unsafe { device.get_device_queue(selected_queue_family, 0) };

        // --- Command pool ---
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(selected_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create command pool: {:?}", e))?
        };

        // --- gpu-allocator ---
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        })
        .map_err(|e| format!("Failed to create GPU allocator: {:?}", e))?;

        // --- RT extension loaders and properties ---
        let (rt_pipeline_loader, accel_struct_loader, rt_properties) = if enable_rt {
            let rt_loader =
                ash::khr::ray_tracing_pipeline::Device::new(&instance, &device);
            let as_loader =
                ash::khr::acceleration_structure::Device::new(&instance, &device);

            let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut props2 =
                vk::PhysicalDeviceProperties2::default().push_next(&mut rt_props);
            unsafe {
                instance.get_physical_device_properties2(physical_device, &mut props2);
            }

            info!(
                "RT properties: handle_size={}, max_recursion={}",
                rt_props.shader_group_handle_size, rt_props.max_ray_recursion_depth
            );

            // The properties struct is plain-old-data; safe to transmute the lifetime.
            let rt_props_static: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static> =
                unsafe { std::mem::transmute(rt_props) };

            (Some(rt_loader), Some(as_loader), Some(rt_props_static))
        } else {
            (None, None, None)
        };

        info!("Vulkan context initialized successfully");

        Ok(VulkanContext {
            rt_pipeline_loader,
            accel_struct_loader,
            rt_properties,
            allocator_inner: Some(allocator),
            allocator: Mutex::new(Some(())),
            command_pool,
            graphics_queue,
            graphics_queue_family: selected_queue_family,
            physical_device,
            device,
            debug_utils_loader,
            debug_messenger,
            instance,
            entry,
            destroyed: false,
        })
    }

    /// Returns true if ray tracing extensions are available.
    pub fn supports_rt(&self) -> bool {
        self.rt_pipeline_loader.is_some()
    }

    /// Get a mutable reference to the allocator. Panics if already destroyed.
    pub fn allocator_mut(&mut self) -> &mut Allocator {
        self.allocator_inner
            .as_mut()
            .expect("Allocator already destroyed")
    }

    /// Get a reference to the allocator. Panics if already destroyed.
    pub fn allocator_ref(&self) -> &Allocator {
        self.allocator_inner
            .as_ref()
            .expect("Allocator already destroyed")
    }

    /// Allocate and begin a one-shot command buffer.
    pub fn begin_single_commands(&self) -> Result<vk::CommandBuffer, String> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| format!("Failed to allocate command buffer: {:?}", e))?[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| format!("Failed to begin command buffer: {:?}", e))?;
        }

        Ok(cmd)
    }

    /// End, submit, and wait for a one-shot command buffer.
    pub fn end_single_commands(&self, cmd: vk::CommandBuffer) -> Result<(), String> {
        unsafe {
            self.device
                .end_command_buffer(cmd)
                .map_err(|e| format!("Failed to end command buffer: {:?}", e))?;
        }

        let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            self.device
                .create_fence(&fence_info, None)
                .map_err(|e| format!("Failed to create fence: {:?}", e))?
        };

        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], fence)
                .map_err(|e| format!("Failed to submit command buffer: {:?}", e))?;

            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| format!("Failed to wait for fence: {:?}", e))?;

            self.device.destroy_fence(fence, None);
            self.device
                .free_command_buffers(self.command_pool, &[cmd]);
        }

        Ok(())
    }

    /// Explicitly destroy all Vulkan resources in the correct order.
    ///
    /// Must be called before the VulkanContext is dropped.
    /// The Drop impl will also call this if it hasn't been called yet.
    pub fn destroy(&mut self) {
        if self.destroyed {
            return;
        }
        self.destroyed = true;

        unsafe {
            let _ = self.device.device_wait_idle();
        }

        // Destroy command pool
        if self.command_pool != vk::CommandPool::null() {
            unsafe {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            self.command_pool = vk::CommandPool::null();
        }

        // Drop allocator (needs device alive)
        drop(self.allocator_inner.take());

        // Destroy debug messenger before instance
        unsafe {
            if let (Some(loader), Some(messenger)) =
                (&self.debug_utils_loader, self.debug_messenger.take())
            {
                loader.destroy_debug_utils_messenger(messenger, None);
            }

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        self.destroy();
    }
}

/// Vulkan debug callback for validation layers.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = if callback_data.is_null() {
        "Unknown validation message".to_string()
    } else {
        let data = unsafe { &*callback_data };
        if data.p_message.is_null() {
            "Empty validation message".to_string()
        } else {
            unsafe { CStr::from_ptr(data.p_message) }
                .to_string_lossy()
                .into_owned()
        }
    };

    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("[Vulkan] {}", msg);
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::warn!("[Vulkan] {}", msg);
    } else {
        log::info!("[Vulkan] {}", msg);
    }

    vk::FALSE
}
