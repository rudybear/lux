//! Vulkan initialization: instance, device, queues, allocator, command pool.
//!
//! Supports optional ray tracing and mesh shader extensions with graceful fallback.

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use log::{info, warn};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
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

    // Optional mesh shader extension loader and properties
    pub mesh_shader_supported: bool,
    pub mesh_shader_loader: Option<ash::ext::mesh_shader::Device>,
    pub mesh_shader_properties: Option<vk::PhysicalDeviceMeshShaderPropertiesEXT<'static>>,

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

    // Surface/swapchain (interactive mode only)
    surface_loader: Option<ash::khr::surface::Instance>,
    swapchain_loader: Option<ash::khr::swapchain::Device>,
    pub surface: Option<vk::SurfaceKHR>,
    pub swapchain: Option<vk::SwapchainKHR>,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,

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
        let mut mesh_shader_available = false;

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

                let has_mesh = ext_names.contains(&"VK_EXT_mesh_shader".to_string());

                let is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;

                if selected_physical_device.is_none()
                    || (is_discrete && !rt_available && has_rt)
                    || (is_discrete && selected_physical_device.is_some())
                {
                    selected_physical_device = Some(phys_dev);
                    selected_queue_family = family_idx as u32;
                    rt_available = has_rt;
                    mesh_shader_available = has_mesh;

                    let dev_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
                    info!(
                        "Selected GPU: {} (Vulkan {}.{}, RT: {}, Mesh: {})",
                        dev_name.to_string_lossy(),
                        vk::api_version_major(api_version),
                        vk::api_version_minor(api_version),
                        if has_rt { "yes" } else { "no" },
                        if has_mesh { "yes" } else { "no" }
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
        let enable_mesh = mesh_shader_available;

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
        if enable_mesh {
            device_extensions.push(CString::new("VK_EXT_mesh_shader").unwrap());
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

        let mut mesh_shader_features =
            vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
                .mesh_shader(true)
                .task_shader(true);

        let mut features2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan_12_features);

        if enable_rt {
            features2 = features2
                .push_next(&mut accel_features)
                .push_next(&mut rt_pipeline_features);
        }
        if enable_mesh {
            features2 = features2.push_next(&mut mesh_shader_features);
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

        // --- Mesh shader extension loader and properties ---
        let (mesh_shader_loader, mesh_shader_properties) = if enable_mesh {
            let ms_loader = ash::ext::mesh_shader::Device::new(&instance, &device);

            let mut ms_props = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
            let mut props2 =
                vk::PhysicalDeviceProperties2::default().push_next(&mut ms_props);
            unsafe {
                instance.get_physical_device_properties2(physical_device, &mut props2);
            }

            info!(
                "Mesh shader properties: max_output_vertices={}, max_output_primitives={}, max_work_group_invocations={}",
                ms_props.max_mesh_output_vertices,
                ms_props.max_mesh_output_primitives,
                ms_props.max_mesh_work_group_invocations
            );

            let ms_props_static: vk::PhysicalDeviceMeshShaderPropertiesEXT<'static> =
                unsafe { std::mem::transmute(ms_props) };

            (Some(ms_loader), Some(ms_props_static))
        } else {
            (None, None)
        };

        info!("Vulkan context initialized successfully");

        Ok(VulkanContext {
            rt_pipeline_loader,
            accel_struct_loader,
            rt_properties,
            mesh_shader_supported: enable_mesh,
            mesh_shader_loader,
            mesh_shader_properties,
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
            surface_loader: None,
            swapchain_loader: None,
            surface: None,
            swapchain: None,
            swapchain_images: Vec::new(),
            swapchain_image_views: Vec::new(),
            swapchain_format: vk::Format::UNDEFINED,
            swapchain_extent: vk::Extent2D { width: 0, height: 0 },
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

    /// Create a VulkanContext with a window surface for interactive rendering.
    pub fn new_with_window(
        window: &(impl HasDisplayHandle + HasWindowHandle),
        request_rt: bool,
    ) -> Result<Self, String> {
        let entry = unsafe {
            ash::Entry::load().map_err(|e| format!("Failed to load Vulkan: {}", e))?
        };

        let display_handle = window
            .display_handle()
            .map_err(|e| format!("Failed to get display handle: {}", e))?;
        let window_handle = window
            .window_handle()
            .map_err(|e| format!("Failed to get window handle: {}", e))?;

        // Get required surface extensions for this platform
        let surface_extensions = ash_window::enumerate_required_extensions(display_handle.as_raw())
            .map_err(|e| format!("Failed to enumerate surface extensions: {:?}", e))?;

        // --- Instance ---
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"lux-playground")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"lux")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let mut extension_names: Vec<*const i8> = surface_extensions.to_vec();

        let enable_validation = cfg!(debug_assertions);
        let mut layer_names: Vec<CString> = Vec::new();
        let mut extra_extensions: Vec<CString> = Vec::new();
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
                extra_extensions.push(CString::new("VK_EXT_debug_utils").unwrap());
                info!("Validation layers enabled");
            }
        }

        let layer_name_ptrs: Vec<*const i8> =
            layer_names.iter().map(|n| n.as_ptr()).collect();
        for ext in &extra_extensions {
            extension_names.push(ext.as_ptr());
        }

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_name_ptrs)
            .enabled_extension_names(&extension_names);

        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .map_err(|e| format!("Failed to create Vulkan instance: {:?}", e))?
        };

        // --- Debug messenger ---
        let (debug_utils_loader, debug_messenger) = if enable_validation
            && extra_extensions
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
                loader.create_debug_utils_messenger(&messenger_info, None).ok()
            };
            (Some(loader), messenger)
        } else {
            (None, None)
        };

        // --- Create surface ---
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
            .map_err(|e| format!("Failed to create Vulkan surface: {:?}", e))?
        };
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        // --- Physical device selection (must support presentation) ---
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
        let mut mesh_shader_available = false;

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
            let graphics_family = queue_families.iter().enumerate().find(|(idx, props)| {
                let supports_graphics = props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let supports_present = unsafe {
                    surface_loader
                        .get_physical_device_surface_support(phys_dev, *idx as u32, surface)
                        .unwrap_or(false)
                };
                supports_graphics && supports_present
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

                let has_mesh = ext_names.contains(&"VK_EXT_mesh_shader".to_string());

                let is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;

                if selected_physical_device.is_none()
                    || (is_discrete && !rt_available && has_rt)
                    || (is_discrete && selected_physical_device.is_some())
                {
                    selected_physical_device = Some(phys_dev);
                    selected_queue_family = family_idx as u32;
                    rt_available = has_rt;
                    mesh_shader_available = has_mesh;

                    let dev_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
                    info!(
                        "Selected GPU: {} (Vulkan {}.{}, RT: {}, Mesh: {})",
                        dev_name.to_string_lossy(),
                        vk::api_version_major(api_version),
                        vk::api_version_minor(api_version),
                        if has_rt { "yes" } else { "no" },
                        if has_mesh { "yes" } else { "no" }
                    );
                }
            }
        }

        let physical_device = selected_physical_device
            .ok_or("No suitable GPU found (need Vulkan 1.2+ with graphics+present queue)")?;

        let enable_rt = request_rt && rt_available;
        let enable_mesh = mesh_shader_available;

        // --- Device creation (with VK_KHR_swapchain) ---
        let queue_priority = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(selected_queue_family)
            .queue_priorities(&queue_priority);
        let queue_create_infos = [queue_create_info];

        let mut device_extensions: Vec<CString> = Vec::new();
        device_extensions.push(CString::new("VK_KHR_swapchain").unwrap());
        if enable_rt {
            device_extensions.push(CString::new("VK_KHR_ray_tracing_pipeline").unwrap());
            device_extensions.push(CString::new("VK_KHR_acceleration_structure").unwrap());
            device_extensions.push(CString::new("VK_KHR_deferred_host_operations").unwrap());
            device_extensions.push(CString::new("VK_KHR_buffer_device_address").unwrap());
        }
        if enable_mesh {
            device_extensions.push(CString::new("VK_EXT_mesh_shader").unwrap());
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
        let mut mesh_shader_features =
            vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
                .mesh_shader(true)
                .task_shader(true);

        let mut features2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan_12_features);
        if enable_rt {
            features2 = features2
                .push_next(&mut accel_features)
                .push_next(&mut rt_pipeline_features);
        }
        if enable_mesh {
            features2 = features2.push_next(&mut mesh_shader_features);
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

        // --- Allocator ---
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        })
        .map_err(|e| format!("Failed to create GPU allocator: {:?}", e))?;

        // --- RT extension loaders ---
        let (rt_pipeline_loader, accel_struct_loader, rt_properties) = if enable_rt {
            let rt_loader = ash::khr::ray_tracing_pipeline::Device::new(&instance, &device);
            let as_loader = ash::khr::acceleration_structure::Device::new(&instance, &device);
            let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut rt_props);
            unsafe { instance.get_physical_device_properties2(physical_device, &mut props2); }
            let rt_props_static: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static> =
                unsafe { std::mem::transmute(rt_props) };
            (Some(rt_loader), Some(as_loader), Some(rt_props_static))
        } else {
            (None, None, None)
        };

        // --- Mesh shader extension loader and properties ---
        let (mesh_shader_loader, mesh_shader_properties) = if enable_mesh {
            let ms_loader = ash::ext::mesh_shader::Device::new(&instance, &device);
            let mut ms_props = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut ms_props);
            unsafe { instance.get_physical_device_properties2(physical_device, &mut props2); }
            info!(
                "Mesh shader properties: max_output_vertices={}, max_output_primitives={}, max_work_group_invocations={}",
                ms_props.max_mesh_output_vertices,
                ms_props.max_mesh_output_primitives,
                ms_props.max_mesh_work_group_invocations
            );
            let ms_props_static: vk::PhysicalDeviceMeshShaderPropertiesEXT<'static> =
                unsafe { std::mem::transmute(ms_props) };
            (Some(ms_loader), Some(ms_props_static))
        } else {
            (None, None)
        };

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        info!("Vulkan context (with window) initialized successfully");

        let mut ctx = VulkanContext {
            rt_pipeline_loader,
            accel_struct_loader,
            rt_properties,
            mesh_shader_supported: enable_mesh,
            mesh_shader_loader,
            mesh_shader_properties,
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
            surface_loader: Some(surface_loader),
            swapchain_loader: Some(swapchain_loader),
            surface: Some(surface),
            swapchain: None,
            swapchain_images: Vec::new(),
            swapchain_image_views: Vec::new(),
            swapchain_format: vk::Format::UNDEFINED,
            swapchain_extent: vk::Extent2D { width: 0, height: 0 },
            destroyed: false,
        };

        // Create initial swapchain
        ctx.create_swapchain(800, 600)?;

        Ok(ctx)
    }

    /// Create or recreate the swapchain for the given dimensions.
    pub fn create_swapchain(&mut self, width: u32, height: u32) -> Result<(), String> {
        let surface = self.surface.ok_or("No surface for swapchain creation")?;
        let surface_loader = self
            .surface_loader
            .as_ref()
            .ok_or("No surface loader")?;
        let swapchain_loader = self
            .swapchain_loader
            .as_ref()
            .ok_or("No swapchain loader")?;

        // Wait for device idle before recreating
        unsafe { let _ = self.device.device_wait_idle(); }

        // Destroy old swapchain image views
        for view in self.swapchain_image_views.drain(..) {
            unsafe { self.device.destroy_image_view(view, None); }
        }
        let old_swapchain = self.swapchain.take().unwrap_or(vk::SwapchainKHR::null());

        // Query surface capabilities
        let caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, surface)
                .map_err(|e| format!("Failed to get surface capabilities: {:?}", e))?
        };

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(self.physical_device, surface)
                .map_err(|e| format!("Failed to get surface formats: {:?}", e))?
        };

        // Choose format: prefer B8G8R8A8_UNORM or B8G8R8A8_SRGB
        let surface_format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first())
            .ok_or("No surface formats available")?;

        self.swapchain_format = surface_format.format;

        // Choose extent
        self.swapchain_extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };

        let image_count = (caps.min_image_count + 1).min(
            if caps.max_image_count > 0 {
                caps.max_image_count
            } else {
                u32::MAX
            },
        );

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(self.swapchain_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO) // vsync
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&create_info, None)
                .map_err(|e| format!("Failed to create swapchain: {:?}", e))?
        };

        // Destroy old swapchain after new one is created
        if old_swapchain != vk::SwapchainKHR::null() {
            unsafe { swapchain_loader.destroy_swapchain(old_swapchain, None); }
        }

        // Get swapchain images
        self.swapchain_images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .map_err(|e| format!("Failed to get swapchain images: {:?}", e))?
        };

        // Create image views
        for &image in &self.swapchain_images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(self.swapchain_format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            let view = unsafe {
                self.device
                    .create_image_view(&view_info, None)
                    .map_err(|e| format!("Failed to create swapchain image view: {:?}", e))?
            };
            self.swapchain_image_views.push(view);
        }

        self.swapchain = Some(swapchain);

        info!(
            "Swapchain created: {}x{} format={:?} images={}",
            self.swapchain_extent.width,
            self.swapchain_extent.height,
            self.swapchain_format,
            self.swapchain_images.len()
        );

        Ok(())
    }

    /// Acquire the next swapchain image. Returns (image_index, suboptimal).
    pub fn acquire_next_image(&self, semaphore: vk::Semaphore) -> Result<(u32, bool), vk::Result> {
        let swapchain = self.swapchain.expect("No swapchain");
        let swapchain_loader = self.swapchain_loader.as_ref().expect("No swapchain loader");
        unsafe {
            swapchain_loader.acquire_next_image(swapchain, u64::MAX, semaphore, vk::Fence::null())
        }
    }

    /// Present a swapchain image. Returns true if suboptimal.
    pub fn queue_present(
        &self,
        image_index: u32,
        wait_semaphore: vk::Semaphore,
    ) -> Result<bool, vk::Result> {
        let swapchain = self.swapchain.expect("No swapchain");
        let swapchain_loader = self.swapchain_loader.as_ref().expect("No swapchain loader");
        let swapchains = [swapchain];
        let image_indices = [image_index];
        let wait_semaphores = [wait_semaphore];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        unsafe { swapchain_loader.queue_present(self.graphics_queue, &present_info) }
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

        // Destroy swapchain resources
        for view in self.swapchain_image_views.drain(..) {
            unsafe { self.device.destroy_image_view(view, None); }
        }
        if let (Some(swapchain_loader), Some(swapchain)) =
            (&self.swapchain_loader, self.swapchain.take())
        {
            unsafe { swapchain_loader.destroy_swapchain(swapchain, None); }
        }

        // Destroy surface
        if let (Some(surface_loader), Some(surface)) =
            (&self.surface_loader, self.surface.take())
        {
            unsafe { surface_loader.destroy_surface(surface, None); }
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
