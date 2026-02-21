//! CLI entry point for the Lux shader playground.
//!
//! Supports scene/pipeline architecture with auto-detection of render paths.
//! Can run headless (offscreen render to PNG) or interactively (winit window).

mod camera;
pub mod gltf_loader;
mod raster_renderer;
pub mod reflected_pipeline;
mod rt_renderer;
mod scene;
pub mod scene_manager;
mod screenshot;
mod spv_loader;
mod vulkan_context;

use clap::Parser;
use log::{error, info};
use std::path::Path;

/// Lux shader playground — render SPIR-V shaders using Vulkan.
#[derive(Parser)]
#[command(name = "lux-playground", about = "Lux shader playground")]
struct Args {
    /// Scene source: sphere, fullscreen, triangle, or path to .glb/.gltf
    #[arg(long, required_unless_present = "shader")]
    scene: Option<String>,

    /// Pipeline shader base path (auto-resolved from scene if not given)
    #[arg(long)]
    pipeline: Option<String>,

    /// [DEPRECATED: use --scene/--pipeline] Base path to .spv shader files
    #[arg(index = 1)]
    shader: Option<String>,

    /// [DEPRECATED: use --scene instead]
    #[arg(long, value_parser = ["triangle", "fullscreen", "pbr", "rt"])]
    mode: Option<String>,

    /// Output image width in pixels.
    #[arg(long, default_value = "512")]
    width: u32,

    /// Output image height in pixels.
    #[arg(long, default_value = "512")]
    height: u32,

    /// Output PNG file path.
    #[arg(long, default_value = "output.png")]
    output: String,

    /// Open an interactive preview window instead of headless rendering.
    #[arg(long)]
    interactive: bool,

    /// Force headless (offscreen) rendering. This is the default.
    #[arg(long)]
    headless: bool,

    /// IBL environment name (default: auto-detect pisa/neutral from assets/ibl/).
    #[arg(long)]
    ibl: Option<String>,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    if let Err(e) = run(args) {
        error!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Resolve a default pipeline shader base path from the scene source.
fn resolve_default_pipeline(scene: &str) -> Result<String, String> {
    if scene.ends_with(".glb") || scene.ends_with(".gltf") {
        // Try shadercache/gltf_pbr first (where compiled .spv files live),
        // fall back to examples/gltf_pbr
        if Path::new("shadercache/gltf_pbr.vert.spv").exists() {
            Ok("shadercache/gltf_pbr".to_string())
        } else {
            Ok("examples/gltf_pbr".to_string())
        }
    } else if scene == "fullscreen" {
        Err("--pipeline required for fullscreen scenes".to_string())
    } else if scene == "triangle" {
        Ok("shadercache/hello_triangle".to_string())
    } else {
        Ok("shadercache/pbr_basic".to_string())
    }
}

/// Auto-detect the render path from which .spv files exist for a pipeline base.
fn detect_render_path(pipeline_base: &str) -> &'static str {
    if Path::new(&format!("{}.rgen.spv", pipeline_base)).exists() {
        "rt"
    } else if Path::new(&format!("{}.vert.spv", pipeline_base)).exists()
        && Path::new(&format!("{}.frag.spv", pipeline_base)).exists()
    {
        "raster"
    } else if Path::new(&format!("{}.frag.spv", pipeline_base)).exists() {
        "fullscreen"
    } else {
        panic!(
            "No shader files found for pipeline base: {}",
            pipeline_base
        );
    }
}

fn run(args: Args) -> Result<(), String> {
    // Handle backward compatibility: old --mode maps to new --scene
    let scene_source = if let Some(scene) = &args.scene {
        scene.clone()
    } else if let Some(mode) = &args.mode {
        match mode.as_str() {
            "triangle" => "triangle".to_string(),
            "fullscreen" => "fullscreen".to_string(),
            "pbr" => "sphere".to_string(),
            "rt" => "sphere".to_string(),
            _ => "sphere".to_string(),
        }
    } else {
        "sphere".to_string() // default
    };

    let pipeline_base = if let Some(p) = &args.pipeline {
        p.clone()
    } else if let Some(shader) = &args.shader {
        shader.clone()
    } else {
        resolve_default_pipeline(&scene_source).unwrap_or_else(|e| panic!("{}", e))
    };

    let render_path = detect_render_path(&pipeline_base);

    info!("Scene: {}", scene_source);
    info!("Pipeline: {}", pipeline_base);
    info!("Render path: {}", render_path);
    info!("Resolution: {}x{}", args.width, args.height);
    info!("Output: {}", args.output);

    let ibl_name = args.ibl.as_deref().unwrap_or("");

    if args.interactive && !args.headless {
        // Use larger window for interactive mode if user didn't specify
        let (iw, ih) = if args.width == 512 && args.height == 512 {
            (1024, 768)
        } else {
            (args.width, args.height)
        };
        run_interactive(&pipeline_base, &scene_source, render_path, iw, ih, ibl_name)?;
    } else {
        run_headless(&pipeline_base, &scene_source, render_path, args.width, args.height, &args.output, ibl_name)?;
    }

    Ok(())
}

/// Run in headless mode: create context, render offscreen, save PNG.
fn run_headless(
    pipeline_base: &str,
    scene_source: &str,
    render_path: &str,
    width: u32,
    height: u32,
    output: &str,
    ibl_name: &str,
) -> Result<(), String> {
    let enable_rt = render_path == "rt";

    info!("Creating Vulkan context (RT: {})...", enable_rt);
    let mut ctx = vulkan_context::VulkanContext::new(enable_rt)?;

    let output_path = Path::new(output);

    let result = match render_path {
        "raster" => {
            info!("Rendering raster scene '{}' with pipeline '{}'...", scene_source, pipeline_base);
            raster_renderer::render_raster(
                &mut ctx,
                pipeline_base,
                scene_source,
                width,
                height,
                output_path,
                ibl_name,
            )
        }
        "fullscreen" => {
            info!("Rendering fullscreen pipeline '{}'...", pipeline_base);
            raster_renderer::render_fullscreen(
                &mut ctx,
                pipeline_base,
                width,
                height,
                output_path,
                ibl_name,
            )
        }
        "rt" => {
            if !ctx.supports_rt() {
                return Err(
                    "Ray tracing mode requested but GPU does not support \
                     VK_KHR_ray_tracing_pipeline. Please use a different mode or a GPU \
                     with ray tracing support."
                        .to_string(),
                );
            }
            info!("Rendering ray traced image...");
            rt_renderer::render_rt(&mut ctx, pipeline_base, scene_source, width, height, output_path, ibl_name)
        }
        _ => Err(format!("Unknown render path: {}", render_path)),
    };

    // Wait for GPU idle before cleanup
    unsafe {
        let _ = ctx.device.device_wait_idle();
    }

    ctx.destroy();

    match &result {
        Ok(()) => info!("Render complete: {}", output),
        Err(e) => error!("Render failed: {}", e),
    }

    result
}

/// Orbit camera state for interactive viewing.
struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: glam::Vec3,
    up: glam::Vec3,
    fov_y: f32,
    near_plane: f32,
    far_plane: f32,
    dragging: bool,
    last_x: f64,
    last_y: f64,
}

impl OrbitCamera {
    fn new() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.15,
            distance: 3.0,
            target: glam::Vec3::ZERO,
            up: glam::Vec3::new(0.0, 1.0, 0.0),
            fov_y: 45.0f32.to_radians(),
            near_plane: 0.1,
            far_plane: 100.0,
            dragging: false,
            last_x: 0.0,
            last_y: 0.0,
        }
    }

    fn init_from_auto_camera(
        &mut self,
        eye: glam::Vec3,
        target: glam::Vec3,
        up: glam::Vec3,
        far: f32,
    ) {
        self.target = target;
        self.up = up;
        self.far_plane = far;

        let dir = eye - target;
        self.distance = dir.length();
        if self.distance > 1e-6 {
            let d = dir / self.distance;
            self.yaw = d.x.atan2(d.z);
            self.pitch = d.y.asin();
        }
    }

    fn eye(&self) -> glam::Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + glam::Vec3::new(x, y, z)
    }
}

/// Run in interactive mode with a winit window and orbit camera.
fn run_interactive(
    pipeline_base: &str,
    scene_source: &str,
    render_path: &str,
    width: u32,
    height: u32,
    ibl_name: &str,
) -> Result<(), String> {
    use ash::vk;
    use scene_manager::Renderer;
    use winit::application::ApplicationHandler;
    use winit::event::{ElementState, MouseButton, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::keyboard::{KeyCode, PhysicalKey};
    use winit::window::{Window, WindowId};

    let use_rt = render_path == "rt";

    struct App {
        window: Option<Window>,
        pipeline_base: String,
        scene_source: String,
        width: u32,
        height: u32,
        use_rt: bool,
        ibl_name: String,
        // Vulkan state (initialized after window creation)
        ctx: Option<vulkan_context::VulkanContext>,
        renderer: Option<Box<dyn Renderer>>,
        orbit: OrbitCamera,
        // Frame sync
        image_available_sem: vk::Semaphore,
        render_finished_sem: vk::Semaphore,
        in_flight_fence: vk::Fence,
        initialized: bool,
    }

    impl App {
        fn init_vulkan(&mut self) {
            let window = match self.window.as_ref() {
                Some(w) => w,
                None => return,
            };

            info!("Initializing Vulkan with window surface...");
            let mut ctx = match vulkan_context::VulkanContext::new_with_window(window, self.use_rt) {
                Ok(c) => c,
                Err(e) => {
                    error!("Failed to create Vulkan context: {}", e);
                    return;
                }
            };

            // Create sync objects
            let sem_info = vk::SemaphoreCreateInfo::default();
            let fence_info = vk::FenceCreateInfo::default()
                .flags(vk::FenceCreateFlags::SIGNALED);

            let image_available = unsafe {
                ctx.device.create_semaphore(&sem_info, None).unwrap()
            };
            let render_finished = unsafe {
                ctx.device.create_semaphore(&sem_info, None).unwrap()
            };
            let fence = unsafe {
                ctx.device.create_fence(&fence_info, None).unwrap()
            };

            // Create the renderer (unified via Renderer trait)
            let renderer_result: Result<Box<dyn Renderer>, String> = if self.use_rt {
                info!("Initializing RT renderer...");
                rt_renderer::RTRenderer::new(
                    &mut ctx,
                    &self.pipeline_base,
                    &self.scene_source,
                    self.width,
                    self.height,
                    &self.ibl_name,
                ).map(|r| Box::new(r) as Box<dyn Renderer>)
            } else {
                info!("Initializing persistent renderer...");
                raster_renderer::PersistentRenderer::init(
                    &mut ctx,
                    &self.pipeline_base,
                    &self.scene_source,
                    self.width,
                    self.height,
                    &self.ibl_name,
                ).map(|r| Box::new(r) as Box<dyn Renderer>)
            };

            let renderer = match renderer_result {
                Ok(r) => r,
                Err(e) => {
                    error!("Failed to init renderer: {}", e);
                    unsafe {
                        ctx.device.destroy_semaphore(image_available, None);
                        ctx.device.destroy_semaphore(render_finished, None);
                        ctx.device.destroy_fence(fence, None);
                    }
                    ctx.destroy();
                    return;
                }
            };

            // Init orbit camera from auto-camera
            if renderer.has_scene_bounds() {
                self.orbit.init_from_auto_camera(
                    renderer.auto_eye(),
                    renderer.auto_target(),
                    renderer.auto_up(),
                    renderer.auto_far(),
                );
            }

            self.image_available_sem = image_available;
            self.render_finished_sem = render_finished;
            self.in_flight_fence = fence;
            self.renderer = Some(renderer);
            self.ctx = Some(ctx);
            self.initialized = true;

            info!("Interactive mode ready. Drag mouse to orbit, scroll to zoom, ESC to exit.");
        }

        fn render_frame(&mut self) {
            let ctx = match self.ctx.as_ref() {
                Some(c) => c,
                None => return,
            };

            // Wait for previous frame
            unsafe {
                let _ = ctx.device.wait_for_fences(&[self.in_flight_fence], true, u64::MAX);
                let _ = ctx.device.reset_fences(&[self.in_flight_fence]);
            }

            // Acquire swapchain image
            let (image_index, _suboptimal) = match ctx.acquire_next_image(self.image_available_sem) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    info!("Swapchain out of date, skipping frame");
                    return;
                }
                Err(e) => {
                    error!("Failed to acquire swapchain image: {:?}", e);
                    return;
                }
            };

            let extent = ctx.swapchain_extent;

            let renderer = match self.renderer.as_mut() {
                Some(r) => r,
                None => return,
            };

            // Update camera
            let aspect = extent.width as f32 / extent.height as f32;
            renderer.update_camera(
                self.orbit.eye(),
                self.orbit.target,
                self.orbit.up,
                self.orbit.fov_y,
                aspect,
                self.orbit.near_plane,
                self.orbit.far_plane,
            );

            // Render (returns a command buffer that is still recording)
            let cmd = match renderer.render(ctx) {
                Ok(c) => c,
                Err(e) => {
                    error!("Render failed: {}", e);
                    return;
                }
            };

            // Blit to swapchain image
            let swapchain_image = ctx.swapchain_images[image_index as usize];
            renderer.blit_to_swapchain(&ctx.device, cmd, swapchain_image, extent);

            // End and submit
            unsafe {
                let _ = ctx.device.end_command_buffer(cmd);
            }

            let wait_semaphores = [self.image_available_sem];
            let signal_semaphores = [self.render_finished_sem];
            let wait_stages = [renderer.wait_stage()];
            let cmd_bufs = [cmd];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmd_bufs)
                .signal_semaphores(&signal_semaphores);

            unsafe {
                let _ = ctx.device.queue_submit(
                    ctx.graphics_queue,
                    &[submit_info],
                    self.in_flight_fence,
                );
            }

            // Present
            match ctx.queue_present(image_index, self.render_finished_sem) {
                Ok(_) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    info!("Swapchain out of date during present");
                }
                Err(e) => {
                    error!("Present failed: {:?}", e);
                }
            }

            // Free the command buffer after GPU is done
            unsafe {
                let _ = ctx.device.device_wait_idle();
                ctx.device.free_command_buffers(ctx.command_pool, &cmd_bufs);
            }
        }

        fn cleanup_vulkan(&mut self) {
            if let Some(ref ctx) = self.ctx {
                unsafe { let _ = ctx.device.device_wait_idle(); }
            }

            if let Some(ref mut renderer) = self.renderer {
                if let Some(ref mut ctx) = self.ctx {
                    renderer.destroy(ctx);
                }
            }

            if let Some(ref ctx) = self.ctx {
                unsafe {
                    if self.image_available_sem != vk::Semaphore::null() {
                        ctx.device.destroy_semaphore(self.image_available_sem, None);
                    }
                    if self.render_finished_sem != vk::Semaphore::null() {
                        ctx.device.destroy_semaphore(self.render_finished_sem, None);
                    }
                    if self.in_flight_fence != vk::Fence::null() {
                        ctx.device.destroy_fence(self.in_flight_fence, None);
                    }
                }
            }

            if let Some(mut ctx) = self.ctx.take() {
                ctx.destroy();
            }
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_none() {
                let window_attrs = Window::default_attributes()
                    .with_title("Lux Playground")
                    .with_inner_size(winit::dpi::LogicalSize::new(self.width, self.height))
                    .with_resizable(false);

                match event_loop.create_window(window_attrs) {
                    Ok(window) => {
                        info!("Window created: {}x{}", self.width, self.height);
                        self.window = Some(window);
                        self.init_vulkan();
                    }
                    Err(e) => {
                        error!("Failed to create window: {}", e);
                        event_loop.exit();
                    }
                }
            }
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => {
                    info!("Window close requested");
                    self.cleanup_vulkan();
                    event_loop.exit();
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                        info!("Escape pressed, closing");
                        self.cleanup_vulkan();
                        event_loop.exit();
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        self.orbit.dragging = state == ElementState::Pressed;
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if self.orbit.dragging {
                        let dx = position.x - self.orbit.last_x;
                        let dy = position.y - self.orbit.last_y;
                        self.orbit.yaw += dx as f32 * 0.005;
                        self.orbit.pitch += dy as f32 * 0.005;
                        self.orbit.pitch = self.orbit.pitch.clamp(-1.5, 1.5);
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                    }
                    self.orbit.last_x = position.x;
                    self.orbit.last_y = position.y;
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                    };
                    self.orbit.distance *= 1.0 - scroll * 0.1;
                    self.orbit.distance = self.orbit.distance.clamp(0.01, 1000.0);
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                WindowEvent::RedrawRequested => {
                    if self.initialized {
                        self.render_frame();
                    }
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                _ => {}
            }
        }
    }

    let event_loop = EventLoop::new().map_err(|e| format!("Failed to create event loop: {}", e))?;

    let mut app = App {
        window: None,
        pipeline_base: pipeline_base.to_string(),
        scene_source: scene_source.to_string(),
        width,
        height,
        use_rt,
        ibl_name: ibl_name.to_string(),
        ctx: None,
        renderer: None,
        orbit: OrbitCamera::new(),
        image_available_sem: vk::Semaphore::null(),
        render_finished_sem: vk::Semaphore::null(),
        in_flight_fence: vk::Fence::null(),
        initialized: false,
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("Event loop error: {}", e))?;

    Ok(())
}
