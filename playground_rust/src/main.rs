//! CLI entry point for the Lux shader playground.
//!
//! Supports rasterization (triangle, fullscreen, PBR) and ray tracing modes.
//! Can run headless (offscreen render to PNG) or interactively (winit window).

mod camera;
mod raster_renderer;
mod rt_renderer;
mod scene;
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
    /// Base name or path to .spv files (without stage extension).
    ///
    /// For example, "shaders/my_shader" will look for:
    ///   - shaders/my_shader.vert.spv + shaders/my_shader.frag.spv (triangle/pbr)
    ///   - shaders/my_shader.frag.spv (fullscreen)
    ///   - shaders/my_shader.rgen.spv + .rchit.spv + .rmiss.spv (rt)
    shader: String,

    /// Rendering mode: triangle, fullscreen, pbr, or rt.
    ///
    /// If not specified, auto-detected from which .spv files exist.
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
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    if let Err(e) = run(args) {
        error!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    let shader_base = &args.shader;

    // Detect or use provided mode
    let mode = match args.mode.as_deref() {
        Some(m) => m.to_string(),
        None => detect_mode(shader_base).to_string(),
    };

    info!("Mode: {}", mode);
    info!("Shader base: {}", shader_base);
    info!("Resolution: {}x{}", args.width, args.height);
    info!("Output: {}", args.output);

    if args.interactive && !args.headless {
        run_interactive(shader_base, &mode, args.width, args.height)?;
    } else {
        run_headless(shader_base, &mode, args.width, args.height, &args.output)?;
    }

    Ok(())
}

/// Auto-detect the rendering mode from which .spv files exist.
fn detect_mode(base: &str) -> &str {
    if Path::new(&format!("{}.rgen.spv", base)).exists() {
        "rt"
    } else if Path::new(&format!("{}.vert.spv", base)).exists()
        && Path::new(&format!("{}.frag.spv", base)).exists()
    {
        "triangle"
    } else if Path::new(&format!("{}.frag.spv", base)).exists() {
        "fullscreen"
    } else {
        panic!(
            "No .spv files found for base: {}. Expected one of:\n  \
             {0}.rgen.spv (ray tracing)\n  \
             {0}.vert.spv + {0}.frag.spv (triangle/pbr)\n  \
             {0}.frag.spv (fullscreen)",
            base
        );
    }
}

/// Run in headless mode: create context, render offscreen, save PNG.
fn run_headless(
    shader_base: &str,
    mode: &str,
    width: u32,
    height: u32,
    output: &str,
) -> Result<(), String> {
    let enable_rt = mode == "rt";

    info!("Creating Vulkan context (RT: {})...", enable_rt);
    let mut ctx = vulkan_context::VulkanContext::new(enable_rt)?;

    let output_path = Path::new(output);

    let result = match mode {
        "triangle" => {
            info!("Rendering triangle...");
            raster_renderer::render_raster(
                &mut ctx,
                raster_renderer::RasterMode::Triangle,
                shader_base,
                width,
                height,
                output_path,
            )
        }
        "fullscreen" => {
            info!("Rendering fullscreen quad...");
            raster_renderer::render_raster(
                &mut ctx,
                raster_renderer::RasterMode::Fullscreen,
                shader_base,
                width,
                height,
                output_path,
            )
        }
        "pbr" => {
            info!("Rendering PBR sphere...");
            raster_renderer::render_raster(
                &mut ctx,
                raster_renderer::RasterMode::Pbr,
                shader_base,
                width,
                height,
                output_path,
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
            rt_renderer::render_rt(&mut ctx, shader_base, width, height, output_path)
        }
        _ => Err(format!("Unknown mode: {}", mode)),
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

/// Run in interactive mode with a winit window.
fn run_interactive(
    shader_base: &str,
    mode: &str,
    width: u32,
    height: u32,
) -> Result<(), String> {
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::keyboard::{KeyCode, PhysicalKey};
    use winit::window::{Window, WindowId};

    struct App {
        window: Option<Window>,
        shader_base: String,
        mode: String,
        width: u32,
        height: u32,
        rendered: bool,
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
                    event_loop.exit();
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                        info!("Escape pressed, closing");
                        event_loop.exit();
                    }
                }
                WindowEvent::RedrawRequested => {
                    if !self.rendered {
                        self.rendered = true;

                        // In interactive mode, we do an offscreen render and display
                        // the result info. A full swapchain-based interactive loop
                        // would require surface creation and presentation.
                        info!(
                            "Interactive mode: rendering {} with mode '{}'",
                            self.shader_base, self.mode
                        );

                        let enable_rt = self.mode == "rt";
                        match vulkan_context::VulkanContext::new(enable_rt) {
                            Ok(mut ctx) => {
                                let output_path = Path::new("interactive_output.png");
                                let result = match self.mode.as_str() {
                                    "triangle" => raster_renderer::render_raster(
                                        &mut ctx,
                                        raster_renderer::RasterMode::Triangle,
                                        &self.shader_base,
                                        self.width,
                                        self.height,
                                        output_path,
                                    ),
                                    "fullscreen" => raster_renderer::render_raster(
                                        &mut ctx,
                                        raster_renderer::RasterMode::Fullscreen,
                                        &self.shader_base,
                                        self.width,
                                        self.height,
                                        output_path,
                                    ),
                                    "pbr" => raster_renderer::render_raster(
                                        &mut ctx,
                                        raster_renderer::RasterMode::Pbr,
                                        &self.shader_base,
                                        self.width,
                                        self.height,
                                        output_path,
                                    ),
                                    "rt" => rt_renderer::render_rt(
                                        &mut ctx,
                                        &self.shader_base,
                                        self.width,
                                        self.height,
                                        output_path,
                                    ),
                                    _ => Err(format!("Unknown mode: {}", self.mode)),
                                };

                                unsafe {
                                    let _ = ctx.device.device_wait_idle();
                                }
                                ctx.destroy();

                                match result {
                                    Ok(()) => info!("Interactive render saved to {:?}", output_path),
                                    Err(e) => error!("Interactive render failed: {}", e),
                                }
                            }
                            Err(e) => {
                                error!("Failed to create Vulkan context: {}", e);
                            }
                        }
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
        shader_base: shader_base.to_string(),
        mode: mode.to_string(),
        width,
        height,
        rendered: false,
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("Event loop error: {}", e))?;

    Ok(())
}
