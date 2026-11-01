//! Windowed game: winit event loop, input mapping, surface presentation.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::Vec2;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::game::Controls;
use crate::gpu::Gpu;
use crate::run::RunConfig;
use crate::session::{Session, SessionConfig};

/// Mass granted per press of the developer key (H, with --dev).
const DEV_GRANT: f64 = 100.0;

#[derive(Clone, Debug)]
pub struct AppOptions {
    pub agents: u32,
    pub capacity: u32,
    pub seed: u64,
    /// Enables developer keys.
    pub dev: bool,
    /// Include the mop hazard.
    pub hazard: bool,
}

pub fn run(opts: AppOptions) -> Result<()> {
    let event_loop = EventLoop::new().context("creating event loop")?;
    let mut app = App {
        opts,
        running: None,
        error: None,
    };
    event_loop.run_app(&mut app)?;
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Raw input state, turned into [`Controls`] each frame.
#[derive(Default)]
struct Input {
    up: bool,
    down: bool,
    left: bool,
    right: bool,
    gather: bool,
    spread: bool,
    mouse_px: Option<Vec2>,
    mouse_down: bool,
}

impl Input {
    fn key(&mut self, code: KeyCode, pressed: bool) {
        match code {
            KeyCode::KeyW | KeyCode::ArrowUp => self.up = pressed,
            KeyCode::KeyS | KeyCode::ArrowDown => self.down = pressed,
            KeyCode::KeyA | KeyCode::ArrowLeft => self.left = pressed,
            KeyCode::KeyD | KeyCode::ArrowRight => self.right = pressed,
            KeyCode::Space => self.gather = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.spread = pressed,
            _ => {}
        }
    }

    fn controls(&self, session: &Session, viewport: Vec2) -> Controls {
        let axis = |neg: bool, pos: bool| (pos as i32 - neg as i32) as f32;
        let pointer = match (self.mouse_down, self.mouse_px) {
            (true, Some(px)) => Some(session.run.game.camera.screen_to_world(px, viewport)),
            _ => None,
        };
        Controls {
            move_dir: Vec2::new(axis(self.left, self.right), axis(self.down, self.up)),
            gather: axis(self.spread, self.gather),
            pointer,
        }
    }
}

struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    session: Session,
    input: Input,
    dev: bool,
    last_frame: Instant,
    fps_window_start: Instant,
    fps_frames: u32,
}

impl Running {
    fn new(event_loop: &ActiveEventLoop, opts: &AppOptions) -> Result<Self> {
        let attrs = Window::default_attributes()
            .with_title("Myriad")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(event_loop.create_window(attrs)?);
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        desc.display = Some(Box::new(event_loop.owned_display_handle()));
        let instance = wgpu::Instance::new(desc);
        let surface = instance.create_surface(window.clone())?;
        let gpu = pollster::block_on(Gpu::from_instance(instance, Some(&surface)))?;
        log::info!("adapter: {}", gpu.adapter_summary());

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1))
            .context("surface is not supported by the adapter")?;
        // Prefer an sRGB format; the shaders output linear colour.
        let caps = surface.get_capabilities(&gpu.adapter);
        if let Some(srgb) = caps.formats.iter().copied().find(|f| f.is_srgb()) {
            config.format = srgb;
        }
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&gpu.device, &config);

        let cfg = SessionConfig {
            run: RunConfig {
                initial_critters: opts.agents,
                capacity: opts.capacity.max(opts.agents),
                hazard_delay: opts.hazard.then_some(30.0),
            },
            seed: opts.seed,
        };
        let aspect = config.width as f32 / config.height as f32;
        let session = Session::new(&gpu, config.format, aspect, &cfg);
        let now = Instant::now();
        Ok(Self {
            window,
            surface,
            config,
            gpu,
            session,
            input: Input::default(),
            dev: opts.dev,
            last_frame: now,
            fps_window_start: now,
            fps_frames: 0,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        self.session.run.game.camera.aspect = width as f32 / height as f32;
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        let viewport = Vec2::new(self.config.width as f32, self.config.height as f32);
        let controls = self.input.controls(&self.session, viewport);
        self.session.advance(&self.gpu, &controls, dt);

        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Validation => {
                log::error!("surface validation error");
                return;
            }
        };
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.session
            .draw(&self.gpu, &view, self.config.width, self.config.height);
        self.window.pre_present_notify();
        self.gpu.queue.present(texture);

        self.fps_frames += 1;
        let elapsed = (now - self.fps_window_start).as_secs_f32();
        if elapsed >= 0.5 {
            let fps = self.fps_frames as f32 / elapsed;
            self.window.set_title(&format!(
                "Myriad | mass {} | {fps:.0} fps | WASD/mouse steer, Space gather, Shift spread, R restart",
                self.session.run.colony.mass()
            ));
            self.fps_window_start = now;
            self.fps_frames = 0;
        }
    }
}

struct App {
    opts: AppOptions,
    running: Option<Running>,
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.running.is_some() {
            return;
        }
        match Running::new(event_loop, &self.opts) {
            Ok(r) => self.running = Some(r),
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(r) = self.running.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => r.resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;
                    if code == KeyCode::Escape && pressed {
                        event_loop.exit();
                    } else if code == KeyCode::KeyR && pressed && !event.repeat {
                        r.session.restart(&r.gpu);
                    } else if code == KeyCode::KeyH && pressed && !event.repeat && r.dev {
                        r.session.grant(DEV_GRANT);
                    } else {
                        r.input.key(code, pressed);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                r.input.mouse_px = Some(Vec2::new(position.x as f32, position.y as f32));
            }
            WindowEvent::CursorLeft { .. } => r.input.mouse_px = None,
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => r.input.mouse_down = state == ElementState::Pressed,
            WindowEvent::RedrawRequested => r.frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(r) = &self.running {
            r.window.request_redraw();
        }
    }
}
