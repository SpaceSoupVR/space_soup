use ash::vk::Handle;
use log::info;
use openxr as xr;

use crate::xr::{VkContext, XrContext};

pub struct Headset {
    pub session: xr::Session<xr::Vulkan>,
    pub frame_waiter: xr::FrameWaiter,
    pub frame_stream: xr::FrameStream<xr::Vulkan>,
    pub stage: xr::Space,
    pub running: bool,
}

impl Headset {
    pub fn new(xr: &XrContext, vk: &VkContext) -> Result<Self, Box<dyn std::error::Error>> {
        let (session, frame_waiter, frame_stream) = unsafe {
            xr.instance.create_session::<xr::Vulkan>(
                xr.system,
                &xr::vulkan::SessionCreateInfo {
                    instance: vk.instance.handle().as_raw() as _,
                    physical_device: vk.physical_device.as_raw() as _,
                    device: vk.device.handle().as_raw() as _,
                    queue_family_index: vk.queue_family_index,
                    queue_index: 0,
                },
            )?
        };

        let stage =
            session.create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY)?;

        info!("Headset session created");

        Ok(Self {
            session,
            frame_waiter,
            frame_stream,
            stage,
            running: false,
        })
    }

    pub fn handle_state_change(
        &mut self,
        state: xr::SessionState,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        info!("Session state -> {:?}", state);
        match state {
            xr::SessionState::READY => {
                self.session
                    .begin(xr::ViewConfigurationType::PRIMARY_STEREO)?;
                self.running = true;
            }
            xr::SessionState::STOPPING => {
                self.session.end()?;
                self.running = false;
            }
            xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                return Ok(true);
            }
            _ => {}
        }
        Ok(false)
    }
}
