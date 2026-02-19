//! SPIR-V file loading and shader module creation.

use ash::vk;
use std::fs;
use std::path::Path;

/// The SPIR-V magic number (little-endian).
const SPIRV_MAGIC: u32 = 0x07230203;

/// Read a SPIR-V binary file and return its contents as a Vec<u32>.
///
/// Validates the magic number and alignment.
pub fn load_spirv(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("Failed to read {:?}: {}", path, e))?;

    if bytes.len() < 4 {
        return Err(format!("{:?}: file too small to be valid SPIR-V", path));
    }

    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{:?}: file size {} is not a multiple of 4",
            path,
            bytes.len()
        ));
    }

    // Reinterpret as u32 slice
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    if words[0] != SPIRV_MAGIC {
        return Err(format!(
            "{:?}: bad SPIR-V magic 0x{:08X} (expected 0x{:08X})",
            path, words[0], SPIRV_MAGIC
        ));
    }

    Ok(words)
}

/// Create a VkShaderModule from SPIR-V words.
pub fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule, String> {
    let create_info = vk::ShaderModuleCreateInfo::default().code(code);

    unsafe {
        device
            .create_shader_module(&create_info, None)
            .map_err(|e| format!("Failed to create shader module: {:?}", e))
    }
}

/// Detect the shader stage from a filename's extension pattern.
///
/// Recognized patterns:
/// - `*.vert.spv` -> VERTEX
/// - `*.frag.spv` -> FRAGMENT
/// - `*.rgen.spv` -> RAYGEN_KHR
/// - `*.rchit.spv` -> CLOSEST_HIT_KHR
/// - `*.rmiss.spv` -> MISS_KHR
/// - `*.comp.spv` -> COMPUTE
pub fn detect_stage(filename: &str) -> Result<vk::ShaderStageFlags, String> {
    let lower = filename.to_lowercase();
    if lower.ends_with(".vert.spv") {
        Ok(vk::ShaderStageFlags::VERTEX)
    } else if lower.ends_with(".frag.spv") {
        Ok(vk::ShaderStageFlags::FRAGMENT)
    } else if lower.ends_with(".rgen.spv") {
        Ok(vk::ShaderStageFlags::RAYGEN_KHR)
    } else if lower.ends_with(".rchit.spv") {
        Ok(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
    } else if lower.ends_with(".rmiss.spv") {
        Ok(vk::ShaderStageFlags::MISS_KHR)
    } else if lower.ends_with(".comp.spv") {
        Ok(vk::ShaderStageFlags::COMPUTE)
    } else {
        Err(format!(
            "Cannot detect shader stage from filename: {}",
            filename
        ))
    }
}
