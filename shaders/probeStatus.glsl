#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "host_device.h"
#include "probeUtil.glsl"


layout(push_constant) uniform _PushConstantOffset {
  PushConstantOffset pcStatus;
};


layout(std430, set = 0, binding = eStatus) buffer ProbeStatusSSBO {
  uint probe_status[];
};


layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

void main() {
  uint  first_frame = pcStatus.first_frame;
  ivec3 coords = ivec3(gl_GlobalInvocationID.xyz);

  int offset = 0;

  int probe_index = coords.x;

  int   closest_backface_index    = -1;
  float closest_backface_distance = 100000000.f;

  int   closest_frontface_index    = -1;
  float closest_frontface_distance = 100000000.f;

  int   farthest_frontface_index    = -1;
  float farthest_frontface_distance = 0;

  int  backfaces_count = 0;
  uint flag            = first_frame == 1 ? PROBE_STATUS_UNINITIALIZED : probe_status[probe_index];

  // Worst case, view and normal contribute in the same direction, so need 2x self-shadow bias.
  vec3 outerBounds = normalize(probe_spacing) * (length(probe_spacing) + (2.0f * self_shadow_bias));

  for(int ray_index = 0; ray_index < probe_rays; ++ray_index) {
    ivec2 ray_tex_coord = ivec2(ray_index, probe_index);

    // Distance is negative if we hit a backface
    float d_front = texelFetch(global_textures[nonuniformEXT( radiance_output_index )], ray_tex_coord, 0).w;
    float d_back  = -d_front;

    // Backface test backface -> position.w < 0.0f
    if(d_back > 0.0f) {
      backfaces_count += 1;
      if(d_back < closest_backface_distance) {
        // This distance is negative on a backface hit
        closest_backface_distance = d_back;
        // Recompute ray direction
        closest_backface_index = ray_index;
      }
    }

    if(d_front > 0.0f) {
      // Check all frontfaces to see if any are wihtin shading range
      vec3 frontFaceDirection = d_front * normalize(mat3(random_rotation) * spherical_fibonacci(ray_index, probe_rays));
      if(all(lessThan(abs(frontFaceDirection), outerBounds))) {
        // There is a static surface being shaded by this probe. Make it "just vigilant"
        flag = PROBE_STATUS_ACTIVE;
      }
      if(d_front < closest_frontface_distance) {
        closest_frontface_distance = d_front;
        closest_frontface_index    = ray_index;
      }
      else if(d_front > farthest_frontface_distance) {
        farthest_frontface_distance = d_front;
        farthest_frontface_index    = ray_index;
      }
    }
  }

  // If there's a close backface AND you see more than 25% backfaces, assume you're inside something.
  if(closest_backface_index != -1 && (float(backfaces_count) / probe_rays) > 0.25f) {
    // At this point, we were just in a wall, so set probe to "Off".
    flag = PROBE_STATUS_OFF;
  }
  else if(closest_frontface_index == -1) {
    // Probe sees only backfaces and sky, so set probe to "Off".
    flag = PROBE_STATUS_OFF;
  }
  else if(closest_frontface_distance < 0.05f) {
    // We hit no backfaces and a close frontface (within 2 cm). Set to "Newly Vigilant".
    flag = PROBE_STATUS_ACTIVE;
  }

  // Write probe status
  probe_status[probe_index] = flag;
}