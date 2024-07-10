#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

//#include "host_device.h"
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

  // Invoke this shader for each probe
  int       probe_index  = coords.x;
  const int total_probes = probe_counts.x * probe_counts.y * probe_counts.z;

  // Early out if index is not valid
  if(probe_index >= total_probes) {
    return;
  }

  int   closest_backface_index    = -1;
  float closest_backface_distance = 100000000.f;

  int   closest_frontface_index    = -1;
  float closest_frontface_distance = 100000000.f;

  int   farthest_frontface_index    = -1;
  float farthest_frontface_distance = 0;

  int backfaces_count = 0;
  // For each ray cache front/backfaces index and distances.
  for(int ray_index = 0; ray_index < probe_rays; ++ray_index) {

    ivec2 ray_tex_coord = ivec2(ray_index, probe_index);

    float ray_distance = texelFetch(global_textures[nonuniformEXT(radiance_output_index)], ray_tex_coord, 0).w;
    // Negative distance is stored for backface hits in the Ray-Tracing Hit shader.
    if(ray_distance <= 0.0f) {
      ++backfaces_count;
      // Distance is a positive value, thus negate ray_distance as it is negative already if
      // we are inside this branch.
      if((-ray_distance) < closest_backface_distance) {
        closest_backface_distance = ray_distance;
        closest_backface_index    = ray_index;
      }
    }
    else {
      // Cache either closest or farther distance and indices for this ray.
      if(ray_distance < closest_frontface_distance) {
        closest_frontface_distance = ray_distance;
        closest_frontface_index    = ray_index;
      }
      else if(ray_distance > farthest_frontface_distance) {
        farthest_frontface_distance = ray_distance;
        farthest_frontface_index    = ray_index;
      }
    }
  }

  vec3 full_offset       = vec3(10000.f);
  vec3 cell_offset_limit = max_probe_offset * probe_spacing;

  vec4 current_offset = vec4(0);
  // Read previous offset after the first frame.
  if( first_frame == 0 ) {
    const int probe_counts_xy                   = probe_counts.x * probe_counts.y;
    ivec2     probe_offset_sampling_coordinates = ivec2(probe_index % probe_counts_xy, probe_index / probe_counts_xy);
    current_offset.rgb = texelFetch(global_textures[nonuniformEXT(probe_offset_texture_index)], probe_offset_sampling_coordinates, 0).rgb;
  }

  // Check if a fourth of the rays was a backface, we can assume the probe is inside a geometry.
  const bool inside_geometry = (float(backfaces_count) / probe_rays) > 0.25f;

  if(inside_geometry && (closest_backface_index != -1)) {
    // Calculate the backface direction
    // Distance is always positive
    const vec3 closest_backface_direction = closest_backface_distance * normalize(mat3(random_rotation) * spherical_fibonacci(closest_backface_index, probe_rays));

    // Find the maximum offset inside the cell.
    const vec3 positive_offset = (current_offset.xyz + cell_offset_limit) / closest_backface_direction;
    const vec3 negative_offset = (current_offset.xyz - cell_offset_limit) / closest_backface_direction;
    const vec3 maximum_offset = vec3(max(positive_offset.x, negative_offset.x), max(positive_offset.y, negative_offset.y),
                                     max(positive_offset.z, negative_offset.z));

    // Get the smallest of the offsets to scale the direction
    const float direction_scale_factor = min(min(maximum_offset.x, maximum_offset.y), maximum_offset.z) - 0.001f;

    // Move the offset in the opposite direction of the backface one.
    full_offset = current_offset.xyz - closest_backface_direction * direction_scale_factor;
  }
  else if(closest_frontface_distance < 0.05f) { // In this case we have a very small hit distance.

    // Ensure that we never move through the farthest frontface
    // Move minimum distance to ensure not moving on a future iteration.
    const vec3 farthest_direction = min(0.2f, farthest_frontface_distance) * normalize(mat3(random_rotation) * spherical_fibonacci(farthest_frontface_index, probe_rays));
    const vec3 closest_direction = normalize(mat3(random_rotation) * spherical_fibonacci(closest_frontface_index, probe_rays));
    
    /* The farthest frontface may also be the closest if the probe can only
    see one surface. If this is the case, don't move the probe */
    if(dot(farthest_direction, closest_direction) < 0.5f) {
      full_offset = current_offset.xyz + farthest_direction;
    }
  }

  // Move the probe only if the newly calculated offset is within the cell.
  if(all(lessThan(abs(full_offset), cell_offset_limit))) {
    current_offset.xyz = full_offset;
  }

  // Write probe offset
  const int probe_counts_xy = probe_counts.x * probe_counts.y;

  const int probe_texel_x = (probe_index % probe_counts_xy);
  const int probe_texel_y = probe_index / probe_counts_xy;

  imageStore(global_images_2d[probe_offset_texture_index], ivec2(probe_texel_x, probe_texel_y), current_offset);
}
