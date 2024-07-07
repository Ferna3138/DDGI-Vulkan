#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "probeUtil.glsl"
#include "wavefront.glsl"

layout(location = 0) flat out int probe_index;
layout(location = 1) out vec4 normal_edge_factor;
layout(location = 2) flat out uint probe_status;
layout(location = 3) out vec2 fragOffset;

layout(set = 1, binding = eGlobals) uniform _GlobalUniforms { GlobalUniforms uni; };

layout(std430, set = 0, binding = eStatus) buffer ProbeStatusSSBO {
  uint probe_statuses[];
};


const vec2 OFFSETS[6] = vec2[](
  vec2(-1.0, -1.0),
  vec2(-1.0, 1.0),
  vec2(1.0, -1.0),
  vec2(1.0, -1.0),
  vec2(-1.0, 1.0),
  vec2(1.0, 1.0)
);



void main() {
    fragOffset = OFFSETS[gl_VertexIndex];

    probe_index = gl_InstanceIndex;
    probe_status = probe_statuses[probe_index];

    const ivec3 probe_grid_indices = probe_index_to_grid_indices(int(probe_index));
    const vec3 probe_position = grid_indices_to_world(probe_grid_indices, probe_index);

    vec3 cameraRightWorld = vec3(uni.viewInverse[0][0], uni.viewInverse[1][0], uni.viewInverse[2][0]);
    vec3 cameraUpWorld = vec3(uni.viewInverse[0][1], uni.viewInverse[1][1], uni.viewInverse[2][1]);

    vec3 positionWorld = probe_position 
                        + fragOffset.x * probe_sphere_scale * cameraRightWorld 
                        + fragOffset.y * probe_sphere_scale * cameraUpWorld;

    //gl_Position = uni.viewProj * vec4(positionWorld, 1.0);
    gl_Position = uni.viewProj * vec4(positionWorld, 1.0);
}
