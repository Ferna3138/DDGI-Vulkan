#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "wavefront.glsl"
#include "probeUtil.glsl"


layout (location = 0) flat in int probe_index;
layout (location = 1) in vec4 normal_edge_factor;
layout(location = 2) flat in uint probe_status;
layout(location = 3) in vec2 fragOffset;

layout(location = 0) out vec4 o_color;


void main() {
        float dis = sqrt(dot(fragOffset, fragOffset));
        if(dis >= 1.0){
            discard;
        }
        if (probe_status == PROBE_STATUS_OFF) {
            o_color = vec4(1.0, 0.0, 0.0, 1.0);
        }
        else if (probe_status == PROBE_STATUS_UNINITIALIZED) {
            o_color = vec4(0.0, 0.0, 1.0, 1.0);
        }
        else if (probe_status == PROBE_STATUS_ACTIVE) {
            o_color = vec4(0.0, 1.0, 0.0, 1.0);
        }
        else {
            o_color = vec4(1.0, 1.0, 1.0, 1.0);
        }
}
