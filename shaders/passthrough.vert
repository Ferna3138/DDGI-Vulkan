#version 450

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) out vec2 outUV;

layout(location = 1) flat out int probe_index;
layout(location = 2) out vec4 normal_edge_factor;
layout(location = 3) flat out uint probe_status;


out gl_PerVertex {
  vec4 gl_Position;
};


void main() {
	outUV       = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(outUV * 2.0f - 1.0f, 1.0f, 1.0f);
}
