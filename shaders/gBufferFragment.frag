#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "wavefront.glsl"



layout(push_constant) uniform _PushConstantRaster {
  PushConstantRaster pcRaster;
};

// clang-format off
// Incoming 
layout(location = 1) in vec3 i_worldPos;
layout(location = 2) in vec3 i_worldNrm;
layout(location = 3) in vec3 i_viewDir;
layout(location = 4) in vec2 i_texCoord;

// Outgoing
layout(location = 0) out vec4 o_color;



layout(buffer_reference, scalar) buffer Vertices {Vertex v[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer Indices {uint i[]; }; // Triangle indices
layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[]; }; // Array of all materials on an object
layout(buffer_reference, scalar) buffer MatIndices {int i[]; }; // Material ID for each triangle

layout(binding = eGlobals) uniform _GlobalUniforms { GlobalUniforms uni; };

layout(binding = eObjDescs, scalar) buffer ObjDesc_ { ObjDesc i[]; } objDesc;
layout(binding = eTextures) uniform sampler2D[] textureSamplers;
// clang-format on


float sign_not_zero(in float k) {
  return (k >= 0.0) ? 1.0 : -1.0;
}


vec2 sign_not_zero2(in vec2 v) {
  return vec2(sign_not_zero(v.x), sign_not_zero(v.y));
}


vec2 octahedral_encode(vec3 n) {
  // Project the sphere onto the octahedron, and then onto the xy plane
  vec2 p = n.xy * (1.0f / (abs(n.x) + abs(n.y) + abs(n.z)));
  // Reflect the folds of the lower hemisphere over the diagonals
  return (n.z < 0.0f) ? ((1.0 - abs(p.yx)) * sign_not_zero2(p)) : p;
}


void main() {
o_color = vec4(octahedral_encode(i_worldNrm), 0.0, 0.0);
 // o_color = vec4(i_worldNrm, 1.0);
}
