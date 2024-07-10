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


// Assumes that v is a unit vector. The result is an octahedral vector on the [-1, +1] square.
vec2 oct_encode(in vec3 v) {
  float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
  vec2  result = v.xy * (1.0 / l1norm);
  if(v.z < 0.0) {
    result = (1.0 - abs(result.yx)) * sign_not_zero2(result.xy);
  }
  return result;
}


void main() {
	o_color = vec4(oct_encode(i_worldNrm), 0.0, 1.0);
 // o_color = vec4(i_worldNrm, 1.0);
}
