#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "raycommon.glsl"
#include "wavefront.glsl"
#include "probeUtil.glsl"

layout(location = 0) rayPayloadInEXT ProbeRayPayload prd;

layout(push_constant) uniform _PushConstantRay {
  PushConstantRay pcRay;
};

void main() {
  prd.radiance = vec3( 0.529, 0.807, 0.921 );
  //prd.radiance = vec3( 0.0, 0.0, 0.0 );

  prd.distance = 10000.0f;
}
