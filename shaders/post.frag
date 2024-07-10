#version 450

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "probeUtil.glsl"

layout(location = 0) in vec2 outUV;
layout(location = 1) flat in int probe_index;
layout(location = 2) in vec4 normal_edge_factor;
layout(location = 3) flat in uint probe_status;

layout(location = 0) out vec4 fragColor;

layout(set = 2, binding = 0) uniform sampler2D noisyTxt;
layout(set = 2, binding = 1) uniform sampler2D indirectTxt;
layout(set = 2, binding = 2) uniform sampler2D debugTxt;



layout(push_constant) uniform shaderInformation {
	uint useIndirect;
    uint showProbes;
    uint debugTexture;
    uint show_textures;
	float aspectRatio;
} pushc;

void main() {
    vec2  uv    = outUV;
    float gamma = 1.0 / 2.2;
    vec4 finalColor;
    
    if (pushc.useIndirect == 1) {
        vec4 noisyColor = texture(noisyTxt, uv).rgba;
        //vec4 indirectColor = texture(global_textures[6], uv).rgba;
        vec4 indirectColor = texture(global_textures[6], uv).rgba * 3.0;


        vec4 blendedColor;

        if(pushc.showProbes == 0){
            blendedColor = indirectColor * noisyColor;
        }else{
            vec4 debugColor = texture(debugTxt, uv).rgba;
            blendedColor = indirectColor * noisyColor + debugColor;
        }

        // Apply gamma correction
        finalColor = pow(blendedColor, vec4(gamma));
    }else {
        finalColor = pow(texture(noisyTxt, uv).rgba, vec4(gamma));
    }

    fragColor = finalColor;


    if(pushc.show_textures == 1){
        // Determine the overlay position and size
        float overlaySize = 1.0 / 2.0; // Overlay size (1/8th of the screen)
        vec2 overlayStart = vec2(0.0, 0.0); 

        if (uv.x < overlaySize && uv.y < overlaySize) {
            // Map the UV coordinates to the overlay texture space
            vec2 overlayUV = uv / overlaySize;
            vec4 overlayColor = texture(global_textures[pushc.debugTexture], overlayUV).rgba;
            fragColor = overlayColor;
        } else {
            fragColor = finalColor;
        }
    }else{
        fragColor = finalColor;
    }

}
