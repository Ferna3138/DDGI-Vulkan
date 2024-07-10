#ifndef COMMON_HOST_DEVICE
#define COMMON_HOST_DEVICE


#ifdef __cplusplus
#include <glm/glm.hpp>
// GLSL Type
using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;
using mat4 = glm::mat4;
using uint = unsigned int;
#endif

// clang-format off
#ifdef __cplusplus // Descriptor binding helper for C++ and GLSL
 #define START_BINDING(a) enum a {
 #define END_BINDING() }
#else
 #define START_BINDING(a)  const uint
 #define END_BINDING() 
#endif

//#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require


START_BINDING(SceneBindings)
  eGlobals  = 0,  // Global uniform containing camera matrices
  eObjDescs = 1,  // Access to the object descriptions
  eTextures = 2   // Access to textures
END_BINDING();

START_BINDING(RtxBindings)
  eTlas     = 0,		// Top-level acceleration structure
  eOutImage = 1,		// Ray tracer output image
  eConstants = 2,		// Constants Buffer
  eStatus = 3,			// Status Buffer
  eStorageImages = 4,	// Storage Images
  eGlobalTextures = 5,	// Global Textures
  eIrradianceImage = 6,	// Irradiance Image for Probe Update
  eVisibilityImage = 7  // Visibility Image for Probe Update
END_BINDING();

 // clang-format on


// Information of a obj model when referenced in a shader
struct ObjDesc {
  int      txtOffset;             // Texture index offset in the array of textures
  uint64_t vertexAddress;         // Address of the Vertex buffer
  uint64_t indexAddress;          // Address of the index buffer
  uint64_t materialAddress;       // Address of the material buffer
  uint64_t materialIndexAddress;  // Address of the triangle material index buffer
};

// Uniform buffer set at each frame
struct GlobalUniforms {
  mat4 viewProj;     // Camera view * projection
  mat4 viewInverse;  // Camera inverse view matrix
  mat4 projInverse;  // Camera inverse projection matrix
  mat4 view;
  mat4 projection;
  vec3 position;
};

// Push constant structure for the raster
struct PushConstantRaster {
  mat4  modelMatrix;  // matrix of the instance
  vec3  lightPosition;
  uint  objIndex;
  float lightIntensity;
  int   lightType;
};

struct PushConstantOffset {
  uint first_frame;
};

struct PushConstantStatus {
  uint first_frame;
};

struct PushConstantSample {
  uint output_resolution_half;
};

struct PushConstantPost {
  uint indirect_enabled;
  uint debug_enabled;
  uint  debug_texture;
  uint  show_textures;
  float aspectRatio;
};

// Push constant structure for the ray tracer
struct PushConstantRay {
  vec4  clearColor;
  vec3  lightPosition;
  float lightIntensity;
  int   lightType;
};

struct Vertex { // See ObjLoader, copy of VertexObj, could be compressed for device
  vec3 pos;
  vec3 nrm;
  vec3 color;
  vec2 texCoord;
};

struct WaveFrontMaterial { // See ObjLoader, copy of MaterialObj, could be compressed for device
  vec3  ambient;
  vec3  diffuse;
  vec3  specular;
  vec3  transmittance;
  vec3  emission;
  float shininess;
  float ior;       // index of refraction
  float dissolve;  // 1 == opaque; 0 == fully transparent
  int   illum;     // illumination model (see http://www.fileformat.info/format/material/)
  int   textureId;
};


#endif
