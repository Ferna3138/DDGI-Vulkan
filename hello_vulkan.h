#pragma once

#include "nvvkhl/appbase_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "shaders/host_device.h"
//#include "ProbeVolume.h"

// #VKRay
#include "nvvk/raytraceKHR_vk.hpp"

//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class HelloVulkan : public nvvkhl::AppBaseVk {

public:
  void setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily) override;
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void loadModel(const std::string& filename, glm::mat4 transform = glm::mat4(1), float scaleFactor = 1);
  void updateDescriptorSet();
  void createUniformBuffer();
  void createObjDescriptionBuffer();
  void createTextureImages(const VkCommandBuffer& cmdBuf, const std::vector<std::string>& textures);
  void updateUniformBuffer(const VkCommandBuffer& cmdBuf);
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();
  void rasterize(const VkCommandBuffer& cmdBuff);

  // The OBJ model
  struct ObjModel
  {
    uint32_t     nbIndices{0};
    uint32_t     nbVertices{0};
    nvvk::Buffer vertexBuffer;    // Device buffer of all 'Vertex'
    nvvk::Buffer indexBuffer;     // Device buffer of the indices forming triangles
    nvvk::Buffer matColorBuffer;  // Device buffer of array of 'Wavefront material'
    nvvk::Buffer matIndexBuffer;  // Device buffer of array of 'Wavefront material'
  };

  struct ObjInstance {
    glm::mat4 transform;    // Matrix of the instance
    glm::mat4 invTransform = glm::inverse(transform);
    uint32_t  objIndex{0};  // Model index reference
  };


  // Information pushed at each draw call
  PushConstantRaster m_pcRaster{
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},  // Identity matrix
      {-4.0f, 1.f, 0.3f},                                 // light position
      0,                                                 // instance Id
      5.f,                                             // light intensity
      0                                                  // light type
  };

  float     randomFloat(float min, float max);
  glm::vec3 randomUnitVector();
  glm::mat4 randomRotationMatrix();
  void      glm_euler_xyz2(glm::vec3 angles, glm::mat4 dest);
  glm::mat4 glms_euler_xyz(glm::vec3 angles);

  // Array of objects and instances in the scene
  std::vector<ObjModel>    m_objModel;   // Model on host
  std::vector<ObjDesc>     m_objDesc;    // Model description for device access
  std::vector<ObjInstance> m_instances;  // Scene model instances

  // Graphic pipeline
  VkPipelineLayout            m_pipelineLayout;
  VkPipeline                  m_graphicsPipeline;
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  VkDescriptorPool            m_descPool;
  VkDescriptorSetLayout       m_descSetLayout;
  VkDescriptorSet             m_descSet;

  nvvk::Buffer m_bGlobals;  // Device-Host of the camera matrices
  nvvk::Buffer m_bObjDesc;  // Device buffer of the OBJ descriptions


  std::vector<nvvk::Texture> m_textures;  // vector of all textures of the scene


  nvvk::ResourceAllocatorDma m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil            m_debug;  // Utility to name objects


  // #Post - Draw the rendered image on a quad using a tonemapper
  void createOffscreenRender();
  void createPostPipeline();
  void createPostDescriptor();
  void updatePostDescriptorSet();
  void drawPost(VkCommandBuffer cmdBuf, bool useIndirect, bool showProbes);

  PushConstantPost pcPost{};

  nvvk::DescriptorSetBindings m_postDescSetLayoutBind;
  VkDescriptorPool            m_postDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout       m_postDescSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet             m_postDescSet{VK_NULL_HANDLE};
  VkPipeline                  m_postPipeline{VK_NULL_HANDLE};
  VkPipelineLayout            m_postPipelineLayout{VK_NULL_HANDLE};
  VkRenderPass                m_offscreenRenderPass{VK_NULL_HANDLE};
  VkFramebuffer               m_offscreenFramebuffer{VK_NULL_HANDLE};
  nvvk::Texture               m_offscreenColor;
  nvvk::Texture               m_offscreenDepth;
  VkFormat                    m_offscreenColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat                    m_offscreenDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};





  //////////////////////////////////////////////////////////////////////////
  // Ray Tracing
  //////////////////////////////////////////////////////////////////////////

  void initRayTracing();
  auto objectToVkGeometryKHR(const ObjModel& model);
  void createBottomLevelAS();
  void createTopLevelAS();
  void createRtDescriptorSet();
  void updateRtDescriptorSet();
  void createRtPipeline();
  void createRtShaderBindingTable();
  void raytrace(const VkCommandBuffer& cmdBuf, const glm::vec4& clearColor);


  VkPhysicalDeviceRayTracingPipelinePropertiesKHR   m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::RaytracingBuilderKHR                        m_rtBuilder;
  nvvk::DescriptorSetBindings                       m_rtDescSetLayoutBind;
  VkDescriptorPool                                  m_rtDescPool;
  VkDescriptorSetLayout                             m_rtDescSetLayout;
  VkDescriptorSet                                   m_rtDescSet;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  VkPipelineLayout                                  m_rtPipelineLayout;
  VkPipeline                                        m_rtPipeline;

  nvvk::Buffer                    m_rtSBTBuffer;
  VkStridedDeviceAddressRegionKHR m_rgenRegion{};
  VkStridedDeviceAddressRegionKHR m_missRegion{};
  VkStridedDeviceAddressRegionKHR m_hitRegion{};
  VkStridedDeviceAddressRegionKHR m_callRegion{};

  // Push constant for ray tracer
  PushConstantRay m_pcRay{};



  //////////////////////////////////////////////////////////////////////////
  // Irradiance Fields
  //////////////////////////////////////////////////////////////////////////

  struct renderSceneDDGI {
    bool      gi_show_probes = false;
    //glm::vec3 gi_probe_grid_position{-56.0f, -2.0f, -35.f};
    //glm::vec3 gi_probe_spacing{6.f, 2.5f, 4.f};

    //Scene scale = 1
    //glm::vec3 gi_probe_grid_position{-16.0f, -2.0f, -11.47f};
    //glm::vec3 gi_probe_spacing{1.7f, 1.12f, 1.27f};   
    

    //Scene scale = 1.5
    glm::vec3 gi_probe_grid_position{-24.0f, -2.0f, -13.804f};
    glm::vec3 gi_probe_spacing{2.5f, 1.120f, 2.25f};


    float     gi_probe_sphere_scale          = 0.1f;
    float     gi_max_probe_offset            = 0.5f;
    float     gi_self_shadow_bias            = 0.3f;
    float     gi_hysteresis                  = 0.95f;
    bool      gi_debug_border                = false;
    bool      gi_debug_border_type           = false;
    bool      gi_debug_border_source         = false;
    uint32_t  gi_total_probes                = 0;
    float     gi_intensity                   = 0.8f;
    bool      gi_use_visibility              = true;
    bool      gi_use_backface_smoothing      = true;
    bool      gi_use_perceptual_encoding     = true;
    bool      gi_use_backface_blending       = false;
    bool      gi_use_probe_offsetting        = false;
    bool      gi_recalculate_offsets         = false;  // When moving grid or changing spaces, recalculate offsets.
    bool      gi_use_probe_status            = false;
    bool      gi_use_half_resolution         = false;
    bool      gi_use_infinite_bounces        = false;
    float     gi_infinite_bounces_multiplier = 0.75f;
    int32_t   gi_per_frame_probes_update     = 1000;
  };



  uint32_t probe_count_x = 20;
  uint32_t probe_count_y = 20;
  uint32_t probe_count_z = 12;

  int32_t per_frame_probe_updates = 0;
  int32_t probe_update_offset     = 0;

  int32_t probe_rays = 128;

  int32_t irradiance_atlas_width;
  int32_t irradiance_atlas_height;
  int32_t irradiance_probe_size = 6;  // Irradiance is a 6x6 quad with 1 pixel borders for bilinear filtering, total 8x8

  int32_t visibility_atlas_width;
  int32_t visibility_atlas_height;
  int32_t visibility_probe_size = 6;

  bool half_resolution_output = false;

  uint32_t get_total_probes() { return probe_count_x * probe_count_y * probe_count_z; }
  uint32_t get_total_rays() { return probe_rays * probe_count_x * probe_count_y * probe_count_z; }


  struct alignas(16) GpuDDGIConstants {
    uint32_t radiance_output_index;
    uint32_t grid_irradiance_output_index;
    uint32_t indirect_output_index;
    uint32_t normal_texture_index;

    uint32_t depth_pyramid_texture_index;
    uint32_t depth_fullscreen_texture_index;
    uint32_t grid_visibility_texture_index;
    uint32_t probe_offset_texture_index;

    float   hysteresis;
    float   infinte_bounces_multiplier;
    int32_t probe_update_offset;
    int32_t probe_update_count;

    glm::vec3 probe_grid_position;
    float     probe_sphere_scale;

    glm::vec3 probe_spacing;
    float     max_probe_offset;  // [0,0.5] max offset for probes

    glm::vec3 reciprocal_probe_spacing;
    float     self_shadow_bias;

    int32_t  probe_counts[3];
    uint32_t debug_options;

    int32_t irradiance_texture_width;
    int32_t irradiance_texture_height;
    int32_t irradiance_side_length;
    int32_t probe_rays;

    int32_t  visibility_texture_width;
    int32_t  visibility_texture_height;
    int32_t  visibility_side_length;
    uint32_t pad1;

    glm::mat4 random_rotation;
    glm::vec2 resolution;
  };  // struct DDGIConstants


  VkRenderPass  m_DDGIRenderPass{VK_NULL_HANDLE};
  VkFramebuffer m_DDGIFramebuffer{VK_NULL_HANDLE};


  void createDDGIPipeline();
  void createDDGIShaderBindingTable();
  void createComputePipeline(const std::string& shaderPath,
                             std::vector<VkDescriptorSetLayout> DDGIDescSetLayouts,
                             VkPipelineLayout& pipelineLayout,
                             VkPipeline& pipeline,
                             const void* pushConstants,
                             uint32_t pushConstantsSize);


  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_DDGIShaderGroups;
  VkPipelineLayout                                  m_DDGIPipelineLayout;
  VkPipeline                                        m_DDGIPipeline;

  nvvk::Buffer                    m_DDGISBTBuffer;
  VkStridedDeviceAddressRegionKHR m_DDGIrgenRegion{};
  VkStridedDeviceAddressRegionKHR m_DDGImissRegion{};
  VkStridedDeviceAddressRegionKHR m_DDGIhitRegion{};
  VkStridedDeviceAddressRegionKHR m_DDGIcallRegion{};

  // Push constant for ray tracer
  PushConstantOffset m_pcProbeOffsets{};
  PushConstantStatus m_pcProbeStatus{};
  PushConstantSample m_pcSampleIrradiance{};

  // Textures Vector
  std::vector<nvvk::Texture> m_DDGIImages;
  std::vector<VkImageView> m_DDGIImageViews;
  
  std::vector<nvvk::Texture> m_globalTextures;
  std::vector<VkImageView> m_globalTexturesView;
  std::vector<VkSampler> m_globalTextureSamplers;


  nvvk::Image              m_indirectImage;
  nvvk::Texture            m_indirectTexture;

  nvvk::Image              m_radianceImage;
  nvvk::Texture            m_radianceTexture;

  nvvk::Image              m_offsetsImage;
  nvvk::Texture            m_offsetsTexture;
  
  nvvk::Image              m_irradianceImage;
  nvvk::Texture            m_irradianceTexture;
  
  nvvk::Image              m_visibilityImage;
  nvvk::Texture            m_visibilityTexture;

  nvvk::Image createStorageImage(const VkCommandBuffer& cmdBuf, VkDevice device, VkPhysicalDevice physicalDevice, uint32_t width, uint32_t height, VkFormat format);

  // Buffers
  nvvk::Buffer m_bDDGIConstants;
  nvvk::Buffer m_bDDGIStatus;

  // Compute Pipelines
  VkPipelineLayout m_probeOffsetsPipelineLayout;
  VkPipeline       m_probeOffsetsPipeline;

  VkPipelineLayout m_probeStatusPipelineLayout;
  VkPipeline       m_probeStatusPipeline;

  VkPipelineLayout m_probeUpdateIrradiancePipelineLayout;
  VkPipeline       m_probeUpdateIrradiancePipeline;
  
  VkPipelineLayout m_probeUpdateVisibilityPipelineLayout;
  VkPipeline       m_probeUpdateVisibilityPipeline;

  VkPipelineLayout m_sampleIrradiancePipelineLayout;
  VkPipeline       m_sampleIrradiancePipeline;

  void createDDGIConstantsBuffer();
  void createDDGIStatusBuffer();

  void updateDDGIConstantsBuffer(const VkCommandBuffer& cmdBuf, renderSceneDDGI& scene);

  //Analogue to the raytrace function
  void DDGIBegin(const VkCommandBuffer& cmdBuf, const glm::vec4& clearColor, renderSceneDDGI& scene);

  void prepareDraws( renderSceneDDGI& scene);
  void addNormalsDepthTextures();



  //////////////////////////////////////////////////////////////////////////
  // G Buffer
  //////////////////////////////////////////////////////////////////////////
  VkPipelineLayout m_gBufferPipelineLayout;
  VkPipeline       m_gBufferPipeline;
  VkRenderPass     m_gBufferRenderPass{VK_NULL_HANDLE};
  VkFramebuffer    m_gBufferFramebuffer{VK_NULL_HANDLE};

  void createGBufferRender();
  void gBufferBegin(const VkCommandBuffer& cmdBuff);
  void createGBufferPipeline();

  nvvk::Texture m_gBufferNormals;
  nvvk::Texture m_gBufferDepth;
  VkFormat      m_gBufferNormalFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat      m_gBufferDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};


  //////////////////////////////////////////////////////////////////////////
  // G Buffer Depth
  //////////////////////////////////////////////////////////////////////////
  VkPipelineLayout m_gBufferDepthPipelineLayout;
  VkPipeline       m_gBufferDepthPipeline;
  VkRenderPass     m_gBufferDepthRenderPass{VK_NULL_HANDLE};
  VkFramebuffer    m_gBufferDepthFramebuffer{VK_NULL_HANDLE};

  void createGBufferDepthRender();
  void gBufferDepthBegin(const VkCommandBuffer& cmdBuff);
  void createGBufferDepthPipeline();

  nvvk::Texture m_depthTexture;
  nvvk::Texture m_gBufferDepth2;
  VkFormat      m_depthTextureFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat      m_gBufferDepth2Format{VK_FORMAT_X8_D24_UNORM_PACK32};



  //////////////////////////////////////////////////////////////////////////
  // Debug Pass
  //////////////////////////////////////////////////////////////////////////
  VkPipelineLayout m_debugPipelineLayout;
  VkPipeline       m_debugPipeline;
  VkRenderPass     m_debugRenderPass{VK_NULL_HANDLE};
  VkFramebuffer    m_debugFramebuffer{VK_NULL_HANDLE};

  void createDebugRender();
  void createDebugPipeline();
  void drawDebug(VkCommandBuffer cmdBuf);

  nvvk::Texture m_debugTexture;
  nvvk::Texture m_debugDepth;
  VkFormat      m_debugTextureFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat      m_debugDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};

};