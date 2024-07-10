#include <sstream>

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "obj_loader.h"
#include "stb_image.h"

#include "hello_vulkan.h"
#include "nvh/alignment.hpp"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/buffers_vk.hpp"

//#include "ProbeVolume.h"

extern std::vector<std::string> defaultSearchPaths;


//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily) {
  AppBaseVk::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(instance, device, physicalDevice);
  m_debug.setup(m_device);
  m_offscreenDepthFormat = nvvk::findDepthFormat(physicalDevice);
  m_gBufferDepthFormat   = nvvk::findDepthFormat(physicalDevice);
  m_gBufferDepth2Format   = nvvk::findDepthFormat(physicalDevice);
  m_debugDepthFormat     = nvvk::findDepthFormat(physicalDevice);
}



//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer(const VkCommandBuffer& cmdBuf) {
  // Prepare new UBO contents on host.
  const float    aspectRatio = m_size.width / static_cast<float>(m_size.height);
  GlobalUniforms hostUBO     = {};
  const auto&    view        = CameraManip.getMatrix();
  glm::vec3      pos         = CameraManip.getEye();
  glm::mat4      proj        = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspectRatio, 0.1f, 1000.0f);
  proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).

  hostUBO.viewProj    = proj * view;
  hostUBO.viewInverse = glm::inverse(view);
  hostUBO.projInverse = glm::inverse(proj);
  hostUBO.view        = view;
  hostUBO.projection  = proj;
  hostUBO.position    = pos;

  // UBO on the device, and what stages access it.
  VkBuffer deviceUBO      = m_bGlobals.buffer;
  auto     uboUsageStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier beforeBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  beforeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  beforeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  beforeBarrier.buffer        = deviceUBO;
  beforeBarrier.offset        = 0;
  beforeBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, uboUsageStages, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &beforeBarrier, 0, nullptr);


  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_bGlobals.buffer, 0, sizeof(GlobalUniforms), &hostUBO);

  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier afterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  afterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  afterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  afterBarrier.buffer        = deviceUBO;
  afterBarrier.offset        = 0;
  afterBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, uboUsageStages, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &afterBarrier, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout() {
  auto nbTxt = static_cast<uint32_t>(m_textures.size());

  // Camera matrices
  m_descSetLayoutBind.addBinding(SceneBindings::eGlobals, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);
  // Obj descriptions
  m_descSetLayoutBind.addBinding(SceneBindings::eObjDescs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
  // Textures
  m_descSetLayoutBind.addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nbTxt,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);


  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet() {
  std::vector<VkWriteDescriptorSet> writes;

  // Camera matrices and scene description
  VkDescriptorBufferInfo dbiUnif{m_bGlobals.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eGlobals, &dbiUnif));

  VkDescriptorBufferInfo dbiSceneDesc{m_bObjDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eObjDescs, &dbiSceneDesc));

  // All texture samplers
  std::vector<VkDescriptorImageInfo> diit;
  for(auto& texture : m_textures) {
    diit.emplace_back(texture.descriptor);
  }
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, SceneBindings::eTextures, diit.data()));

  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}



//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline() {
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster)};

  // Creating the Pipeline Layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_descSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_pipelineLayout);


  // Creating the Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_offscreenRenderPass);
  gpb.depthStencilState.depthTestEnable = true;
  gpb.addShader(nvh::loadFile("spv/vert_shader.vert.spv", true, paths, true), VK_SHADER_STAGE_VERTEX_BIT);
  gpb.addShader(nvh::loadFile("spv/frag_shader.frag.spv", true, paths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  gpb.addBindingDescription({0, sizeof(VertexObj)});
  
  gpb.addAttributeDescriptions({
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, pos))},
      {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, nrm))},
      {2, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, color))},
      {3, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, texCoord))},
  });

  std::array<VkPipelineColorBlendAttachmentState, 2> colorBlendAttachments = {};
  for(auto& blendAttachment : colorBlendAttachments) {
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachment.blendEnable = VK_FALSE;
  }

  VkPipelineColorBlendStateCreateInfo colorBlending = {};
  colorBlending.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable                       = VK_FALSE;
  colorBlending.attachmentCount                     = static_cast<uint32_t>(colorBlendAttachments.size());
  colorBlending.pAttachments                        = colorBlendAttachments.data();
  gpb.colorBlendState                               = colorBlending;


  m_graphicsPipeline = gpb.createPipeline();
  m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}


//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadModel(const std::string& filename, glm::mat4 transform, float scaleFactor) {
  LOGI("Loading File:  %s \n", filename.c_str());
  ObjLoader loader;
  loader.loadModel(filename);

  // Converting from Srgb to linear
  for(auto& m : loader.m_materials) {
    m.ambient  = glm::pow(m.ambient, glm::vec3(2.2f));
    m.diffuse  = glm::pow(m.diffuse, glm::vec3(2.2f));
    m.specular = glm::pow(m.specular, glm::vec3(2.2f));
  }

  ObjModel model;
  model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool  cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer    cmdBuf          = cmdBufGet.createCommandBuffer();
  VkBufferUsageFlags flag            = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBufferUsageFlags rayTracingFlags =  // used also for building acceleration structures
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  model.vertexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags);
  model.indexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags);
  model.matColorBuffer = m_alloc.createBuffer(cmdBuf, loader.m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  model.matIndexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_matIndx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  // Creates all textures found and find the offset for this model
  auto txtOffset = static_cast<uint32_t>(m_textures.size());
  createTextureImages(cmdBuf, loader.m_textures);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  std::string objNb = std::to_string(m_objModel.size());
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb)));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb)));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb)));
  m_debug.setObjectName(model.matIndexBuffer.buffer, (std::string("matIdx_" + objNb)));

  // Keeping transformation matrix of the instance
  ObjInstance instance;

  glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(scaleFactor));
  transform             = transform * scaleMatrix;

  
  instance.transform = transform;
  instance.objIndex  = static_cast<uint32_t>(m_objModel.size());
  m_instances.push_back(instance);

  // Creating information for device access
  ObjDesc desc;
  desc.txtOffset            = txtOffset;
  desc.vertexAddress        = nvvk::getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
  desc.indexAddress         = nvvk::getBufferDeviceAddress(m_device, model.indexBuffer.buffer);
  desc.materialAddress      = nvvk::getBufferDeviceAddress(m_device, model.matColorBuffer.buffer);
  desc.materialIndexAddress = nvvk::getBufferDeviceAddress(m_device, model.matIndexBuffer.buffer);

  // Keeping the obj host model and device description
  m_objModel.emplace_back(model);
  m_objDesc.emplace_back(desc);
}

//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer() {
  m_bGlobals = m_alloc.createBuffer(sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_bGlobals.buffer, "Globals");
}

//--------------------------------------------------------------------------------------------------
// Create a storage buffer containing the description of the scene elements
// - Which geometry is used by which instance
// - Transformation
// - Offset for texture
//
void HelloVulkan::createObjDescriptionBuffer() {
  nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);

  auto cmdBuf = cmdGen.createCommandBuffer();
  m_bObjDesc  = m_alloc.createBuffer(cmdBuf, m_objDesc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  cmdGen.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
  m_debug.setObjectName(m_bObjDesc.buffer, "ObjDescs");
}


//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void HelloVulkan::createTextureImages(const VkCommandBuffer& cmdBuf, const std::vector<std::string>& textures) {
  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.maxLod     = FLT_MAX;

  VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;

  // If no textures are present, create a dummy one to accommodate the pipeline layout
  if(textures.empty() && m_textures.empty()) {
    nvvk::Texture texture;

    std::array<uint8_t, 4> color{255u, 255u, 255u, 255u};
    VkDeviceSize           bufferSize      = sizeof(color);
    auto                   imgSize         = VkExtent2D{1, 1};
    auto                   imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format);

    // Creating the dummy texture
    nvvk::Image           image  = m_alloc.createImage(cmdBuf, bufferSize, color.data(), imageCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    texture                      = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

    // The image format must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    nvvk::cmdBarrierImageLayout(cmdBuf, texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_textures.push_back(texture);
  }
  else {
    // Uploading all images
    for(const auto& texture : textures) {
      std::stringstream o;
      int               texWidth, texHeight, texChannels;
      o << "media/textures/" << texture;
      std::string txtFile = nvh::findFile(o.str(), defaultSearchPaths, true);

      stbi_uc* stbi_pixels = stbi_load(txtFile.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

      std::array<stbi_uc, 4> color{255u, 0u, 255u, 255u};

      stbi_uc* pixels = stbi_pixels;
      // Handle failure
      if(!stbi_pixels) {
        texWidth = texHeight = 1;
        texChannels          = 4;
        pixels               = reinterpret_cast<stbi_uc*>(color.data());
      }

      VkDeviceSize bufferSize      = static_cast<uint64_t>(texWidth) * texHeight * sizeof(uint8_t) * 4;
      auto         imgSize         = VkExtent2D{(uint32_t)texWidth, (uint32_t)texHeight};
      auto         imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);

      {
        nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, pixels, imageCreateInfo);
        nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
        VkImageViewCreateInfo ivInfo  = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
        nvvk::Texture         texture = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

        m_textures.push_back(texture);
      }

      stbi_image_free(stbi_pixels);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources() {
  vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

  m_alloc.destroy(m_bGlobals);
  m_alloc.destroy(m_bObjDesc);
  m_alloc.destroy(m_bDDGIConstants);
  m_alloc.destroy(m_bDDGIStatus);


  for(auto& m : m_objModel) {
    m_alloc.destroy(m.vertexBuffer);
    m_alloc.destroy(m.indexBuffer);
    m_alloc.destroy(m.matColorBuffer);
    m_alloc.destroy(m.matIndexBuffer);
  }
  
  for(auto& t : m_textures) {
    m_alloc.destroy(t);
  }
  
  m_alloc.destroy(m_radianceTexture);
  m_alloc.destroy(m_irradianceTexture);
  m_alloc.destroy(m_offsetsTexture);
  m_alloc.destroy(m_visibilityTexture);
  m_alloc.destroy(m_indirectTexture);
  
  
  for(auto& sample : m_globalTextureSamplers) {
    vkDestroySampler(m_device, sample, nullptr);
  }


  // G Buffer
  m_alloc.destroy(m_gBufferNormals);
  m_alloc.destroy(m_gBufferDepth);
  vkDestroyPipeline(m_device, m_gBufferPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_gBufferPipelineLayout, nullptr);
  vkDestroyRenderPass(m_device, m_gBufferRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_gBufferFramebuffer, nullptr);

  // G Buffer depth
  m_alloc.destroy(m_depthTexture);
  m_alloc.destroy(m_gBufferDepth2);
  vkDestroyPipeline(m_device, m_gBufferDepthPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_gBufferDepthPipelineLayout, nullptr);
  vkDestroyRenderPass(m_device, m_gBufferDepthRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_gBufferDepthFramebuffer, nullptr);

  //#Post
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  vkDestroyPipeline(m_device, m_postPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);
  vkDestroyRenderPass(m_device, m_offscreenRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);

  // DDGI
  vkDestroyPipeline(m_device, m_DDGIPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_DDGIPipelineLayout, nullptr);
  // Compute
  vkDestroyPipeline(m_device, m_probeOffsetsPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_probeOffsetsPipelineLayout, nullptr);
  vkDestroyPipeline(m_device, m_probeStatusPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_probeStatusPipelineLayout, nullptr);
  vkDestroyPipeline(m_device, m_probeUpdateIrradiancePipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_probeUpdateIrradiancePipelineLayout, nullptr);
  vkDestroyPipeline(m_device, m_probeUpdateVisibilityPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_probeUpdateVisibilityPipelineLayout, nullptr);
  vkDestroyPipeline(m_device, m_sampleIrradiancePipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_sampleIrradiancePipelineLayout, nullptr);

  vkDestroyRenderPass(m_device, m_DDGIRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_DDGIFramebuffer, nullptr);


  // Debug
  m_alloc.destroy(m_debugTexture);
  m_alloc.destroy(m_debugDepth);
  vkDestroyPipeline(m_device, m_debugPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_debugPipelineLayout, nullptr);
  vkDestroyRenderPass(m_device, m_debugRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_debugFramebuffer, nullptr);


  // #VKRay
  m_rtBuilder.destroy();
  vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_rtDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_rtDescSetLayout, nullptr);
  m_alloc.destroy(m_rtSBTBuffer);


  
  m_alloc.deinit();
}


//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const VkCommandBuffer& cmdBuf) {
  VkDeviceSize offset{0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  // Dynamic Viewport
  setViewport(cmdBuf);

  // Drawing all triangles
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);


  for(const HelloVulkan::ObjInstance& inst : m_instances) {
    auto& model            = m_objModel[inst.objIndex];
    m_pcRaster.objIndex    = inst.objIndex;  // Telling which object is drawn
    m_pcRaster.modelMatrix = inst.transform;

    vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster), &m_pcRaster);
    vkCmdBindVertexBuffers(cmdBuf, 0, 1, &model.vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmdBuf, model.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmdBuf, model.nbIndices, 1, 0, 0, 0);
  }
  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/) {
  createOffscreenRender();
  createGBufferRender();
  createGBufferDepthRender();
  //createDebugRender();
  updatePostDescriptorSet();
  updateRtDescriptorSet();
  
}


//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////


//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//

void HelloVulkan::createOffscreenRender() {
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);

  // Creating the color image
  {
    auto        colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
                                                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                                                  | VK_IMAGE_USAGE_STORAGE_BIT);
    nvvk::Image image           = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    VkSamplerCreateInfo   sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_offscreenColor                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Creating the depth buffer
  {
    auto depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);
    VkImageViewCreateInfo depthStencilView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthStencilView.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format           = m_offscreenDepthFormat;
    depthStencilView.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    depthStencilView.image            = image.image;

    m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
  }


  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a render pass for the offscreen
  if(!m_offscreenRenderPass) {
    m_offscreenRenderPass = nvvk::createRenderPass (m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat, 1, true, true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  }

  // Creating the framebuffer for offscreen
  std::vector<VkImageView> attachments = {m_offscreenColor.descriptor.imageView,
                                          m_offscreenDepth.descriptor.imageView};

  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);
  VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  info.renderPass      = m_offscreenRenderPass;
  info.attachmentCount = static_cast<uint32_t>(attachments.size());
  info.pAttachments    = attachments.data();
  info.width           = m_size.width;
  info.height          = m_size.height;
  info.layers          = 1;
  vkCreateFramebuffer(m_device, &info, nullptr, &m_offscreenFramebuffer);
}



//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void HelloVulkan::createPostPipeline() {
  // Push constants in the fragment shader
  //VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float)};
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantPost)};

  std::vector<VkDescriptorSetLayout> postDescSetLayouts2 = {m_rtDescSetLayout, m_descSetLayout, m_postDescSetLayout};
  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = postDescSetLayouts2.size();
  createInfo.pSetLayouts            = postDescSetLayouts2.data();
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_postPipelineLayout);

  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_indirectTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Pipeline: completely generic, no vertices
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout, m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("spv/passthrough.vert.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_VERTEX_BIT);
  pipelineGenerator.addShader(nvh::loadFile("spv/post.frag.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;
  m_postPipeline                                = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createPostDescriptor() {
  m_postDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_postDescSetLayoutBind.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_postDescSetLayoutBind.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);

  //VkDescriptorImageInfo imageInfo{{}, m_indirectTexture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  //VkDescriptorImageInfo debugImageInfo{{}, m_debugColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};


  // Add the indirect texture
  m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
  m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);

  //std::vector<VkWriteDescriptorSet> writes;
  //writes.emplace_back(m_postDescSetLayoutBind.makeWrite(m_postDescSet, 1, &imageInfo));
}


//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet() {
  VkWriteDescriptorSet writeDescriptorSets[3];
  writeDescriptorSets[0] = m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor);
  // Descriptor info for indirect texture
  writeDescriptorSets[1] = m_postDescSetLayoutBind.makeWrite(m_postDescSet, 1, &m_indirectTexture.descriptor);

  writeDescriptorSets[2] = m_postDescSetLayoutBind.makeWrite(m_postDescSet, 2, &m_debugTexture.descriptor);
  //writeDescriptorSets[2] = m_postDescSetLayoutBind.makeWrite(m_postDescSet, 2, &m_debugColor.descriptor);
  // Update both descriptors at once
  vkUpdateDescriptorSets(m_device, 3, writeDescriptorSets, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void HelloVulkan::drawPost(VkCommandBuffer cmdBuf, bool useIndirect, bool showProbes) {
  m_debug.beginLabel(cmdBuf, "Post");

  pcPost.indirect_enabled = useIndirect == true ? 1 : 0;
  pcPost.debug_enabled    = showProbes == true ? 1 : 0;
  pcPost.debug_texture    = m_currentTextureDebug;
  pcPost.show_textures    = m_showDebugTextures == true ? 1 : 0;
  setViewport(cmdBuf);

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);

  pcPost.aspectRatio = aspectRatio;

  std::vector<VkDescriptorSet> descSets{m_rtDescSet, m_descSet, m_postDescSet};

  vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantPost), &pcPost);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, descSets.size(), descSets.data(), 0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}



//////////////////////////////////////////////////////////////////////////
// G Buffer - Normals
//////////////////////////////////////////////////////////////////////////
void HelloVulkan::createGBufferRender() {
  m_alloc.destroy(m_gBufferNormals);
  m_alloc.destroy(m_gBufferDepth);

  // Creating the normal image
  {
    auto        colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_gBufferNormalFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    nvvk::Image image           = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    VkSamplerCreateInfo   sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_gBufferNormals                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_gBufferNormals.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }


  // Creating the depth buffer
  {
    auto depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_gBufferDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);

    VkImageViewCreateInfo depthStencilView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthStencilView.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format           = m_gBufferDepthFormat;
    depthStencilView.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    depthStencilView.image            = image.image;

    m_gBufferDepth = m_alloc.createTexture(image, depthStencilView);
  }

  // Setting the image layout for all attachments
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_gBufferNormals.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_gBufferDepth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a render pass for the offscreen
  if(!m_gBufferRenderPass) {
    m_gBufferRenderPass = nvvk::createRenderPass(m_device, {m_gBufferNormalFormat}, m_gBufferDepthFormat, 1, true, true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  }

  // Creating the framebuffer for offscreen
  std::vector<VkImageView> attachments = {m_gBufferNormals.descriptor.imageView, m_gBufferDepth.descriptor.imageView};


  vkDestroyFramebuffer(m_device, m_gBufferFramebuffer, nullptr);
  VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  info.renderPass      = m_gBufferRenderPass;
  info.attachmentCount = static_cast<uint32_t>(attachments.size());
  info.pAttachments    = attachments.data();
  info.width           = m_size.width;
  info.height          = m_size.height;
  info.layers          = 1;
  vkCreateFramebuffer(m_device, &info, nullptr, &m_gBufferFramebuffer);
}

void HelloVulkan::gBufferBegin(const VkCommandBuffer& cmdBuf) {
    VkDeviceSize offset{0};

    m_debug.beginLabel(cmdBuf, "GBuffer");

      // Dynamic Viewport
    setViewport(cmdBuf);

      
      // Drawing all triangles
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gBufferPipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gBufferPipelineLayout, 0, 1, &m_descSet, 0, nullptr);

    for(const HelloVulkan::ObjInstance& inst : m_instances) {
        auto& model            = m_objModel[inst.objIndex];
        m_pcRaster.objIndex    = inst.objIndex;  // Telling which object is drawn
        m_pcRaster.modelMatrix = inst.transform;

        vkCmdPushConstants(cmdBuf, m_gBufferPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster), &m_pcRaster);
        vkCmdBindVertexBuffers(cmdBuf, 0, 1, &model.vertexBuffer.buffer, &offset);
        vkCmdBindIndexBuffer(cmdBuf, model.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmdBuf, model.nbIndices, 1, 0, 0, 0);
    }
      
    m_debug.endLabel(cmdBuf);
}

void HelloVulkan::createGBufferPipeline() {
      VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                sizeof(PushConstantRaster)};

      // Creating the Pipeline Layout
      VkPipelineLayoutCreateInfo gBufferCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      gBufferCreateInfo.setLayoutCount         = 1;
      gBufferCreateInfo.pSetLayouts            = &m_descSetLayout;
      gBufferCreateInfo.pushConstantRangeCount = 1;
      gBufferCreateInfo.pPushConstantRanges    = &pushConstantRanges;
      vkCreatePipelineLayout(m_device, &gBufferCreateInfo, nullptr, &m_gBufferPipelineLayout);


      // Creating the Pipeline
      std::vector<std::string>                paths = defaultSearchPaths;
      nvvk::GraphicsPipelineGeneratorCombined gBufferGpb(m_device, m_gBufferPipelineLayout, m_gBufferRenderPass);
      gBufferGpb.depthStencilState.depthTestEnable = true;
      gBufferGpb.addShader(nvh::loadFile("spv/gBufferVertex.vert.spv", true, paths, true), VK_SHADER_STAGE_VERTEX_BIT);
      gBufferGpb.addShader(nvh::loadFile("spv/gBufferFragment.frag.spv", true, paths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
      gBufferGpb.addBindingDescription({0, sizeof(VertexObj)});

      gBufferGpb.addAttributeDescriptions({
          {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, pos))},
          {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, nrm))},
          {2, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, color))},
          {3, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, texCoord))},
      });

      // Define color blend attachment states for the two color attachments
      
      std::array<VkPipelineColorBlendAttachmentState, 2> colorBlendAttachments = {};
      for(auto& blendAttachment : colorBlendAttachments) {
        blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blendAttachment.blendEnable = VK_FALSE;
      }
      
      // Create color blend state create info with the correct number of attachments
      VkPipelineColorBlendStateCreateInfo gBufferColorBlending = {};
      gBufferColorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
      gBufferColorBlending.logicOpEnable   = VK_FALSE;
      gBufferColorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size());
      gBufferColorBlending.pAttachments    = colorBlendAttachments.data();

      gBufferGpb.colorBlendState = gBufferColorBlending;
       
      m_gBufferPipeline = gBufferGpb.createPipeline();

      m_debug.setObjectName(m_gBufferPipeline, "GBuffer");
}




//////////////////////////////////////////////////////////////////////////
// G Buffer - Depth
//////////////////////////////////////////////////////////////////////////
void HelloVulkan::createGBufferDepthRender() {
      m_alloc.destroy(m_depthTexture);
      m_alloc.destroy(m_gBufferDepth2);

      // Creating the normal image
      {
        auto        colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_depthTextureFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        nvvk::Image image           = m_alloc.createImage(colorCreateInfo);

        VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
        VkSamplerCreateInfo   sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        m_depthTexture                         = m_alloc.createTexture(image, ivInfo, sampler);
        m_depthTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      }


      // Creating the depth buffer
      {
        auto        depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_gBufferDepth2Format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        nvvk::Image image = m_alloc.createImage(depthCreateInfo);

        VkImageViewCreateInfo depthStencilView2{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        depthStencilView2.viewType        = VK_IMAGE_VIEW_TYPE_2D;
        depthStencilView2.format          = m_gBufferDepth2Format;
        depthStencilView2.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        depthStencilView2.image            = image.image;

        m_gBufferDepth2 = m_alloc.createTexture(image, depthStencilView2);
      }

      // Setting the image layout for all attachments
      {
        nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
        auto              cmdBuf = genCmdBuf.createCommandBuffer();
        nvvk::cmdBarrierImageLayout(cmdBuf, m_depthTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        nvvk::cmdBarrierImageLayout(cmdBuf, m_gBufferDepth2.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        genCmdBuf.submitAndWait(cmdBuf);
      }

      // Creating a render pass for the offscreen
      if(!m_gBufferDepthRenderPass) {
        m_gBufferDepthRenderPass = nvvk::createRenderPass(m_device, {m_depthTextureFormat}, m_gBufferDepth2Format, 1,
                                                          true, true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
      }

      // Creating the framebuffer for offscreen
      std::vector<VkImageView> depthAttachments = {m_depthTexture.descriptor.imageView, m_gBufferDepth2.descriptor.imageView};


      vkDestroyFramebuffer(m_device, m_gBufferDepthFramebuffer, nullptr);
      VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      info.renderPass      = m_gBufferDepthRenderPass;
      info.attachmentCount = static_cast<uint32_t>(depthAttachments.size());
      info.pAttachments    = depthAttachments.data();
      info.width           = m_size.width;
      info.height          = m_size.height;
      info.layers          = 1;
      vkCreateFramebuffer(m_device, &info, nullptr, &m_gBufferDepthFramebuffer);
}

void HelloVulkan::gBufferDepthBegin(const VkCommandBuffer& cmdBuf) {
      VkDeviceSize offset{0};

      m_debug.beginLabel(cmdBuf, "Depth");

      // Dynamic Viewport
      setViewport(cmdBuf);

      // Drawing all triangles
      vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gBufferDepthPipeline);
      vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gBufferDepthPipelineLayout, 0, 1, &m_descSet, 0, nullptr);

      for(const HelloVulkan::ObjInstance& inst : m_instances) {
        auto& model            = m_objModel[inst.objIndex];
        m_pcRaster.objIndex    = inst.objIndex;  // Telling which object is drawn
        m_pcRaster.modelMatrix = inst.transform;

        vkCmdPushConstants(cmdBuf, m_gBufferDepthPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster), &m_pcRaster);
        vkCmdBindVertexBuffers(cmdBuf, 0, 1, &model.vertexBuffer.buffer, &offset);
        vkCmdBindIndexBuffer(cmdBuf, model.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmdBuf, model.nbIndices, 1, 0, 0, 0);
      }

      m_debug.endLabel(cmdBuf);
}

void HelloVulkan::createGBufferDepthPipeline() {
      VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster)};

      // Creating the Pipeline Layout
      VkPipelineLayoutCreateInfo gBufferDepthCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      gBufferDepthCreateInfo.setLayoutCount         = 1;
      gBufferDepthCreateInfo.pSetLayouts            = &m_descSetLayout;
      gBufferDepthCreateInfo.pushConstantRangeCount = 1;
      gBufferDepthCreateInfo.pPushConstantRanges    = &pushConstantRanges;
      vkCreatePipelineLayout(m_device, &gBufferDepthCreateInfo, nullptr, &m_gBufferDepthPipelineLayout);


      // Creating the Pipeline
      std::vector<std::string>                paths = defaultSearchPaths;
      nvvk::GraphicsPipelineGeneratorCombined depthGpb(m_device, m_gBufferDepthPipelineLayout, m_gBufferDepthRenderPass);
      depthGpb.depthStencilState.depthTestEnable = true;
      depthGpb.addShader(nvh::loadFile("spv/gBufferDepthVertex.vert.spv", true, paths, true), VK_SHADER_STAGE_VERTEX_BIT);
      depthGpb.addShader(nvh::loadFile("spv/gBufferDepthFragment.frag.spv", true, paths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
      depthGpb.addBindingDescription({0, sizeof(VertexObj)});

      depthGpb.addAttributeDescriptions({
          {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, pos))},
          {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, nrm))},
          {2, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, color))},
          {3, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, texCoord))},
      });

      // Define color blend attachment states for the two color attachments

      std::array<VkPipelineColorBlendAttachmentState, 1> depthColorBlendAttachments = {};
      depthColorBlendAttachments[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      depthColorBlendAttachments[0].blendEnable = VK_FALSE;

      // Create color blend state create info with the correct number of attachments
      VkPipelineColorBlendStateCreateInfo gBufferDepthColorBlending = {};
      gBufferDepthColorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
      gBufferDepthColorBlending.logicOpEnable   = VK_FALSE;
      gBufferDepthColorBlending.attachmentCount = static_cast<uint32_t>(depthColorBlendAttachments.size());
      gBufferDepthColorBlending.pAttachments    = depthColorBlendAttachments.data();

      depthGpb.colorBlendState = gBufferDepthColorBlending;

      m_gBufferDepthPipeline = depthGpb.createPipeline();

      m_debug.setObjectName(m_gBufferDepthPipeline, "Depth");
}



//////////////////////////////////////////////////////////////////////////
// Ray Tracing
//////////////////////////////////////////////////////////////////////////
// #VKRay
void HelloVulkan::initRayTracing() {
  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext = &m_rtProperties;
  vkGetPhysicalDeviceProperties2(m_physicalDevice, &prop2);

  m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
}

//--------------------------------------------------------------------------------------------------
// Convert an OBJ model into the ray tracing geometry used to build the BLAS
//
auto HelloVulkan::objectToVkGeometryKHR(const ObjModel& model) {
  // BLAS builder requires raw device addresses.
  VkDeviceAddress vertexAddress = nvvk::getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
  VkDeviceAddress indexAddress  = nvvk::getBufferDeviceAddress(m_device, model.indexBuffer.buffer);

  uint32_t maxPrimitiveCount = model.nbIndices / 3;

  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
  triangles.vertexData.deviceAddress = vertexAddress;
  triangles.vertexStride             = sizeof(VertexObj);
  // Describe index data (32-bit unsigned int)
  triangles.indexType               = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress = indexAddress;
  // Indicate identity transform by setting transformData to null device pointer.
  //triangles.transformData = {};
  triangles.maxVertex = model.nbVertices - 1;

  // Identify the above data as containing opaque triangles.
  VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  asGeom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeom.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
  asGeom.geometry.triangles = triangles;

  // The entire array will be used to build the BLAS.
  VkAccelerationStructureBuildRangeInfoKHR offset;
  offset.firstVertex     = 0;
  offset.primitiveCount  = maxPrimitiveCount;
  offset.primitiveOffset = 0;
  offset.transformOffset = 0;

  // Our blas is made from only one geometry, but could be made of many geometries
  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);

  return input;
}

//--------------------------------------------------------------------------------------------------
//
//
void HelloVulkan::createBottomLevelAS() {
  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
  allBlas.reserve(m_objModel.size());
  for(const auto& obj : m_objModel) {
    auto blas = objectToVkGeometryKHR(obj);

    // We could add more geometry in each BLAS, but we add only one for now
    allBlas.emplace_back(blas);
  }
  m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
//
//
void HelloVulkan::createTopLevelAS() {
  std::vector<VkAccelerationStructureInstanceKHR> tlas;
  tlas.reserve(m_instances.size());
  for(const HelloVulkan::ObjInstance& inst : m_instances) {
    VkAccelerationStructureInstanceKHR rayInst{};
    rayInst.transform                      = nvvk::toTransformMatrixKHR(inst.transform);  // Position of the instance
    rayInst.instanceCustomIndex            = inst.objIndex;                               // gl_InstanceCustomIndexEXT
    rayInst.accelerationStructureReference = m_rtBuilder.getBlasDeviceAddress(inst.objIndex);
    rayInst.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    rayInst.mask                           = 0xFF;       //  Only be hit if rayMask & instance.mask != 0
    rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
    tlas.emplace_back(rayInst);
  }
  m_rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void HelloVulkan::createRtDescriptorSet() {
  // Top-level acceleration structure, usable by both the ray generation and the closest hit (to shoot shadow rays)
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);  // TLAS
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR);  // Output image

  //DDGI
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eConstants, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                   VK_SHADER_STAGE_VERTEX_BIT |VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT); // Constants
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eStatus, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);  // Probe Status buffer

  m_rtDescSetLayoutBind.addBinding(RtxBindings::eStorageImages, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, static_cast<uint32_t>(m_DDGIImages.size()),
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);  // Global Images 2D

  m_rtDescSetLayoutBind.addBinding(RtxBindings::eGlobalTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(m_globalTextures.size()),
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);  // Global Textures
  
  // Binded Images for Probe Update
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eIrradianceImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);  // Irradiance image
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eVisibilityImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);  // Visibility image


  m_rtDescPool      = m_rtDescSetLayoutBind.createPool(m_device);
  m_rtDescSetLayout = m_rtDescSetLayoutBind.createLayout(m_device);


  VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocateInfo.descriptorPool     = m_rtDescPool;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts        = &m_rtDescSetLayout;
  vkAllocateDescriptorSets(m_device, &allocateInfo, &m_rtDescSet);


  VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
  VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  descASInfo.accelerationStructureCount = 1;
  descASInfo.pAccelerationStructures    = &tlas;

  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};



  // Prepare buffer info for DDGI buffers
  VkDescriptorBufferInfo constantsBufferInfo{};
  constantsBufferInfo.buffer = m_bDDGIConstants.buffer;  // Assuming m_constantsBuffer is a VkBuffer handle for the constants buffer
  constantsBufferInfo.offset = 0;
  constantsBufferInfo.range  = VK_WHOLE_SIZE;

  
  VkDescriptorBufferInfo statusBufferInfo{};
  statusBufferInfo.buffer = m_bDDGIStatus.buffer;  // Assuming m_statusBuffer is a VkBuffer handle for the status buffer
  statusBufferInfo.offset = 0;
  statusBufferInfo.range  = VK_WHOLE_SIZE;

  // Global Images 2D
  std::vector<VkDescriptorImageInfo> imageInfos(m_DDGIImages.size());
  m_DDGIImageViews.resize(m_DDGIImages.size());

  for(size_t i = 0; i < m_DDGIImages.size(); ++i) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = m_DDGIImages[i].image;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = m_DDGIImages[i].format;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    if(vkCreateImageView(m_device, &viewInfo, nullptr, &m_DDGIImageViews[i]) != VK_SUCCESS){
      throw std::runtime_error("failed to create texture image view!");
    }

    imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfos[i].imageView   = m_DDGIImageViews[i];
    imageInfos[i].sampler     = VK_NULL_HANDLE;
  }

  // Global Textures
  std::vector<VkDescriptorImageInfo> globalTextureInfos(m_globalTextures.size());
  m_globalTexturesView.resize(m_globalTextures.size());
  m_globalTextureSamplers.resize(m_globalTextures.size());

  for(size_t i = 0; i < m_globalTextures.size(); ++i) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = m_globalTextures[i].image;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = m_globalTextures[i].format;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    if(vkCreateImageView(m_device, &viewInfo, nullptr, &m_globalTexturesView[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create texture image view!");
    }

    // Create or reuse a sampler for global textures
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter               = VK_FILTER_LINEAR;
    samplerInfo.minFilter               = VK_FILTER_LINEAR;
    samplerInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable        = VK_TRUE;
    samplerInfo.maxAnisotropy           = 16;
    samplerInfo.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable           = VK_FALSE;
    samplerInfo.compareOp               = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias              = 0.0f;
    samplerInfo.minLod                  = 0.0f;
    samplerInfo.maxLod                  = 0.0f;

    if(vkCreateSampler(m_device, &samplerInfo, nullptr, &m_globalTextureSamplers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create texture sampler!");
    }

    globalTextureInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    globalTextureInfos[i].imageView   = m_globalTexturesView[i];
    globalTextureInfos[i].sampler     = m_globalTextureSamplers[i];
  }

  // Binded images
  VkDescriptorImageInfo irradianceImageInfo{{}, m_irradianceTexture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorImageInfo visibilityImageInfo{{}, m_visibilityTexture.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  

  // Writes
  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eTlas, &descASInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eConstants, &constantsBufferInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eStatus, &statusBufferInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eIrradianceImage, &irradianceImageInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eVisibilityImage, &visibilityImageInfo));
  
  // Global Images 2D
  VkWriteDescriptorSet writeStorageImages = {};
  writeStorageImages.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeStorageImages.dstSet               = m_rtDescSet;
  writeStorageImages.dstBinding           = RtxBindings::eStorageImages;
  writeStorageImages.dstArrayElement      = 0;
  writeStorageImages.descriptorType       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  writeStorageImages.descriptorCount      = static_cast<uint32_t>(m_DDGIImages.size());
  writeStorageImages.pImageInfo           = imageInfos.data();
  writes.emplace_back(writeStorageImages);

   // Global Textures
  VkWriteDescriptorSet writeGlobalTextures = {};
  writeGlobalTextures.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeGlobalTextures.dstSet               = m_rtDescSet;
  writeGlobalTextures.dstBinding           = RtxBindings::eGlobalTextures;
  writeGlobalTextures.dstArrayElement      = 0;
  writeGlobalTextures.descriptorType       = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writeGlobalTextures.descriptorCount      = static_cast<uint32_t>(m_globalTextures.size());
  writeGlobalTextures.pImageInfo           = globalTextureInfos.data();
  writes.emplace_back(writeGlobalTextures);

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void HelloVulkan::updateRtDescriptorSet() {
  // (1) Output buffer
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet  wds = m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo);
  vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);
}




//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void HelloVulkan::createRtPipeline() {
    enum StageIndices {
    eRaygen,
    eMiss,
    eMiss2,
    eClosestHit,
    eShaderGroupCount
  };

  // All stages
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pName = "main";  // All the same entry point
  // Raygen
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rgen.spv", true, defaultSearchPaths, true));
  stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eRaygen] = stage;
  // Miss
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rmiss.spv", true, defaultSearchPaths, true));
  stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss] = stage;
  // The second miss shader is invoked when a shadow ray misses the geometry. It simply indicates that no occlusion has been found
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytraceShadow.rmiss.spv", true, defaultSearchPaths, true));
  stage.stage    = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss2] = stage;
  // Hit Group - Closest Hit
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rchit.spv", true, defaultSearchPaths, true));
  stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHit] = stage;


  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  m_rtShaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  m_rtShaderGroups.push_back(group);

  // Shadow Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss2;
  m_rtShaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  m_rtShaderGroups.push_back(group);

  // Push constant: we want to be able to update constants used by the shaders
  VkPushConstantRange pushConstant{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                                   0, sizeof(PushConstantRay)};


  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
  pipelineLayoutCreateInfo.setLayoutCount             = static_cast<uint32_t>(rtDescSetLayouts.size());
  pipelineLayoutCreateInfo.pSetLayouts                = rtDescSetLayouts.data();

  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout);


  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rayPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());  // Stages are shaders
  rayPipelineInfo.pStages    = stages.data();

  // In this case, m_rtShaderGroups.size() == 4: we have one raygen group,
  // two miss shader groups, and one hit group.
  rayPipelineInfo.groupCount = static_cast<uint32_t>(m_rtShaderGroups.size());
  rayPipelineInfo.pGroups    = m_rtShaderGroups.data();
  rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
  rayPipelineInfo.layout                       = m_rtPipelineLayout;

  vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_rtPipeline);


  // Spec only guarantees 1 level of "recursion". Check for that sad possibility here.
  if(m_rtProperties.maxRayRecursionDepth <= 1) {
    throw std::runtime_error("Device fails to support ray recursion (m_rtProperties.maxRayRecursionDepth <= 1)");
  }

  for(auto& s : stages)
    vkDestroyShaderModule(m_device, s.module, nullptr);

}


//--------------------------------------------------------------------------------------------------
// The Shader Binding Table (SBT)
// - getting all shader handles and write them in a SBT buffer
// - Besides exception, this could be always done like this
//
void HelloVulkan::createRtShaderBindingTable() {
  uint32_t missCount{2};
  uint32_t hitCount{1};
  auto     handleCount = 1 + missCount + hitCount;
  uint32_t handleSize  = m_rtProperties.shaderGroupHandleSize;

  // The SBT (buffer) need to have starting groups to be aligned and handles in the group to be aligned.
  uint32_t handleSizeAligned = nvh::align_up(handleSize, m_rtProperties.shaderGroupHandleAlignment);

  m_rgenRegion.stride = nvh::align_up(handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);
  m_rgenRegion.size = m_rgenRegion.stride;  // The size member of pRayGenShaderBindingTable must be equal to its stride member
  m_missRegion.stride = handleSizeAligned;
  m_missRegion.size   = nvh::align_up(missCount * handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);
  m_hitRegion.stride  = handleSizeAligned;
  m_hitRegion.size    = nvh::align_up(hitCount * handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);

  // Get the shader group handles
  uint32_t             dataSize = handleCount * handleSize;
  std::vector<uint8_t> handles(dataSize);
  auto result = vkGetRayTracingShaderGroupHandlesKHR(m_device, m_rtPipeline, 0, handleCount, dataSize, handles.data());
  assert(result == VK_SUCCESS);

  // Allocate a buffer for storing the SBT.
  VkDeviceSize sbtSize = m_rgenRegion.size + m_missRegion.size + m_hitRegion.size + m_callRegion.size;
  m_rtSBTBuffer        = m_alloc.createBuffer(sbtSize,
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                  | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_debug.setObjectName(m_rtSBTBuffer.buffer, std::string("SBT"));  // Give it a debug name for NSight.

  // Find the SBT addresses of each group
  VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, m_rtSBTBuffer.buffer};
  VkDeviceAddress           sbtAddress = vkGetBufferDeviceAddress(m_device, &info);
  m_rgenRegion.deviceAddress           = sbtAddress;
  m_missRegion.deviceAddress           = sbtAddress + m_rgenRegion.size;
  m_hitRegion.deviceAddress            = sbtAddress + m_rgenRegion.size + m_missRegion.size;

  // Helper to retrieve the handle data
  auto getHandle = [&](int i) { return handles.data() + i * handleSize; };

  // Map the SBT buffer and write in the handles.
  auto*    pSBTBuffer = reinterpret_cast<uint8_t*>(m_alloc.map(m_rtSBTBuffer));
  uint8_t* pData{nullptr};
  uint32_t handleIdx{0};
  // Raygen
  pData = pSBTBuffer;
  memcpy(pData, getHandle(handleIdx++), handleSize);
  // Miss
  pData = pSBTBuffer + m_rgenRegion.size;
  for(uint32_t c = 0; c < missCount; c++) {
    memcpy(pData, getHandle(handleIdx++), handleSize);
    pData += m_missRegion.stride;
  }
  // Hit
  pData = pSBTBuffer + m_rgenRegion.size + m_missRegion.size;
  for(uint32_t c = 0; c < hitCount; c++) {
    memcpy(pData, getHandle(handleIdx++), handleSize);
    pData += m_hitRegion.stride;
  }

  m_alloc.unmap(m_rtSBTBuffer);
  m_alloc.finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void HelloVulkan::raytrace(const VkCommandBuffer& cmdBuf, const glm::vec4& clearColor) {
  m_debug.beginLabel(cmdBuf, "Ray trace");
  // Initializing push constant values
  m_pcRay.clearColor     = clearColor;
  m_pcRay.lightPosition  = m_pcRaster.lightPosition;
  m_pcRay.lightIntensity = m_pcRaster.lightIntensity;
  m_pcRay.lightType      = m_pcRaster.lightType;

  

  std::vector<VkDescriptorSet> descSets{m_rtDescSet, m_descSet};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
                     VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                     0, sizeof(PushConstantRay), &m_pcRay);


  vkCmdTraceRaysKHR(cmdBuf, &m_rgenRegion, &m_missRegion, &m_hitRegion, &m_callRegion, m_size.width, m_size.height, 1);


  m_debug.endLabel(cmdBuf);

}



//////////////////////////////////////////////////////////////////////////
// Irradiance Field
//////////////////////////////////////////////////////////////////////////


void HelloVulkan::prepareDraws(renderSceneDDGI& scene) {
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = cmdBufGet.createCommandBuffer();

  per_frame_probe_updates   = scene.gi_per_frame_probes_update;
  const uint32_t num_probes = get_total_probes();
  scene.gi_total_probes     = num_probes;

  half_resolution_output = scene.gi_use_half_resolution;
  
  createDDGIConstantsBuffer();
  createDDGIStatusBuffer();
  


  // Texture creation
  //-----------------
  // Radiance Texture
  const uint32_t num_rays = probe_rays;
  auto           radianceCreateInfo = nvvk::makeImage2DCreateInfo({num_rays, num_probes}, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
  m_radianceImage  = m_alloc.createImage(radianceCreateInfo);
  nvvk::cmdBarrierImageLayout(cmdBuf, m_radianceImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  //m_radianceImage = createStorageImage(cmdBuf, m_device, m_physicalDevice, num_rays, num_probes, VK_FORMAT_R16G16B16A16_SFLOAT);
  VkImageViewCreateInfo radianceIvInfo = nvvk::makeImageViewCreateInfo(m_radianceImage.image, radianceCreateInfo);
  VkSamplerCreateInfo radianceSampler { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
  m_radianceTexture                        = m_alloc.createTexture(m_radianceImage, radianceIvInfo, radianceSampler);
  m_radianceTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  m_DDGIImages.push_back(m_radianceTexture);        // Global Images array
  m_globalTextures.push_back(m_radianceTexture);    // Global Textures array


  
  //----------------------
  // Probe offsets texture
  auto offsetsCreateInfo = nvvk::makeImage2DCreateInfo({probe_count_x * probe_count_y, probe_count_z}, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT );
  m_offsetsImage = createStorageImage(cmdBuf, m_device, m_physicalDevice, probe_count_x * probe_count_y, probe_count_z, VK_FORMAT_R16G16B16A16_SFLOAT);
  VkImageViewCreateInfo offsetsIvInfo = nvvk::makeImageViewCreateInfo(m_offsetsImage.image, offsetsCreateInfo);
  VkSamplerCreateInfo offsetsSampler { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
  m_offsetsTexture                        = m_alloc.createTexture(m_offsetsImage, offsetsIvInfo, offsetsSampler);
  m_offsetsTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  m_DDGIImages.push_back(m_offsetsTexture);         // Global Images array
  m_globalTextures.push_back(m_offsetsTexture);     // Global Textures array




  //----------------------
  // Irradiance Texture 6x6 plus 2 additional pixel border to allow bilinear interpolation
  const int octahedral_irradiance_size = irradiance_probe_size + 2;
  irradiance_atlas_width               = (octahedral_irradiance_size * probe_count_x * probe_count_y);
  irradiance_atlas_height              = (octahedral_irradiance_size * probe_count_z);
  //m_irradianceImage = createStorageImage(cmdBuf, m_device, m_physicalDevice, irradiance_atlas_width, irradiance_atlas_height, VK_FORMAT_R16G16B16A16_SFLOAT);
  auto irradianceCreateInfo = nvvk::makeImage2DCreateInfo({static_cast<uint32_t>(irradiance_atlas_width), static_cast<uint32_t>(irradiance_atlas_height)}, VK_FORMAT_R16G16B16A16_SFLOAT,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
  m_irradianceImage = m_alloc.createImage(irradianceCreateInfo);
  nvvk::cmdBarrierImageLayout(cmdBuf, m_irradianceImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  VkImageViewCreateInfo irradianceIvInfo = nvvk::makeImageViewCreateInfo(m_irradianceImage.image, irradianceCreateInfo);
  VkSamplerCreateInfo   irradianceSampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  m_irradianceTexture                       = m_alloc.createTexture(m_irradianceImage, irradianceIvInfo, irradianceSampler);
  m_irradianceTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  
  m_globalTextures.push_back(m_irradianceTexture);  // Global Textures array


  // Visibility Texture
  const int octahedral_visibility_size = visibility_probe_size + 2;
  visibility_atlas_width                   = (octahedral_visibility_size * probe_count_x * probe_count_y);
  visibility_atlas_height                  = (octahedral_visibility_size * probe_count_z);
  //m_visibilityImage = createStorageImage(cmdBuf, m_device, m_physicalDevice, visibility_atlas_width, visibility_atlas_height, VK_FORMAT_R16G16_SFLOAT);
  auto visibilityCreateInfo = nvvk::makeImage2DCreateInfo( {static_cast<uint32_t>(visibility_atlas_width), static_cast<uint32_t>(visibility_atlas_height)}, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
  m_visibilityImage = m_alloc.createImage(visibilityCreateInfo);
  nvvk::cmdBarrierImageLayout(cmdBuf, m_visibilityImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  VkImageViewCreateInfo visibilityIvInfo = nvvk::makeImageViewCreateInfo(m_visibilityImage.image, visibilityCreateInfo);
  VkSamplerCreateInfo   visibilitySampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  m_visibilityTexture = m_alloc.createTexture(m_visibilityImage, visibilityIvInfo, visibilitySampler);
  m_visibilityTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  m_globalTextures.push_back(m_visibilityTexture);      // Global Textures array
  

  // Indirect Texture
  uint32_t adjusted_width  = scene.gi_use_half_resolution ? m_size.width / 2 : m_size.width;
  uint32_t adjusted_height = scene.gi_use_half_resolution ? m_size.height / 2 : m_size.height;
  //m_indirectImage = createStorageImage(cmdBuf, m_device, m_physicalDevice, adjusted_width, adjusted_height, VK_FORMAT_R16G16B16A16_SFLOAT);
  auto indirectCreateInfo = nvvk::makeImage2DCreateInfo({adjusted_width, adjusted_height}, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
  m_indirectImage = m_alloc.createImage(indirectCreateInfo);

  nvvk::cmdBarrierImageLayout(cmdBuf, m_indirectImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  VkImageViewCreateInfo indirectIvInfo = nvvk::makeImageViewCreateInfo(m_indirectImage.image, indirectCreateInfo);
  VkSamplerCreateInfo   indirectSampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  m_indirectTexture = m_alloc.createTexture(m_indirectImage, indirectIvInfo, indirectSampler);
  m_indirectTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  
  m_DDGIImages.push_back(m_indirectTexture);  // Global Images array


}

void HelloVulkan::addNormalsDepthTextures() {
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = cmdBufGet.createCommandBuffer();
  m_globalTextures.push_back(m_gBufferNormals);
  m_globalTextures.push_back(m_depthTexture);
  m_globalTextures.push_back(m_indirectTexture);
}


void HelloVulkan::DDGIBegin(const VkCommandBuffer& cmdBuf, const glm::vec4& clearColor, renderSceneDDGI& scene) {
  m_debug.beginLabel(cmdBuf, "DDGI Begin");

  // Initializing push constant values
  m_pcRay.clearColor     = clearColor;
  m_pcRay.lightPosition  = m_pcRaster.lightPosition;
  m_pcRay.lightIntensity = m_pcRaster.lightIntensity;
  m_pcRay.lightType      = m_pcRaster.lightType;

  // Sample Irradiance Push Constant
  m_pcSampleIrradiance.output_resolution_half = (scene.gi_use_half_resolution == true) ? 1 : 0;

  uint32_t   first_frame;
  static int offsets_calculations_count = 24;

  if(scene.gi_recalculate_offsets) {
    offsets_calculations_count = 24;
  }



  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_radianceTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    genCmdBuf.submitAndWait(cmdBuf);
  }
  // Ray Tracing
  std::vector<VkDescriptorSet> descSets{m_rtDescSet, m_descSet};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_DDGIPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_DDGIPipelineLayout, 0, (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_DDGIPipelineLayout, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 0, sizeof(PushConstantRay), &m_pcRay);
  const uint32_t probe_count = offsets_calculations_count >= 0 ? get_total_probes() : per_frame_probe_updates;
  vkCmdTraceRaysKHR(cmdBuf, &m_DDGIrgenRegion, &m_DDGImissRegion, &m_DDGIhitRegion, &m_DDGIcallRegion, probe_rays, get_total_probes(), 1);

  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_radianceTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    genCmdBuf.submitAndWait(cmdBuf);
  }
  m_debug.endLabel(cmdBuf);

  
  if(offsets_calculations_count >= 0) {
    --offsets_calculations_count;
    first_frame = offsets_calculations_count == 23 ? 1 : 0;

    m_pcProbeOffsets.first_frame = first_frame;

    m_debug.beginLabel(cmdBuf, "Offsets Compute Begin");

    {
      nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
      auto              cmdBuf = genCmdBuf.createCommandBuffer();
      nvvk::cmdBarrierImageLayout(cmdBuf, m_offsetsTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
      genCmdBuf.submitAndWait(cmdBuf);
    }

    // Probe Offsets Compute Pipeline
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_probeOffsetsPipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_probeOffsetsPipelineLayout, 0,
                            (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
    vkCmdPushConstants(cmdBuf, m_probeOffsetsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantOffset),
                       &m_pcProbeOffsets);
    vkCmdDispatch(cmdBuf, glm::ceil(probe_count / 32.0f), 1, 1);

    m_debug.endLabel(cmdBuf);
  }

  
  m_debug.beginLabel(cmdBuf, "Status Compute Begin");
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_radianceTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    genCmdBuf.submitAndWait(cmdBuf);
  }
  // Probe Status
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_probeStatusPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_probeStatusPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  first_frame = 0;
  m_pcProbeStatus.first_frame = first_frame;
  vkCmdPushConstants(cmdBuf, m_probeStatusPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantStatus), &m_pcProbeStatus);
  vkCmdDispatch(cmdBuf, glm::ceil(probe_count / 32.0f), 1, 1);

  m_debug.endLabel(cmdBuf);

  

  m_debug.beginLabel(cmdBuf, "Irradiance Compute Begin");
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_irradianceTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    genCmdBuf.submitAndWait(cmdBuf);
  }
  // Probe Update Irradiance
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_probeUpdateIrradiancePipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_probeUpdateIrradiancePipelineLayout, 0, (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_probeUpdateIrradiancePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(PushConstantOffset), &m_pcProbeOffsets);
  vkCmdDispatch(cmdBuf, glm::ceil(irradiance_atlas_width / 8.0f), glm::ceil(irradiance_atlas_height / 8.0f), 1);
  m_debug.endLabel(cmdBuf);

  

  m_debug.beginLabel(cmdBuf, "Visibility Compute Begin");
  // Probe Update Visibility
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_visibilityTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    genCmdBuf.submitAndWait(cmdBuf);
  }
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_probeUpdateVisibilityPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_probeUpdateVisibilityPipelineLayout, 0, (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_probeUpdateVisibilityPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(PushConstantOffset), &m_pcProbeOffsets);
  vkCmdDispatch(cmdBuf, glm::ceil(visibility_atlas_width / 8.0f), glm::ceil(visibility_atlas_height / 8.0f), 1);
  
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_irradianceTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_visibilityTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    genCmdBuf.submitAndWait(cmdBuf);
  }
  m_debug.endLabel(cmdBuf);
  


  m_debug.beginLabel(cmdBuf, "Sample Compute Begin");
 
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_indirectTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    genCmdBuf.submitAndWait(cmdBuf);
  }
  const float resolution_divider = scene.gi_use_half_resolution ? 0.5f : 1.0f;


  // Sample Irradiance
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_sampleIrradiancePipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_sampleIrradiancePipelineLayout, 0, (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_sampleIrradiancePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantSample), &m_pcSampleIrradiance);
  vkCmdDispatch(cmdBuf, m_size.width * resolution_divider, m_size.height * resolution_divider, 1);
  
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_indirectTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  m_debug.endLabel(cmdBuf);
}


void HelloVulkan::createDDGIPipeline() {
  enum StageIndices {
    eRaygen,
    eMiss,
    eClosestHit,
    eShaderGroupCount
  };

  // All stages
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> DDGIstages{};
  VkPipelineShaderStageCreateInfo DDGIstage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  DDGIstage.pName = "main";  // All the same entry point

  // DDGI Raygen
  DDGIstage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytraceDDGI.rgen.spv", true, defaultSearchPaths, true));
  DDGIstage.stage         = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  DDGIstages[eRaygen] = DDGIstage;

  // DDGI Miss
  DDGIstage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytraceDDGI.rmiss.spv", true, defaultSearchPaths, true));
  DDGIstage.stage       = VK_SHADER_STAGE_MISS_BIT_KHR;
  DDGIstages[eMiss] = DDGIstage;

  // DDGI Closest Hit
  DDGIstage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytraceDDGI.rchit.spv", true, defaultSearchPaths, true));
  DDGIstage.stage             = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  DDGIstages[eClosestHit] = DDGIstage;

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR DDGIgroup{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  DDGIgroup.anyHitShader   = VK_SHADER_UNUSED_KHR;
  DDGIgroup.closestHitShader = VK_SHADER_UNUSED_KHR;
  DDGIgroup.generalShader    = VK_SHADER_UNUSED_KHR;
  DDGIgroup.intersectionShader = VK_SHADER_UNUSED_KHR;

  // DDGI Raygen
  DDGIgroup.type      = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  DDGIgroup.generalShader = eRaygen;
  m_DDGIShaderGroups.push_back(DDGIgroup);

  // DDGI Miss
  DDGIgroup.type      = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  DDGIgroup.generalShader = eMiss;
  m_DDGIShaderGroups.push_back(DDGIgroup);

  // DDGI Closest Hit
  DDGIgroup.type         = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  DDGIgroup.generalShader = VK_SHADER_UNUSED_KHR;
  DDGIgroup.closestHitShader = eClosestHit;
  m_DDGIShaderGroups.push_back(DDGIgroup);


  // Push constant: we want to be able to update constants used by the shaders
  VkPushConstantRange pushConstant{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                                   0, sizeof(PushConstantRay)};

  VkPipelineLayoutCreateInfo DDGIpipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  DDGIpipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  DDGIpipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

  // Descriptor sets: specific to DDGI
  std::vector<VkDescriptorSetLayout> DDGIDescSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
  DDGIpipelineLayoutCreateInfo.setLayoutCount           = static_cast<uint32_t>(DDGIDescSetLayouts.size());
  DDGIpipelineLayoutCreateInfo.pSetLayouts              = DDGIDescSetLayouts.data();

  vkCreatePipelineLayout(m_device, &DDGIpipelineLayoutCreateInfo, nullptr, &m_DDGIPipelineLayout);

  // Assemble the shader stages and recursion depth info into the DDGI ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR DDGIPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
  DDGIPipelineInfo.stageCount = static_cast<uint32_t>(DDGIstages.size());  // Stages are shaders
  DDGIPipelineInfo.pStages    = DDGIstages.data();
  DDGIPipelineInfo.groupCount = static_cast<uint32_t>(m_DDGIShaderGroups.size());
  DDGIPipelineInfo.pGroups    = m_DDGIShaderGroups.data();
  DDGIPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
  DDGIPipelineInfo.layout                       = m_DDGIPipelineLayout;

  vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &DDGIPipelineInfo, nullptr, &m_DDGIPipeline);

  
  // Spec only guarantees 1 level of "recursion". Check for that sad possibility here.
  if(m_rtProperties.maxRayRecursionDepth <= 1) {
    throw std::runtime_error("Device fails to support ray recursion (m_DDGIProperties.maxRayRecursionDepth <= 1)");
  }
  
  for(auto& s : DDGIstages) {
    vkDestroyShaderModule(m_device, s.module, nullptr);
  }



  // Compute Pipelines
  VkPushConstantRange pushConstantOffset{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(PushConstantOffset)};

  VkPushConstantRange pushConstantSample{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
                                         0, sizeof(PushConstantSample)};

  
  createComputePipeline("spv/probeOffsets.glsl.spv", DDGIDescSetLayouts, m_probeOffsetsPipelineLayout,
                        m_probeOffsetsPipeline, &pushConstantOffset, sizeof(pushConstantOffset));

  createComputePipeline("spv/probeStatus.glsl.spv", DDGIDescSetLayouts, m_probeStatusPipelineLayout,
                        m_probeStatusPipeline, &pushConstantOffset, sizeof(pushConstantOffset));

  createComputePipeline("spv/probeUpdateIrradiance.glsl.spv", DDGIDescSetLayouts, m_probeUpdateIrradiancePipelineLayout,
                        m_probeUpdateIrradiancePipeline, &pushConstant, sizeof(pushConstant));

  createComputePipeline("spv/probeUpdateVisibility.glsl.spv", DDGIDescSetLayouts, m_probeUpdateVisibilityPipelineLayout,
                        m_probeUpdateVisibilityPipeline, &pushConstant, sizeof(pushConstant));
  
  createComputePipeline("spv/sampleIrradiance.glsl.spv", DDGIDescSetLayouts, m_sampleIrradiancePipelineLayout,
                        m_sampleIrradiancePipeline, &pushConstantSample, sizeof(pushConstantSample));
  
}


void HelloVulkan::createComputePipeline(const std::string&    shaderPath,
                                        std::vector<VkDescriptorSetLayout> DDGIDescSetLayouts,
                                        VkPipelineLayout&     pipelineLayout,
                                        VkPipeline&           pipeline,
                                        const void*           pushConstants,
                                        uint32_t              pushConstantsSize) {

  // Compile compute shader and package as stage.
  VkShaderModule computeShader = nvvk::createShaderModule(m_device, nvh::loadFile(shaderPath, true, defaultSearchPaths, true));
  VkPipelineShaderStageCreateInfo stageInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
  stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = computeShader;
  stageInfo.pName  = "main";

  // Set up push constant and pipeline layout.
  VkPushConstantRange        pushCRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantsSize };
  
  VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutInfo.setLayoutCount         = static_cast<uint32_t>(DDGIDescSetLayouts.size());
  layoutInfo.pSetLayouts            = DDGIDescSetLayouts.data();
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges    = &pushCRange;
  vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &pipelineLayout);

  // Create compute pipeline.
  VkComputePipelineCreateInfo pipelineInfo { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
  pipelineInfo.stage  = stageInfo;
  pipelineInfo.layout = pipelineLayout;
  vkCreateComputePipelines(m_device, {}, 1, &pipelineInfo, nullptr, &pipeline);

  vkDestroyShaderModule(m_device, computeShader, nullptr);
}

//---------------------------------
// Buffers

void HelloVulkan::createDDGIConstantsBuffer() {
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = cmdBufGet.createCommandBuffer();

  using Usage      = VkBufferUsageFlagBits;
  m_bDDGIConstants = m_alloc.createBuffer(sizeof(GpuDDGIConstants), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_bDDGIConstants.buffer, "DDGIConstantsBuffer");

}

void HelloVulkan::createDDGIStatusBuffer() {
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = cmdBufGet.createCommandBuffer();

  const uint32_t    num_probes = get_total_probes();
  using Usage   = VkBufferUsageFlagBits;
  m_bDDGIStatus = m_alloc.createBuffer(sizeof(uint32_t) * num_probes, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_bDDGIStatus.buffer, "DDGIStausBuffer");
}

void HelloVulkan::updateDDGIConstantsBuffer(const VkCommandBuffer& cmdBuf, renderSceneDDGI& scene) {
  GpuDDGIConstants hostDDGIConstBuffer = {};
  
  hostDDGIConstBuffer.radiance_output_index        = 0;
  hostDDGIConstBuffer.grid_irradiance_output_index = 2;
  hostDDGIConstBuffer.indirect_output_index        = 2;
  hostDDGIConstBuffer.normal_texture_index         = 4;
  
  hostDDGIConstBuffer.depth_pyramid_texture_index    = 6;
  hostDDGIConstBuffer.depth_fullscreen_texture_index = 5;
  hostDDGIConstBuffer.grid_visibility_texture_index  = 3;
  hostDDGIConstBuffer.probe_offset_texture_index     = 1;

  hostDDGIConstBuffer.hysteresis                 = scene.gi_hysteresis;
  hostDDGIConstBuffer.infinte_bounces_multiplier = scene.gi_infinite_bounces_multiplier;
  hostDDGIConstBuffer.probe_update_offset        = probe_update_offset;
  hostDDGIConstBuffer.probe_update_count         = per_frame_probe_updates;

  hostDDGIConstBuffer.probe_grid_position        = scene.gi_probe_grid_position;
  hostDDGIConstBuffer.probe_sphere_scale         = scene.gi_probe_sphere_scale;

  hostDDGIConstBuffer.probe_spacing              = scene.gi_probe_spacing;
  hostDDGIConstBuffer.max_probe_offset           = scene.gi_max_probe_offset;

  hostDDGIConstBuffer.reciprocal_probe_spacing   = {1.f / scene.gi_probe_spacing.x, 1.f / scene.gi_probe_spacing.y, 1.f / scene.gi_probe_spacing.z};
  hostDDGIConstBuffer.self_shadow_bias           = scene.gi_self_shadow_bias;

  hostDDGIConstBuffer.probe_counts[0]            = probe_count_x;
  hostDDGIConstBuffer.probe_counts[1]            = probe_count_y;
  hostDDGIConstBuffer.probe_counts[2]            = probe_count_z;

  // Debug options for probes
  hostDDGIConstBuffer.debug_options = ((scene.gi_debug_border ? 1 : 0)) 
                                        | ((scene.gi_debug_border_type ? 1 : 0) << 1)
                                        | ((scene.gi_debug_border_source ? 1 : 0) << 2) 
                                        | ((scene.gi_use_visibility ? 1 : 0) << 3)
                                        | ((scene.gi_use_backface_smoothing ? 1 : 0) << 4) 
                                        | ((scene.gi_use_perceptual_encoding ? 1 : 0) << 5)
                                        | ((scene.gi_use_backface_blending ? 1 : 0) << 6) 
                                        | ((scene.gi_use_probe_offsetting ? 1 : 0) << 7)
                                        | ((scene.gi_use_probe_status ? 1 : 0) << 8) 
                                        | ((scene.gi_use_infinite_bounces ? 1 : 0) << 9);

  hostDDGIConstBuffer.irradiance_texture_width  = irradiance_atlas_width;
  hostDDGIConstBuffer.irradiance_texture_height = irradiance_atlas_height;
  hostDDGIConstBuffer.irradiance_side_length    = irradiance_probe_size;

  hostDDGIConstBuffer.probe_rays                = probe_rays;

  hostDDGIConstBuffer.visibility_texture_width  = visibility_atlas_width;
  hostDDGIConstBuffer.visibility_texture_height = visibility_atlas_height;
  hostDDGIConstBuffer.visibility_side_length    = visibility_probe_size;

  hostDDGIConstBuffer.probe_update_offset       = probe_update_offset;
  hostDDGIConstBuffer.probe_update_count        = per_frame_probe_updates;

  const float rotation_scaler                     = 0.001f; 
  hostDDGIConstBuffer.random_rotation           = glms_euler_xyz ( glm::vec3(randomFloat(-1.0f, 1.0f) * rotation_scaler, randomFloat(-1, 1) * rotation_scaler, randomFloat(-1, 1) * rotation_scaler));
  //hostDDGIConstBuffer.random_rotation           = randomRotationMatrix();
  hostDDGIConstBuffer.resolution                = glm::vec2(m_size.width, m_size.height);

  const uint32_t num_probes    = probe_count_x * probe_count_y * probe_count_z;
  probe_update_offset     = (probe_update_offset + per_frame_probe_updates) % num_probes;
  per_frame_probe_updates = scene.gi_per_frame_probes_update;


  VkBuffer deviceDDGIConstBuffer = m_bDDGIConstants.buffer;
  auto     DDGIConstUsageStages  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier beforeBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  beforeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  beforeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  beforeBarrier.buffer        = deviceDDGIConstBuffer;
  beforeBarrier.offset        = 0;
  beforeBarrier.size          = sizeof(hostDDGIConstBuffer);

  vkCmdPipelineBarrier(cmdBuf, DDGIConstUsageStages, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &beforeBarrier, 0, nullptr);


  // Schedule the host-to-device upload. (hostDDGIConstants is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_bDDGIConstants.buffer, 0, sizeof(GpuDDGIConstants), &hostDDGIConstBuffer);
  
  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier afterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  afterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  afterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  afterBarrier.buffer        = deviceDDGIConstBuffer;
  afterBarrier.offset        = 0;
  afterBarrier.size          = sizeof(hostDDGIConstBuffer);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, DDGIConstUsageStages, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &afterBarrier, 0, nullptr);
                       
}





// Shader Binding Table
void HelloVulkan::createDDGIShaderBindingTable() {
  uint32_t missCount{1};
  uint32_t hitCount{1};
  auto     handleCount = 1 + missCount + hitCount;
  uint32_t handleSize  = m_rtProperties.shaderGroupHandleSize;

  // The SBT (buffer) need to have starting groups to be aligned and handles in the group to be aligned.
  uint32_t handleSizeAligned = nvh::align_up(handleSize, m_rtProperties.shaderGroupHandleAlignment);

  m_DDGIrgenRegion.stride = nvh::align_up(handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);
  m_DDGIrgenRegion.size = m_DDGIrgenRegion.stride;  // The size member of pRayGenShaderBindingTable must be equal to its stride member
  m_DDGImissRegion.stride = handleSizeAligned;
  m_DDGImissRegion.size   = nvh::align_up(missCount * handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);
  m_DDGIhitRegion.stride  = handleSizeAligned;
  m_DDGIhitRegion.size    = nvh::align_up(hitCount * handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);

  // Get the shader group handles
  uint32_t             dataSize = handleCount * handleSize;
  std::vector<uint8_t> handles(dataSize);
  auto result = vkGetRayTracingShaderGroupHandlesKHR(m_device, m_DDGIPipeline, 0, handleCount, dataSize, handles.data());
  assert(result == VK_SUCCESS);

  // Allocate a buffer for storing the SBT.
  VkDeviceSize sbtSize = m_DDGIrgenRegion.size + m_DDGImissRegion.size + m_DDGIhitRegion.size + m_DDGIcallRegion.size;
  m_DDGISBTBuffer      = m_alloc.createBuffer(sbtSize,
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                  | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_debug.setObjectName(m_DDGISBTBuffer.buffer, std::string("SBT"));  // Give it a debug name for NSight.

  // Find the SBT addresses of each group
  VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, m_DDGISBTBuffer.buffer};
  VkDeviceAddress           sbtAddress = vkGetBufferDeviceAddress(m_device, &info);
  m_DDGIrgenRegion.deviceAddress       = sbtAddress;
  m_DDGImissRegion.deviceAddress       = sbtAddress + m_DDGIrgenRegion.size;
  m_DDGIhitRegion.deviceAddress        = sbtAddress + m_DDGIrgenRegion.size + m_DDGImissRegion.size;

  // Helper to retrieve the handle data
  auto getHandle = [&](int i) { return handles.data() + i * handleSize; };

  // Map the SBT buffer and write in the handles.
  auto*    pSBTBuffer = reinterpret_cast<uint8_t*>(m_alloc.map(m_DDGISBTBuffer));
  uint8_t* pData{nullptr};
  uint32_t handleIdx{0};
  // Raygen
  pData = pSBTBuffer;
  memcpy(pData, getHandle(handleIdx++), handleSize);
  // Miss
  pData = pSBTBuffer + m_DDGIrgenRegion.size;
  for(uint32_t c = 0; c < missCount; c++) {
    memcpy(pData, getHandle(handleIdx++), handleSize);
    pData += m_DDGImissRegion.stride;
  }
  // Hit
  pData = pSBTBuffer + m_DDGIrgenRegion.size + m_missRegion.size;
  for(uint32_t c = 0; c < hitCount; c++) {
    memcpy(pData, getHandle(handleIdx++), handleSize);
    pData += m_DDGIhitRegion.stride;
  }

  m_alloc.unmap(m_DDGISBTBuffer);
  m_alloc.finalizeAndReleaseStaging();
}




nvvk::Image HelloVulkan::createStorageImage(const VkCommandBuffer& cmdBuf,
                                            VkDevice               device,
                                            VkPhysicalDevice       physicalDevice,
                                            uint32_t               width,
                                            uint32_t               height,
                                            VkFormat               format)
{
  nvvk::Image image;


  std::array<uint8_t, 4> color{255u, 255u, 255u, 255u};
  VkDeviceSize           bufferSize = sizeof(color);

  // Image creation
  VkImageCreateInfo imageCreateInfo{};
  imageCreateInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageCreateInfo.imageType     = VK_IMAGE_TYPE_2D;
  imageCreateInfo.extent.width  = width;
  imageCreateInfo.extent.height = height;
  imageCreateInfo.extent.depth  = 1;
  imageCreateInfo.mipLevels     = 1;
  imageCreateInfo.arrayLayers   = 1;
  imageCreateInfo.format        = format;
  imageCreateInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
  imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageCreateInfo.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageCreateInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
  imageCreateInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

  image                        = m_alloc.createImage(cmdBuf, bufferSize, color.data(), imageCreateInfo);
  image.format = format;  // Store the format


  VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);

  // Transition image layout to VK_IMAGE_LAYOUT_GENERAL
  nvvk::cmdBarrierImageLayout(cmdBuf, image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);

  return image;
}



void HelloVulkan::createDebugRender() {
  m_alloc.destroy(m_debugTexture);
  m_alloc.destroy(m_debugDepth);

  // Creating the normal image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_debugTextureFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    nvvk::Image           image           = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo          = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    VkSamplerCreateInfo   sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_debugTexture                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_debugTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }


  // Creating the depth buffer
  {
    auto depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_debugDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);

    VkImageViewCreateInfo depthStencilView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthStencilView.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format           = m_debugDepthFormat;
    depthStencilView.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    depthStencilView.image            = image.image;

    m_debugDepth = m_alloc.createTexture(image, depthStencilView);
  }

  // Setting the image layout for all attachments
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_debugTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_debugDepth.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a render pass for the offscreen
  if(!m_debugRenderPass) {
    m_debugRenderPass = nvvk::createRenderPass(m_device, {m_debugTextureFormat}, m_debugDepthFormat, 1, true, true,
                                                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  }

  // Creating the framebuffer for offscreen
  std::vector<VkImageView> attachments = {m_debugTexture.descriptor.imageView, m_debugDepth.descriptor.imageView};


  vkDestroyFramebuffer(m_device, m_debugFramebuffer, nullptr);
  VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  info.renderPass      = m_debugRenderPass;
  info.attachmentCount = static_cast<uint32_t>(attachments.size());
  info.pAttachments    = attachments.data();
  info.width           = m_size.width;
  info.height          = m_size.height;
  info.layers          = 1;
  vkCreateFramebuffer(m_device, &info, nullptr, &m_debugFramebuffer);
}

void HelloVulkan::drawDebug(VkCommandBuffer cmdBuf) {
  VkDeviceSize offset{0};

  m_debug.beginLabel(cmdBuf, "Debug");

  // Dynamic Viewport
  setViewport(cmdBuf);

  std::vector<VkDescriptorSet> descSets{m_rtDescSet, m_descSet};

  // Drawing all triangles
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_debugPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_debugPipelineLayout, 0, (uint32_t)descSets.size(),
                          descSets.data(), 0, nullptr);

  for(const HelloVulkan::ObjInstance& inst : m_instances) {
    auto& model            = m_objModel[inst.objIndex];
    m_pcRaster.objIndex    = inst.objIndex;  // Telling which object is drawn
    m_pcRaster.modelMatrix = inst.transform;

    vkCmdPushConstants(cmdBuf, m_debugPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstantRaster), &m_pcRaster);
    vkCmdBindVertexBuffers(cmdBuf, 0, 1, &model.vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmdBuf, model.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmdBuf, 6, get_total_probes(), 0, 0, 0);
  }

  m_debug.endLabel(cmdBuf);
}

void HelloVulkan::createDebugPipeline() {
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster)};

  std::vector<VkDescriptorSetLayout> debugSetLayouts = {m_rtDescSetLayout, m_descSetLayout};

  // Creating the Pipeline Layout
  VkPipelineLayoutCreateInfo debugCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  debugCreateInfo.setLayoutCount           = static_cast<uint32_t>(debugSetLayouts.size());
  debugCreateInfo.pSetLayouts              = debugSetLayouts.data();
  debugCreateInfo.pushConstantRangeCount   = 1;
  debugCreateInfo.pPushConstantRanges      = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &debugCreateInfo, nullptr, &m_debugPipelineLayout);


  // Creating the Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined debugGpb(m_device, m_debugPipelineLayout, m_debugRenderPass);
  debugGpb.depthStencilState.depthTestEnable = true;
  debugGpb.addShader(nvh::loadFile("spv/debugVertex.vert.spv", true, paths, true), VK_SHADER_STAGE_VERTEX_BIT);
  debugGpb.addShader(nvh::loadFile("spv/debugFragment.frag.spv", true, paths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  debugGpb.addBindingDescription({0, sizeof(VertexObj)});

  debugGpb.addAttributeDescriptions({
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, pos))},
      {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, nrm))},
      {2, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, color))},
      {3, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, texCoord))},
  });

  // Define color blend attachment states for the two color attachments

  std::array<VkPipelineColorBlendAttachmentState, 2> colorBlendAttachments = {};
  for(auto& blendAttachment : colorBlendAttachments) {
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachment.blendEnable = VK_FALSE;
  }

  // Create color blend state create info with the correct number of attachments
  VkPipelineColorBlendStateCreateInfo debugColorBlending = {};
  debugColorBlending.sType                                 = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  debugColorBlending.logicOpEnable                         = VK_FALSE;
  debugColorBlending.attachmentCount                       = static_cast<uint32_t>(colorBlendAttachments.size());
  debugColorBlending.pAttachments                          = colorBlendAttachments.data();

  debugGpb.colorBlendState = debugColorBlending;

  m_debugPipeline = debugGpb.createPipeline();

  m_debug.setObjectName(m_debugPipeline, "Debug");
}


// Function to generate a random float between min and max
float HelloVulkan::randomFloat(float min, float max)
{
  return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

// Function to generate a random unit vector (axis of rotation)
glm::vec3 HelloVulkan::randomUnitVector()
{
  float theta  = randomFloat(0.0f, 2.0f * glm::pi<float>());  // Random angle in radians
  float z      = randomFloat(-1.0f, 1.0f);                    // Random z coordinate between -1 and 1
  float radius = sqrt(1.0f - z * z);                          // Radius in the x-y plane

  float x = radius * cos(theta);
  float y = radius * sin(theta);

  return glm::vec3(x, y, z);
}

// Function to generate a random rotation matrix
glm::mat4 HelloVulkan::randomRotationMatrix()
{
  glm::vec3 axis  = randomUnitVector();
  float     angle = randomFloat(0.0f, 2.0f * glm::pi<float>());  // Random angle in radians
  return glm::rotate(glm::mat4(1.0f), angle, axis);
}


glm::mat4 HelloVulkan::glms_euler_xyz(glm::vec3 angles)
{
  glm::mat4 dest = glm::mat4(1.0f);
  glm_euler_xyz2(angles, dest);
  return dest;
}

void HelloVulkan::glm_euler_xyz2(glm::vec3 angles, glm::mat4 dest)
{
  float cx, cy, cz, sx, sy, sz, czsx, cxcz, sysz;

  sx = sinf(angles.x);
  cx = cosf(angles.x);
  sy = sinf(angles.y);
  cy = cosf(angles.y);
  sz = sinf(angles.z);
  cz = cosf(angles.z);

  czsx = cz * sx;
  cxcz = cx * cz;
  sysz = sy * sz;

  dest[0][0] = cy * cz;
  dest[0][1] = czsx * sy + cx * sz;
  dest[0][2] = -cxcz * sy + sx * sz;
  dest[1][0] = -cy * sz;
  dest[1][1] = cxcz - sx * sysz;
  dest[1][2] = czsx + cx * sysz;
  dest[2][0] = sy;
  dest[2][1] = -cy * sx;
  dest[2][2] = cx * cy;
  dest[0][3] = 0.0f;
  dest[1][3] = 0.0f;
  dest[2][3] = 0.0f;
  dest[3][0] = 0.0f;
  dest[3][1] = 0.0f;
  dest[3][2] = 0.0f;
  dest[3][3] = 1.0f;
}
