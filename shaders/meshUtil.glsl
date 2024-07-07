struct MeshDraw {

  // x = diffuse index, y = roughness index, z = normal index, w = occlusion index.
  // Occlusion and roughness are encoded in the same texture
  uvec4 textures;
  vec4  emissive;
  vec4  base_color_factor;
  vec4  metallic_roughness_occlusion_factor;

  uint  flags;
  float alpha_cutoff;
  uint  vertexOffset;  // == meshes[meshIndex].vertexOffset, helps data locality in mesh shader
  uint  meshIndex;

  uint meshlet_offset;
  uint meshlet_count;
  uint meshlet_index_count;
  uint pad001;

  uint64_t position_buffer;
  uint64_t uv_buffer;
  uint64_t index_buffer;
  uint64_t normals_buffer;
};