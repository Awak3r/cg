#include "application.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

namespace application {
namespace {

constexpr float pi = 3.14159265f;

struct Vec3 { float x, y, z; };
struct Mat4 { float a[16]{}; }; // Column-major, like GLSL mat4.
struct Vertex { Vec3 position; Vec3 normal; Vec3 color; };

// std140: mat4 = 64 bytes, vec4 = 16 bytes, both 16-byte aligned.
struct alignas(16) GlobalUniforms {
    Mat4 model;
    Mat4 view;
    Mat4 proj;
    float tint[4]; // vec4 keeps std140 alignment simple.
};

Mat4 identity() {
    Mat4 m{};
    for (int i = 0; i < 4; ++i) m.a[i * 5] = 1.0f;
    return m;
}

Mat4 multiply(const Mat4& x, const Mat4& y) {
    Mat4 result{};
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k)
                result.a[col * 4 + row] += x.a[k * 4 + row] * y.a[col * 4 + k];
    return result;
}

Mat4 translation(Vec3 p) {
    Mat4 m = identity();
    m.a[12] = p.x; m.a[13] = p.y; m.a[14] = p.z;
    return m;
}

Mat4 scale(Vec3 s) {
    Mat4 m = identity();
    m.a[0] = s.x; m.a[5] = s.y; m.a[10] = s.z;
    return m;
}

Mat4 rotation(float x, float y, float z) {
    x *= pi / 180.0f; y *= pi / 180.0f; z *= pi / 180.0f;
    Mat4 rx = identity(), ry = identity(), rz = identity();
    rx.a[5] = std::cos(x); rx.a[6] = std::sin(x);
    rx.a[9] = -std::sin(x); rx.a[10] = std::cos(x);
    ry.a[0] = std::cos(y); ry.a[2] = -std::sin(y);
    ry.a[8] = std::sin(y); ry.a[10] = std::cos(y);
    rz.a[0] = std::cos(z); rz.a[1] = std::sin(z);
    rz.a[4] = -std::sin(z); rz.a[5] = std::cos(z);
    return multiply(rz, multiply(ry, rx));
}

Mat4 perspective(float aspect) {
    const float f = 1.0f / std::tan(45.0f * pi / 360.0f);
    const float near = 0.1f, far = 100.0f;
    Mat4 m{};
    m.a[0] = f / aspect;
    m.a[5] = -f; // Vulkan's screen Y points down.
    m.a[10] = far / (near - far);
    m.a[11] = -1.0f;
    m.a[14] = far * near / (near - far);
    return m;
}

Mat4 orthographic(float aspect) {
    const float h = 3.0f, w = h * aspect;
    const float near = 0.1f, far = 100.0f;
    Mat4 m = identity();
    m.a[0] = 1.0f / w;
    m.a[5] = -1.0f / h;
    m.a[10] = 1.0f / (near - far);
    m.a[14] = near / (near - far);
    return m;
}

// Parallelepiped with half extents hx, hy, hz centered at the origin.
constexpr float hx = 1.0f, hy = 0.7f, hz = 0.5f;

// Outward normals of the 6 faces, in the same order as the corners below.
constexpr Vec3 faceNormals[6] = {
    {0, 0,  1}, // front +Z
    {0, 0, -1}, // back -Z
    { 1, 0, 0}, // right +X
    {-1, 0, 0}, // left -X
    {0,  1, 0}, // top +Y
    {0, -1, 0}, // bottom -Y
};

// 4 corners per face, counter-clockwise when seen from outside the box.
constexpr Vec3 faceCorners[6][4] = {
    {{-hx, -hy,  hz}, { hx, -hy,  hz}, { hx,  hy,  hz}, {-hx,  hy,  hz}},
    {{ hx, -hy, -hz}, {-hx, -hy, -hz}, {-hx,  hy, -hz}, { hx,  hy, -hz}},
    {{ hx, -hy,  hz}, { hx, -hy, -hz}, { hx,  hy, -hz}, { hx,  hy,  hz}},
    {{-hx, -hy, -hz}, {-hx, -hy,  hz}, {-hx,  hy,  hz}, {-hx,  hy, -hz}},
    {{-hx,  hy,  hz}, { hx,  hy,  hz}, { hx,  hy, -hz}, {-hx,  hy, -hz}},
    {{-hx, -hy, -hz}, { hx, -hy, -hz}, { hx, -hy,  hz}, {-hx, -hy,  hz}},
};

// Procedural vertex color: normalized local position, X->R, Y->G, Z->B.
constexpr Vec3 vertexColor(Vec3 p) {
    return {(p.x + hx) / (2.0f * hx), (p.y + hy) / (2.0f * hy), (p.z + hz) / (2.0f * hz)};
}

// Flat shading: every face carries its own outward normal, hence 24 vertices
// instead of the 8 unique corners.
constexpr std::array<Vertex, 24> makeVertices() {
    std::array<Vertex, 24> result{};
    for (int face = 0; face < 6; ++face)
        for (int corner = 0; corner < 4; ++corner) {
            const Vec3 p = faceCorners[face][corner];
            result[face * 4 + corner] = {p, faceNormals[face], vertexColor(p)};
        }
    return result;
}

constexpr std::array<Vertex, 24> vertices = makeVertices();

// Two CCW triangles per face over the 4 duplicated corners.
constexpr std::array<uint16_t, 36> makeIndices() {
    std::array<uint16_t, 36> result{};
    for (int face = 0; face < 6; ++face) {
        const uint16_t base = static_cast<uint16_t>(face * 4);
        result[face * 6 + 0] = base;
        result[face * 6 + 1] = static_cast<uint16_t>(base + 1);
        result[face * 6 + 2] = static_cast<uint16_t>(base + 2);
        result[face * 6 + 3] = base;
        result[face * 6 + 4] = static_cast<uint16_t>(base + 2);
        result[face * 6 + 5] = static_cast<uint16_t>(base + 3);
    }
    return result;
}

constexpr std::array<uint16_t, 36> indices = makeIndices();

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkBuffer vertex_buffer = VK_NULL_HANDLE, index_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_memory = VK_NULL_HANDLE, index_memory = VK_NULL_HANDLE;

// One global uniform buffer: model/view/proj/tint. Persistently mapped;
// render() only memcpy's, so there is no map/unmap per frame.
VkBuffer uniform_buffer = VK_NULL_HANDLE;
VmaAllocation uniform_memory = VK_NULL_HANDLE;
GlobalUniforms* uniform_data = nullptr;

VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

// Object transform state, edited from the ImGui panel.
struct ObjectState {
    float position[3] = {0, 0, 0};
    float angles[3] = {18, 25, 0};
    float size[3] = {1, 1, 1};
    float color[3] = {1.0f, 1.0f, 1.0f}; // Tint multiplied onto vertex color.
};
ObjectState state;

bool use_perspective = true;
bool playing = false;
bool spin = false;
float laps_per_second = 0.15f, orbit_radius = 1.2f;
float phase = 0.0f; // Angle along the circular orbit, in radians.
double previous_time = -1.0;

Vec3 orbitPoint(float t) {
    return {orbit_radius * std::cos(t), 0.0f, orbit_radius * std::sin(t)};
}

bool makeBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, const void* data,
                VkBuffer& buffer, VmaAllocation& allocation, void** mapped) {
    auto& ctx = graphics::internal::context;
    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo memory = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VmaAllocationInfo result{};
    if (vmaCreateBuffer(ctx.allocator, &info, &memory, &buffer, &allocation, &result) != VK_SUCCESS)
        return false;
    if (data) std::memcpy(result.pMappedData, data, static_cast<size_t>(bytes));
    if (mapped) *mapped = result.pMappedData;
    vmaFlushAllocation(ctx.allocator, allocation, 0, bytes);
    return true;
}

std::vector<char> readFile(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto length = file.tellg();
    if (length <= 0 || length % 4 != 0) return {};
    std::vector<char> bytes(static_cast<size_t>(length));
    file.seekg(0);
    file.read(bytes.data(), length);
    return bytes;
}

VkShaderModule shaderModule(const char* path) {
    const auto bytes = readFile(path);
    if (bytes.empty()) {
        std::cerr << "Cannot read shader " << path << " (run from project root)\n";
        return VK_NULL_HANDLE;
    }
    const VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = bytes.size(),
        .pCode = reinterpret_cast<const uint32_t*>(bytes.data()),
    };
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS)
        std::cerr << "Cannot create shader module: " << path << '\n';
    return module;
}

bool createPipeline() {
    auto& ctx = graphics::internal::context;

    VkShaderModule vert = shaderModule("shaders/box.vert.spv");
    VkShaderModule frag = shaderModule("shaders/box.frag.spv");
    if (!vert || !frag) return false;

    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &set_layout,
    };
    if (vkCreatePipelineLayout(ctx.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS)
        return false;

    const VkPipelineShaderStageCreateInfo stages[] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag, .pName = "main"},
    };
    const VkVertexInputBindingDescription binding = {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription attributes[] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
        {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)},
    };
    const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attributes,
    };
    const VkPipelineInputAssemblyStateCreateInfo assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1,
    };
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dynamic_states,
    };
    // Faces are wound CCW seen from outside; the projection only negates Y,
    // and Vulkan measures facing in framebuffer coordinates (Y down), which
    // cancels the flip: the outward faces remain CCW on screen.
    const VkPipelineRasterizationStateCreateInfo raster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = graphics::internal::msaa_samples,
    };
    const VkPipelineDepthStencilStateCreateInfo depth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };
    const VkPipelineColorBlendAttachmentState blend_attachment = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &blend_attachment,
    };
    const VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &raster,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth,
        .pColorBlendState = &blend,
        .pDynamicState = &dynamic,
        .layout = pipeline_layout, .renderPass = ctx.render_pass,
    };
    const VkResult result = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1,
                                                       &pipeline_info, nullptr, &pipeline);
    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
    if (result != VK_SUCCESS) {
        std::cerr << "Cannot create graphics pipeline: " << result << '\n';
        return false;
    }
    return true;
}

void resetState() {
    state.position[0] = state.position[1] = state.position[2] = 0.0f;
    state.angles[0] = 18.0f; state.angles[1] = 25.0f; state.angles[2] = 0.0f;
    state.size[0] = state.size[1] = state.size[2] = 1.0f;
    state.color[0] = state.color[1] = state.color[2] = 1.0f;
    use_perspective = true;
    playing = false; spin = false;
    laps_per_second = 0.15f; orbit_radius = 1.2f;
    phase = 0.0f;
}

} // namespace

bool initialize() {
    auto& ctx = graphics::internal::context;

    if (!makeBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data(),
                    vertex_buffer, vertex_memory, nullptr) ||
        !makeBuffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.data(),
                    index_buffer, index_memory, nullptr)) {
        std::cerr << "Cannot create vertex/index buffer\n";
        return false;
    }

    if (!makeBuffer(sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
                    uniform_buffer, uniform_memory,
                    reinterpret_cast<void**>(&uniform_data))) {
        std::cerr << "Cannot create uniform buffer\n";
        return false;
    }

    const VkDescriptorSetLayoutBinding binding = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    const VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &binding,
    };
    if (vkCreateDescriptorSetLayout(ctx.device, &layout_info, nullptr, &set_layout) != VK_SUCCESS) {
        std::cerr << "Cannot create descriptor set layout\n";
        return false;
    }

    const VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1,
    };
    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size,
    };
    if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Cannot create descriptor pool\n";
        return false;
    }

    const VkDescriptorSetAllocateInfo set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1, .pSetLayouts = &set_layout,
    };
    if (vkAllocateDescriptorSets(ctx.device, &set_info, &descriptor_set) != VK_SUCCESS) {
        std::cerr << "Cannot allocate descriptor set\n";
        return false;
    }

    const VkDescriptorBufferInfo buffer_info = {uniform_buffer, 0, sizeof(GlobalUniforms)};
    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &buffer_info,
    };
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);

    if (!createPipeline()) return false;
    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, set_layout, nullptr);
    if (vertex_buffer) vmaDestroyBuffer(ctx.allocator, vertex_buffer, vertex_memory);
    if (index_buffer) vmaDestroyBuffer(ctx.allocator, index_buffer, index_memory);
    if (uniform_buffer) vmaDestroyBuffer(ctx.allocator, uniform_buffer, uniform_memory);
}

void update(double time) {
    static double fps = 0.0;
    if (previous_time >= 0.0) {
        const double dt = time - previous_time;
        if (dt > 0.0) fps = fps <= 0.0 ? 1.0 / dt : fps * 0.9 + (1.0 / dt) * 0.1;
        if (playing) {
            // Clamp long stalls (e.g. a dragged WSL window) so there is no jump.
            const double step = std::fmax(0.0, std::fmin(dt, 0.05));
            phase = std::fmod(phase + static_cast<float>(step) * laps_per_second * 2.0f * pi,
                              2.0f * pi);
        }
    }
    previous_time = time;

    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lab 1 - Parallelepiped (variant 3)");
    ImGui::Text("FPS: %.0f", fps);
    ImGui::Separator();

    const char* projections[] = {"Perspective", "Orthographic"};
    int projection_index = use_perspective ? 0 : 1;
    if (ImGui::Combo("Projection", &projection_index, projections, 2))
        use_perspective = (projection_index == 0);

    ImGui::DragFloat3("Position", state.position, 0.05f, -3.0f, 3.0f);
    ImGui::DragFloat3("Rotation (degrees)", state.angles, 0.5f, -180.0f, 180.0f);
    ImGui::DragFloat3("Scale", state.size, 0.01f, 0.1f, 2.0f);
    ImGui::ColorEdit3("Tint", state.color);
    ImGui::Separator();

    ImGui::Text("Orbit animation");
    ImGui::Checkbox("Play / pause motion", &playing);
    ImGui::SliderFloat("Speed (laps/second)", &laps_per_second, 0.03f, 0.6f, "%.2f");
    ImGui::SliderFloat("Radius", &orbit_radius, 0.0f, 2.0f, "%.2f");

    ImGui::Checkbox("Also rotate along orbit", &spin);
    if (ImGui::Button("Restart phase")) phase = 0.0f;
    ImGui::SameLine();
    if (ImGui::Button("Reset all")) resetState();
    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;
    vkResetCommandBuffer(fd.command_buffer, 0);
    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(fd.command_buffer, &begin);

    const VkClearValue clear[] = {
        {.color = {.float32 = {0.10f, 0.13f, 0.18f, 1.0f}}},
        {.depthStencil = {1.0f, 0}},
    };
    const VkRenderPassBeginInfo pass = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = ctx.render_pass, .framebuffer = fd.framebuffer,
        .renderArea = {.extent = ctx.swapchain_extent},
        .clearValueCount = 2, .pClearValues = clear,
    };
    vkCmdBeginRenderPass(fd.command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);

    const float aspect = static_cast<float>(ctx.swapchain_extent.width) / ctx.swapchain_extent.height;

    // Orbit is added on top of the manual position from the UI.
    const Vec3 point = orbitPoint(phase);
    const Vec3 position = {state.position[0] + point.x, state.position[1] + point.y,
                           state.position[2] + point.z};
    const Mat4 model = multiply(
        translation(position),
        multiply(rotation(state.angles[0],
                          state.angles[1] + (spin ? phase * 180.0f / pi : 0.0f),
                          state.angles[2]),
                 scale({state.size[0], state.size[1], state.size[2]})));
    const Mat4 view = translation({0, 0, -5.0f});
    const Mat4 projection = use_perspective ? perspective(aspect) : orthographic(aspect);
    std::memcpy(&uniform_data->model, &model, sizeof(Mat4));
    std::memcpy(&uniform_data->view, &view, sizeof(Mat4));
    std::memcpy(&uniform_data->proj, &projection, sizeof(Mat4));
    uniform_data->tint[0] = state.color[0];
    uniform_data->tint[1] = state.color[1];
    uniform_data->tint[2] = state.color[2];
    uniform_data->tint[3] = 1.0f;
    vmaFlushAllocation(ctx.allocator, uniform_memory, 0, sizeof(GlobalUniforms));

    const VkViewport viewport = {0, 0, static_cast<float>(ctx.swapchain_extent.width),
                                 static_cast<float>(ctx.swapchain_extent.height), 0, 1};
    const VkRect2D scissor = {{0, 0}, ctx.swapchain_extent};
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, &offset);
    vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
