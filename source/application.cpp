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
struct Vertex { Vec3 position; Vec3 color; };

// std140: mat4 = 64 bytes, vec4 = 16 bytes, both 16-byte aligned.
struct alignas(16) CameraUniforms {
    Mat4 view;
    Mat4 proj;
};
struct alignas(16) ObjectUniforms {
    Mat4 model;
    float tint[4];
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

// Procedural vertex color: the normalized local position, so X -> red,
// Y -> green, Z -> blue (extra task 5).
constexpr Vec3 proceduralColor(float x, float y, float z) {
    return {(x + hx) / (2.0f * hx), (y + hy) / (2.0f * hy), (z + hz) / (2.0f * hz)};
}

const std::array<Vertex, 8> vertices = {{
    {{-hx, -hy, -hz}, proceduralColor(-hx, -hy, -hz)},
    {{ hx, -hy, -hz}, proceduralColor( hx, -hy, -hz)},
    {{ hx,  hy, -hz}, proceduralColor( hx,  hy, -hz)},
    {{-hx,  hy, -hz}, proceduralColor(-hx,  hy, -hz)},
    {{-hx, -hy,  hz}, proceduralColor(-hx, -hy,  hz)},
    {{ hx, -hy,  hz}, proceduralColor( hx, -hy,  hz)},
    {{ hx,  hy,  hz}, proceduralColor( hx,  hy,  hz)},
    {{-hx,  hy,  hz}, proceduralColor(-hx,  hy,  hz)},
}};

// Counter-clockwise when seen from outside the box.
const std::array<uint16_t, 36> indices = {{
    4,5,6, 4,6,7, // front +Z
    1,0,3, 1,3,2, // back -Z
    5,1,2, 5,2,6, // right +X
    0,4,7, 0,7,3, // left -X
    7,6,2, 7,2,3, // top +Y
    0,1,5, 0,5,4, // bottom -Y
}};

constexpr int object_count = 2;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkBuffer vertex_buffer = VK_NULL_HANDLE, index_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_memory = VK_NULL_HANDLE, index_memory = VK_NULL_HANDLE;

// Two uniform buffers: one global (camera) + one per object.
// Persistently mapped; render() only memcpy's, so no map/unmap per frame.
VkBuffer camera_buffer = VK_NULL_HANDLE;
VmaAllocation camera_memory = VK_NULL_HANDLE;
VkBuffer object_buffers[object_count] = {};
VmaAllocation object_memories[object_count] = {};

VkDescriptorSetLayout camera_layout = VK_NULL_HANDLE, object_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet camera_set = VK_NULL_HANDLE;
VkDescriptorSet object_sets[object_count] = {};

CameraUniforms* camera_data = nullptr;
ObjectUniforms* object_data[object_count] = {};

// Per-object UI state: the selected object is edited in ImGui.
struct ObjectState {
    float position[3] = {0, 0, 0};
    float angles[3] = {18, 25, 0};
    float size[3] = {1, 1, 1};
    float color[3] = {1, 1, 1};
};
ObjectState objects[object_count];
int selected = 0;

bool use_perspective = true;
bool playing = false;
bool spin = false;
float laps_per_second = 0.15f, orbit_radius = 1.2f, orbit_height = 0.0f;
float phase = 0.0f; // Angle along the circular orbit, in radians.
double previous_time = -1.0;

Vec3 orbitPoint(float t) {
    return {orbit_radius * std::cos(t), orbit_height * std::sin(t), orbit_radius * std::sin(t)};
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

void updateDescriptorSet(VkDescriptorSet set, VkBuffer buffer, VkDeviceSize range) {
    auto& ctx = graphics::internal::context;
    const VkDescriptorBufferInfo buffer_info = {buffer, 0, range};
    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &buffer_info,
    };
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
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

    const VkDescriptorSetLayout layouts[] = {camera_layout, object_layout};
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2, .pSetLayouts = layouts,
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
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)},
    };
    const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attributes,
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
    // The projection matrix negates Y, which flips winding order,
    // so CLOCKWISE in clip space matches the CCW-outward index order.
    const VkPipelineRasterizationStateCreateInfo raster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
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

    if (!makeBuffer(sizeof(CameraUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
                    camera_buffer, camera_memory,
                    reinterpret_cast<void**>(&camera_data)) ||
        !makeBuffer(sizeof(ObjectUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
                    object_buffers[0], object_memories[0],
                    reinterpret_cast<void**>(&object_data[0])) ||
        !makeBuffer(sizeof(ObjectUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
                    object_buffers[1], object_memories[1],
                    reinterpret_cast<void**>(&object_data[1]))) {
        std::cerr << "Cannot create uniform buffers\n";
        return false;
    }

    const VkDescriptorSetLayoutBinding camera_binding = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    };
    const VkDescriptorSetLayoutBinding object_binding = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    const VkDescriptorSetLayoutCreateInfo camera_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &camera_binding,
    };
    const VkDescriptorSetLayoutCreateInfo object_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &object_binding,
    };
    if (vkCreateDescriptorSetLayout(ctx.device, &camera_layout_info, nullptr, &camera_layout) !=
            VK_SUCCESS ||
        vkCreateDescriptorSetLayout(ctx.device, &object_layout_info, nullptr, &object_layout) !=
            VK_SUCCESS) {
        std::cerr << "Cannot create descriptor set layouts\n";
        return false;
    }

    // 3 sets x 1 uniform buffer each = 3 descriptors of this type.
    const VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 3,
    };
    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 3, .poolSizeCount = 1, .pPoolSizes = &pool_size,
    };
    if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Cannot create descriptor pool\n";
        return false;
    }

    const VkDescriptorSetLayout set_layouts[] = {
        camera_layout, object_layout, object_layout,
    };
    const VkDescriptorSetAllocateInfo set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = object_count + 1,
        .pSetLayouts = set_layouts,
    };
    VkDescriptorSet sets[object_count + 1] = {};
    if (vkAllocateDescriptorSets(ctx.device, &set_info, sets) != VK_SUCCESS) {
        std::cerr << "Cannot allocate descriptor sets\n";
        return false;
    }
    camera_set = sets[0];
    object_sets[0] = sets[1];
    object_sets[1] = sets[2];

    updateDescriptorSet(camera_set, camera_buffer, sizeof(CameraUniforms));
    updateDescriptorSet(object_sets[0], object_buffers[0], sizeof(ObjectUniforms));
    updateDescriptorSet(object_sets[1], object_buffers[1], sizeof(ObjectUniforms));

    if (!createPipeline()) return false;
    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, camera_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, object_layout, nullptr);
    if (vertex_buffer) vmaDestroyBuffer(ctx.allocator, vertex_buffer, vertex_memory);
    if (index_buffer) vmaDestroyBuffer(ctx.allocator, index_buffer, index_memory);
    if (camera_buffer) vmaDestroyBuffer(ctx.allocator, camera_buffer, camera_memory);
    for (int i = 0; i < object_count; ++i)
        if (object_buffers[i])
            vmaDestroyBuffer(ctx.allocator, object_buffers[i], object_memories[i]);
}

void update(double time) {
    if (previous_time >= 0.0 && playing) {
        // Clamp long pauses (e.g. when moving the WSL window) so there is no jump.
        const double dt = std::fmax(0.0, std::fmin(time - previous_time, 0.05));
        phase = std::fmod(phase + static_cast<float>(dt) * laps_per_second * 2.0f * pi,
                          2.0f * pi);
    }
    previous_time = time;

    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lab 1 - Parallelepiped (variant 3)");

    const char* projections[] = {"Perspective", "Orthographic"};
    int projection_index = use_perspective ? 0 : 1;
    if (ImGui::Combo("Projection", &projection_index, projections, 2))
        use_perspective = (projection_index == 0);

    const char* names[] = {"Box 1", "Box 2"};
    ImGui::Combo("Object", &selected, names, object_count);
    ObjectState& state = objects[selected];
    ImGui::DragFloat3("Position", state.position, 0.05f, -3.0f, 3.0f);
    ImGui::DragFloat3("Rotation (degrees)", state.angles, 0.5f, -180.0f, 180.0f);
    ImGui::DragFloat3("Scale", state.size, 0.01f, 0.1f, 2.0f);
    ImGui::ColorEdit3("Color multiplier", state.color);
    ImGui::Separator();

    ImGui::Text("Orbit animation (both boxes, phase shifted by half a lap)");
    ImGui::Checkbox("Play / pause motion", &playing);
    ImGui::SliderFloat("Speed (laps/second)", &laps_per_second, 0.03f, 0.6f, "%.2f");
    ImGui::SliderFloat("Radius", &orbit_radius, 0.0f, 2.0f, "%.2f");
    ImGui::SliderFloat("Height", &orbit_height, 0.0f, 1.0f, "%.2f");
    ImGui::Checkbox("Also rotate boxes", &spin);
    if (ImGui::Button("Restart phase")) phase = 0.0f;
    ImGui::SameLine();
    if (ImGui::Button("Reset all")) {
        for (ObjectState& o : objects) {
            o.position[0] = o.position[1] = o.position[2] = 0.0f;
            o.angles[0] = 18.0f; o.angles[1] = 25.0f; o.angles[2] = 0.0f;
            o.size[0] = o.size[1] = o.size[2] = 1.0f;
            o.color[0] = o.color[1] = o.color[2] = 1.0f;
        }
        selected = 0;
        phase = 0.0f; playing = false; spin = false; use_perspective = true;
        laps_per_second = 0.15f; orbit_radius = 1.2f; orbit_height = 0.0f;
    }
    ImGui::TextDisabled("Vertex colors: normalized local position (procedural).");
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
        {.color = {.float32 = {0.07f, 0.10f, 0.15f, 1.0f}}},
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

    const Mat4 view = translation({0, 0, -6.0f});
    const Mat4 projection = use_perspective ? perspective(aspect) : orthographic(aspect);
    std::memcpy(&camera_data->view, &view, sizeof(Mat4));
    std::memcpy(&camera_data->proj, &projection, sizeof(Mat4));
    vmaFlushAllocation(ctx.allocator, camera_memory, 0, sizeof(CameraUniforms));

    for (int i = 0; i < object_count; ++i) {
        const ObjectState& state = objects[i];
        const float orbit_shift = phase + pi * static_cast<float>(i);
        const Vec3 point = orbitPoint(orbit_shift);
        // Orbit is added on top of the manual position from the UI.
        const Vec3 position = {state.position[0] + point.x, state.position[1] + point.y,
                               state.position[2] + point.z};
        const Mat4 model = multiply(
            translation(position),
            multiply(rotation(state.angles[0],
                              state.angles[1] + (spin ? orbit_shift * 180.0f / pi : 0.0f),
                              state.angles[2]),
                     scale({state.size[0], state.size[1], state.size[2]})));
        std::memcpy(&object_data[i]->model, &model, sizeof(Mat4));
        object_data[i]->tint[0] = state.color[0];
        object_data[i]->tint[1] = state.color[1];
        object_data[i]->tint[2] = state.color[2];
        object_data[i]->tint[3] = 1.0f;
        vmaFlushAllocation(ctx.allocator, object_memories[i], 0, sizeof(ObjectUniforms));
    }

    const VkViewport viewport = {0, 0, static_cast<float>(ctx.swapchain_extent.width),
                                 static_cast<float>(ctx.swapchain_extent.height), 0, 1};
    const VkRect2D scissor = {{0, 0}, ctx.swapchain_extent};
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    // Set 0 (camera) is shared by both objects, set 1 differs per object.
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout, 0, 1, &camera_set, 0, nullptr);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, &offset);
    vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT16);

    for (int i = 0; i < object_count; ++i) {
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout, 1, 1, &object_sets[i], 0, nullptr);
        vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
