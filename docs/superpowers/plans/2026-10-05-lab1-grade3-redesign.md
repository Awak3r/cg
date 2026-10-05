# Lab 1 Grade-3 Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-subagent-driven-development (recommended) or superpowers-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce lab1 to the grade-3 baseline (single parallelepiped, MVP, depth test, back-face culling) while fixing the winding bug so the box renders correctly and looks polished.

**Architecture:** Rewrite `application.cpp` down to a single uniform buffer (model/view/proj), one descriptor set, one draw call, static camera; fix `frontFace` to `CCW`; simplify both shaders (single uniform block, material constant in fragment shader). `graphics_internal.*`, `main.cpp`, `CMakeLists.txt` stay as-is (MSAA 4x and WSL fixes are intentionally kept).

**Tech Stack:** Vulkan, GLFW, ImGui, VMA, C++20, CMake presets, glslc.

**Spec:** `docs/superpowers/specs/2026-10-05-lab1-grade3-redesign.md`

**Build/verify commands (run from `/home/bogdanoff/study/university/cg/lab1`):**

```bash
cmake --preset debug && cmake --build build-debug --parallel
cmake --preset release && cmake --build build-release --parallel
./run-wsl.sh    # launches app from project root; validation output goes to console/log
```

This repo is NOT a git repo — there are no commit steps. Verification is compile + run + log inspection instead of tests.

---

### Task 1: Rewrite `source/application.cpp` to grade-3 baseline

**Files:**
- Modify (full rewrite): `source/application.cpp`
- Unchanged: `source/application.hpp`, `source/graphics_internal.hpp`, `source/graphics_internal.cpp`, `source/main.cpp`

- [ ] **Step 1: Replace the entire content of `source/application.cpp` with the code below**

```cpp
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
struct Vertex { Vec3 position; Vec3 normal; };

// std140: each mat4 is 16-byte aligned, 64 bytes.
struct alignas(16) GlobalUniforms {
    Mat4 model;
    Mat4 view;
    Mat4 proj;
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

// Flat shading: every face carries its own outward normal, hence 24 vertices
// instead of the 8 unique corners.
constexpr std::array<Vertex, 24> makeVertices() {
    std::array<Vertex, 24> result{};
    for (int face = 0; face < 6; ++face)
        for (int corner = 0; corner < 4; ++corner)
            result[face * 4 + corner] = {faceCorners[face][corner], faceNormals[face]};
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

// One global uniform buffer: model/view/proj. Persistently mapped; render()
// only memcpy's, so there is no map/unmap per frame.
VkBuffer uniform_buffer = VK_NULL_HANDLE;
VmaAllocation uniform_memory = VK_NULL_HANDLE;
GlobalUniforms* uniform_data = nullptr;

VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

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
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
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
    static double previous_time = -1.0;
    static double fps = 0.0;
    if (previous_time >= 0.0) {
        const double dt = time - previous_time;
        if (dt > 0.0) fps = fps <= 0.0 ? 1.0 / dt : fps * 0.9 + (1.0 / dt) * 0.1;
    }
    previous_time = time;

    ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lab 1 - Parallelepiped (variant 3)");
    ImGui::Text("Vulkan + GLFW + ImGui");
    ImGui::Text("2.0 x 1.4 x 1.0, flat shading");
    ImGui::Text("FPS: %.0f", fps);
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

    const Mat4 model = rotation(18.0f, 25.0f, 0.0f);
    const Mat4 view = translation({0, 0, -5.0f});
    const Mat4 projection = perspective(aspect);
    std::memcpy(&uniform_data->model, &model, sizeof(Mat4));
    std::memcpy(&uniform_data->view, &view, sizeof(Mat4));
    std::memcpy(&uniform_data->proj, &projection, sizeof(Mat4));
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
```

- [ ] **Step 2: Verify the file compiles later (covered by Task 4 build step). Quick grep sanity check now**

Run: `grep -c "vkDestroy\|vmaDestroyBuffer" source/application.cpp`
Expected: at least 5 destroy calls (pipeline, pipeline layout, descriptor pool, descriptor set layout, 3 vma buffers).

---

### Task 2: Rewrite `shaders/box.vert` — single uniform block

**Files:**
- Modify (full rewrite): `shaders/box.vert`

- [ ] **Step 1: Replace the entire content of `shaders/box.vert` with:**

```glsl
#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 0) out vec3 worldNormal;

layout(set = 0, binding = 0) uniform GlobalUniforms {
	mat4 model;
	mat4 view;
	mat4 proj;
} g;

void main() {
	gl_Position = g.proj * g.view * g.model * vec4(inPosition, 1.0);
	// Inverse-transpose so normals survive any model matrix.
	worldNormal = mat3(transpose(inverse(g.model))) * inNormal;
}
```

---

### Task 3: Rewrite `shaders/box.frag` — material constant, no tint

**Files:**
- Modify (full rewrite): `shaders/box.frag`

- [ ] **Step 1: Replace the entire content of `shaders/box.frag` with:**

```glsl
#version 450
layout(location = 0) in vec3 worldNormal;
layout(location = 0) out vec4 outColor;

// Key light from above-right-front + soft fill from below-left, both fixed
// in world space. Fill keeps the opposite faces readable instead of black.
const vec3 keyLightDirection = normalize(vec3(0.3, 0.85, 0.45));
const vec3 fillLightDirection = normalize(vec3(-0.45, -0.25, 0.35));
const float keyIntensity = 0.8;
const float fillIntensity = 0.35;
const vec3 material = vec3(0.35, 0.55, 0.85);

void main() {
	// Lambert: hemisphere ambient (brighter for up-facing surfaces) + diffuse.
	const vec3 normal = normalize(worldNormal);
	const float hemi = 0.5 + 0.5 * normal.y;
	const vec3 ambient = mix(vec3(0.16), vec3(0.30), hemi);
	const float diffuse = keyIntensity * max(dot(normal, keyLightDirection), 0.0) +
	                      fillIntensity * max(dot(normal, fillLightDirection), 0.0);
	outColor = vec4(material * min(ambient + diffuse, vec3(1.0)), 1.0);
}
```

---

### Task 4: Rebuild both presets

**Files:** none modified; build only.

- [ ] **Step 1: Reconfigure + build debug**

Run:
```bash
cmake --preset debug && cmake --build build-debug --parallel
```
Expected: build finishes, `shaders/box.vert.spv` and `shaders/box.frag.spv` regenerated, no compiler errors.

- [ ] **Step 2: Reconfigure + build release**

Run:
```bash
cmake --preset release && cmake --build build-release --parallel
```
Expected: build finishes with no errors.

---

### Task 5: Run and verify validation-clean

**Files:** none modified; runtime verification.

- [ ] **Step 1: Launch the app from project root, capture logs**

Run:
```bash
pkill -f vulkan-starter-app 2>/dev/null; sleep 1
(./build-release/vulkan-starter-app > /tmp/opencode/cg/run.log 2>&1 &)
sleep 5
pgrep -a vulkan-starter-app
```
Expected: process running; log file contains no "Validation" / "validation layer" error lines.

- [ ] **Step 2: Check the log for validation errors**

Run: `grep -iE "validation|error|VUID" /tmp/opencode/cg/run.log || echo CLEAN`
Expected: `CLEAN` (no validation output).

- [ ] **Step 3: Close the app cleanly and verify shutdown**

Run:
```bash
pkill -INT -f vulkan-starter-app 2>/dev/null || pkill -f vulkan-starter-app
sleep 2
grep -iE "validation|error|VUID" /tmp/opencode/cg/run.log || echo CLEAN_SHUTDOWN
```
Expected: process exits; `CLEAN_SHUTDOWN` printed (no destroy-time validation errors).

- [ ] **Step 4: Visual acceptance by the user**

Ask the user to look at the window: the box should show exactly 3 faces (front, left, top) with distinct lighting, no interior visible, smooth edges (MSAA 4x). CLI screenshots come out black in WSLg, so the user confirms visually.

---

### Task 6: Rewrite `README.md` for the grade-3 scope

**Files:**
- Modify (full rewrite): `README.md`

- [ ] **Step 1: Replace the entire content of `README.md` with:**

```markdown
# Лабораторная работа 1 — Параллелепипед (вариант 3)

Vulkan + GLFW + ImGui, C++20. Реализация на базе стартового репозитория
[vkadeemerr/vulkan-starter-app](https://github.com/vkadeemerr/vulkan-starter-app).

## Вариант

**Вариант 3: параллелепипед** — прямоугольный параллелепипед 2.0 x 1.4 x 1.0
(8 вершин углов, 24 вершины с нормалями граней, 12 треугольников, 36 индексов),
центр в начале координат.

## Сборка

Нужны: компилятор C++20, CMake 3.20+, Vulkan SDK (`glslc` в PATH).

```bash
cmake --preset debug          # Linux (GCC/Clang)
cmake --build build-debug --parallel
```

Windows: `cmake --preset msvc-debug` (Visual Studio) или `cmake --preset mingw-debug` (MinGW),
затем `cmake --build build-debug --parallel`.

## Запуск

Рабочая директория — корень проекта (пути к шейдерам относительные):

```bash
./build-debug/vulkan-starter-app
```

Под WSL: `./run-wsl.sh` (ждёт готовности WSLg и запускает приложение).

## Что реализовано (базовый уровень)

- Окно GLFW, инициализация Vulkan, graphics pipeline.
- Параллелепипед: vertex buffer + index buffer (VMA), рисование `vkCmdDrawIndexed`.
- MVP-матрицы: model/view/projection через uniform buffer (`std140`, 192 байта).
- Перспективная проекция (45°), вид подобран так, что видны три грани.
- Depth test (`VK_COMPARE_OP_LESS`, clear depth = 1.0) — внутренность не видна.
- Back-face culling (`VK_CULL_MODE_BACK_BIT`, лицевые — CCW-наружу грани).
- Dynamic viewport/scissor (корректный resize окна).
- Flat-освещение граней в фрагментном шейдере (нормали — outward нормаль грани).
- MSAA 4x — сглаженные рёбра.
- ImGui-окно с информацией о лабораторной (интеграция UI-библиотеки).
- Все `vkCreate*` имеют парные `vkDestroy*` / `vmaDestroyBuffer`,
  validation layers чистые.

## Шейдеры

`shaders/box.vert`, `shaders/box.frag` — компилируются через `glslc` на этапе сборки
(CMake target `shaders`).

## Управление

Окно закрывается крестиком. ImGui-панель информационная (название, вариант, FPS).
```

- [ ] **Step 2: Final full rebuild to confirm nothing broke**

Run: `cmake --build build-debug --parallel && cmake --build build-release --parallel`
Expected: up-to-date / builds clean.

---

## Self-Review Results

1. **Spec coverage:**
   - §2 winding fix → Task 1 (`frontFace = COUNTER_CLOCKWISE`, comment updated).
   - §3 deletions → Task 1 rewrite (all 4/5-level code gone; only GlobalUniforms + 1 set remain).
   - §4.1–4.5 math/geometry/uniform/descriptors/scene → Task 1 (view -5, rotation 18/25, perspective 45).
   - §4.6 pipeline (single change frontFace) → Task 1.
   - §4.7 ImGui info window + FPS → Task 1 `update()`.
   - §4.8 render() → Task 1.
   - §5 shaders → Tasks 2, 3.
   - §6 README → Task 6.
   - §8 verification → Tasks 4, 5.
   - §3 "keep" list (MSAA, WSL, glslc changes) → no task touches those files, as required.
2. **Placeholder scan:** none — every step has complete code or exact commands.
3. **Type consistency:** `GlobalUniforms` (model/view/proj) matches in Task 1 C++ and Task 2 GLSL; `set = 0, binding = 0` matches layout binding 0; descriptor uses `sizeof(GlobalUniforms)`; `msaa_samples` referenced exactly as declared in `graphics_internal.hpp`; `application.hpp` API (initialize/shutdown/update/render) unchanged.
