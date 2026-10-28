#include "application.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <cstdio>
#include <iostream>
#include <vector>

#include <imgui.h>

namespace application {
namespace {

constexpr float pi = 3.14159265f;

// Максимальное число источников в storage-буферах (лимит — не в шейдере,
// поэтому массивы в GLSL остаются динамической длины).
constexpr size_t max_lights = 256;

struct Vec3 { float x, y, z; };
struct Mat4 { float a[16]{}; }; // Column-major, like GLSL mat4.
struct Vertex { Vec3 position; Vec3 normal; Vec3 color; };

// std140-зеркало блока SceneUniforms из object.vert/object.frag.
// Только mat4 и vec4 — выравнивание совпадает с std430 в storage-буферах.
struct alignas(16) SceneUniforms {
	Mat4 view;                       // 64
	Mat4 proj;                       // 64
	float cameraPosition[4];         // 16: xyz, w — запас
	float ambientLight[4];           // 16: rgb, a — запас
	float dirLightDirection[4];      // 16: xyz — направление (normalized)
	float dirLightColorIntensity[4]; // 16: rgb, a — интенсивность
	float lightCounts[4];            // 16: x — точечные, y — прожекторные, z — exposure
};                                   // итого 208 байт

// std140-зеркало блока ModelUniforms.
struct alignas(16) ModelUniforms {
	Mat4 model;                   // 64
	float albedoColor[4];         // 16: rgb, w — запас
	float specularShininess[4];   // 16: rgb — цвет блика, w — коэффициент блеска
	float flags[4];               // 16: x — шахматный пол (отключён), yzw — запас
};                                // итого 128 байт

// std430-зеркало struct PointLight из object.frag.
struct alignas(16) GpuPointLight {
	float position[4];            // xyz, w — запас
	float colorIntensity[4];      // rgb, a — интенсивность
};
static_assert(sizeof(GpuPointLight) == 32, "std430 mirror broken");

// std430-зеркало struct SpotLight из object.frag.
struct alignas(16) GpuSpotLight {
	float position[4];            // xyz, w — запас
	float direction[4];           // xyz — ось конуса (normalized)
	float colorIntensity[4];      // rgb, a — интенсивность
	float angles[4];              // x — cos(inner), y — cos(outer)
};
static_assert(sizeof(GpuSpotLight) == 64, "std430 mirror broken");

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

Mat4 rotationX(float radians) {
	Mat4 m = identity();
	m.a[5] = std::cos(radians); m.a[6] = std::sin(radians);
	m.a[9] = -std::sin(radians); m.a[10] = std::cos(radians);
	return m;
}

Mat4 rotationY(float radians) {
	Mat4 m = identity();
	m.a[0] = std::cos(radians); m.a[2] = -std::sin(radians);
	m.a[8] = std::sin(radians); m.a[10] = std::cos(radians);
	return m;
}

Mat4 rotationZ(float radians) {
	Mat4 m = identity();
	m.a[0] = std::cos(radians); m.a[1] = std::sin(radians);
	m.a[4] = -std::sin(radians); m.a[5] = std::cos(radians);
	return m;
}

// Эйлерово вращение в градусах: R = Rz * Ry * Rx.
Mat4 rotation(float xDegrees, float yDegrees, float zDegrees) {
	return multiply(rotationZ(zDegrees * pi / 180.0f),
	       multiply(rotationY(yDegrees * pi / 180.0f),
	                rotationX(xDegrees * pi / 180.0f)));
}

// Перспективная проекция с настраиваемым вертикальным FOV (в градусах).
Mat4 perspective(float fovDegrees, float aspect) {
	const float f = 1.0f / std::tan(fovDegrees * pi / 360.0f);
	const float near = 0.05f, far = 120.0f;
	Mat4 m{};
	m.a[0] = f / aspect;
	m.a[5] = -f; // Vulkan's screen Y points down.
	m.a[10] = far / (near - far);
	m.a[11] = -1.0f;
	m.a[14] = far * near / (near - far);
	return m;
}

// Матрица вида = обратная матрица модели камеры.
// M_cam = T(pos) * Ry(yaw) * Rx(pitch), значит
// V = Rx(-pitch) * Ry(-yaw) * T(-pos) — без общего инверса 4x4.
Mat4 viewFromCamera(const Vec3 position, const float yawDegrees, const float pitchDegrees) {
	const Mat4 inverseRotation =
	    multiply(rotationX(-pitchDegrees * pi / 180.0f),
	             rotationY(-yawDegrees * pi / 180.0f));
	return multiply(inverseRotation, translation({-position.x, -position.y, -position.z}));
}

Vec3 normalized(const Vec3 v) {
	const float length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	if (length < 0.0001f) return {0.0f, 0.0f, 0.0f};
	return {v.x / length, v.y / length, v.z / length};
}

// ---- Геометрия варианта 3: параллелепипед ------------------------------

// Полуразмеры параллелепипеда, в центре координат.
constexpr float hx = 1.0f, hy = 0.7f, hz = 0.5f;

// Наружные нормали 6 граней в порядке следования углов ниже.
constexpr Vec3 faceNormals[6] = {
	{0, 0,  1}, // front +Z
	{0, 0, -1}, // back -Z
	{ 1, 0, 0}, // right +X
	{-1, 0, 0}, // left -X
	{0,  1, 0}, // top +Y
	{0, -1, 0}, // bottom -Y
};

// 4 угла на грань, против часовой при взгляде снаружи коробки.
constexpr Vec3 faceCorners[6][4] = {
	{{-hx, -hy,  hz}, { hx, -hy,  hz}, { hx,  hy,  hz}, {-hx,  hy,  hz}},
	{{ hx, -hy, -hz}, {-hx, -hy, -hz}, {-hx,  hy, -hz}, { hx,  hy, -hz}},
	{{ hx, -hy,  hz}, { hx, -hy, -hz}, { hx,  hy, -hz}, { hx,  hy,  hz}},
	{{-hx, -hy, -hz}, {-hx, -hy,  hz}, {-hx,  hy,  hz}, {-hx,  hy, -hz}},
	{{-hx,  hy,  hz}, { hx,  hy,  hz}, { hx,  hy, -hz}, {-hx,  hy, -hz}},
	{{-hx, -hy, -hz}, { hx, -hy, -hz}, { hx, -hy,  hz}, {-hx, -hy,  hz}},
};

// Flat shading: каждая грань держит свою наружную нормаль, поэтому 24 вершины
// вместо 8 уникальных углов. Цвет вершины белый — окраску даёт альбедо
// материала из UI (вершинный атрибут цвета оставлен в конвейере по лабе 1).
constexpr std::array<Vertex, 24> makeBoxVertices() {
	std::array<Vertex, 24> result{};
	for (int face = 0; face < 6; ++face)
		for (int corner = 0; corner < 4; ++corner) {
			const Vec3 p = faceCorners[face][corner];
			result[face * 4 + corner] = {p, faceNormals[face], {1.0f, 1.0f, 1.0f}};
		}
	return result;
}

constexpr std::array<Vertex, 24> vertices = makeBoxVertices();

// Два треугольника против часовой на каждой из 4-х граней.
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

// set = 0: UBO сцены + два storage-буфера для источников света.
VkBuffer scene_ubo = VK_NULL_HANDLE;
VmaAllocation scene_ubo_memory = VK_NULL_HANDLE;
SceneUniforms* scene_ubo_data = nullptr;

// Storage-буферы светов: persistently mapped, каждый кадр переписываются
// (первые count элементов), дескрипторы не пересоздаются.
VkBuffer point_lights_ssbo = VK_NULL_HANDLE;
VmaAllocation point_lights_memory = VK_NULL_HANDLE;
GpuPointLight* point_lights_data = nullptr;

VkBuffer spot_lights_ssbo = VK_NULL_HANDLE;
VmaAllocation spot_lights_memory = VK_NULL_HANDLE;
GpuSpotLight* spot_lights_data = nullptr;

// set = 1: UBO модели.
VkBuffer model_ubo = VK_NULL_HANDLE;
VmaAllocation model_ubo_memory = VK_NULL_HANDLE;
ModelUniforms* model_ubo_data = nullptr;

VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;
VkDescriptorSet model_set = VK_NULL_HANDLE;

// ---- Состояние, правится из ImGui -------------------------------------

struct CameraState {
	float position[3] = {0.0f, 0.30f, 4.9f};
	float yawDegrees = 0.0f;
	float pitchDegrees = -2.0f;
	float fovDegrees = 42.0f;
};
CameraState camera;

struct DirectionalLightState {
	float direction[3] = {-0.55f, -0.75f, -0.35f}; // куда светит
	float color[3] = {0.85f, 0.92f, 1.0f};
	float intensity = 0.50f;
};
DirectionalLightState dirLight;

struct PointLightState {
	float position[3] = {1.8f, 0.9f, 1.2f};
	float color[3] = {1.0f, 0.6f, 0.3f};
	float intensity = 5.0f;
};
std::vector<PointLightState> pointLights;

struct SpotLightState {
	float position[3] = {0.4f, 2.4f, 0.6f};
	float direction[3] = {0.0f, -1.0f, -0.1f};
	float color[3] = {1.0f, 0.95f, 0.85f};
	float intensity = 6.0f;
	float innerAngleDegrees = 10.0f;
	float outerAngleDegrees = 24.0f;
};
std::vector<SpotLightState> spotLights;

struct ObjectState {
	float position[3] = {0, 0, 0};
	float angles[3] = {18, 25, 0};
	float size[3] = {1, 1, 1};
};
ObjectState object;

struct MaterialState {
	float albedo[3] = {0.04f, 0.42f, 0.60f};
	float specular[3] = {0.55f, 0.55f, 0.55f};
	float shininess = 96.0f;
};
MaterialState material;

float ambientIntensity = 0.03f;
float exposure = 0.90f;
double previous_time = -1.0;

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

	VkShaderModule vert = shaderModule("shaders/object.vert.spv");
	VkShaderModule frag = shaderModule("shaders/object.frag.spv");
	if (!vert || !frag) return false;

	// Два набора: индекс 0 — сцена (set = 0 в шейдерах), 1 — модель (set = 1).
	const VkDescriptorSetLayout set_layouts[] = {scene_set_layout, model_set_layout};
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2, .pSetLayouts = set_layouts,
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
	// Грани обёрнуты CCW снаружи; проекция лишь переворачивает Y, а Vulkan
	// меряет лицевую сторону в координатах фреймбуфера (Y вниз), что взаимно
	// гасится: наружные грани остаются CCW на экране.
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

// Пул дескрипторов: должен вмещать и UBO, и STORAGE дескрипторы обоих наборов.
bool createDescriptors() {
	auto& ctx = graphics::internal::context;

	// set = 0: UBO сцены + два storage-буфера светов (только фрагментный шейдер).
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};
	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 3, .pBindings = scene_bindings,
	};
	if (vkCreateDescriptorSetLayout(ctx.device, &scene_layout_info, nullptr,
	                                &scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Cannot create scene descriptor set layout\n";
		return false;
	}

	// set = 1: UBO модели.
	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1, .pBindings = &model_binding,
	};
	if (vkCreateDescriptorSetLayout(ctx.device, &model_layout_info, nullptr,
	                                &model_set_layout) != VK_SUCCESS) {
		std::cerr << "Cannot create model descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_sizes[] = {
		{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2},
		{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2},
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2, .poolSizeCount = 2, .pPoolSizes = pool_sizes,
	};
	if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Cannot create descriptor pool\n";
		return false;
	}

	VkDescriptorSetLayout set_layouts[] = {scene_set_layout, model_set_layout};
	const VkDescriptorSetAllocateInfo set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 2, .pSetLayouts = set_layouts,
	};
	VkDescriptorSet sets[2] = {};
	if (vkAllocateDescriptorSets(ctx.device, &set_info, sets) != VK_SUCCESS) {
		std::cerr << "Cannot allocate descriptor sets\n";
		return false;
	}
	scene_set = sets[0];
	model_set = sets[1];
	return true;
}

void writeDescriptorSet(const VkDescriptorSet set, const uint32_t binding,
                        const VkDescriptorType type, const VkBuffer buffer,
                        const VkDeviceSize range) {
	const VkDescriptorBufferInfo buffer_info = {buffer, 0, range};
	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set, .dstBinding = binding, .descriptorCount = 1,
		.descriptorType = type, .pBufferInfo = &buffer_info,
	};
	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

void resetState() {
	camera = CameraState();
	dirLight = DirectionalLightState();
	object = ObjectState();
	material = MaterialState();
	ambientIntensity = 0.03f;
	exposure = 0.90f;
	// Стартовый набор источников тот же, что и в initialize().
	pointLights.clear();
	pointLights.push_back({{2.8f, 0.8f, 2.6f}, {1.0f, 0.50f, 0.28f}, 3.5f});
	pointLights.push_back({{-2.8f, 0.5f, 2.2f}, {0.35f, 0.55f, 1.0f}, 2.5f});
	spotLights.clear();
	spotLights.push_back({{0.2f, 2.6f, 1.6f}, {0.0f, -1.0f, -0.55f},
	                      {1.0f, 0.97f, 0.92f}, 6.0f, 9.0f, 27.0f});
}

} // namespace

bool initialize() {
	if (!makeBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data(),
	                vertex_buffer, vertex_memory, nullptr) ||
	    !makeBuffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.data(),
	                index_buffer, index_memory, nullptr)) {
		std::cerr << "Cannot create vertex/index buffer\n";
		return false;
	}

	if (!makeBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
	                scene_ubo, scene_ubo_memory,
	                reinterpret_cast<void**>(&scene_ubo_data))) {
		std::cerr << "Cannot create scene uniform buffer\n";
		return false;
	}

	if (!makeBuffer(VkDeviceSize(max_lights * sizeof(GpuPointLight)),
	                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr,
	                point_lights_ssbo, point_lights_memory,
	                reinterpret_cast<void**>(&point_lights_data))) {
		std::cerr << "Cannot create point lights storage buffer\n";
		return false;
	}

	if (!makeBuffer(VkDeviceSize(max_lights * sizeof(GpuSpotLight)),
	                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr,
	                spot_lights_ssbo, spot_lights_memory,
	                reinterpret_cast<void**>(&spot_lights_data))) {
		std::cerr << "Cannot create spot lights storage buffer\n";
		return false;
	}

	if (!makeBuffer(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
	                model_ubo, model_ubo_memory,
	                reinterpret_cast<void**>(&model_ubo_data))) {
		std::cerr << "Cannot create model uniform buffer\n";
		return false;
	}

	if (!createDescriptors()) return false;

	// Привязка буферов к наборам дескрипторов: единожды, буферы не меняются.
	writeDescriptorSet(scene_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	                   scene_ubo, sizeof(SceneUniforms));
	writeDescriptorSet(scene_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                   point_lights_ssbo, max_lights * sizeof(GpuPointLight));
	writeDescriptorSet(scene_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                   spot_lights_ssbo, max_lights * sizeof(GpuSpotLight));
	writeDescriptorSet(model_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	                   model_ubo, sizeof(ModelUniforms));

	// Стартовая сцена: холодный ключевой направленный сверху-слева; тёплый point
	// справа-спереди и холодный слева-спереди расщепляют видимые грани по цвету
	// (тёплые/холодные переливы, как в термосъёмке тора); спот сверху-спереди
	// даёт блестящую «корону» на верхней грани. Интенсивности небольшие:
	// затухание 1/D² и гамма легко вымывают цвет в пастель.
	pointLights.push_back({{2.8f, 0.8f, 2.6f}, {1.0f, 0.50f, 0.28f}, 3.5f});
	pointLights.push_back({{-2.8f, 0.5f, 2.2f}, {0.35f, 0.55f, 1.0f}, 2.5f});
	spotLights.push_back({{0.2f, 2.6f, 1.6f}, {0.0f, -1.0f, -0.55f},
	                      {1.0f, 0.97f, 0.92f}, 6.0f, 9.0f, 27.0f});

	if (!createPipeline()) return false;
	return true;
}

void shutdown() {
	auto& ctx = graphics::internal::context;
	vkQueueWaitIdle(ctx.graphics_queue);
	vkDestroyPipeline(ctx.device, pipeline, nullptr);
	vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device, scene_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device, model_set_layout, nullptr);
	if (vertex_buffer) vmaDestroyBuffer(ctx.allocator, vertex_buffer, vertex_memory);
	if (index_buffer) vmaDestroyBuffer(ctx.allocator, index_buffer, index_memory);
	if (scene_ubo) vmaDestroyBuffer(ctx.allocator, scene_ubo, scene_ubo_memory);
	if (point_lights_ssbo)
		vmaDestroyBuffer(ctx.allocator, point_lights_ssbo, point_lights_memory);
	if (spot_lights_ssbo)
		vmaDestroyBuffer(ctx.allocator, spot_lights_ssbo, spot_lights_memory);
	if (model_ubo) vmaDestroyBuffer(ctx.allocator, model_ubo, model_ubo_memory);
}

void update(double time) {
	static double fps = 0.0;
	if (previous_time >= 0.0) {
		const double dt = time - previous_time;
		if (dt > 0.0) fps = fps <= 0.0 ? 1.0 / dt : fps * 0.9 + (1.0 / dt) * 0.1;
	}
	previous_time = time;

	ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(340, 540), ImGuiCond_FirstUseEver);
	ImGui::Begin("CG Lab2 - Variant 3 - Bogdanov/Vulkan");

	ImGui::TextDisabled("Student: Bogdanov, group 314");
	ImGui::TextDisabled("Variant 3: parallelepiped");
	ImGui::TextDisabled("Vulkan | C++20 | GLFW | ImGUI");
	ImGui::Text("FPS: %.0f", fps);
	ImGui::Spacing();

	// Камера: позиция, поворот, угол обзора — требование базового задания.
	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position##cam", camera.position, 0.05f, -10.0f, 10.0f);
		ImGui::SliderFloat("Yaw (degrees)", &camera.yawDegrees, -180.0f, 180.0f, "%.0f");
		ImGui::SliderFloat("Pitch (degrees)", &camera.pitchDegrees, -85.0f, 85.0f, "%.0f");
		ImGui::SliderFloat("FOV (degrees)", &camera.fovDegrees, 20.0f, 120.0f, "%.0f");
		ImGui::Spacing();
	}

	// Освещение: позиция, цвет + интенсивность и уникальные параметры
	// каждого типа источника (ТЗ), плюс добавление/удаление из UI (доп. 2-3).
	if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Text("Directional light:");
		ImGui::DragFloat3("Direction", dirLight.direction, 0.01f, -1.0f, 1.0f);
		ImGui::ColorEdit3("Color##dir", dirLight.color);
		ImGui::SliderFloat("Intensity##dir", &dirLight.intensity, 0.0f, 5.0f);
		ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0.0f, 1.0f);
		ImGui::Spacing();

		ImGui::SeparatorText("Point lights");
		if (ImGui::Button("+ Add##point") && pointLights.size() < max_lights)
			pointLights.push_back({});
		ImGui::SameLine();
		if (ImGui::Button("Clear##point")) pointLights.clear();
		for (size_t i = 0; i < pointLights.size(); ++i) {
			ImGui::PushID(static_cast<int>(i));
			char header[32];
			std::snprintf(header, sizeof header, "Point light #%zu", i);
			if (ImGui::CollapsingHeader(header)) {
				ImGui::DragFloat3("Position", pointLights[i].position, 0.05f, -10.0f, 10.0f);
				ImGui::ColorEdit3("Color", pointLights[i].color);
				ImGui::SliderFloat("Intensity", &pointLights[i].intensity, 0.0f, 30.0f);
				if (ImGui::Button("Remove"))
					pointLights.erase(pointLights.begin() + static_cast<long>(i));
			}
			ImGui::PopID();
		}
		ImGui::Spacing();

		ImGui::SeparatorText("Spot lights");
		if (ImGui::Button("+ Add##spot") && spotLights.size() < max_lights)
			spotLights.push_back({});
		ImGui::SameLine();
		if (ImGui::Button("Clear##spot")) spotLights.clear();
		for (size_t i = 0; i < spotLights.size(); ++i) {
			ImGui::PushID(static_cast<int>(i + 1000));
			char header[32];
			std::snprintf(header, sizeof header, "Spot light #%zu", i);
			if (ImGui::CollapsingHeader(header)) {
				ImGui::DragFloat3("Position", spotLights[i].position, 0.05f, -10.0f, 10.0f);
				ImGui::DragFloat3("Direction", spotLights[i].direction, 0.01f, -1.0f, 1.0f);
				ImGui::ColorEdit3("Color", spotLights[i].color);
				ImGui::SliderFloat("Intensity", &spotLights[i].intensity, 0.0f, 40.0f);
				// Уникальные параметры прожектора: внутренний и внешний углы
				// мягких краёв (ТЗ). Внешний не может быть меньше внутреннего.
				ImGui::SliderFloat("Inner angle", &spotLights[i].innerAngleDegrees, 0.0f, 89.0f, "%.0f deg");
				ImGui::SliderFloat("Outer angle", &spotLights[i].outerAngleDegrees, 0.0f, 89.0f, "%.0f deg");
				if (spotLights[i].outerAngleDegrees < spotLights[i].innerAngleDegrees)
					spotLights[i].outerAngleDegrees = spotLights[i].innerAngleDegrees;
				if (ImGui::Button("Remove"))
					spotLights.erase(spotLights.begin() + static_cast<long>(i));
			}
			ImGui::PopID();
		}
	}

	// Материал: альбедо, цвет блика, коэффициент блика (ТЗ).
	if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Albedo", material.albedo);
		ImGui::ColorEdit3("Specular color", material.specular);
		ImGui::SliderFloat("Shininess", &material.shininess, 1.0f, 256.0f, "%.0f");
		ImGui::Spacing();
	}

	ImGui::Separator();
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
		{.color = {.float32 = {0.04f, 0.055f, 0.09f, 1.0f}}},
		{.depthStencil = {1.0f, 0}},
	};
	const VkRenderPassBeginInfo pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = ctx.render_pass, .framebuffer = fd.framebuffer,
		.renderArea = {.extent = ctx.swapchain_extent},
		.clearValueCount = 2, .pClearValues = clear,
	};
	vkCmdBeginRenderPass(fd.command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);

	const float aspect = static_cast<float>(ctx.swapchain_extent.width) /
	                     static_cast<float>(ctx.swapchain_extent.height);

	// ---- Матрицы: model / view / proj ----
	const Mat4 model = multiply(
		translation({object.position[0], object.position[1], object.position[2]}),
		multiply(rotation(object.angles[0], object.angles[1], object.angles[2]),
		         scale({object.size[0], object.size[1], object.size[2]})));
	// Матрица вида через обратную модель камеры, проекция с настраиваемым FOV.
	const Mat4 view = viewFromCamera(
		{camera.position[0], camera.position[1], camera.position[2]},
		camera.yawDegrees, camera.pitchDegrees);
	const Mat4 proj = perspective(camera.fovDegrees, aspect);

	// ---- UBO сцены ----
	const Vec3 dirDirection = normalized({dirLight.direction[0], dirLight.direction[1],
	                                      dirLight.direction[2]});
	std::memcpy(&scene_ubo_data->view, &view, sizeof(Mat4));
	std::memcpy(&scene_ubo_data->proj, &proj, sizeof(Mat4));
	scene_ubo_data->cameraPosition[0] = camera.position[0];
	scene_ubo_data->cameraPosition[1] = camera.position[1];
	scene_ubo_data->cameraPosition[2] = camera.position[2];
	scene_ubo_data->cameraPosition[3] = 1.0f;
	scene_ubo_data->ambientLight[0] = ambientIntensity;
	scene_ubo_data->ambientLight[1] = ambientIntensity;
	scene_ubo_data->ambientLight[2] = ambientIntensity;
	scene_ubo_data->ambientLight[3] = 1.0f;
	scene_ubo_data->dirLightDirection[0] = dirDirection.x;
	scene_ubo_data->dirLightDirection[1] = dirDirection.y;
	scene_ubo_data->dirLightDirection[2] = dirDirection.z;
	scene_ubo_data->dirLightDirection[3] = 0.0f;
	scene_ubo_data->dirLightColorIntensity[0] = dirLight.color[0];
	scene_ubo_data->dirLightColorIntensity[1] = dirLight.color[1];
	scene_ubo_data->dirLightColorIntensity[2] = dirLight.color[2];
	scene_ubo_data->dirLightColorIntensity[3] = dirLight.intensity;
	scene_ubo_data->lightCounts[0] = static_cast<float>(pointLights.size());
	scene_ubo_data->lightCounts[1] = static_cast<float>(spotLights.size());
	scene_ubo_data->lightCounts[2] = exposure;
	scene_ubo_data->lightCounts[3] = 0.0f;
	vmaFlushAllocation(ctx.allocator, scene_ubo_memory, 0, sizeof(SceneUniforms));

	// ---- Storage-буфер точечных источников ----
	GpuPointLight gpuPoints[max_lights];
	for (size_t i = 0; i < pointLights.size(); ++i) {
		gpuPoints[i] = {};
		gpuPoints[i].position[0] = pointLights[i].position[0];
		gpuPoints[i].position[1] = pointLights[i].position[1];
		gpuPoints[i].position[2] = pointLights[i].position[2];
		gpuPoints[i].colorIntensity[0] = pointLights[i].color[0];
		gpuPoints[i].colorIntensity[1] = pointLights[i].color[1];
		gpuPoints[i].colorIntensity[2] = pointLights[i].color[2];
		gpuPoints[i].colorIntensity[3] = pointLights[i].intensity;
	}
	if (!pointLights.empty()) {
		std::memcpy(point_lights_data, gpuPoints, pointLights.size() * sizeof(GpuPointLight));
		vmaFlushAllocation(ctx.allocator, point_lights_memory, 0,
		                   pointLights.size() * sizeof(GpuPointLight));
	}

	// ---- Storage-буфер прожекторных источников ----
	GpuSpotLight gpuSpots[max_lights];
	for (size_t i = 0; i < spotLights.size(); ++i) {
		const Vec3 axis = normalized({spotLights[i].direction[0], spotLights[i].direction[1],
		                              spotLights[i].direction[2]});
		gpuSpots[i] = {};
		gpuSpots[i].position[0] = spotLights[i].position[0];
		gpuSpots[i].position[1] = spotLights[i].position[1];
		gpuSpots[i].position[2] = spotLights[i].position[2];
		gpuSpots[i].direction[0] = axis.x;
		gpuSpots[i].direction[1] = axis.y;
		gpuSpots[i].direction[2] = axis.z;
		gpuSpots[i].colorIntensity[0] = spotLights[i].color[0];
		gpuSpots[i].colorIntensity[1] = spotLights[i].color[1];
		gpuSpots[i].colorIntensity[2] = spotLights[i].color[2];
		gpuSpots[i].colorIntensity[3] = spotLights[i].intensity;
		gpuSpots[i].angles[0] = std::cos(spotLights[i].innerAngleDegrees * pi / 180.0f);
		gpuSpots[i].angles[1] = std::cos(spotLights[i].outerAngleDegrees * pi / 180.0f);
	}
	if (!spotLights.empty()) {
		std::memcpy(spot_lights_data, gpuSpots, spotLights.size() * sizeof(GpuSpotLight));
		vmaFlushAllocation(ctx.allocator, spot_lights_memory, 0,
		                   spotLights.size() * sizeof(GpuSpotLight));
	}

	// ---- UBO модели: матрица + материал ----
	std::memcpy(&model_ubo_data->model, &model, sizeof(Mat4));
	model_ubo_data->albedoColor[0] = material.albedo[0];
	model_ubo_data->albedoColor[1] = material.albedo[1];
	model_ubo_data->albedoColor[2] = material.albedo[2];
	model_ubo_data->albedoColor[3] = 1.0f;
	model_ubo_data->specularShininess[0] = material.specular[0];
	model_ubo_data->specularShininess[1] = material.specular[1];
	model_ubo_data->specularShininess[2] = material.specular[2];
	model_ubo_data->specularShininess[3] = material.shininess;
	model_ubo_data->flags[0] = 0.0f; // шахматный пол отключён, обычный материал
	model_ubo_data->flags[1] = 0.0f;
	model_ubo_data->flags[2] = 0.0f;
	model_ubo_data->flags[3] = 0.0f;
	vmaFlushAllocation(ctx.allocator, model_ubo_memory, 0, sizeof(ModelUniforms));

	// ---- Запись отрисовки ----
	const VkViewport viewport = {0, 0, static_cast<float>(ctx.swapchain_extent.width),
	                             static_cast<float>(ctx.swapchain_extent.height), 0, 1};
	const VkRect2D scissor = {{0, 0}, ctx.swapchain_extent};
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	// Привязываем наборы: сцена с индекса 0, модель с индекса 1.
	const VkDescriptorSet bind_sets[] = {scene_set, model_set};
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        pipeline_layout, 0, 2, bind_sets, 0, nullptr);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, &offset);
	vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT16);
	vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
