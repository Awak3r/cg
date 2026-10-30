#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 f_worldPos;
layout(location = 1) out vec3 f_worldNormal;
layout(location = 2) out vec3 f_vertexColor;

// Единый набор дескрипторов (set = 0):
//   binding 0 — общий UBO сцены (матрицы камеры, освещение),
//   binding 1 — UBO модели (матрица трансформации, материал).
// Структуры — зеркала struct SceneUniforms/ModelUniforms из application.cpp.
layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view;                   // матрица вида (обратная матрица модели камеры)
	mat4 proj;                   // матрица проекции (FOV берётся из ImGui)
	vec4 cameraPosition;         // позиция камеры в мире, нужна фрагментному для V
	vec4 ambientLight;           // rgb — рассеянный свет, a — запас
	vec4 dirLightDirection;      // xyz — направление направленного источника
	vec4 dirLightColorIntensity; // rgb — цвет, a — интенсивность
	vec4 display;                // x — экспозиция тонемаппинга, yzw — запас
} s;

layout(set = 0, binding = 1) uniform ModelUniforms {
	mat4 model;                // T*R*S блока
	vec4 albedoColor;          // rgb — альбедо материала
	vec4 specularShininess;    // rgb — цвет блика, w — коэффициент блеска
} m;

void main() {
	const vec4 worldPos = m.model * vec4(inPosition, 1.0);
	f_worldPos = worldPos.xyz;

	// Нормаль в мир: только mat3(model) безопасен при равномерном масштабе,
	// поэтому используем inverse-transpose (масштаб неравномерный).
	f_worldNormal = mat3(transpose(inverse(m.model))) * inNormal;

	f_vertexColor = inColor;
	gl_Position = s.proj * s.view * worldPos;
}
