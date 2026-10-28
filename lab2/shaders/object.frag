#version 450

layout(location = 0) in vec3 f_worldPos;
layout(location = 1) in vec3 f_worldNormal;
layout(location = 2) in vec3 f_vertexColor;
layout(location = 0) out vec4 outColor;

// set = 0: общие данные сцены — зеркалит struct SceneUniforms из application.cpp.
layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view;
	mat4 proj;
	vec4 cameraPosition;
	vec4 ambientLight;
	vec4 dirLightDirection;      // xyz — направление направленного источника
	vec4 dirLightColorIntensity; // rgb — цвет, a — интенсивность
	vec4 lightCounts;            // x — количество точечных, y — прожекторных
} s;

// Параметры точечного источника: 2 vec4 = 32 байта, зеркалит struct GpuPointLight.
struct PointLight {
	vec4 position;               // xyz — позиция, w — запас
	vec4 colorIntensity;         // rgb — цвет, a — интенсивность
};

// Массив точечных источников в storage-буфере: std430 и НЕфиксированная длина
// массива — число задействованных элементов приходит через lightCounts.x.
layout(std430, set = 0, binding = 1) buffer PointLightsBuffer {
	PointLight lights[];
} pointLights;

// Параметры прожекторного источника: 4 vec4 = 64 байта, зеркалит struct GpuSpotLight.
struct SpotLight {
	vec4 position;               // xyz — позиция источника
	vec4 direction;              // xyz — ось конуса (normalized)
	vec4 colorIntensity;         // rgb — цвет, a — интенсивность
	vec4 angles;                 // x — cos(inner), y — cos(outer)
};

layout(std430, set = 0, binding = 2) buffer SpotLightsBuffer {
	SpotLight lights[];
} spotLights;

// set = 1: данные конкретной модели.
layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	vec4 albedoColor;
	vec4 specularShininess;      // rgb — цвет блика, w — коэффициент блеска
	vec4 flags;                  // x — шахматный пол (0/1)
} m;

// Гамма монитора: считаем свет в линейном пространстве, а на выходе
// возвращаем его в sRGB — иначе тёмные области уходят в грязную черноту.
const float displayGamma = 2.2;

struct Material {
	vec3 albedo;
	vec3 specular;
	float shininess;
};

// Вклад одного источника по Блинну-Фонгу для уже нормализованных N, V и toLight.
vec3 blinnPhong(Material mat, vec3 N, vec3 V, vec3 lightColor, vec3 toLight) {
	// Диффузная часть: закон Ламберта.
	const float diffuse = max(dot(N, toLight), 0.0);
	// Половинный вектор между «на свет» и «на камеру», блик через него.
	const vec3 halfVector = normalize(toLight + V);
	const float specular = pow(max(dot(N, halfVector), 0.0), mat.shininess);
	return lightColor * (diffuse * mat.albedo + specular * mat.specular);
}

void main() {
	const vec3 N = normalize(f_worldNormal);
	// Из точки поверхности к камере — камера живёт в мировых координатах.
	const vec3 V = normalize(s.cameraPosition.xyz - f_worldPos);

	Material mat;
	mat.specular  = m.specularShininess.rgb;
	mat.shininess = max(m.specularShininess.w, 0.0);
	mat.albedo    = m.albedoColor.rgb * f_vertexColor;

	// Шахматный пол: альбедо берётся из мировой позиции, материал UI не трогаем.
	if (m.flags.x > 0.5) {
		const float chess = mod(floor(f_worldPos.x * 0.75) + floor(f_worldPos.z * 0.75), 2.0);
		mat.albedo = mix(vec3(0.42, 0.44, 0.48), vec3(0.13, 0.14, 0.17), chess);
		mat.specular = vec3(0.15);
		mat.shininess = 16.0;
	}

	// Hemisphere ambient: «небо» освещает верхние грани ярче пола — дёшево даёт объём.
	const vec3 ambient = s.ambientLight.rgb * mat.albedo *
	                     (0.55 + 0.45 * (N.y * 0.5 + 0.5));

	// Направленный источник («солнце»): у лучей нет затухания с расстоянием.
	const vec3 dirToLight = -normalize(s.dirLightDirection.xyz);
	vec3 result = ambient +
	    blinnPhong(mat, N, V, s.dirLightColorIntensity.rgb *
	                  s.dirLightColorIntensity.a, dirToLight);

	// Точечные источники с затуханием по закону обратных квадратов.
	for (uint i = 0; i < uint(s.lightCounts.x); ++i) {
		const vec3 toSource = pointLights.lights[i].position.xyz - f_worldPos;
		const float distance = length(toSource);
		const vec3 toLight = toSource / max(distance, 0.0001);
		const float falloff = 1.0 / max(distance * distance, 0.0001);
		const vec3 lightColor = pointLights.lights[i].colorIntensity.rgb *
		                        pointLights.lights[i].colorIntensity.a;
		result += falloff * blinnPhong(mat, N, V, lightColor, toLight);
	}

	// Прожекторные источники: как точечные + конус с мягкими краями.
	for (uint i = 0; i < uint(s.lightCounts.y); ++i) {
		const vec3 toSource = spotLights.lights[i].position.xyz - f_worldPos;
		const float distance = length(toSource);
		const vec3 toLight = toSource / max(distance, 0.0001);
		const float falloff = 1.0 / max(distance * distance, 0.0001);

		// cos угла между осью конуса и направлением «на точку» от прожектора.
		const float theta = dot(normalize(spotLights.lights[i].direction.xyz), -toLight);
		const float cosOuter = spotLights.lights[i].angles.y;
		const float epsilon = spotLights.lights[i].angles.x - cosOuter;
		// 1 внутри внутреннего угла, 0 снаружи, плавный спад между ними.
		// При epsilon <= 0 мягких краёв нет — резкая граница, без деления на 0.
		const float cone = epsilon > 0.0
		    ? clamp((theta - cosOuter) / epsilon, 0.0, 1.0)
		    : step(cosOuter, theta);

		const vec3 lightColor = spotLights.lights[i].colorIntensity.rgb *
		                        spotLights.lights[i].colorIntensity.a;
		result += cone * falloff * blinnPhong(mat, N, V, lightColor, toLight);
	}

	// Рим-свет: тонкое голубоватое свечение по краям (Френель) отделяет
	// объект от фона. Скала видна только на поверхностях, отвёрнутых от камеры.
	const float rim = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 4.0);
	result += rim * 0.18 * vec3(0.40, 0.55, 0.85);

	// Тонемаппинг (exponential, как в LearnOpenGL HDR) вместо жёсткого min()
	// — яркие области мягко замыливаются, а не вырезаются белым.
	const vec3 mapped = vec3(1.0) - exp(-result * max(s.lightCounts.z, 0.01));

	// Гамма-коррекция: монитор показывает цвета через кривую 2.2,
	// поэтому перед выводом берём обратную степень 1/2.2.
	outColor = vec4(pow(mapped, vec3(1.0 / displayGamma)), 1.0);
}
