#version 450

layout(location = 0) in vec3 f_worldPos;
layout(location = 1) in vec3 f_worldNormal;
layout(location = 2) in vec3 f_vertexColor;
layout(location = 0) out vec4 outColor;

// Единый набор дескрипторов (set = 0): зеркалит структуры из application.cpp.
layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view;
	mat4 proj;
	vec4 cameraPosition;         // xyz — позиция камеры (для V)
	vec4 ambientLight;           // rgb — рассеянный свет
	vec4 dirLightDirection;      // xyz — направление направленного источника
	vec4 dirLightColorIntensity; // rgb — цвет, a — интенсивность
	vec4 display;                // x — экспозиция тонемаппинга
} s;

layout(set = 0, binding = 1) uniform ModelUniforms {
	mat4 model;
	vec4 albedoColor;          // rgb — альбедо материала
	vec4 specularShininess;    // rgb — цвет блика, w — коэффициент блеска
} m;

// Гамма монитора: считаем свет в линейном пространстве, а на выходе
// возвращаем его в sRGB — иначе тёмные области уходят в грязную черноту.
const float displayGamma = 2.2;

struct Material {
	vec3 albedo;
	vec3 specular;
	float shininess;
};

void main() {
	const vec3 N = normalize(f_worldNormal);
	// Из точки поверхности к камере — камера живёт в мировых координатах.
	const vec3 V = normalize(s.cameraPosition.xyz - f_worldPos);

	Material mat;
	mat.specular  = m.specularShininess.rgb;
	mat.shininess = max(m.specularShininess.w, 0.0);
	mat.albedo    = m.albedoColor.rgb * f_vertexColor;

	// Полусферный ambient: «небо» освещает верхние грани ярче пола —
	// дёшево даёт объём.
	const vec3 ambient = s.ambientLight.rgb * mat.albedo *
	                     (0.55 + 0.45 * (N.y * 0.5 + 0.5));

	// Направленный источник («солнце»): у лучей нет затухания с расстоянием.
	const vec3 toLight = -normalize(s.dirLightDirection.xyz);
	const float diffuse = max(dot(N, toLight), 0.0);
	// Полувектор между «на свет» и «на камеру» — блик через него.
	const vec3 halfVector = normalize(toLight + V);
	const float specular = pow(max(dot(N, halfVector), 0.0), mat.shininess);

	const vec3 result = ambient +
	    s.dirLightColorIntensity.rgb * s.dirLightColorIntensity.a *
	    (mat.albedo * diffuse + mat.specular * specular);

	// Рим-свет: тонкое голубоватое свечение по краям (Френель) отделяет
	// объект от фона. Сама limitless видна только на поверхностях,
	// отвёрнутых от камеры.
	const float rim = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 4.0);
	const vec3 lit = result + rim * 0.18 * vec3(0.40, 0.55, 0.85);

	// Тонемаппинг (exponential) вместо жёсткого min() — яркие области
	// мягко замыливаются, а не вырезаются белым.
	const vec3 mapped = vec3(1.0) - exp(-lit * max(s.display.x, 0.01));

	// Гамма-коррекция: монитор показывает цвета через кривую 2.2,
	// поэтому перед выводом берём обратную степень 1/2.2.
	outColor = vec4(pow(mapped, vec3(1.0 / displayGamma)), 1.0);
}
