#version 450
layout(location = 0) in vec3 worldNormal;
layout(location = 1) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform GlobalUniforms {
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 tint;
} g;

// Key light from above-right-front + soft fill from below-left, both fixed
// in world space. Fill keeps the opposite faces readable instead of black.
const vec3 keyLightDirection = normalize(vec3(0.3, 0.85, 0.45));
const vec3 fillLightDirection = normalize(vec3(-0.45, -0.25, 0.35));
const float keyIntensity = 0.8;
const float fillIntensity = 0.35;

void main() {
	// Lambert: hemisphere ambient (brighter for up-facing surfaces) + diffuse.
	const vec3 normal = normalize(worldNormal);
	const float hemi = 0.5 + 0.5 * normal.y;
	const vec3 ambient = mix(vec3(0.16), vec3(0.30), hemi);
	const float diffuse = keyIntensity * max(dot(normal, keyLightDirection), 0.0) +
	                      fillIntensity * max(dot(normal, fillLightDirection), 0.0);
	outColor = vec4(vertexColor * g.tint.rgb * min(ambient + diffuse, vec3(1.0)), 1.0);
}