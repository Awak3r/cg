#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 0) out vec3 worldNormal;
layout(location = 1) out vec3 vertexColor;

layout(set = 0, binding = 0) uniform GlobalUniforms {
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 tint;
} g;

void main() {
	gl_Position = g.proj * g.view * g.model * vec4(inPosition, 1.0);
	// Inverse-transpose so normals survive non-uniform scale (the Scale slider).
	worldNormal = mat3(transpose(inverse(g.model))) * inNormal;
	vertexColor = inColor;
}
