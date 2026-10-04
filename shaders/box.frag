#version 450
layout(location = 0) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0) uniform ObjectUniforms {
	mat4 model;
	vec4 tint;
} object;

void main() {
	outColor = vec4(vertexColor * object.tint.rgb, 1.0);
}
