#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 vertexColor;

layout(set = 0, binding = 0) uniform CameraUniforms {
	mat4 view;
	mat4 proj;
} camera;

layout(set = 1, binding = 0) uniform ObjectUniforms {
	mat4 model;
	vec4 tint;
} object;

void main() {
	gl_Position = camera.proj * camera.view * object.model * vec4(inPosition, 1.0);
	vertexColor = inColor;
}
