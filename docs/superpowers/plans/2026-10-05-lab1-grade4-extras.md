# Lab 1 Grade-4 Extras Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-subagent-driven-development (recommended) or superpowers-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add all five grade-4 extras from the TZ to the grade-3 baseline: projection switch, transform UI, orbit animation, tint color, procedural vertex colors.

**Architecture:** Extend the single uniform buffer with a `tint` vec4 (std140-safe, 208 bytes), add a third vertex attribute (procedural color), add `orthographic()` and orbit animation state; rewrite the affected files in one pass since they change together.

**Tech Stack:** Vulkan, GLFW, ImGui, VMA, C++20, glslc.

**Spec:** user-approved chat decisions (2026-10-05): color × tint × lighting; orbit + optional spin; single box (no descriptor-split — that's grade 5).

---

### Task 1: `source/application.cpp` — full rewrite with grade-4 features

**Files:**
- Modify (full rewrite): `source/application.cpp`

- [x] **Step 1: Rewrite the file** — implemented changes:

- `struct Vertex { Vec3 position; Vec3 normal; Vec3 color; }` + `vertexColor()` procedural mapping (X→R, Y→G, Z→B) baked into `makeVertices()` (Доп. 5);
- third attribute `{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)}` in pipeline vertex input;
- `GlobalUniforms` gains `float tint[4]` after `proj` (std140: vec4-aligned, size 208 B), descriptor stage flags now include `VK_SHADER_STAGE_FRAGMENT_BIT` (Доп. 4);
- `scale(Vec3)` restored; `orthographic(aspect)` added (Доп. 1);
- `ObjectState` with position/angles(18,25,0)/size/color(1,1,1) default (Доп. 2);
- orbit state: `playing, spin, laps_per_second, orbit_radius, orbit_height, phase`; `orbitPoint(t)` = `{r·cos t, h·sin t, r·sin t}`; `update()` advances `phase` with 0.05 s stall clamp (Доп. 3);
- `update()` UI: Projection Combo, DragFloat3 ×3, ColorEdit3 tint, orbit sliders/checkboxes, Restart phase + Reset all buttons;
- `render()`: model = `translation(position + orbitPoint) · rotation(angles + spin·phase) · scale(size)`, projection switch, tint write, flush;
- `resetState()` restores all defaults.

- [x] **Step 2: Verify destroy symmetry** — `grep -c "vkDestroy|vmaDestroyBuffer"` = 9.

### Task 2: `shaders/box.vert`

- [x] **Step 1:** added `location 2` `in vec3 inColor`, out `location 1` `vertexColor`; uniform block gains `vec4 tint`.

### Task 3: `shaders/box.frag`

- [x] **Step 1:** in `location 1` `vertexColor`; uniform block mirrored; `outColor = vertexColor * g.tint.rgb * min(ambient + diffuse, 1)`.

### Task 4: Build

- [x] **Step 1:** `cmake --build build-debug --parallel` — clean (shaders + app relink).
- [x] **Step 2:** `cmake --build build-release --parallel` — clean.

### Task 5: Run + validation

- [x] **Step 1:** launch release build, process alive, log empty, validation hits = 0.
- [x] **Step 2:** SIGTERM shutdown, process exits, validation hits after shutdown = 0.

### Task 6: README

- [x] **Step 1:** rewritten with grade-3 + grade-4 checklists, controls table, shader color formula.

---

## Self-Review Results

1. **TZ coverage (оценка 4):** Доп. 1 → ortho+combo; Доп. 2 → DragFloat3 ×3 + model composition T·R·S; Доп. 3 → orbit with play/pause/speed/radius/height/spin/restart; Доп. 4 → ColorEdit3 + tint multiply in fragment; Доп. 5 → procedural colors. Grade-3 features untouched (depth, culling CCW fix, MSAA, dynamic viewport, validation-clean).
2. **Placeholder scan:** none — all code present in the executed steps.
3. **Type consistency:** `GlobalUniforms` layout identical in C++ (`Mat4×3 + float[4]`, 208 B) and GLSL (`mat4×3 + vec4`); descriptor range `sizeof(GlobalUniforms)`; attribute location 2 matches `layout(location = 2)`.
