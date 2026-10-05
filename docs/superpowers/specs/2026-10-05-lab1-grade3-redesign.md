# Lab 1 — Parallelepiped (variant 3): базовый уровень на оценку 3, безупречный вид

Дата: 2026-10-05
Статус: утверждён пользователем (дизайн), ожидает ревью спеки

## 1. Цель

Привести lab1 к **базовому уровню ТЗ (оценка 3)** — параллелепипед + MVP + depth test +
back-face culling + ImGui + Vulkan — и сделать картинку безупречной:

- не видно внутренней/задних граней бокса;
- грани визуально различимы (flat-освещение);
- рёбра сглажены (MSAA 4x).

Всё, что делалось под оценки 4/5 и больше, — удалить как лишнее.

## 2. Root cause визуального бага

В `application.cpp: createPipeline()` задано `frontFace = VK_FRONT_FACE_CLOCKWISE`
с комментарием, что проекция негативит Y, поэтому нужен CW.

Проверка знака:

- геометрия: все 6 граней обведены CCW при взгляде снаружи (`faceCorners`);
- view — чистый translation по Z (детерминант +1, winding не меняет);
- projection негативит Y → один флип → в framebuffer-координатах
  мировые CCW-наружу грани дают положительную signed area, что по спецификации
  Vulkan значит **CCW**.

С `CLOCKWISE` конвейер считает лицевыми задние грани → внешние грани отсекаются
culling'ом → видна внутренность. Фикс: `VK_FRONT_FACE_COUNTER_CLOCKWISE`.
Тот же фикс чинит освещение (раньше нормали смотрели внутрь).

## 3. Что удаляем из текущей версии

Из `application.cpp`:

- второй объект (`object_count`, `object_buffers/object_memories/object_sets`,
  выбор бокса в UI, сдвиг фазы);
- орбитальную анимацию (play/pause, speed, radius, height, spin, phase);
- UI трансформаций (Position/Rotation/Scale DragFloat3, ColorEdit);
- переключение Perspective/Orthographic;
- второй uniform-буфер на объект, второй descriptor set layout, pool maxSets=3;
- tint в uniform'ах и шейдерах.

Из шейдеров: `ObjectUniforms` (set 1), tint.

Файлы `graphics_internal.hpp/cpp`, `main.cpp`, `CMakeLists.txt`, `CMakePresets.json`
**не откатываются к шаблону** — в них остаются только принятые практические правки:

- MSAA 4x в `graphics_internal.cpp/hpp` (решено голосованием: оставить);
- WSL X11-хинт и ShowWindow/FocusWindow в `main.cpp` (оставить);
- `glslc REQUIRED` + `find_program` в `CMakeLists.txt` (оставить).

## 4. Оставляемая архитектура (application.cpp)

### 4.1. Математика (собственная, без GLM)

`Vec3`, `Mat4` (column-major), `identity`, `multiply`, `translation`, `scale`,
`rotation(x,y,z)` в градусах (порядок Rz·Ry·Rx), `perspective(aspect)` 45°/0.1/100
с негативным Y (Vulkan screen space, depth [0,1] через 0..1 формулу),
`orthographic` — **удаляется** (не нужен на тройку; проекция по ТЗ — перспективная).

### 4.2. Геометрия

Без изменений: 24 вершины `{position, normal}` (4 угла × 6 граней, нормаль грани),
36 индексов (CCW наружу), полуразмеры `hx=1.0, hy=0.7, hz=0.5`.

### 4.3. Uniform

Один uniform-буфер, std140, persistently mapped:

```cpp
struct alignas(16) GlobalUniforms {
    Mat4 model; // 64 B
    Mat4 view;  // 64 B
    Mat4 proj;  // 64 B
};              // 192 B
```

### 4.4. Дескрипторы

- 1 `VkDescriptorSetLayout`: binding 0, `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER`,
  stageFlags = `VK_SHADER_STAGE_VERTEX_BIT` (tint'а во фрагментном больше нет);
- 1 `VkDescriptorPool` (maxSets=1, descriptorCount=1);
- 1 `VkDescriptorSet`.

### 4.5. Сцена и ракурс

- model = `rotation(18°, 25°, 0°)` — статичный ракурс, видно 3 грани;
- view = `translation({0, 0, -5})` — бокс крупнее в кадре, чем с -6;
- proj = perspective 45°, near 0.1, far 100.

### 4.6. Pipeline

Единственное изменение относительно текущего кода:

- `frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE` (фикс из §2).

Остальное остаётся как сейчас (зафиксировано для ясности):

- rasterization: cullMode = BACK_BIT, polygonMode = FILL;
- depth: test+write, `VK_COMPARE_OP_LESS`; clear depth 1.0;
- MSAA: `graphics::internal::msaa_samples` (4x);
- vertex input: location 0 = position, location 1 = normal.

### 4.7. ImGui

Минимальное окно, доказывающее интеграцию ImGui, без контролов трансформаций:

```
Lab 1 — Parallelepiped (variant 3)
Vulkan + GLFW + ImGui
FPS: <усреднённый за ~0.5 с>
```

FPS считается в `update` по `time` (скользящее среднее), только для информации.

### 4.8. render()

Единственная запись uniform'ов (model/view/proj) + привязка одного set +
один `vkCmdDrawIndexed` на 36 индексов. Орбитальной логики нет.

## 5. Шейдеры

`shaders/box.vert`:

- in: `location 0` position, `location 1` normal;
- uniform set 0 binding 0: `GlobalUniforms { model, view, proj }`;
- out: `location 0` `vec3 worldNormal = mat3(transpose(inverse(model))) * inNormal;`
  (корректно при любых матрицах; стоимость мизерная);
- `gl_Position = proj * view * model * vec4(position, 1)`.

`shaders/box.frag`:

- in: `location 0` worldNormal;
- flat Lambert: hemisphere ambient (0.16..0.30 по normal.y) +
  key (dir 0.3/0.85/0.45, int 0.8) + fill (dir -0.45/-0.25/0.35, int 0.35);
- out = material * min(ambient+diffuse, 1);
- material — константа в шейдере `vec3(0.35, 0.55, 0.85)` (светло-синий, как раньше).

Компилируются через glslc (уже подключено в CMake: `compile_shader(box.vert/frag)`).

## 6. README.md

Переписывается под тройку:

- вариант (3 — параллелепипед) и размеры 2.0×1.4×1.0;
- сборка/запуск (Linux/WSL: presets debug/release, `./run-wsl.sh`), рабочая директория — корень;
- что реализовано: только базовый чеклист оценки 3 из ТЗ
  (окно, Vulkan, GLFW, ImGui, vertex/index/uniform буферы, MVP, depth test,
  back-face culling, dynamic viewport/scissor, resize, чистая валидация,
  парные destroy);
- управление: окно закрывается крестиком; ImGui-панель информационная;
- без упоминаний уровней 4/5.

## 7. Не входит в объём (осознанно)

- Переключение проекции, трансформации, цвет, анимация, второй объект,
  несколько descriptor sets (уровни 4/5 из ТЗ).
- GLM, любые новые зависимости.
- Изменения в `graphics_internal.cpp/hpp`, `main.cpp`, `CMakeLists.txt`
  (кроме уже принятых правок, перечисленных в §3).
- `theotry/` не трогаем.

## 8. Верификация

1. `cmake --preset debug` + `cmake --build build-debug --parallel` — без ошибок.
2. `cmake --preset release` + build release — без ошибок.
3. Запуск из корня lab1 (через `./run-wsl.sh`): окно открывается,
   консоль без сообщений validation layers (stdout/stderr логируем в файл и проверяем).
4. Аналитическая проверка winding уже сделана (§2); финальную картинку
   подтверждает пользователь (CLI-скриншоты в WSLg получаются чёрными —
   содержимое окна недоступно для автоскриншота).
5. Корректное закрытие: без ошибок валидации при destroy.

## 9. Риски

- Если после фикса winding картинка всё ещё выглядит неверно — временно ставим
  `VK_CULL_MODE_NONE` для диагностики (приём из ТЗ §15.2), затем выбираем
  согласованный `frontFace`.
- CLI-скриншоты не работают → визуальную приемку делает пользователь.
