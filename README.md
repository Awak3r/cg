# Лабораторная работа 1 — Параллелепипед (вариант 3)

Vulkan + GLFW + ImGui, C++20. Реализация на базе стартового репозитория
[vkadeemerr/vulkan-starter-app](https://github.com/vkadeemerr/vulkan-starter-app).

## Вариант

**Вариант 3: параллелепипед** — прямоугольный параллелепипед 2.0 x 1.4 x 1.0
(8 вершин углов, 24 вершины с нормалями и цветами граней, 12 треугольников,
36 индексов), центр в начале координат.

## Сборка

Нужны: компилятор C++20, CMake 3.20+, Vulkan SDK (`glslc` в PATH).

```bash
cmake --preset debug          # Linux (GCC/Clang)
cmake --build build-debug --parallel
```

Windows: `cmake --preset msvc-debug` (Visual Studio) или `cmake --preset mingw-debug` (MinGW),
затем `cmake --build build-debug --parallel`.

## Запуск

Рабочая директория — корень проекта (пути к шейдерам относительные):

```bash
./build-debug/vulkan-starter-app
```

Под WSL: `./run-wsl.sh` (ждёт готовности WSLg и запускает приложение).

## Что реализовано

### Базовый уровень (оценка 3)

- [x] Окно GLFW, инициализация Vulkan, graphics pipeline.
- [x] Параллелепипед: vertex buffer + index buffer (VMA), рисование `vkCmdDrawIndexed`.
- [x] MVP-матрицы: model/view/projection через uniform buffer (`std140`, 208 байт).
- [x] Depth test (`VK_COMPARE_OP_LESS`, clear depth = 1.0) — внутренность не видна.
- [x] Back-face culling (`VK_CULL_MODE_BACK_BIT`, лицевые — CCW-наружу грани).
- [x] Dynamic viewport/scissor (корректный resize окна).
- [x] Flat-освещение граней в фрагментном шейдере (нормали — outward нормаль грани).
- [x] MSAA 4x — сглаженные рёбра.
- [x] Все `vkCreate*` имеют парные `vkDestroy*` / `vmaDestroyBuffer`,
  validation layers чистые.

### Дополнительные задания (оценка 4)

- [x] **Доп. 1** — переключение перспективной/ортографической проекции (ImGui Combo).
- [x] **Доп. 2** — position / rotation / scale (DragFloat3), модель = T·Rz·Ry·Rx·S.
- [x] **Доп. 3** — анимация по круговой орбите: play/pause, скорость (оборотов/сек),
      радиус, высота орбиты, опциональное вращение бокса вдоль орбиты,
      перезапуск фазы, clamp длинных пауз.
- [x] **Доп. 4** — `ImGui::ColorEdit3`, tint умножается на цвет вершины во фрагментном шейдере.
- [x] **Доп. 5** — процедурные цвета вершин: нормированная локальная позиция
      `(pos + halfExtent) / (2 * halfExtent)`, т.е. X→R, Y→G, Z→B.

### Шейдеры

`shaders/box.vert`, `shaders/box.frag` — компилируются через `glslc` на этапе сборки
(CMake target `shaders`).

Финальный цвет фрагмента: `vertexColor * tint * lighting`
(освещение — Lambert: hemisphere ambient + key/fill источники).

## Управление

| Элемент | Назначение |
|---|---|
| Projection | Perspective / Orthographic |
| Position, Rotation, Scale | трансформации бокса |
| Tint | цвет, умножаемый на процедурный цвет вершины |
| Play / pause motion | запуск/пауза орбитальной анимации |
| Speed, Radius, Height | параметры орбиты |
| Also rotate along orbit | дополнительное вращение бокса вдоль орбиты |
| Restart phase / Reset all | сброс фазы / полный сброс состояния |
