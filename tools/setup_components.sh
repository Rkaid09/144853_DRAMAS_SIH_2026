#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p components && cd components

[ -d esp-tflite-micro ] || git clone --depth 1 \
    https://github.com/espressif/tflite-micro-esp-examples esp-tflite-micro
[ -d esp-nn ] || git clone --depth 1 https://github.com/espressif/esp-nn

python3 - <<'PY'
import io, os, re

for p in ("esp-tflite-micro/idf_component.yml", "esp-nn/idf_component.yml"):
    if not os.path.exists(p):
        continue
    s = io.open(p, encoding="utf-8").read()
    if "dependencies:" in s:
        s = re.sub(r"\ndependencies:\n(?:[ \t]+.*\n|\n)*", "\n", s)
        io.open(p, "w", encoding="utf-8").write(s)
        print("patched (1):", p)

p = "esp-tflite-micro/CMakeLists.txt"
s = io.open(p, encoding="utf-8").read()
if "set(pub_req" not in s:
    s = s.replace("set(priv_req)",
                  "# PRAHARI: upstream never sets pub_req; it relies on the IDF\n"
                  "# component manager to inject espressif/esp-nn. We vendor\n"
                  "# esp-nn locally and stripped that manifest dependency, so\n"
                  "# the link is declared by hand here.\n"
                  "set(pub_req esp-nn)\n\nset(priv_req)", 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched (2): pub_req esp-nn")

if "PRAHARI_NO_ESP_NN" not in s:
    s = io.open(p, encoding="utf-8").read()
    old_rm = '# remove sources which will be provided by esp_nn\nlist(REMOVE_ITEM srcs_kernels'
    if old_rm in s:
        s = s.replace(old_rm,
            'if(PRAHARI_NO_ESP_NN)\n'
            '    message(STATUS "PRAHARI: ESP-NN DISABLED - reference kernels")\n'
            '    set(esp_nn_kernels "")\n'
            'else()\n'
            '# remove sources which will be provided by esp_nn\n'
            'list(REMOVE_ITEM srcs_kernels', 1)
        s = s.replace('FILE(GLOB esp_nn_kernels\n          "${tfmicro_kernels_dir}/esp_nn/*.cc")',
                      'FILE(GLOB esp_nn_kernels\n          "${tfmicro_kernels_dir}/esp_nn/*.cc")\nendif()', 1)
        s = s.replace('target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)',
                      'if(NOT PRAHARI_NO_ESP_NN)\n'
                      '    target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)\n'
                      'endif()', 1)
        io.open(p, "w", encoding="utf-8").write(s)
        print("patched (3): ESP-NN switch")
PY

echo
echo "Components ready."
echo "REMINDER: after any change under components/, touch the ROOT"
echo "CMakeLists.txt or PlatformIO will silently reuse its stale CMake cache."
