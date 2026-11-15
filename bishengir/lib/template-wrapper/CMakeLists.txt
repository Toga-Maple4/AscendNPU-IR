cmake_minimum_required(VERSION 3.16)
project(bishengir_template NONE)

# Standalone build of just the BiShengIR template bitcode (meta_op*.bc).
# Uses ccec/llvm-link from the CANN bisheng_compiler; no LLVM monorepo needed.
# Placed 3 levels under AscendNPU-IR-Dev so the Template CMakeLists's
# ${CMAKE_SOURCE_DIR}/../../../bishengir/... relative include (catlass) resolves.
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../Template
  ${CMAKE_BINARY_DIR}/Template
)
