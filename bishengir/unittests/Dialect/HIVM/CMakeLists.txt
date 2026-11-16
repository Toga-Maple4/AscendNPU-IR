add_bishengir_unittest(BiShengIRL1ProofTests
  L1ProofTest.cpp
)

target_include_directories(BiShengIRL1ProofTests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/Dialect/HIVM/Transforms
)
target_link_libraries(BiShengIRL1ProofTests PRIVATE
  BiShengIRHIVMTransforms
  MLIRParser
)
