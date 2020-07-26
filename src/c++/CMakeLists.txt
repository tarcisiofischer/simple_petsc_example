find_package(pybind11 REQUIRED)
find_package(PETSc REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(
    simple_petsc_example
    SHARED
        _simple_petsc_example.cpp
        solver.hpp
        solver.cpp
)

target_link_libraries(
	simple_petsc_example
	PRIVATE
		pybind11::module
        PETSc::PETSc
)

set_target_properties(
	simple_petsc_example
	PROPERTIES
		OUTPUT_NAME _simple_petsc_example
		PREFIX ""
)

MESSAGE(STATUS "Artifacts path: $ENV{ARTIFACTS_PATH}")

install(
	TARGETS simple_petsc_example
	DESTINATION $ENV{ARTIFACTS_PATH}
)
