find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.SABIModule)

Python_add_library(_mineru_llama_cpp MODULE USE_SABI 3.11 WITH_SOABI
    binding.cpp
    engine_core.cpp
)
set_target_properties(_mineru_llama_cpp PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

# dladdr() (engine_core.cpp's load_packaged_backends()) is a glibc extension;
# its declaration in <dlfcn.h> is hidden unless _GNU_SOURCE is defined --
# same requirement ggml itself already has for its own dladdr() use
# (ggml/src/CMakeLists.txt adds this for Linux/Android). Not needed on
# Darwin, where dladdr() is always visible in the system <dlfcn.h>.
if (CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Android")
    target_compile_definitions(_mineru_llama_cpp PRIVATE _GNU_SOURCE)
endif()

# These includes mirror the ones proven to work in the technical-validation
# spike's CMakeLists.txt, just re-pathed to the submodule. We add them
# explicitly rather than relying on transitive include propagation from the
# server-context/mtmd/etc. CMake targets, since that propagation was never
# verified for this llama.cpp version -- this explicit form is known-good.
target_include_directories(_mineru_llama_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/common
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/tools/server
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor
)

target_link_libraries(_mineru_llama_cpp PRIVATE
    server-context
    # server-common 使用 httplib::hosted_at，上游 server-context 未传递此静态库。
    cpp-httplib
    llama-common-base
    llama-common
    mtmd
    llama
    ggml
    ggml-base
)

# 将构建时的子模块提交写入扩展，避免诊断误读工作目录中的其他版本。
execute_process(COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/third_party/llama.cpp"
    OUTPUT_VARIABLE MINERU_LLAMA_CPP_LLAMA_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE MINERU_LLAMA_CPP_REVISION_RESULT ERROR_QUIET)
execute_process(COMMAND git rev-parse --show-toplevel
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/third_party/llama.cpp"
    OUTPUT_VARIABLE MINERU_LLAMA_CPP_GIT_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
file(READ "${CMAKE_SOURCE_DIR}/llama-cpp-revision.txt" MINERU_LLAMA_CPP_EXPECTED_COMMIT)
string(STRIP "${MINERU_LLAMA_CPP_EXPECTED_COMMIT}" MINERU_LLAMA_CPP_EXPECTED_COMMIT)
file(REAL_PATH "${CMAKE_SOURCE_DIR}/third_party/llama.cpp" MINERU_LLAMA_CPP_SOURCE_ROOT)
# sdist 没有子模块 Git 元数据；禁止误读解压目录外层仓库的提交。
if (NOT MINERU_LLAMA_CPP_REVISION_RESULT EQUAL 0 OR
    NOT MINERU_LLAMA_CPP_GIT_ROOT STREQUAL MINERU_LLAMA_CPP_SOURCE_ROOT)
    set(MINERU_LLAMA_CPP_LLAMA_COMMIT "${MINERU_LLAMA_CPP_EXPECTED_COMMIT}")
elseif (NOT MINERU_LLAMA_CPP_LLAMA_COMMIT STREQUAL MINERU_LLAMA_CPP_EXPECTED_COMMIT)
    message(FATAL_ERROR "llama.cpp checkout differs from llama-cpp-revision.txt; update both together")
endif()
target_compile_definitions(_mineru_llama_cpp PRIVATE
    MINERU_LLAMA_CPP_LLAMA_COMMIT="${MINERU_LLAMA_CPP_LLAMA_COMMIT}")

# ggml-cpu/ggml-metal are ggml_add_backend_library() targets (see
# ggml/src/CMakeLists.txt). With GGML_BACKEND_DL=ON (forced in the top-level
# CMakeLists.txt) they build as standalone MODULE libraries meant to be
# dlopen'd at runtime via load_packaged_backends() (engine_core.cpp), not
# linked at build time -- linking a MODULE library directly isn't the
# supported usage. This project always builds with GGML_BACKEND_DL=ON, so
# these branches are effectively dead today; kept so the target still links
# correctly if that ever changes back to OFF.
if (NOT GGML_BACKEND_DL)
    target_link_libraries(_mineru_llama_cpp PRIVATE ggml-cpu)
endif()

# GGML_METAL is a CACHE variable set by ggml's own CMakeLists.txt (via
# add_subdirectory(third_party/llama.cpp) in the top-level CMakeLists.txt),
# auto-detected per-platform. We deliberately do not set it ourselves (design
# spec §7) — we only react to whatever it resolved to.
if (GGML_METAL AND NOT GGML_BACKEND_DL)
    target_link_libraries(_mineru_llama_cpp PRIVATE ggml-metal)
endif()

if (APPLE)
    target_link_libraries(_mineru_llama_cpp PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics"
    )
endif()

# BUILD_RPATH points at the build tree so the .so can load its sibling
# ggml/llama/mtmd dylibs straight out of wherever llama.cpp's own
# CMAKE_RUNTIME_OUTPUT_DIRECTORY put them (shared across the whole build tree
# since CMAKE_BINARY_DIR is top-level-wide) -- this is what editable
# "inplace" installs run against, since there's no install step to relocate
# anything. A generator expression is used rather than a hardcoded path, so
# this keeps working if llama.cpp's internal output-directory layout changes.
#
# INSTALL_RPATH is relative (@loader_path/lib on macOS, $ORIGIN/lib on
# Linux) -- no ".." -- since CMAKE_INSTALL_LIBDIR is now
# "mineru_llama_cpp/lib" (top-level CMakeLists.txt), making lib/ a *sibling*
# of mineru_llama_cpp/_mineru_llama_cpp.so, not one level up from it as it
# was when lib/ sat at the wheel root. Baking in the build-tree absolute path
# here would make every wheel unusable on any machine other than the one it
# was built on. This overrides the directory-scoped CMAKE_INSTALL_RPATH set
# in the top-level CMakeLists.txt, since explicit target properties take
# precedence -- that top-level value (unchanged: $ORIGIN;$ORIGIN/../lib /
# @loader_path;@loader_path/../lib) is still correct for llama-server and
# the backend MODULEs, which now live in mineru_llama_cpp/bin/ (one level
# above mineru_llama_cpp/lib/, same as before) -- only this target's own
# nesting level changed.
#
# Windows branch leaves INSTALL_RPATH empty: Windows has no $ORIGIN, DLLs
# resolve via the default search order (loaded-module dir + PATH). See the
# top-level CMakeLists.txt's CMAKE_INSTALL_RPATH comment for the full story.
set_target_properties(_mineru_llama_cpp PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:llama>"
    INSTALL_RPATH "$<IF:$<BOOL:${APPLE}>,@loader_path/lib,$<IF:$<BOOL:${WIN32}>,,$ORIGIN/lib>>"
)

# editable inplace 不执行 install，将扩展放回源码目录以便导入和类型检查。
# 普通 wheel 在构建目录产出，再由 install 复制当前 Python ABI 的扩展。
if (SKBUILD_STATE STREQUAL "editable")
    set_target_properties(_mineru_llama_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/mineru_llama_cpp")
endif()

install(TARGETS _mineru_llama_cpp DESTINATION mineru_llama_cpp)

# llama-server links against this at runtime (@rpath/libllama-server-impl.dylib),
# but llama.cpp's own tools/server/CMakeLists.txt only installs it when
# LLAMA_TOOLS_INSTALL is ON, which defaults to LLAMA_STANDALONE (OFF here,
# since we add_subdirectory() llama.cpp instead of building it standalone).
# Installing it here directly avoids flipping LLAMA_TOOLS_INSTALL globally,
# which would also install a dozen unrelated llama.cpp CLI tools we don't
# ship (llama-tts, llama-mtmd-cli, llama-fit-params, etc).
install(TARGETS llama-server-impl LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# libgomp.so.1 bundling for Linux is handled in the top-level CMakeLists.txt
# (after add_subdirectory(third_party/llama.cpp)) -- it needs top-level scope
# to reliably see ggml's OpenMP state (OpenMP::OpenMP_CXX imported target).
# See that file for the find_library + install(FILES) logic.
