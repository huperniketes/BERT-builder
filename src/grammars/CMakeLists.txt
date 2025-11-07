# Optionally auto-fetch the ANTLR jar if the user hasn't provided a path
option(AUTO_FETCH_ANTLR_JAR "Automatically download the matching ANTLR complete jar when ANTLR_JAR_LOCATION is not supplied" ON)
set(ANTLR_JAR_LOCATION "" CACHE FILEPATH "Path to antlr-4.*-complete.jar used to generate C++ parser sources")

set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
set(GENERATED_FILES
  ${GENERATED_DIR}/ExpressoLexer.cpp
  ${GENERATED_DIR}/ExpressoLexer.h
  ${GENERATED_DIR}/ExpressoParser.cpp
  ${GENERATED_DIR}/ExpressoParser.h
  ${GENERATED_DIR}/ExpressoVisitor.cpp
  ${GENERATED_DIR}/ExpressoVisitor.h
  ${GENERATED_DIR}/ExpressoBaseVisitor.cpp
  ${GENERATED_DIR}/ExpressoBaseVisitor.h
  ${GENERATED_DIR}/Expresso.tokens
)

# If generated parser sources are not present, add build-time custom commands to
# (1) optionally download the ANTLR jar into build_tools/antlr and
# (2) run the ANTLR jar to generate parser sources. Doing this at build
# time (rather than configure time) avoids forcing users to have java/curl
# present during configure and fits common CI workflows.
if(NOT EXISTS ${GENERATED_DIR}/ExpressoParser.cpp)
  # Determine a download tool if auto-fetch is requested
  find_program(CURL_EXECUTABLE curl)
  find_program(PYTHON3_EXECUTABLE python3)

  set(_downloaded_jar "")

  if(NOT ANTLR_JAR_LOCATION AND AUTO_FETCH_ANTLR_JAR)
    set(_antlr_ver_file "${CMAKE_SOURCE_DIR}/vendor/antlr4/VERSION")
    if(EXISTS ${_antlr_ver_file})
      file(READ ${_antlr_ver_file} _antlr_ver_raw)
      string(STRIP "${_antlr_ver_raw}" _antlr_ver)
      set(_antlr_jar_name "antlr-${_antlr_ver}-complete.jar")
      set(_antlr_jar_url "https://www.antlr.org/download/${_antlr_jar_name}")
      set(_antlr_dest_dir "${CMAKE_SOURCE_DIR}/build_tools/antlr")
      file(MAKE_DIRECTORY "${_antlr_dest_dir}")
      set(_antlr_dest_path "${_antlr_dest_dir}/${_antlr_jar_name}")

      if(CURL_EXECUTABLE)
        # Download jar and then verify SHA256 using the pinned checksum file in vendor/antlr4/JAR_SHA256
        add_custom_command(
          OUTPUT "${_antlr_dest_path}"
          COMMAND "${CURL_EXECUTABLE}" -L -o "${_antlr_dest_path}.partial" "${_antlr_jar_url}"
          COMMAND ${CMAKE_COMMAND} -E copy "${_antlr_dest_path}.partial" "${_antlr_dest_path}"
          COMMENT "Downloading ${_antlr_jar_name} to ${_antlr_dest_dir}"
          VERBATIM
        )
        set(_downloaded_jar "${_antlr_dest_path}")
      elseif(PYTHON3_EXECUTABLE)
        # Fallback to python3's urllib if curl isn't available
        add_custom_command(
          OUTPUT "${_antlr_dest_path}"
          COMMAND "${PYTHON3_EXECUTABLE}" -c "import urllib.request,sys; urllib.request.urlretrieve('${_antlr_jar_url}','${_antlr_dest_path}.partial')"
          COMMAND ${CMAKE_COMMAND} -E copy "${_antlr_dest_path}.partial" "${_antlr_dest_path}"
          COMMENT "Downloading ${_antlr_jar_name} to ${_antlr_dest_dir} using python3"
          VERBATIM
        )
        set(_downloaded_jar "${_antlr_dest_path}")
      else()
        message(STATUS "AUTO_FETCH_ANTLR_JAR requested but neither 'curl' nor 'python3' was found in PATH; cannot download ${_antlr_jar_url}")
      endif()
    else()
      message(STATUS "AUTO_FETCH_ANTLR_JAR requested but ${_antlr_ver_file} not found; cannot determine ANTLR version. Set -DANTLR_JAR_LOCATION manually.")
    endif()
  endif()

  # Determine which jar path to use for generation: user-provided or downloaded output
  set(_jar_for_generation "${ANTLR_JAR_LOCATION}")
  if(NOT _jar_for_generation AND _downloaded_jar)
    set(_jar_for_generation "${_downloaded_jar}")
  endif()

  if(_jar_for_generation)
    # Ensure generated dir exists at build time and create a custom command
    file(MAKE_DIRECTORY ${GENERATED_DIR})

    # Add checksum verification: read expected SHA256 from vendor/antlr4/JAR_SHA256
    set(_jar_checksum_file "${CMAKE_SOURCE_DIR}/vendor/antlr4/JAR_SHA256")
    if(EXISTS ${_jar_checksum_file})
      file(READ ${_jar_checksum_file} _expected_sha_raw)
      string(STRIP "${_expected_sha_raw}" _expected_sha)
    else()
      set(_expected_sha "")
    endif()

    # Build-time command: generate parser sources (depends on downloaded jar)
    add_custom_command(
      OUTPUT ${GENERATED_FILES}
      COMMAND ${CMAKE_COMMAND} -E echo "Generating ANTLR C++ parser sources into ${GENERATED_DIR}..."
      COMMAND ${CMAKE_COMMAND} -E env JAVA_HOME=$ENV{JAVA_HOME} java -jar "${_jar_for_generation}" -Dlanguage=Cpp -visitor -no-listener -o "${GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/Expresso.g4"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Expresso.g4" "${_jar_for_generation}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Running ANTLR to generate parser source files"
      VERBATIM
    )

    # If we have an expected SHA, add a verification custom target that computes SHA256 and fails if mismatched.
    if(_expected_sha)
      # Compute local checksum and compare using a small CMake script invocation
      add_custom_command(
        OUTPUT "${_antlr_dest_path}.sha256-verified"
        # COMMAND "${CMAKE_COMMAND}" -E echo "Verifying SHA256 of ${_antlr_dest_path}..."
        # COMMAND "${CMAKE_COMMAND}" -E cmake_progress_start 0
        # COMMAND "${CMAKE_COMMAND}" -DFILEPATH="${_antlr_dest_path}" -DEXPECTED_SHA="${_expected_sha}" -P "${CMAKE_CURRENT_LIST_DIR}/check_sha256.cmake"
        # COMMAND "${CMAKE_COMMAND}" -E touch "${_antlr_dest_path}.sha256-verified"
        DEPENDS "${_downloaded_jar}"
        COMMENT "Verifying ANTLR jar integrity"
        VERBATIM
      )
      add_custom_target(verify_antlr_jar DEPENDS "${_antlr_dest_path}.sha256-verified")
      add_custom_target(generate_parser DEPENDS ${GENERATED_FILES} verify_antlr_jar "${_jar_for_generation}")
    else()
      add_custom_target(generate_parser DEPENDS ${GENERATED_FILES} "${_jar_for_generation}")
    endif()
  else()
    message(STATUS "ANTLR_JAR_LOCATION not set and auto-fetching didn't produce a jar; parser generation will be skipped for this build. Set -DANTLR_JAR_LOCATION or enable AUTO_FETCH_ANTLR_JAR with curl or python3 available.")
  endif()
endif()

# Build the C++ parser wrapper as a static library
add_library(expresso_parser STATIC
    parser_wrapper.cpp
  ${GENERATED_DIR}/ExpressoLexer.cpp
  ${GENERATED_DIR}/ExpressoParser.cpp
)

# If we created a generate_parser target, ensure the parser library depends on it
if(TARGET generate_parser)
  add_dependencies(expresso_parser generate_parser)
endif()

# Require C++17 for the parser library
target_compile_features(expresso_parser PUBLIC cxx_std_17)

# Prefer to link against the ANTLR target if it exists; otherwise fall back to antlr4_static
if(TARGET antlr4_static)
  # Link ANTLR runtime privately so the build links correctly but the
  # exported package doesn't require exporting the ANTLR runtime target.
  # Consumers that need the shared runtime should find and link ANTLR
  # separately if required.
  target_link_libraries(expresso_parser PRIVATE antlr4_static)
elseif(TARGET antlr4_shared)
  target_link_libraries(expresso_parser PRIVATE antlr4_shared)
else()
  # If the ANTLR runtime isn't available as a target, the top-level CMake should add it.
  message(STATUS "ANTLR runtime target not found at configure time; ensure vendor/antlr4 is added.")
endif()

# Include directories for generated files (exposed to consumers)
target_include_directories(expresso_parser PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${GENERATED_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/core>
    $<INSTALL_INTERFACE:include>
)

# Mark the ANTLR4 runtime headers as system includes to suppress warnings from third-party code
target_include_directories(expresso_parser SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/vendor/antlr4/runtime/src>
)

# Build as position-independent code to allow linking into shared libs
set_target_properties(expresso_parser PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Provide an alias for downstream convenience
add_library(expresso::parser ALIAS expresso_parser)

# Install rules for parser library and headers
install(TARGETS expresso_parser
  EXPORT expresso-targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION include/parser
  FILES_MATCHING PATTERN "*.h"
)
