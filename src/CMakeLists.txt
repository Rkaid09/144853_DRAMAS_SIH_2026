idf_component_register(
    SRCS "main.cpp"
         "../firmware/core/prahari_logmel.c"
         "../firmware/core/prahari_link.c"
    INCLUDE_DIRS "." "../firmware/core" "../firmware/hal"
    REQUIRES esp-tflite-micro driver lwip
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++14>)

target_compile_options(${COMPONENT_LIB} PRIVATE -O3)

if(PRAHARI_NO_ESP_NN)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE PRAHARI_NO_ESP_NN=1)
endif()
