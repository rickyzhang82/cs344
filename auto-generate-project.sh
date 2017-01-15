mkdir -p cs344-project/bin
cd cs344-project
#cmake -G "Eclipse CDT4 - Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug -D CMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE ../src/
cmake -G "Eclipse CDT4 - Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug ../src/
