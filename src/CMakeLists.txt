# a simple program to test scaling of response matrix
# operations in petsc and slepc on parallel systems

##===========================================================================##
## project                                                                   ##
##===========================================================================##

project(scaling)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

##===========================================================================##
## cmake modules                                                             ##
##===========================================================================##

# extra cmake files
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}" )

##===========================================================================##
## user options                                                              ##
##===========================================================================##

# set the default build type (debug)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE DEBUG CACHE STRING
        "Choose the type of build, options are: DEBUG RELEASE." FORCE)
endif()

##===========================================================================##
## external libraries                                                        ##
##===========================================================================##

# get petsc
find_package(PETSc REQUIRED)
include_directories(${PETSC_INCLUDES})
add_definitions(${PETSC_DEFINITIONS})

# get slepc
find_package(SLEPc REQUIRED)
include_directories(${SLEPC_INCLUDES})

##===========================================================================##
## source                                                                    ##
##===========================================================================##

set(SRC
    driver.cc
    data.cc
)

##===========================================================================##
## executable                                                                ##
##===========================================================================##

ADD_EXECUTABLE(scaling ${SRC})

message(" petsc --> ${PETSC_LIBRARIES} ")
message(" slepc --> ${SLEPC_LIBRARIES} ")

TARGET_LINK_LIBRARIES(scaling
                      ${SLEPC_LIBRARIES}
                      ${PETSC_LIBRARIES} )
