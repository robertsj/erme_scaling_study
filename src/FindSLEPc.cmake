
message("*** SLEPC_DIR = ${SLEPC_DIR}")
 
SET(SLEPC_INCLUDE_DIR_A "${SLEPC_DIR}/${SLEPC_ARCH}/include")
SET(SLEPC_INCLUDE_DIR   "${SLEPC_DIR}/include")
SET(SLEPC_LIB_DIR       "${SLEPC_DIR}/${SLEPC_ARCH}/lib")

FIND_LIBRARY(SLEPC_LIB_SLEPC     slepc 
             HINTS ${SLEPC_DIR}/${SLEPC_ARCH}/lib)

message( "*** SLEPc directory : ${SLEPC_DIR} ${SLEPC_LIB_SLEPC}"  )

if (SLEPC_DIR AND SLEPC_LIB_SLEPC )
  SET(SLEPC_LIBRARIES ${SLEPC_LIB_SLEPC} CACHE STRING "SLEPc libraries" FORCE)
  set(HAVE_SLEPC 1)
  set(SLEPC_FOUND ON)
  message( "-- Found SLEPc: ${SLEPC_LIBRARIES}" )
  set(SLEPC_INCLUDES ${SLEPC_INCLUDE_DIR} ${SLEPC_INCLUDE_DIR_A} CACHE STRING "SLEPc include path" FORCE)
  mark_as_advanced( SLEPC_DIR SLEPC_LIB_SLEPC SLEPC_INCLUDES SLEPC_LIBRARIES )
else()
  message( "-- SLEPc not found!" )
endif()

