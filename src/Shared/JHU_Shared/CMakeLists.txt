#
# Cmake rules for JHU repos
#

# Install AQcGAN package
# ---------------------
  file(GLOB files AQcGAN@/src/*)
  install(FILES ${files} DESTINATION lib/Python/AQcGAN)



#esma_check_if_debug()

#set (subdirectories   
#  AQcGAN    NASA-AQcGAN
#  )
#foreach (subdir ${subdirectories})
#  esma_add_subdirectory (${subdir})
#endforeach()


    
