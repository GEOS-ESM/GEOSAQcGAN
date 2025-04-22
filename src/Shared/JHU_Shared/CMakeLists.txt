esma_check_if_debug()

set (subdirectories   
  AQcGAN    NASA-AQcGAN
  )
foreach (subdir ${subdirectories})
  esma_add_subdirectory (${subdir})
endforeach()
    
