#
# Cmake rules for geosacgan
#


# Install 'forecast' package
# -----------------------
  file(GLOB inference_files inference/*.py)
  file(GLOB scripts_files scripts/*.py)
  install(FILES ${inference_files} DESTINATION lib/Python/NASA_AQcGAN/inference)
  install(FILES ${scripts_files} DESTINATION lib/Python/NASA_AQcGAN/scripts)
  install(DIRECTORY shared DESTINATION lib/Python/NASA_AQcGAN)
  install(DIRECTORY configs DESTINATION etc/NASA_AQcGAN)

# Python executables
# ------------------  
#  install(PROGRAMS forecast.py DESTINATION bin)
