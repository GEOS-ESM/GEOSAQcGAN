#
# Cmake rules for geosacgan
#
add_subdirectory (run)

# Install geosaqcgan
# -----------------------
  install(DIRECTORY preprocess/ DESTINATION lib/Python/NASA_AQcGAN/scripts FILES_MATCHING PATTERN "*.py")
  install(DIRECTORY postprocess/ DESTINATION lib/Python/NASA_AQcGAN/scripts FILES_MATCHING PATTERN "*.py")
  install(DIRECTORY shared DESTINATION lib/Python/NASA_AQcGAN)
  install(DIRECTORY notebooks DESTINATION etc/GEOS_AQcGAN)

