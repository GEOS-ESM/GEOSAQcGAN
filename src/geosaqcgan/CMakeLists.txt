#
# Cmake rules for geosacgan
#


# Install 'forecast' package
# -----------------------
  file(GLOB py_files geosaqcgan/*.py)
  install(FILES ${py_files} DESTINATION lib/Python/geosaqcgan)

# Python executables
# ------------------  
#  install(PROGRAMS forecast.py DESTINATION bin)
