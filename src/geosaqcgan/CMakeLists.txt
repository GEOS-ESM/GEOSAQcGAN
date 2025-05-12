#
# Cmake rules for geosacgan
#


# Install 'forecast' package
# -----------------------
#file(GLOB py_files geosaqcgan/*.py)
  file(GLOB py_files *.py)
  install(FILES ${py_files} DESTINATION lib/Python/geosaqcgan)
  install(DIRECTORY configs DESTINATION lib/Python/geosaqcgan)

# Python executables
# ------------------  
#  install(PROGRAMS forecast.py DESTINATION bin)
