
  install(DIRECTORY configs DESTINATION etc/GEOS_AQcGAN/run)
  file(GLOB script_files scripts/*)
  install(PROGRAMS ${script_files} DESTINATION etc/GEOS_AQcGAN/run/scripts)
  install(PROGRAMS run_setup.py DESTINATION bin)