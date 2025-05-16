esma_set_this()

# Install test
# ---------------------
  file(GLOB files *yaml)
  install(FILES ${files} DESTINATION bin/tests/${this})

  file(GLOB files *sh)
  install(PROGRAMS ${files} DESTINATION bin/tests/${this})

  foreach (forecastfile forecast_setup.py forecast_run.j)
   install (
      PROGRAMS ${forecastfile}
      DESTINATION bin
      )
  endforeach ()
