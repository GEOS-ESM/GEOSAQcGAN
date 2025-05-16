esma_set_this()

# Install test
# ---------------------
  file(GLOB files *yaml)
  install(FILES ${files} DESTINATION bin/tests/${this})

  file(GLOB files *sh)
  install(PROGRAMS ${files} DESTINATION bin/tests/${this})

  install(FILES forecast_setup.py DESTINATION bin/${this})
  install(FILES forecast_run.j DESTINATION bin/${this})
