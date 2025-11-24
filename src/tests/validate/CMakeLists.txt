esma_set_this()

# Install test
# ---------------------
  file(GLOB files *yaml *config)
  install(FILES ${files} DESTINATION bin/tests/${this})

  file(GLOB files *sh)
  install(PROGRAMS ${files} DESTINATION bin/tests/${this})

  file(GLOB files *j)
  install(PROGRAMS ${files} DESTINATION bin)

