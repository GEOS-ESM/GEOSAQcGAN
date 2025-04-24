esma_set_this()

# Install test
# ---------------------
  file(GLOB files *pkl)
  install(FILES ${files} DESTINATION bin/tests/${this})

  file(GLOB files *yaml)
  install(FILES ${files} DESTINATION bin/tests/${this})

  file(GLOB files *sh)
  install(PROGRAMS ${files} DESTINATION bin/tests/${this})

