esma_set_this()

# Install test
# ---------------------
  file(GLOB files *pkl)
  install(FILES ${files} DESTINATION bin/tests/${this})

  file(GLOB files *yaml)
  install(FILES ${files} DESTINATION bin/tests/${this})

  file(GLOB files *sh)
  install(PROGRAMS ${files} DESTINATION bin/tests/${this})

  foreach (valfile validation_run_discover.j validation_run_prism.j)
   install (
      PROGRAMS ${valfile}
      DESTINATION bin
      )
  endforeach ()

