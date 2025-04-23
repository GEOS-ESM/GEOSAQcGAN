#
# Cmake rules for JHU repos
#

# Install AQcGAN package
# ---------------------
  install(DIRECTORY AQcGAN@/src/AQcGAN DESTINATION lib/Python USE_SOURCE_PERMISSIONS)

# Install NASA-AQcGAN package
# --------------------
  install(DIRECTORY NASA-AQcGAN@/inference DESTINATION lib/Python/nasa_AQcGAN USE_SOURCE_PERMISSIONS)
  install(DIRECTORY NASA-AQcGAN@/scripts DESTINATION lib/Python/nasa_AQcGAN USE_SOURCE_PERMISSIONS)

set (PYSCRIPTS
    dataset.py
    main_ens_ic.py
    train_ens_ic.py
    )

install (
    FILES ${PYFILES} 
    DESTINATION lib/Python/nasa_AQcGAN
    )
