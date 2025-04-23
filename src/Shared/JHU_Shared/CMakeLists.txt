#
# Cmake rules for JHU repos
#

# Install AQcGAN package
# ---------------------
  install(DIRECTORY AQcGAN@/src/AQcGAN DESTINATION lib/Python USE_SOURCE_PERMISSIONS)

# Install NASA-AQcGAN package
# --------------------
  install(DIRECTORY NASA-AQcGAN@/inference DESTINATION lib/Python/NASA_AQcGAN USE_SOURCE_PERMISSIONS)
  install(DIRECTORY NASA-AQcGAN@/scripts DESTINATION lib/Python/NASA_AQcGAN USE_SOURCE_PERMISSIONS)

set (PYSCRIPTS
    NASA-AQcGAN@/scripts/merge_geos_cf_ens_data.py
    NASA-AQcGAN@/scripts/preprocess_geos_cf_ens.py
    )

install (
    FILES ${PYSCRIPTS}
    DESTINATION bin/NASA_AQcGAN
    )


set (PYSCRIPTS
    NASA-AQcGAN@/dataset.py
    NASA-AQcGAN@/main_ens_ic.py
    NASA-AQcGAN@/train_ens_ic.py
    __init__.py
    )

install (
    FILES ${PYSCRIPTS} 
    DESTINATION lib/Python/NASA_AQcGAN
    )
