# GEOSAQcGAN

## How to build GEOSAQcGAN

### Preliminary Steps

#### Load Build Modules

In your `.bashrc` or `.tcshrc` or other rc file add a line:

##### NCCS

```
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
```

##### NAS
```
module use -a /nobackup/gmao_SIteam/modulefiles
```

##### GMAO Desktops
On the GMAO desktops, the SI Team modulefiles should automatically be
part of running `module avail` but if not, they are in:

```
module use -a /ford1/share/gmao_SIteam/modulefiles
```

#### PRISM
On the PRISM GPU system

```
ml use -a /home/mathomp4/modulefiles
ml mepo
```

#### On NCCS, NAS, or GMAO Desktops
Also do this in any interactive window you have. This allows you to get module files needed to correctly checkout and build the model.

Now load the `GEOSenv` module:
```
module load GEOSenv
```
which obtains the latest `git`, `CMake`, etc. modules needed to build.

### Obtain the Model

```
mepo clone git@github.com:GEOS-ESM/GEOSAQcGAN.git
```

#### Slow clones

If you notice your clone is taking a while, we recommend running:

```
mepo config set clone.partial blobless
```

This is a one-time command that tells mepo to use blobless clones for all future clones. Blobless clones are much faster than the default clone method, especially for repositories with a large history like MAPL.

### Build the Model
#### For PRISM
Skip this and see instructions below

#### Load Compiler, MPI Stack, and Baselibs
On tcsh:
```
source env@/g5_modules
```
or on bash:
```
source env@/g5_modules.sh
```
#### Run CMake
CMake generates the Makefiles needed to build the model.
```

./cmake_it 
```
This will install to a directory parallel to your `build` directory. If you prefer to install elsewhere change the install path in `cmake_it` to:
```
-DCMAKE_INSTALL_PREFIX=<path>
```
and CMake will install there.

##### Building with Debugging Flags
To build with debugging flags add:
```
-DCMAKE_BUILD_TYPE=Debug
```
to the cmake line.

#### Create Build Directory
We currently do not allow in-source builds of GEOSgcm. So we must make a directory:
```
cd build
```
The advantages of this is that you can build both a Debug and Release version with the same clone if desired.

#### Build and Install with Make
```
make -j6 install
```

#### PRISM Instructions
```
source ./cmake_it_prism
```
```
make -j6 install
```

### Create a run folder
```
cd ../install/bin
./run_setup.py
```
Then follow instructions on the screen.

## Contributing

Please check out our [contributing guidelines](CONTRIBUTING.md).

## License

All files are currently licensed under the Apache-2.0 license, see [`LICENSE`](LICENSE).

Previously, the code was licensed under the [NASA Open Source Agreement, Version 1.3](LICENSE-NOSA).


## AQcGAN v2.2.0 Aerosol Workflow

The AQcGAN v2.2.0 aerosol workflow consists of three stages:

1. Data preprocessing
2. AQcGAN inference
3. Evaluation

### Preprocessing and Inference

The preprocessing and inference workflows are provided by the **NASA-AQcGAN**
repository, which is checked out automatically as part of the `mepo`
workspace.

To simplify evaluation and ensure reproducibility, the preprocessing and
inference steps have already been completed for the AQcGAN v2.2.0 release.

The preprocessed datasets, trained model checkpoints, AQcGAN prediction files,
truth datasets, plotting metadata, and generated evaluation products are
available in the shared directory

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
```

This directory includes

```text
preprocessed_data/                         # Preprocessed input datasets
chkpts/                                   # Trained AQcGAN model checkpoints

test_ens_pred_stats_days{1-10}_chkpt100.npz
test_ens_stats_days{1-10}.npz             # Prediction and truth datasets

plot_metadata/                            # Precomputed plotting metadata

leadmean_global_and_regional_timeseries/  # Lead-time regional diagnostics
leadmean_global_maps/                     # Lead-time global diagnostics
startdate_global_and_regional_timeseries/ # Single-start regional diagnostics
startdate_global_maps/                    # Single-start global diagnostics
```

Users who wish to generate new datasets should follow the preprocessing and
inference documentation in the **NASA-AQcGAN** repository. Most users can
proceed directly to the evaluation workflows described below.

### Evaluation

Before running any of the evaluation and plotting workflows, load the GEOS
Python environment:

```bash
module load python/GEOSpyD/24.11.3-0/3.12
```

The plotting scripts use the Basemap package, which is installed in the shared
`p271` Python package directory. Add this directory to your `PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH:-}:/gpfsm/dnb06/projects/p271/python_packages"
```

The plotting scripts are configured to use Matplotlib's non-interactive `Agg`
backend, making them compatible with Discover login and compute nodes. No
additional environment variables are required.


The evaluation workflows are provided in

```text
src/geosaqcgan/evaluation/v2_2_0/
```

The evaluation package includes:

- Lead-time mean global and regional time-series diagnostics
- Lead-time mean global maps
- Start-date global and regional time-series diagnostics
- Start-date global maps

Each workflow contains its own `README.md` with detailed usage instructions.
