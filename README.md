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

## Contributing

Please check out our [contributing guidelines](CONTRIBUTING.md).

## License

All files are currently licensed under the Apache-2.0 license, see [`LICENSE`](LICENSE).

Previously, the code was licensed under the [NASA Open Source Agreement, Version 1.3](LICENSE-NOSA).
