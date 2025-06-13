# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Option to write validation file during preprocessing
- Option to create predictions in either validation (`val`) or forecast (`fcst`) mode

### Changed
- Improvments to the run script
- Change output filenames and location
- Use flags instead of positional arguments to avoid errors
- Copy AQcGAN yaml files to the experiment folder so we can make changes specific to the experiment
- General cleanup & streamlining of code

### Fixed
- Bug fix to mean and std dev saved in norm_stats

### Removed

### Deprecated

## [v1.0,0] 05-16-2025

### Added
- tests directory to run a single inference test
- cmake rules to install JHU repos
- geos specific preprocess script
- forecast test directory that excercises new goesaqcgan codes
- write outputs to netcdf file
- add setup script for forecast test
- Add in the preproc YAML configuration file a parameter determining the frequency (in hours) we want to read files.
- Added scripts and updated cmake to be able to run the forecast test on PRISM
### Changed
- refactors geosacgan to follow nasa-aqcgan dir structure
- update forcast slurm script to clobber old data files
### Fixed
- must update norm stats file for n_timesteps

