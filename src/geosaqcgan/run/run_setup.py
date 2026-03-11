#!/usr/bin/env python 

"""
  This script is expected to be run from the GEOSAQcGAN/install/bin folder.
  It assumes that the template file forecast_run.j is available in the folder.

  The script asks the user to provide:
    - an experiment name (exp_name)
    - the group id (group_id), i.e., the NCCS sponsor code to be used in SLURM.
 
  It will then create an experiment directory that has a self-contained and ready
  to use SLURM script forecast_run.j.
"""

from pathlib import Path
import sys
import os
import glob
import subprocess
import shutil


def print_message():
    mssg = """
    ---------------------------------------------------------------------------------
    This setup script creates a self-contained experiment directory to run
    a forecast experiment.

    The script is interactive and asks the user to provide:

        - an experiment name (exp_name)
        - the group id (group_id), i.e., the NCCS sponsor code to be used in SLURM.
 
    It will then create an experiment directory that has a self-contained and ready
    to use run scripts
    ---------------------------------------------------------------------------------
    """
    print(mssg)

def search_replace_in_file(template_filename: Path, 
                            target_dir: Path, 
                            dict_words: dict) -> None:
    """
    Take a file template to search and replace collection of words.
    The new file (with the same name) will be created in the target directory.

    Parameters
    ----------
    template_filename : Path
       Template file path.
    target_dir : Path
       Target directory where the new file will be created.
    dict_words : dict
       Dictionary where the keys are old words and the corresponding values
       are the new words.
    """
    new_filename = target_dir / template_filename.name
    shutil.copy(template_filename, new_filename)

    try:
        with open(new_filename, 'r') as fid:
            file_content = fid.read()

        for key in dict_words:
            file_content = file_content.replace(key, dict_words[key])
            print(f"Successfully replaced '{key}' with '{dict_words[key]}' in '{new_filename}'.")

        with open(new_filename, 'w') as file:
            file.write(file_content)

    except FileNotFoundError:
        print(f"Error: File '{new_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_experiment_directory():

    # Get the current directory
    # Will be in the form FULL_PATH/GEOSAQcGAN/install/bin
    current_directory = Path.cwd()

    # Determine the source code main directory
    # Will be FULL_PATH/GEOSAQcGAN
    source_directory = current_directory.parent.parent 

    reference_directory = source_directory.parent

    # Obtain the experiment name
    experiment_name = input("Provide the experiment name (one word):  ")
    experiment_name = experiment_name.strip()

    if not experiment_name:
        print("You need to provide and experiment name")
        sys.exit()

    if len(experiment_name.split()) > 1:
        print(f"The experiment name ({experiment_name}) should be in one word.")
        sys.exit()

    # Create the experiment directory

    experiment_directory = reference_directory / experiment_name
    print(f"The following experiment directory will be created: \n\n\t {experiment_directory}")
    print()

    experiment_directory.mkdir(parents=True, exist_ok=True)

    # Get the sponsor code id

    result = subprocess.run(["groups"], shell=True, capture_output=True, text=True)

    groups = result.stdout.strip().split()

    print(f"The list of available group ids is: \n\n\t {result.stdout.strip()}")
    print()
    my_group = input(f"Provide the group id do you want to use (default: s2825): ")
    my_group = my_group.strip()

    if not my_group:
        my_group = 's2825'

    print()
    print(f"Your group is is: {my_group}")
    print()

    if my_group not in groups:
        print(f"You selected and invalid group.")
        print(f"The group {groups[0]} will be used.")
        print(f"You can change the group id in the SLURM script available in the experiment directory")
        print()
 
    # Copy other scripts
    script_src_dir = current_directory.parent / "etc/GEOS_AQcGAN/run/scripts"

    for file in script_src_dir.glob('*'):
        if file.name.startswith("run_"):
            dict_words = {"@SRCDIR": str(source_directory), "@GROUPID": my_group}
            print(file,experiment_directory)
            search_replace_in_file(file, experiment_directory, dict_words)
        else:
            shutil.copy(file, experiment_directory)


    # Copy the GEOS AQcGAN run config files to the experiment directory.
    config_src_dir = current_directory.parent / "etc/GEOS_AQcGAN/run/configs"
    config_dst_dir = experiment_directory / "config"
    config_dst_dir.mkdir(parents=True, exist_ok=True)

    for config_file in config_src_dir.glob('*'):
        if config_file.is_file():
            if config_file.name == "run.config":
                dict_words = {"@EXPNAME": experiment_name,
                              "@EXPDIR": str(experiment_directory),
                              "@SRCDIR": str(source_directory), 
                              "@GROUPID": my_group
                              }
                search_replace_in_file(config_file, experiment_directory, dict_words)
            else:
                shutil.copy(config_file, config_dst_dir)

    # Copy the GEOS AQcGAN run config files to the experiment directory.
    config_src_dir = current_directory.parent / "etc/GEOS_AQcGAN/run/configs/AQcGAN"
    config_dst_dir = experiment_directory / "config"
    config_dst_dir.mkdir(parents=True, exist_ok=True)

    for root, _ , files in os.walk(config_src_dir):
        for file in files:
            src_file_path = Path(root) / file
            rel_path = src_file_path.relative_to(config_src_dir)
            dst_file_path = config_dst_dir / rel_path
            dst_file_path.parent.mkdir(parents=True, exist_ok=True)
            dict_words = {"@EXPDIR": str(experiment_directory)}
            print(src_file_path,dst_file_path.parent)
            search_replace_in_file(src_file_path, dst_file_path.parent, dict_words)

    # Copy the NASA AQcGAN config files to the run folder
    config_src_dir = current_directory.parent / "etc/NASA_AQcGAN/configs"
    config_dst_dir = experiment_directory / "config" / "NASA_AQcGAN" / "configs_all"
    shutil.copytree(config_src_dir, config_dst_dir, dirs_exist_ok=True)


    print()
    print("-"*70)
    print(f"The experiment directory was created: \n\n\t {experiment_directory}")
    print()
    print(f"Go to the folder, edit the run.config and other config files as needed")
    print()
    print("Then from the experiment directory, issue the command: ")
    print("  sbatch run_{system}.j run.config")
    print("-"*70)
    print()

if __name__ == "__main__":
    print_message()
    create_experiment_directory()
