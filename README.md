# OROCHI Simulator Control and Processing

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Contributing](../CONTRIBUTING.md)

## About <a name = "about"></a>

This repository contains notebooks that walk through image capture and reflectance calibration of images captured with the OROCHI simulator.

The steps are:
1. Reflectance Calibration Imaging
2. Sample Imaging
3. Geometric Calibration Target Imaging
4. Dark Signal Correction
5. Reflectance Calibation
6. Image Co-alignment (TBC)

Images are exported in TIF format.

## Installation <a name = "installation"></a>

Python modules used are:
- ctypes
- pathlib
- matplotlib
- numpy
- opencv
- pandas
- tifffile
- tisgrabber

Control of the OROCHI simulator is via the Python C-Wrapper for the 'tisgrabber' DLL, a library for interfacing with GigE TIS (The Imaging Source) cameras. Note that this DLL is only compatible with Windows. A copy of tisgrabber package is included in this repository.

Setup of a conda environment and the installation of the neccesary packages is as follows.

First, ensure conda and conda develop are installed.

Open a terminal, and navigate to the repository directory, and execute the following commands.

To install an environment with the required packages:
```S
conda env create -f environment.yml
```
Once created, follow the prompt to activate the environment.

To add the path to tisgrabber to the conda environment:
```
conda-develop ./tisgrabber/samples
conda-develop ./src
```

This is all that is required to setup the control software.

## Using the Notebooks

We configure the cameras with camera_config.ipynb, and begin the multichannel_reflectance_imaging.pynb, and then once complete use the multichannel_reflectance_processing.pynb notebook to process these to reflectance.
