**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Rayyan Shaik
  * [LinkedIn](https://www.linkedin.com/in/rayyan-shaik)
* Tested on: Windows 11, AMD Ryzen 7 5700H @ 3.3GHz 32GB, Mobile RTX 3070 (Personal Computer)

## 3D Boids Simulation
![50,000 boids demo](images/boids/50000_boids.gif)

_50,000 boids simulated with `dt=0.2`_

## Overview

This project is a CUDA implementation and visualization of the _Boids Flocking Algorithm_ in 3D. Particles, "boids", are updated using three principle rules of cohesion, separation, and alignment. I implemented 3 different techniques to apply these rules to each boid.
  1. __Naive__: Each boid (each computed by an independent CUDA thread) checks every other boid when computing its velocity update. 
  2. __Scattered Uniform Grid__: The 3D space is split up into a grid of uniform cells and each boid is assigned a cell. Grid cell indices paired with mappings to each boid are sorted so that all boids within a cell can be easily determined. Boids (each computed by an independent CUDA thread) need only search for other boids in their cell and immediate neighbors.
  3. __Coherent Uniform Grid__: On top of the scattered uniform grid approach, boid position and velocities are also reordered by cell to improve memory locality.

## Performance Analysis

### Timing Methodology
Framerate or frames per second (FPS) is defined as the total number of frames divided by elapsed time. Framerate is recorded over a 15 second period after a 5 second warmup (i.e. the first 5 seconds since the programs startup are not included).

### 1. Number of Boids vs. Framerate

### 2. Cuda Block Size & Block Count vs. Framerate

### 3. Cell Width Comparison -- 8 cells vs 27 cells

### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)
