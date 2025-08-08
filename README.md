# Multi-objective optimization algorithm implementation for public transit routes in Cochabamba municipality 

## Overview

This project is an implementation of multi-objective optimization for public transit routes in Cochabamba municipality.

The core components include:
- Pre processing of data to obtain population densities, concurrence, travel times between stops, distance between stops, among other data.
- Generation of stops and the origin destination matrix
- Custom implementation of the Sampler, Problem, Mutation and Crossover for Pymoo's NSGA-2 algorythm
- Validation against a cluster of official Cochabamba public transit routes from 2018, and comparison of network connectivity 

---
