# PMF-RRT Manipulator Planning Demo

This repository provides a simplified research version of the PMF-RRT implementation associated with the manuscript:

**The Path Planning of Locally Learned Directional Guidance for RRT via Probability Mass Function**

The code demonstrates PMF-RRT in a 7-DoF Franka Emika Panda manipulator planning scenario using PyBullet. It is intended for academic verification of the main algorithmic procedure and representative simulation behavior.

This repository is not the complete development codebase. Some unpublished modules, exploratory tuning scripts, environment-specific scripts, and ongoing dissertation-related extensions are not included.

## Overview

PMF-RRT is a sampling-based path planning algorithm that introduces local directional learning into the RRT framework. Each tree node maintains a local probability mass function (PMF) over predefined direction bins. The PMF is updated through positive and negative feedback during tree expansion, allowing the planner to reduce repeated exploration toward collision-prone directions and reinforce locally feasible directions.

The provided implementation includes dynamic goal bias, node maturity judgment, PMF-guided expansion, positive and negative PMF feedback, PMF inheritance from parent nodes, PyBullet-based collision checking, and trajectory visualization for a 7-DoF manipulator.

## Requirements

This implementation requires Python 3.8 or later.

Required Python packages:

```bash
pip install numpy matplotlib pybullet
```

Recommended tested environment:

```text
Python 3.9
numpy 1.24.4
matplotlib 3.7.5
pybullet 3.2.6
```

The `pybullet_data` module is installed automatically with `pybullet` and does not need to be installed separately.

A compatible `requirements.txt` file can be:

```text
numpy>=1.20
matplotlib>=3.5
pybullet>=3.2.5
```

## Notes

The simulation is executed in PyBullet GUI mode:

```python
p.connect(p.GUI)
```

If the code is run on a server or a system without display support, the GUI mode may fail. In that case, the script can be modified to use:

```python
p.connect(p.DIRECT)
```

However, trajectory playback visualization will not be shown in DIRECT mode.

The plotting functions use Chinese labels and specify the `SimHei` font. If this font is not installed, the simulation can still run, but Chinese characters in saved figures may not display correctly.

## Reproducibility

This repository provides a simplified research implementation for academic verification and reproducibility. It is intended to demonstrate the main PMF-RRT planning mechanism and representative manipulator simulation behavior.

The complete development codebase, unpublished modules, exploratory tuning scripts, and ongoing dissertation-related extensions are not publicly released at this stage.

## License

This simplified research version is provided for academic and research use. Please cite the associated manuscript if you use or adapt this implementation.
