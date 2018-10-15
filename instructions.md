# Dynamic Compartmentalization in HTM Dendrites
See file: dendrites.pdf

## Installation and Run instructions:
1) Install Nupic(v1.0.3) and NAB
2) Put this directory (dendrite_experiments) at NAB/nab/detectors/
3) Insert the following code snippet at the end of the file NAB/run.py, just before the call to main:
```python
  if "dendrite" in args.detectors:
    from nab.detectors.dendrite_experiments.dendrite_detector import (
      DendriteDetector)
```
4) Run the detector as:
```bash
$ run.py --detectors dendrite --detect --score --optimize --normalize
```
