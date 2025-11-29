# Requirements

Linux operating system. Some possible options

- Ubuntu 20.04
- WSL under Windows

## miniconda

- Install from [official link](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links)
- Integrate with bash, if you are fish shell user replace `bash` with `fish`
```
conda init bash
```

- Prevent auto-activating base environment
```
conda config --set auto_activate_base false
```
- Restart bash or whichever shell being used

## mamba

- self-update conda, otherwise it installs an extremely outdated version of `mamba`

```
conda update -n base -c defaults conda
```

- install `mamba`
```
conda install mamba -n base -c conda-forge
```

<!-- - integrate with `bash`, if you are `fish` shell user replace `bash` with
 `fish`
```
mamba init bash
``` -->

- reload shell configuration, e.g. in `bash`
```
source ~/.bashrc
```
- `mamba --version` should be >= `0.24`, otherwise
```
conda update mamba -n base -c conda-forge
```
