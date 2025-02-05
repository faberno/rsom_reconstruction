
<a id="readme-top"></a>
<!--
README Template from: https://github.com/othneildrew/Best-README-Template
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
<!--   <a>
    <img src="https://github.com/faberno/vessel_voxelizer/blob/main/files/logo.svg" alt="Logo" width="1000" height="100">
  </a>
 -->
  <h3 align="center">RSOM Reconstruction</h3>

  <p align="center">
    GPU accelerated 3D reconstruction of raster-scan optoacoustic data
    <br /><br />
    <a href="example.py">Demo</a>
    ·
    <a href="https://github.com/faberno/vessel_voxelizer/issues">Report Bug / Request Feature</a>
    ·
    <a href="#documentation">Documentation</a>
  </p>
</div>




[//]: # (<!-- ABOUT THE PROJECT -->)

[//]: # (## About The Project)

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
- a CUDA-capable GPU
- make sure you have one of the following gpu-array libraries installed:
  - cupy (https://docs.cupy.dev/en/stable/install.html)
  - soon supported: pytorch, jax
  
### Installation
From pypi:
```bash
pip install rsom_reconstruction
```

From source:
```bash
git clone https://github.com/faberno/rsom_reconstruction.git
cd rsom_reconstruction
pip install .
```


## Documentation
TODO

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## TODOs
- add proper documentation
- add support for pytorch and jax
- make computed sensitivity field reusable
- add metadata to output file


