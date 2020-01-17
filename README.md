
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">ClassificationTool</h3>

  <p align="center">
    A simple to use QGIS plugin for supervised classification of remote sensing images
    <br />
    <br />
  <img src=images/header.png />
   <br />
  <br />
    <a href="https://github.com/SteveMHill/ClassificationTool/issues">Report Bug</a>
    ·
    <a href="https://github.com/SteveMHill/ClassificationTool/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
QGIS version 3.4.0 or higher 

The ClassificationTool requires the following python packages:

- [scikit-learn] (https://scikit-learn.org)
- [rasterio] (https://rasterio.readthedocs.io/en/stable/)
- [fiona] (https://github.com/Toblerity/Fiona)
- [pandas] (https://pandas.pydata.org/)

### Installation

#### Install additional python packages:

##### Linux

Use pip and install additional packages required:

`pip3 install pandas Fiona rasterio scikit-learn`


##### Windows

1. Close QGIS, if it is open.
2. Start the OSGeo4W Shell with admin rights
3. Enter `call py3_env.bat` to activate the Python 3 environment
4. Enter `pip3 install pandas Fiona rasterio scikit-learn`


#### Install plugin

To install from plugin manager:

1. Click the menu "Plugins" -> "Manage and Install Plugins".
2. Enter 'ClassificationTool' in search box.
3. Select plugin and install it.



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/SteveMHill/ClassificationTool
[contributors-url]: https://github.com/SteveMHill/ClassificationTool/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/SteveMHill/ClassificationTool
[forks-url]: https://github.com/SteveMHill/ClassificationTool/network/members
[stars-shield]: https://img.shields.io/github/stars/SteveMHill/ClassificationTool
[stars-url]: https://github.com/SteveMHill/ClassificationTool/stargazers
[issues-shield]:https://img.shields.io/github/issues/SteveMHill/ClassificationTool
[issues-url]: https://github.com/SteveMHill/ClassificationTool/issues
[license-shield]:https://img.shields.io/github/license/SteveMHill/ClassificationTool
[license-url]: https://github.com/SteveMHill/ClassificationTool/blob/master/LICENSE.txt
