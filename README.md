# Flocking
A Mesa implementation of flocking agents. The agent behavior is inspired by https://doi.org/10.1109/IROS.2014.6943105. The implementation is based on the boid_flockers example of the [Mesa project](https://github.com/projectmesa/mesa).

## Getting Started

### Setup
Create a virtual Python environment:
```shell
python3 -m venv mesa
```
Activate the virtual environment:
```shell
cd mesa
source bin/activate
```
Get the source code:
```shell
git clone https://github.com/tropappar/flocking.git
```
Install requirements:
```shell
cd flocking
pip3 install -r requirements.txt
```

### Run
Switch to the folder containing the virtual environment and activate it:
```shell
cd mesa
source bin/activate
```

Run Mesa in the virtual environment:
```shell
cd flocking
mesa runserver
```
This opens the visualization in your default web browser.
