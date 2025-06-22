# CardShark

[![pip](https://github.com/ewanwm/CardShark/actions/workflows/pip_install.yml/badge.svg)](https://github.com/ewanwm/CardShark/actions/workflows/pip_install.yml)
[![pylint](https://github.com/ewanwm/CardShark/actions/workflows/pylint.yml/badge.svg)](https://github.com/ewanwm/CardShark/actions/workflows/pylint.yml)

A Python based engine for creating card and board games, with bult-in support for [Tensorflows Agents library](https://www.tensorflow.org/agents) allowing you to easily use reinforcement learning to train agents to play your games.

## Features

### Engine
The engine module provides functionality to implement your game. 

### User Interface
The ui module provides functionality to include a graphical user interface for your game

### Reinforcement Learning
You can use the MultiAgent class in the agents module to learn to play your game. For an example of this, see the tests/test_training.py script.

### Examples 
The CardShark/examples directory contains some pre-made example projects using the CardShark engine

## Installation

```
git clone git@github.com:ewanwm/CardShark.git

pip install CardShark/
```

## Requirements

### Python
CardShark currently only supports python versions `3.8`, `3.9` and `3.10`... sorry

## Documentation

Documentation can be built using sphinx:

```
pip install --upgrade pip
pip install sphinx

sphinx-build -M html doc/source <doc_build_directory>
```
