
CardShark is an engine for card and board games which is build from the ground up with support for reinforcement learning, allowing you to easily teach agents to play your games.

============
Installation
============
Git clone this repository then can be installed using pip:

.. code::

    git clone git@github.com:ewanwm/CardShark.git

    pip install CardShark/

========
Features
========

++++++
Engine
++++++
The :ref:`Engine API` module provides functionality to implement your game. 

++++++++++++++
User Interface
++++++++++++++
The ui module provides functionality to include a graphical user interface for your game

++++++++++++++++++++++
Reinforcement Learning
++++++++++++++++++++++
You can use the MultiAgent class in the :ref:`Agent API` module to learn to play your game. For an example of this, see `here <https://github.com/ewanwm/CardShark/blob/main/tests/test_training.py>`_


========
Examples
========

You can find some example projects that use the CardShark engine `here <https://github.com/ewanwm/CardShark/tree/main/cardshark/examples>`_
