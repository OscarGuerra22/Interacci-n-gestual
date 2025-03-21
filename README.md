# Space Hand Adventure

## Overview
Space Hand Adventure is a Python-based game that uses `pygame`, `pymunk`, and `MediaPipe Hands` to create an interactive space adventure. The player controls a spaceship using hand gestures, leveraging the environment to hit a target and progress through three levels.

This project is a fork of the repository from my university account, originally developed as part of an assignment for the course **"Sistemas de Interacción Persona-Computador"** (translated as **Human-Computer Interaction, HCI**).

## Features
- Control a spaceship with hand gestures using **MediaPipe Hands**.
- Obstacle physics powered by **pymunk**.
- Interactive environments with obstacles and targets.
- Three progressively challenging levels.
- Developed with **pygame** for a smooth 2D gaming experience.

## Installation
### Requirements
Make sure you have Python installed (recommended version: 3.8+). Then, install the necessary dependencies:

```sh
pip install pygame pymunk mediapipe
```

### Running the Game
To start the game, simply run the following command:

```sh
python game.py
```


## Controls
- Move your hand up and down to control the spaceship’s vertical position (horizontal movement is not allowed).
- To aim, bring your index and middle fingers together and rotate your hand in the desired direction.
- To shoot, separate both fingers after aiming.
- To reload, bring your index and middle fingers together again.

## Acknowledgments
This project was developed as part of an academic assignment and is a fork of my original university repository.
