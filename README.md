# Checkers AI

A Checkers game developed in Python using Pygame, featuring an AI opponent trained with reinforcement learning techniques.

## Project Overview

This project implements the classic game of Checkers with:

- Graphical user interface using Pygame
- Human vs AI gameplay
- Trained AI models
- Game rendering and animations
- Reinforcement Learning training pipeline

## Project Structure

```text
.
├── assets/          # Images, sounds, and game resources
├── models/          # Saved AI models
├── ai.py            # AI logic and decision making
├── game.py          # Core game rules and mechanics
├── renderer.py      # Board and piece rendering
├── training.py      # AI training script
├── ui.py            # Menus and user interface
├── main.py          # Main entry point
└── main.spec        # PyInstaller build configuration
```

## Requirements

- Python 3.10+
- Pygame

Install dependencies:

```bash
pip install pygame
```

## Running the Game

```bash
python main.py
```

## Training the AI

To train a new model:

```bash
python training.py
```

Trained models are saved in the `models/` directory.

## Features

- Interactive graphical board
- Legal move validation
- Piece promotion to King
- AI opponent
- Reinforcement learning support
- Saved AI models
- Custom game assets

## Building an Executable

If using PyInstaller:

```bash
pyinstaller main.spec
```

The executable will be generated inside the `dist/` folder.

## Screenshots

Add screenshots of the game here.

## Future Improvements

- Stronger AI
- Additional difficulty levels
- Multiplayer support
- Game statistics
- Move history and replay system

## Author

Mohammed Mezour

## License

This project is for educational and personal use.
