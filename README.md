# py-ai-gamer

A Python-based game integrating AI agents with a graphical interface, built using `tkinter`. Train, play, and interact with an intelligent gaming environment.

## Overview

`py-ai-gamer` is an experimental project combining a `tkinter`-powered GUI with an AI agent that learns and interacts within a custom game environment. It’s designed for enthusiasts of AI and game development, offering a sandbox to explore reinforcement learning or similar techniques in a desktop game setting.

## Features

- **Graphical Interface**: Built with `tkinter` for a lightweight, cross-platform UI.
- **AI Agent**: An intelligent agent (`agent.py`) that can be trained (`train.py`) to play the game.
- **Custom Environment**: Defined in `environment.py` for the agent to interact with.
- **Play Mode**: Launch and play the game manually via `play.py`.
- **Logging**: Tracks agent and interface activity in `.log` files.

## Project Structure

```
py-ai-gamer/
├── data/               # Game data (e.g., assets, saved models)
├── screens/            # UI screen definitions or assets
├── .gitignore          # Git ignore file
├── agent.log           # Agent activity log
├── agent.py            # AI agent logic
├── environment.py      # Game environment for the agent
├── game_interface.log  # Interface activity log
├── game_interface.py   # Main GUI implementation
├── game_screenshot.png # Sample game screenshot
├── interface.py        # Additional UI components
├── LICENSE             # MIT License file
├── play.py             # Script to launch the game
├── requirements.txt    # Project dependencies
├── test_windows.py     # Tests for Windows compatibility
└── train.py            # Script to train the AI agent
```

## Requirements

- Python 3.13 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/py-ai-gamer.git
   cd py-ai-gamer
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Python Version**:
   ```bash
   python --version  # Should show 3.13.x
   ```

## Usage

- **Play the Game**:
  ```bash
  python play.py
  ```
  Interact with the game via the `tkinter` interface.

- **Train the AI**:
  ```bash
  python train.py
  ```
  Trains the agent within the environment (check `agent.log` for progress).

- **Run the Interface**:
  ```bash
  python game_interface.py
  ```
  Launches the main GUI independently.

## Development

### Prerequisites
- IDE: PyCharm recommended (`.idea/` included).
- Python 3.13+ configured as the interpreter.

### Running Tests
- For Windows-specific tests:
  ```bash
  python test_windows.py
  ```

### Contributing
1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.