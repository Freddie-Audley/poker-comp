from poker.action import Action
from poker.game_state import GameState


def myAI(state: GameState) -> Action:
    return allIn(state)


def allIn(state: GameState) -> Action:    # My really sophisticated algorithm
    return Action.raise_by(2)
