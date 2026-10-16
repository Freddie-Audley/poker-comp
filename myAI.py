from poker.action import Action
from poker.game_state import GameState


def myAI(state: GameState) -> Action:
    return allIn(state)


def allIn(state: GameState) -> Action:    # My really sophisticated algorithm
    return Action.raise_by(state.my_player_.chips_in_pot // 20)
