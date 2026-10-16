from poker.action import Action
from poker.game_state import GameState


def myAI(state: GameState) -> Action:
    return allIn(state)


def allIn(state: GameState) -> Action:
    if max(player.chips_bet for player in state.players) - state.my_player.chips_bet > state.my_player.chips_in_stack // 2:
        return Action.fold()
    return Action.raise_by(1)
    
