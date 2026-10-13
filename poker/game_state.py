from .player import Player
from .card import Card


class GameState:
    def __init__(self, *, my_player: Player, players: list[Player], chips_in_pot: int = 0, community_cards: set[Card] | None = None):
        self.my_player = my_player
        self.players = players
        self.chips_in_pot = chips_in_pot
        self.community_cards = community_cards if community_cards is not None else set()


    def __repr__(self):
        return f"GameState(my_player={self.my_player}, players={self.players}, chips_in_pot={self.chips_in_pot}, community_cards={self.community_cards})"


    def __eq__(self, other):
        if not isinstance(other, GameState):
            return NotImplemented

        if self.my_player != other.my_player:
            return False

        if self.players != other.players:
            return False

        if self.chips_in_pot != other.chips_in_pot:
            return False

        if self.community_cards != other.community_cards:
            return False

        return True


    def __hash__(self):
        return hash((self.my_player, tuple(self.players), self.chips_in_pot, frozenset(self.community_cards)))
