import json
import random

from poker.action import Action
from poker.card import Card, Rank, Suit
from poker.game_state import GameState
from poker.player import Blind, MyPlayer, Player

from myAI import myAI

# Bots are scored against each other in tournaments, so there is no game to play on a push.
# This only checks that myAI returns an Action on a range of sample states, and gives a fixed score to a bot which always does.
NUM_STATES = 1000
WORKING_SCORE = 10


def sample_state(rng: random.Random) -> GameState:
    deck = [Card(rank, suit) for rank in Rank for suit in Suit]
    rng.shuffle(deck)

    hole_cards = {deck.pop(), deck.pop()}
    community_cards = {deck.pop() for _ in range(rng.choice([0, 3, 4, 5]))}

    blinds = [Blind.SMALL, Blind.BIG] + [None] * rng.randint(0, 4)
    rng.shuffle(blinds)

    my_player = MyPlayer(blind=blinds[0], chips_in_stack=rng.randint(1, 1000), chips_bet=rng.choice([0, rng.randint(1, 100)]), hole_cards=hole_cards)
    others = [Player(blind=blind, has_folded=rng.random() < 0.25, chips_in_stack=rng.randint(0, 1000), chips_bet=rng.choice([0, rng.randint(1, 100)])) for blind in blinds[1:]]

    players = [my_player] + others
    rng.shuffle(players)

    chips_in_pot = sum(player.chips_bet for player in players) + rng.choice([0, rng.randint(1, 500)])

    return GameState(my_player=my_player, players=players, chips_in_pot=chips_in_pot, community_cards=community_cards)


rng = random.Random(0)
failures = []

for i in range(NUM_STATES):
    state = sample_state(rng)

    try:
        result = myAI(state)
    except Exception as e:
        failures.append(f"state {i}: raised {type(e).__name__}: {e}")
        continue

    if not isinstance(result, Action):
        failures.append(f"state {i}: returned {result!r} instead of an Action")

for failure in failures[:10]:
    print(failure)

success_rate = 100 * (NUM_STATES - len(failures)) / NUM_STATES
details = json.dumps({"states": NUM_STATES, "failures": len(failures), "examples": failures[:10]})

with open("score.txt", "w") as f:
    f.write(f"score={0 if failures else WORKING_SCORE}\nsuccess_rate={success_rate:g}\ndetails={details}\n")
