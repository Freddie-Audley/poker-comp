import os
import random
import traceback
from enum import Enum

from poker.action import Action
from poker.game_state import GameState


# ---- CORE: keep feature extraction compatible with trained weights ----

_B5 = 13 ** 5


def _straight_top(mask):
    """Return the highest rank index of a straight, or -1."""
    m = (mask << 1) | (mask >> 12)
    w = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4)
    if not w:
        return -1
    return w.bit_length() - 1 + 3


def _pack(cat, ranks):
    v = 0
    for i in range(5):
        v = v * 13 + (ranks[i] if i < len(ranks) else 0)
    return cat * _B5 + v


def score(cards):
    """Score cards using the original model's hand-scoring implementation."""
    rc = [0] * 13
    sm = [0, 0, 0, 0]

    for c in cards:
        r = c % 13
        rc[r] += 1
        sm[c // 13] |= 1 << r

    allmask = sm[0] | sm[1] | sm[2] | sm[3]

    flush_mask = 0
    for s in range(4):
        if bin(sm[s]).count("1") >= 5:
            flush_mask = sm[s]
            break

    if flush_mask:
        t = _straight_top(flush_mask)
        if t >= 0:
            return _pack(8, [t])

    quads, trips, pairs, singles = [], [], [], []

    for r in range(12, -1, -1):
        n = rc[r]
        if n == 4:
            quads.append(r)
        elif n == 3:
            trips.append(r)
        elif n == 2:
            pairs.append(r)
        elif n == 1:
            singles.append(r)

    if quads:
        kick = [max(trips + pairs + singles)] if (
            trips or pairs or singles
        ) else []
        return _pack(7, [quads[0]] + kick)

    if trips and (len(trips) > 1 or pairs):
        second = trips[1] if len(trips) > 1 else pairs[0]
        if len(trips) > 1 and pairs:
            second = max(trips[1], pairs[0])
        return _pack(6, [trips[0], second])

    if flush_mask:
        top5 = [
            r for r in range(12, -1, -1)
            if flush_mask >> r & 1
        ][:5]
        return _pack(5, top5)

    t = _straight_top(allmask)
    if t >= 0:
        return _pack(4, [t])

    if trips:
        return _pack(3, [trips[0]] + singles[:2] + pairs[:2])

    if len(pairs) >= 2:
        rest = sorted(pairs[2:] + singles, reverse=True)[:1]
        return _pack(2, [pairs[0], pairs[1]] + rest)

    if pairs:
        return _pack(1, [pairs[0]] + singles[:3])

    return _pack(0, singles[:5])


def featurize(hole, board, n_opp):
    """Keep this feature order and length unchanged for the trained model."""
    r1, r2 = hole[0] % 13, hole[1] % 13
    hi, lo = max(r1, r2), min(r1, r2)

    suited = 1.0 if hole[0] // 13 == hole[1] // 13 else 0.0
    nb = len(board)

    f = [
        hi / 12.0,
        lo / 12.0,
        suited,
        1.0 if hi == lo else 0.0,
        (hi - lo) / 12.0,
        (min(max(n_opp, 1), 7) - 1) / 6.0,
    ]

    f += [1.0 if nb == k else 0.0 for k in (0, 3, 4, 5)]

    allc = list(hole) + list(board)
    sc = score(allc)
    cat = sc // _B5

    f += [1.0 if cat == k else 0.0 for k in range(9)]
    f.append(((sc // 13**4) % 13) / 12.0)
    f.append(((sc // 13**3) % 13) / 12.0)

    brank = [c % 13 for c in board]
    bmax = max(brank) if brank else -1

    f.append(
        sum(1 for r in (r1, r2) if r > bmax) / 2.0
        if brank else 0.0
    )
    f.append(
        1.0 if brank and len(set(brank)) < len(brank) else 0.0
    )

    bs = [0, 0, 0, 0]
    for c in board:
        bs[c // 13] += 1
    f.append(max(bs) / 5.0)

    hs = [0, 0, 0, 0]
    for c in allc:
        hs[c // 13] += 1
    f.append(1.0 if nb in (3, 4) and max(hs) == 4 else 0.0)

    mask = 0
    for c in allc:
        mask |= 1 << (c % 13)

    m = (mask << 1) | (mask >> 12)
    wins = sum(
        1 for s in range(9)
        if bin((m >> s) & 31).count("1") >= 4
    )
    f.append(min(wins, 2) / 2.0 if nb in (3, 4) else 0.0)

    f.append(bmax / 12.0 if brank else 0.0)
    f.append(sum(1 for r in (r1, r2) if r in brank) / 2.0)
    f.append(sc / (9.0 * _B5))

    return f


def forward(weights, x):
    """MLP with ReLU hidden layers and sigmoid output."""
    import math

    a = x

    for li, (W, b) in enumerate(weights):
        out = []

        for j in range(len(b)):
            row = W[j]
            s = b[j]
            for i in range(len(a)):
                s += row[i] * a[i]
            out.append(s)

        if li < len(weights) - 1:
            a = [v if v > 0.0 else 0.0 for v in out]
        else:
            z = max(-30.0, min(30.0, out[0]))
            return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------
# Trained model weights
# >>> Replace the line below with your ORIGINAL, UNMODIFIED
# >>> `WEIGHTS = [[[[-0.30614, ...]]]]` definition.
# ---------------------------------------------------------------------

WEIGHTS = [[[[-0.30614,0.91653,-0.08253,0.04543,-2.2398,0.0829,0.28693,0.33068,0.51688,0.61718,-2.58877,-0.83953,1.35042,0.65991,3.89985,1.31384,-0.46415,-4.50478,-1.72329,2.22806,1.42974,1.24017,-0.79153,-0.05016,0.09706,0.04631,-2.38047,-3.36178,1.32931],[-3.14772,-2.82582,-0.00256,-6.04867,-3.57993,-0.25463,-0.01329,2.09846,1.77132,1.44331,-7.65107,-8.54748,1.34994,0.9712,-8.24448,-5.73823,1.59214,3.7617,-7.47352,-0.48807,2.87012,-0.84516,1.23008,-0.92314,0.31393,0.79265,1.38207,-27.99973,3.12495],[1.83914,3.82268,0.00819,-6.39386,-1.36857,0.31435,-5.36293,0.6373,0.94129,0.9392,0.3385,1.46392,1.01284,6.2821,-1.28993,-4.46948,-5.3908,10.00818,1.38066,1.24741,-2.83696,-0.1602,-0.51106,-0.39548,0.40848,-0.20639,-3.36229,-0.51818,-1.75114],[-0.44103,-0.37132,3.44536,-1.04879,0.18364,0.83582,-0.6577,0.73206,0.6947,1.4221,-0.9646,-0.92224,-0.91151,-1.57853,-3.92399,5.94919,-1.86104,-2.84487,0.48981,-0.29086,0.12262,-0.18023,-0.4833,-11.172,2.99629,-0.1138,0.00281,-0.66254,1.8643],[-1.19013,-1.79611,-0.02666,-5.0318,0.23636,0.83313,-2.39839,0.03045,-0.27398,-0.41852,5.03406,1.51131,-1.66325,0.0854,-5.37981,-3.96791,5.61615,7.60042,4.53184,4.73039,-4.06024,1.66503,4.73089,-0.62462,-0.35001,-10.63372,-0.62441,1.66734,-5.43844],[0.0484,0.08537,0.01606,-0.03005,0.13701,0.17562,0.02957,-1.97654,-2.01254,-2.34938,-1.65491,-1.21036,-0.90875,-0.34561,-0.52872,-2.65617,-2.87835,-2.64732,-0.61608,0.21106,-0.07826,0.10588,-0.1308,12.33458,-0.39749,-0.03045,0.07868,-0.10272,-2.1104],[-1.68787,-2.08018,0.00212,2.70464,-1.75426,-0.01728,-2.50195,0.03828,-0.00315,0.01762,-4.84756,-1.49431,-2.46187,-1.82221,5.12565,-5.75793,-4.53143,-9.51765,3.26864,5.53209,3.47995,1.27066,-0.83923,0.06821,-0.22547,2.38554,-5.88658,5.26129,3.67515],[1.54166,8.83427,-0.00368,1.2671,-1.71947,0.01759,-1.98982,-4.31941,0.51715,1.63342,-0.08207,1.3059,-9.84286,0.40923,-1.50079,5.53556,-7.93525,-3.06144,-0.6373,-9.85845,-7.93226,0.89595,-0.38811,-0.03926,0.01475,1.22931,0.21677,2.29367,-1.61538],[0.19989,0.41181,0.00067,2.53811,0.75748,1.26926,-1.51079,-1.02159,-0.78494,-0.39172,0.28109,-5.41049,-7.50012,1.40026,-6.29828,-2.17194,1.13531,3.70735,2.48856,-2.83433,1.83559,-0.35007,3.12572,-1.42123,-0.46177,-1.29917,-0.04596,5.49728,-2.21032],[-0.98425,-1.04372,0.03094,0.79111,0.49662,4.33406,0.86744,1.25205,0.90595,0.92256,1.42291,-0.19941,0.43204,0.46766,6.09526,-1.71861,1.86575,7.29239,-8.61579,-2.10448,0.328,0.25109,-0.13808,0.2664,-0.55485,0.22982,0.51132,-0.02775,2.33589],[3.84989,3.50786,-0.00131,-5.03807,3.01124,0.07174,-0.53323,0.62772,0.54155,0.29964,0.51385,-0.76224,4.24374,-4.323,0.20546,0.83005,-8.03389,-6.33765,0.41718,-7.18998,-0.68196,0.8595,0.10812,0.34405,0.01439,-0.19717,0.04918,-0.67403,-1.57937],[3.73853,3.62867,0.00101,2.62401,3.46019,0.07073,-10.19316,0.48801,0.52584,0.45898,-1.31484,0.76199,-0.10706,-4.03085,2.13019,0.9144,0.03498,3.44193,0.79573,2.84576,-0.80831,-16.99115,-0.04406,-0.0446,-0.00844,0.06977,-9.43151,0.51477,-0.00941],[-1.57289,-0.91769,0.0445,-0.61717,-0.0091,0.36629,-6.54303,-3.93803,0.48502,3.72766,1.89684,1.7349,1.92854,2.08564,-2.30749,2.37087,-4.63104,-2.63714,-2.46121,-1.51661,2.11733,0.59187,-1.2968,-3.58846,-1.5888,-0.05705,-6.65704,-0.65817,2.38131],[0.53519,0.78481,5e-05,-6.4629,0.47829,0.39697,-3.76801,-0.34509,-0.21489,-0.11133,-5.0287,0.99011,1.52084,0.56928,-1.45097,1.30048,1.61502,1.07312,-0.92965,7.37171,0.71959,-0.49741,-0.07461,-0.26205,-0.03662,-0.6813,-8.59917,0.46257,-0.73559],[-0.63297,0.88466,0.38945,0.43509,0.7969,-0.21743,1.64362,-0.60241,1.19339,2.97601,3.90011,3.67207,2.92473,3.35061,-2.46504,-0.58855,-2.39014,-1.79541,-0.80597,-0.1503,-0.4451,0.135,0.25812,1.69566,-0.24084,-0.08887,0.38845,0.56587,-0.5975],[0.22264,0.76767,0.1829,-7.9914,0.08756,0.62728,-4.18085,1.18633,-0.3443,-0.72723,2.6351,-1.22867,-4.37239,-0.21831,-7.40024,0.91738,1.28229,-5.1333,4.65742,-2.06668,-1.60629,1.61444,2.43602,-0.30131,1.7295,1.89056,0.08163,-0.27743,-1.10452],[-0.36345,0.00277,-0.0016,0.24574,0.20391,-13.83347,0.7506,1.0122,1.00877,1.06234,1.15893,0.61752,0.47564,0.71693,5.33145,0.7477,1.55069,-0.38978,-0.28198,-0.05467,-0.17926,0.03847,0.08207,0.42348,-0.26685,-0.13709,0.4115,0.16508,-0.55628],[-1.82213,-1.00715,-0.03285,-1.85953,-0.98166,0.30971,4.98511,-1.31853,-1.13798,-0.54416,-4.73362,-1.41864,0.97725,-1.21234,-0.15484,-3.47695,3.89723,0.03761,-2.64056,1.04074,-0.11313,4.85465,-5.86279,-1.21598,-0.80643,-6.22403,0.28286,6.53032,-0.28905],[-2.12795,-0.44712,0.09248,-0.58482,-4.00836,0.26846,0.07076,0.8365,0.46422,0.10957,2.93453,1.47812,1.22408,3.32219,-3.39659,-0.42103,8.05241,4.51192,7.21026,1.03811,-1.18172,-1.96397,0.94589,0.9592,-7.17564,3.17764,-0.25219,-0.58727,-2.79334],[1.21586,2.87976,-0.0117,0.10364,-0.1583,-0.05049,-0.14984,-1.56203,-1.56967,-2.24797,-0.81072,-1.24369,3.61512,7.41856,2.68378,0.15205,-0.00775,0.18371,-1.87961,-2.02569,1.90293,-0.47014,-7.706,0.09088,-0.07448,7.93043,-3.26823,-0.00214,-3.27896],[-0.42389,-0.04203,0.00032,3.43478,-0.73201,-0.42888,2.60128,0.19479,-0.73704,-0.64221,-2.79323,-1.59416,-0.92752,-0.1831,-5.40267,3.90947,-6.59826,-9.50452,-1.71918,-1.78337,-1.09565,1.69531,0.37259,-0.48488,0.83388,1.40846,2.6811,6.08428,-0.64744],[-2.48559,-5.87937,0.01328,0.14232,2.76143,0.06837,-2.81255,-0.93962,-0.90335,-0.88263,0.65357,-0.91727,2.58891,4.49154,0.77611,-6.71133,3.31662,3.35373,1.30856,8.33445,0.35814,-0.11823,-0.10376,-0.46876,-0.01596,-0.15677,-0.06608,-0.51975,0.33664],[-1.99767,-1.7743,-0.0405,2.09837,-1.2435,-0.36399,-6.85081,-0.55661,-0.37841,0.32124,2.46013,0.30356,0.2575,-2.79155,-4.68605,-3.78182,2.68725,-2.32138,-0.58608,3.52291,5.66988,-0.53119,-3.37614,-0.05003,-0.58975,0.31566,-6.10756,4.14306,-0.05782],[-5.23689,-4.00764,0.01259,-0.043,-3.9271,0.1054,-0.69339,-0.23947,-0.1714,-0.01942,-1.56305,-0.52891,1.13569,-0.22588,-2.19062,-2.38037,0.26444,0.17027,-1.85256,8.88031,0.03275,2.19034,-0.03057,-0.27108,-0.81723,0.10799,-0.02226,0.0596,1.52265],[-0.99912,-1.46304,-0.10239,-2.18271,-1.10671,0.37435,-1.5466,2.21323,2.31324,2.08576,0.55578,0.76504,0.07592,-4.01481,-6.78221,-1.08369,-1.4779,-1.23976,-5.08437,0.45688,0.20661,-0.13398,0.87618,0.04454,-1.94169,-0.74449,-1.91042,-3.47465,3.71133],[-1.35508,-1.01984,0.05708,0.8693,-1.06946,-0.01241,-0.86787,0.42594,0.50748,0.56576,-3.27962,-3.56565,-3.93907,-7.55186,6.43554,1.57111,-3.56855,4.93301,9.131,-0.14772,4.42282,2.85451,-0.42621,-0.51947,-0.10776,-0.01563,0.12057,-2.82695,1.33127],[-2.68421,-2.20867,0.01628,-7.84655,-2.79219,1.02877,-7.02817,1.99867,2.09455,2.15185,3.30217,3.15538,0.06777,3.26016,-6.05433,-5.71826,2.54359,1.65844,-4.83411,1.26501,-2.48201,0.30947,1.124,-0.04595,0.08031,-0.2731,-0.92816,-17.81891,-1.11645],[-0.38799,-0.57902,-0.04475,-0.15186,-0.10979,2.19028,-0.93151,0.60639,0.70045,0.8898,0.71251,0.55857,0.78709,-0.35109,-3.35458,7.389,-0.28402,0.95004,8.87162,-0.05046,0.57279,0.46826,-0.10923,0.25782,0.03923,-0.76855,-1.15901,-0.53575,-0.55706],[-3.67462,-2.50912,-0.04033,0.94248,-4.25589,0.07277,-1.40792,0.83965,0.33239,0.4007,1.46176,1.74782,0.98975,1.54995,-1.96127,-0.1738,-9.33767,-0.2072,-3.99788,5.92589,-1.84122,-7.11269,0.19381,1.52391,0.04752,0.19521,0.14089,-2.6975,1.69197],[0.01183,-0.44441,-0.19315,-0.20208,0.02607,-0.26942,-0.5296,-0.26124,-0.03138,-0.1474,0.1649,-0.38844,-0.77922,0.20055,-0.06636,-0.18605,0.07392,-0.26355,0.20035,-0.00199,-0.00136,0.0634,-0.12441,-0.16176,-0.27261,-0.08089,-0.54046,-0.36368,-0.12824],[-0.38582,-0.01222,0.09826,-1.49391,0.07399,-3.55419,-0.98627,1.38816,1.57337,1.48155,-1.99848,1.75045,2.12973,1.13264,0.92586,0.53535,3.12319,-0.6139,8.72428,2.20791,0.70554,-0.44964,-0.80515,-0.32534,-0.08677,-0.33393,-1.62163,-3.43184,1.25946],[-1.08132,-1.37917,-0.25952,-3.13671,-0.99911,1.92973,-1.23512,-0.08727,0.39298,2.05854,2.2271,2.22404,4.38199,0.36051,-5.75543,-5.04903,-4.72947,3.89743,-2.81696,0.30704,1.04078,-0.33842,0.40295,-0.3943,-3.6083,-12.40983,0.04583,-4.61411,-5.39322]],[0.35439,2.09039,0.69642,-0.97577,-2.54429,-3.37564,-0.25177,-1.03317,-1.11712,1.32254,0.11246,-0.67051,2.69978,-1.05637,-3.96242,0.49194,1.35613,-1.68421,0.13026,-5.02482,-0.9829,-1.89993,-0.9517,-1.2216,3.42238,0.81959,2.88172,-0.2897,0.35633,-0.08442,1.71213,0.25641]],[[[-0.60051,0.50058,-2.21963,3.21757,7.99443,-0.25076,0.1523,2.05988,1.40841,-0.88611,1.92661,3.69841,1.18445,-0.88841,-17.82292,-3.90717,-0.32836,-6.19717,0.94589,-0.93683,-1.06907,-7.97732,-3.55599,-0.96859,-4.68667,1.01095,0.39895,1.51478,0.39952,0.32782,-0.13389,2.70567],[0.62582,-10.41243,0.35153,-7.28535,1.2341,-1.83146,-0.54192,-30.59027,0.73497,0.22806,0.22076,0.626,-1.91138,1.04634,-0.48041,1.29099,-0.00963,-6.79706,0.2589,0.40929,-1.42821,-6.95978,1.0895,-0.91669,-2.01968,0.7024,-5.61098,0.42367,-0.77339,-0.01729,0.33552,-0.38957],[0.85728,-4.38313,1.49411,0.24052,0.26953,0.13138,-0.77176,-6.72354,1.85234,-0.77737,-2.13295,1.17834,0.88314,1.58276,0.25339,1.78206,0.22999,-8.46071,0.51647,0.91843,-1.55067,-62.79664,-2.12815,-16.59926,0.46806,1.76226,-10.53453,-1.21882,0.78441,-0.36578,-1.27652,0.25679],[-1.23081,-0.22843,0.01102,-9.43884,1.17091,3.16948,0.21253,0.64365,-1.40838,-0.04013,-0.49986,-0.28224,3.28892,-2.36469,2.7629,-2.75835,-2.12179,-2.01224,-1.45554,-3.63392,0.37323,-0.3623,-1.4519,-0.15668,-4.575,1.21565,4.0647,0.35584,0.08267,0.24072,-0.73707,3.90662],[-1.54263,1.59542,-0.77461,3.09669,0.42214,0.97432,1.45948,0.44718,-1.80733,0.32372,0.0832,-25.07294,0.22828,-0.60972,0.428,0.42807,0.40284,-0.0819,-0.15631,0.04903,0.93302,-0.18977,-0.91655,0.99177,0.95545,1.39278,1.30534,-1.6357,-0.25677,0.41863,-0.98182,-0.33901],[1.02456,-2.52783,0.8009,-1.3086,2.33658,-0.03765,-0.0224,-9.87682,-0.73729,-0.74339,-14.04568,-1.68163,-1.94111,2.36516,-16.84373,-4.30683,-0.01047,0.32011,3.04055,-0.8475,-0.50069,0.75991,-1.60286,-25.05387,-0.36918,2.73549,-19.60333,0.33024,-2.08059,-0.27623,-1.2774,3.97338],[-0.3499,1.8063,0.44366,-0.63635,-0.08346,0.7321,0.14929,-0.48213,-0.76355,0.78682,-0.16424,2.41897,0.05649,5.0304,-0.02458,-0.11714,0.54099,-1.40943,-0.56488,0.54756,0.0769,-0.02674,1.28924,-0.99942,0.18899,-1.18837,-0.43265,-1.57605,0.42755,0.1626,0.49375,0.66758],[-0.43599,-0.63781,-0.05801,0.80319,0.07325,-0.96565,0.09336,1.45444,1.04353,-0.0963,0.59649,-0.57594,-0.28947,-0.53284,-0.07524,-0.07595,-0.04072,0.78001,-0.45351,-22.84597,0.42675,0.71797,1.62265,-0.64819,-0.26436,-0.64364,-0.21588,0.32439,0.40397,-0.23708,0.54058,-0.2745],[0.56511,-1.34976,-0.40661,1.25781,-1.19234,-0.26168,-51.95879,0.29924,1.44961,-0.07497,0.22661,0.51378,0.35574,-2.72444,0.31615,0.24218,0.00192,-10.71297,-0.56473,7.08575,-0.19369,0.15035,-2.49805,0.81806,1.21043,-0.43912,-1.3455,-0.80958,0.06722,-0.33118,0.05165,0.7693],[1.11579,1.34404,-0.58847,-1.06745,-0.05217,-1.14083,-3.58315,-5.17267,0.2884,0.04349,-0.29494,0.74778,-1.60515,-1.52933,0.89185,1.04442,-0.14229,0.19223,-0.04055,9.27677,-0.03572,-0.35836,1.63574,-7.64788,-0.03764,-0.04436,0.01637,0.79359,-0.77843,-0.36825,0.08563,0.52888],[-0.30162,1.36118,-0.38785,-0.10084,0.21186,0.81003,0.12125,-0.2562,0.64981,-0.68423,0.21537,-0.08019,0.78824,-0.95158,0.12855,0.069,0.98933,-0.82971,-0.1411,0.03537,0.6838,0.00921,0.45512,-0.21286,0.84946,0.13485,-1.19392,-0.57185,0.17712,0.14517,0.34707,0.0507],[0.13662,1.09187,-0.76481,0.71462,1.35004,-0.3526,1.3588,-1.24918,-0.22729,0.26192,0.75596,-23.24959,0.20005,-5.16355,-0.04556,0.06295,0.03171,-1.03538,-0.12812,0.12693,-1.35548,0.12166,-0.68795,0.12884,-0.24012,-0.312,0.95119,-1.20036,0.28679,0.11134,-0.11336,1.27776],[1.12751,-0.40209,1.11548,0.15707,-0.56172,-1.2587,-0.17507,-1.0943,0.49967,-0.39558,-1.00696,0.42299,-0.01025,5.17236,0.1939,-0.44361,-0.26758,0.6826,0.5853,-0.50037,1.24494,0.54682,-1.47315,-0.14998,0.93461,0.23444,-4.7694,1.00498,-1.09806,0.02498,0.07191,-0.04268],[-0.11845,-0.64098,0.18488,0.68087,-0.17325,-2.22651,-0.1739,-2.75586,-1.97038,0.97166,-0.14227,-0.77366,0.35641,-1.01055,-2.25551,0.78994,-0.19865,-1.20278,-0.02215,-0.8189,-0.62363,0.00415,-0.34426,0.45212,-1.34945,-0.67885,1.11743,0.48211,0.06156,0.44055,0.83062,0.21634],[-0.19986,-0.09166,-0.1221,0.20446,-0.29494,-0.61761,-0.5792,-0.34237,-0.15529,-0.12547,-0.01326,0.07805,-0.39651,-0.43454,0.0848,-0.18224,0.05432,-0.28871,-0.17821,-0.14548,-0.67286,-0.42607,0.35684,-0.09241,0.06707,-0.01403,-0.68099,-0.14404,-0.13468,0.14916,-0.55613,-0.35088],[-0.48688,0.86824,0.05209,-1.16739,-0.7218,2.21912,0.45349,1.68741,-0.5062,1.2742,-0.17312,0.16452,0.81789,1.78547,0.24247,-0.40984,-0.38869,-1.04533,-0.6014,-1.72657,0.50101,0.34166,1.06117,0.02469,0.94757,-0.47411,0.45754,0.76137,0.16901,0.17133,-0.32707,-0.01005]],[-0.94947,0.77568,-0.4331,-4.00716,-6.98851,-2.77488,-3.63678,0.86249,-4.81241,-1.30056,0.0382,-0.38869,-0.2453,1.11683,-0.13796,1.78794]],[[[0.12801,0.19985,-0.16531,-0.08486,0.10727,0.22835,0.10294,0.11674,0.15579,-0.06523,0.11775,-0.13332,0.14074,0.11204,0.15736,-0.20915]],[0.28182]]]

# ---------------------------------------------------------------------
# Policy settings
# ---------------------------------------------------------------------

BET_EDGE = 0.20
RAISE_EDGE = 0.30
CALL_MARGIN = 0.03
SHOVE_PENALTY = 0.12
BET_FRACTION = 0.6

PREMIUM_PAIR_MIN_RANK = 8       # TT+ (deuce=0 ... ten=8, ace=12)
SET_MINE_MAX_STACK_FRAC = 0.15  # call cheap preflop bets with any other pair

DEBUG = bool(os.environ.get("POKER_DEBUG"))


# ---------------------------------------------------------------------
# State adapter
# ---------------------------------------------------------------------

_RANK_WORDS = {
    "TWO": 0, "THREE": 1, "FOUR": 2, "FIVE": 3,
    "SIX": 4, "SEVEN": 5, "EIGHT": 6, "NINE": 7,
    "TEN": 8, "JACK": 9, "QUEEN": 10, "KING": 11,
    "ACE": 12,
}

_RANK_CHARS = "23456789TJQKA"

_SUIT_CHARS = {
    "C": 0, "D": 1, "H": 2, "S": 3,
    "♣": 0, "♦": 1, "♥": 2, "♠": 3,
}

_SUIT_WORDS = {
    "CLUB": 0, "CLUBS": 0,
    "DIAMOND": 1, "DIAMONDS": 1,
    "HEART": 2, "HEARTS": 2,
    "SPADE": 3, "SPADES": 3,
}


def _get(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            try:
                value = getattr(obj, name)
                return value() if callable(value) else value
            except Exception:
                continue
    return default


def _unwrap(value):
    if isinstance(value, Enum):
        return value.value
    return value


def _rank_index(rank):
    rank = _unwrap(rank)

    if isinstance(rank, str):
        t = rank.strip().upper()

        if t in _RANK_WORDS:
            return _RANK_WORDS[t]

        if t == "10":
            return 8

        if len(t) == 1 and t in _RANK_CHARS:
            return _RANK_CHARS.index(t)

        if t.isdigit():
            number = int(t)
            if 2 <= number <= 14:
                return number - 2
            if 0 <= number <= 12:
                return number

        raise ValueError("Unknown rank %r" % rank)

    number = int(rank)

    if 0 <= number <= 12:
        return number

    if 13 <= number <= 14:
        return number - 2

    raise ValueError("Unknown integer rank %r" % number)


def _suit_index(suit):
    suit = _unwrap(suit)

    if isinstance(suit, str):
        t = suit.strip().upper()

        if t in _SUIT_WORDS:
            return _SUIT_WORDS[t]

        if t in _SUIT_CHARS:
            return _SUIT_CHARS[t]

        raise ValueError("Unknown suit %r" % suit)

    number = int(suit)
    if 0 <= number <= 3:
        return number

    raise ValueError("Unknown integer suit %r" % number)


def _card_int(card):
    if isinstance(card, int):
        if not 0 <= card <= 51:
            raise ValueError("Card integer out of range: %r" % card)
        return card

    if isinstance(card, str):
        t = card.strip()
        if len(t) < 2:
            raise ValueError("Invalid card string %r" % card)
        return _suit_index(t[-1]) * 13 + _rank_index(t[:-1])

    if isinstance(card, (tuple, list)) and len(card) == 2:
        a, b = card
        try:
            return _rank_index(a) + 13 * _suit_index(b)
        except (ValueError, TypeError):
            return _rank_index(b) + 13 * _suit_index(a)

    rank = _get(card, ("rank", "value", "number", "face"))
    suit = _get(card, ("suit", "colour", "color"))

    if rank is None or suit is None:
        raise ValueError("Cannot parse card %r" % (card,))

    return _rank_index(rank) + 13 * _suit_index(suit)


def _cards(seq):
    return [_card_int(card) for card in (seq or [])]


def _hole(state):
    me = state.my_player

    seq = _get(
        me,
        (
            "hole_cards", "hand", "cards", "hole",
            "holecards", "private_cards",
        ),
    )

    if seq is None:
        seq = _get(state, ("hole_cards", "my_cards", "hand", "cards"))

    return _cards(seq)


def _board(state):
    seq = _get(
        state,
        (
            "community_cards", "board", "community",
            "table_cards", "shared_cards",
        ),
    )
    return _cards(seq)


def _stack(player):
    value = _get(
        player,
        (
            "chips_in_stack", "chips", "stack",
            "chip_count", "money", "balance",
        ),
    )
    return value if isinstance(value, (int, float)) else None


def _is_live(player):
    for name in (
        "folded", "is_folded", "has_folded", "out",
        "is_out", "eliminated", "busted",
    ):
        if _get(player, (name,), False):
            return False

    for name in ("active", "is_active", "in_hand", "in_game"):
        value = _get(player, (name,), None)
        if value is not None and not value:
            return False

    # Busted players have no stack and nothing in the pot.
    # (All-in players have chips_in_pot > 0, so they stay live.)
    stack = _stack(player)
    in_pot = _get(player, ("chips_in_pot",), 0) or 0
    if stack is not None and stack <= 0 and in_pot <= 0:
        return False

    return True


def _pot(state):
    # GameState.chips_in_pot is the engine's authoritative pot size.
    value = getattr(state, "chips_in_pot", None)
    if isinstance(value, (int, float)) and value > 0:
        return value

    total = 0
    for player in state.players:
        v = _get(player, ("chips_in_pot",), None)
        if isinstance(v, (int, float)):
            total += v
    return total


# ---------------------------------------------------------------------
# Decision policy
# ---------------------------------------------------------------------

def _decide(state):
    me = state.my_player
    hole = _hole(state)
    board = _board(state)

    if len(hole) != 2 or len(board) > 5:
        raise ValueError(
            "Unexpected cards: hole=%r board=%r" % (hole, board)
        )

    if len(set(hole + board)) != len(hole + board):
        raise ValueError("Duplicate cards in hole cards and board")

    players = state.players

    max_bet = max(p.chips_bet for p in players)
    to_call = max(0, max_bet - me.chips_bet)

    bb = _get(state, ("big_blind",), 20) or 20
    stack = _stack(me)
    preflop = not board

    ranks = [card % 13 for card in hole]
    is_pair = preflop and ranks[0] == ranks[1]
    is_premium_pair = is_pair and ranks[0] >= PREMIUM_PAIR_MIN_RANK

    if DEBUG:
        print(
            "[myAI] hole=%s board=%s ranks=%s to_call=%s stack=%s"
            % (hole, board, ranks, to_call, stack)
        )

    # Premium pairs (TT+): never fold, open with a raise.
    if is_premium_pair:
        if to_call > 0:
            return Action.call()

        amount = int(2 * bb)
        if stack is not None:
            amount = min(amount, int(stack))

        if amount > 0:
            return Action.raise_by(amount)

        return Action.check()

    # Set-mining: any other pocket pair calls a cheap preflop bet.
    if is_pair and to_call > 0:
        if stack is None or to_call <= SET_MINE_MAX_STACK_FRAC * stack:
            return Action.call()

    # Preflop floor: playable hands never fold to a cheap bet, whatever the model says.
    if preflop and to_call > 0:
        hi, lo = max(ranks), min(ranks)
        suited = hole[0] // 13 == hole[1] // 13
        playable = (
            (hi == 12 and (lo >= 8 or suited))             # AT+, Axs
            or (hi == 11 and lo >= 8)                      # KT+
            or (suited and hi >= 8 and hi - lo <= 3)       # QJs, JTs, T9s, KQs...
            or (hi == 10 and lo >= 9)                      # QJ
        )
        cheap = stack is None or to_call <= PREFLOP_CALL_STACK_FRAC * stack
        if playable and cheap:
            return Action.call()

    # Everything else uses the trained model.
    live_others = [
        p for p in players
        if p is not me and _is_live(p)
    ]
    n_opp = max(1, min(7, len(live_others)))

    equity = forward(WEIGHTS, featurize(hole, board, n_opp))
    fair = 1.0 / (n_opp + 1)
    edge = equity - fair

    pot = _pot(state)

    if DEBUG:
        print(
            "[myAI] n_opp=%d equity=%.3f edge=%.3f pot=%s"
            % (n_opp, equity, edge, pot)
        )

    def raise_size(frac):
        """raise_by is the extra increment on top of calling."""
        amount = int(max(2 * bb, frac * (pot + to_call)))

        if stack is not None:
            max_increment = max(0, int(stack - to_call))
            amount = min(amount, max_increment)

        return amount

    # Nothing to call: bet for value or check.
    if to_call <= 0:
        bet_edge = PREFLOP_RAISE_EDGE if preflop else BET_EDGE
        if edge > bet_edge and random.random() < 0.9:
            amount = raise_size(
                BET_FRACTION + min(0.4, edge - bet_edge)
            )
            if amount > 0:
                return Action.raise_by(amount)

        return Action.check()

    # Facing a bet.
    pot_odds = (
        to_call / float(pot + to_call)
        if pot + to_call > 0 else 1.0
    )

    need = pot_odds + CALL_MARGIN

    if stack is not None and stack > 0:
        need += SHOVE_PENALTY * min(
            1.0, to_call / float(stack)
        )

    # Preflop, pot odds understate a hand's value (blinds, later streets),
    # so compare against fair share when the price is a modest part of the stack.
    if preflop and (stack is None or to_call <= PREFLOP_CALL_STACK_FRAC * stack):
        need = min(need, fair + PREFLOP_CALL_MARGIN)

    raise_edge = PREFLOP_RAISE_EDGE if preflop else RAISE_EDGE

    if edge > raise_edge and random.random() < 0.8:
        amount = raise_size(BET_FRACTION)
        if amount > 0:
            return Action.raise_by(amount)

    if equity > need:
        return Action.call()

    return Action.fold()


def myAI(state: GameState) -> Action:
    try:
        return _decide(state)
    except Exception:
        if DEBUG:
            traceback.print_exc()
        return defaultAI(state)


def defaultAI(state: GameState) -> Action:
    """Safe fallback if the policy raises an exception."""
    try:
        me = state.my_player
        max_bet = max(player.chips_bet for player in state.players)

        if me.chips_bet == max_bet:
            return Action.check()

        return Action.fold()
    except Exception:
        if DEBUG:
            traceback.print_exc()
        return Action.fold()
