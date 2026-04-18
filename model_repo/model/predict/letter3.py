"""Letter-trigram (3-letter-prefix → next-letter) bias layer.

Given the last three letters of the current word buffer, bias the next
character toward common continuations. This sits on top of the 2-letter
digraph layer (trigram.py) and adds conditional specificity for very
common English/Shakespearean 3-letter contexts.

All knowledge is hand-specified from English orthography — no corpus
statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# 3-letter prefix (lowercase) -> {next char: bias}
# Terminator candidates: " " (space), "\n", ",", ".", ";", ":", "!", "?", "'"
_L3: dict[str, dict[str, float]] = {
    # -ing: almost always word-end
    "ing": {" ": 2.5, ",": 1.0, ".": 0.8, "\n": 0.9, ";": 0.5, "s": 0.6,
            "!": 0.4, "?": 0.4, "'": 0.4},
    # -ion (nation, passion, lion)
    "ion": {" ": 2.2, ",": 1.0, ".": 0.8, "s": 1.2, "\n": 0.7, ";": 0.4,
            "a": 0.3, "e": 0.2},
    # -ent (silent, present, went, went, parent)
    "ent": {" ": 1.8, ",": 0.8, ".": 0.6, "\n": 0.6, "s": 0.9, "l": 0.7,
            "i": 0.7, "r": 0.6, "e": 0.4, "h": 0.5, "u": 0.4},
    # -ant (want, giant, grant)
    "ant": {" ": 1.8, ",": 0.7, ".": 0.6, "\n": 0.5, "s": 0.9, "i": 0.5,
            "e": 0.5, "l": 0.5, "h": 0.4, "r": 0.3},
    # -est (best, honest, breast)
    "est": {" ": 2.0, ",": 0.8, ".": 0.6, "\n": 0.6, "s": 0.5, "i": 0.5,
            "e": 0.3, "o": 0.3, "y": 0.3},
    # -ath (hath, death, path, oath)
    "ath": {" ": 1.5, ",": 0.5, ".": 0.5, "\n": 0.5, "e": 0.7, "s": 0.5},
    # -eth (saith, doth-like, death, breath)
    "eth": {" ": 1.6, ",": 0.6, ".": 0.5, "\n": 0.5, "s": 0.4, "i": 0.4,
            "e": 0.4},
    # -ith (with, smith)
    "ith": {" ": 2.0, ",": 0.5, ".": 0.4, "\n": 0.5, "e": 0.4, "s": 0.4},
    # -oth (both, cloth, nothing, other)
    "oth": {"e": 1.4, " ": 1.3, "i": 1.0, "s": 0.7, ",": 0.4, ".": 0.3},
    # -uth (truth, youth)
    "uth": {" ": 1.5, ",": 0.4, "s": 0.4, "e": 0.4},
    # -ous (famous, jealous, ponderous)
    "ous": {" ": 2.0, ",": 0.8, "l": 1.0, ".": 0.6, "\n": 0.5, "e": 0.4,
            "n": 0.4, "t": 0.3, "i": 0.3},
    # -ers (lovers, heavens, answers — with 's')
    "ers": {" ": 2.0, ",": 0.7, ".": 0.5, "\n": 0.5, "e": 0.4, "t": 0.3},
    # -ing endings are above; -ings
    "ngs": {" ": 2.0, ",": 0.7, ".": 0.5, "\n": 0.5, "e": 0.3},
    # -ess (princess, mistress, bless)
    "ess": {" ": 2.0, ",": 0.6, ".": 0.5, "\n": 0.4, "e": 0.5, "i": 0.4,
            "o": 0.3, "n": 0.3},
    # -ies (lies, cries, ladies)
    "ies": {" ": 2.0, ",": 0.8, ".": 0.6, "\n": 0.5, "t": 0.5, "s": 0.3},
    # -ied (cried, denied, married)
    "ied": {" ": 2.0, ",": 0.8, ".": 0.6, "\n": 0.5, ";": 0.4},
    # -ght (night, fought, bright)
    "ght": {" ": 2.0, ",": 0.7, ".": 0.6, "\n": 0.5, "s": 0.5, "e": 0.3,
            "h": 0.3},
    # -igh (bright, might, high — usually followed by t)
    "igh": {"t": 2.5, "s": 0.4, ",": 0.3, " ": 0.3, "l": 0.3, "e": 0.3},
    # -ove (love, move, above, prove)
    "ove": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.4, "d": 0.7, "s": 0.6,
            "r": 0.5, "l": 0.3, "n": 0.3},
    # -ive (give, live, alive, active)
    "ive": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.4, "s": 0.7, "d": 0.6,
            "r": 0.5, "l": 0.3, "n": 0.3},
    # -ave (have, gave, brave, grave)
    "ave": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.4, "s": 0.5, "n": 0.5,
            "d": 0.4, "r": 0.4, "l": 0.3},
    # -ear (hear, dear, fear, tear)
    "ear": {" ": 1.5, ",": 0.5, "s": 1.0, "t": 0.8, "d": 0.6, "n": 0.5,
            "l": 0.5, ".": 0.4, "\n": 0.3, "y": 0.3},
    # -art (heart, apart, part, art, start)
    "art": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.4, "s": 0.5, "h": 0.5,
            "e": 0.3, "i": 0.3},
    # -ord (lord, word, sword, accord)
    "ord": {" ": 2.0, ",": 0.6, ".": 0.5, "\n": 0.5, "s": 0.6, "e": 0.4},
    # -and (and, hand, stand, land, grand)
    "and": {" ": 2.2, ",": 0.8, ".": 0.5, "\n": 0.5, "s": 0.6, "e": 0.3},
    # -end (end, friend, lend, bend)
    "end": {" ": 2.0, ",": 0.6, ".": 0.5, "\n": 0.5, "s": 0.7, "e": 0.3,
            "i": 0.3, "l": 0.3},
    # -ind (find, kind, mind, wind, behind)
    "ind": {" ": 2.0, ",": 0.6, ".": 0.5, "\n": 0.5, "s": 0.6, "n": 0.3,
            "e": 0.3, "i": 0.3, "l": 0.3},
    # -ond (fond, bond, respond, beyond)
    "ond": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.4, "s": 0.5, "e": 0.3},
    # -und (round, sound, found, under-)
    "und": {" ": 1.6, ",": 0.5, ".": 0.4, "\n": 0.4, "s": 0.4, "e": 0.5,
            "r": 0.3, "o": 0.3},
    # -all (all, call, fall, shall, small, tall)
    "all": {" ": 2.2, ",": 0.7, ".": 0.5, "\n": 0.5, "s": 0.5, "o": 0.3,
            "e": 0.3, "y": 0.3},
    # -ell (well, tell, bell, fell, hell)
    "ell": {" ": 2.0, ",": 0.8, ".": 0.6, "\n": 0.5, "s": 0.5, "o": 0.3,
            "'": 0.4, "!": 0.3},
    # -ill (will, still, bill, hill, ill)
    "ill": {" ": 2.2, ",": 0.7, ".": 0.5, "\n": 0.5, "s": 0.5, "e": 0.3,
            "'": 0.4, "y": 0.3},
    # -oll (roll, toll, poll, follow, hollow)
    "oll": {" ": 1.6, "o": 1.0, ",": 0.4, ".": 0.3, "s": 0.4, "i": 0.3,
            "e": 0.3},
    # -ull (full, pull, null)
    "ull": {" ": 1.5, ",": 0.4, "s": 0.5, "y": 0.4},
    # -ome (home, come, some, whom-e)
    "ome": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.4, "s": 0.5, "t": 0.4,
            "r": 0.3, "!": 0.3},
    # -ame (name, same, came, shame)
    "ame": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.4, "s": 0.5, "d": 0.4,
            "r": 0.3, "n": 0.3, "!": 0.3},
    # -ime (time, crime, sometime)
    "ime": {" ": 1.7, ",": 0.5, ".": 0.4, "\n": 0.4, "s": 0.5, "d": 0.3},
    # -ate (hate, fate, late, state, create)
    "ate": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.4, "s": 0.6, "d": 0.5,
            "r": 0.4, "l": 0.3, "n": 0.3},
    # -ite (write, quite, white, recite)
    "ite": {" ": 1.7, ",": 0.5, ".": 0.4, "\n": 0.4, "s": 0.5, "d": 0.4,
            "r": 0.3, "n": 0.3},
    # -ure (sure, pure, nature, creature)
    "ure": {" ": 1.7, ",": 0.5, ".": 0.4, "\n": 0.4, "s": 0.6, "d": 0.5,
            "r": 0.3, "l": 0.3, "n": 0.3},
    # -ere (here, there, were, mere, ere)
    "ere": {" ": 1.8, ",": 0.7, ".": 0.5, "\n": 0.5, "s": 0.5, "d": 0.4,
            "o": 0.3, "'": 0.3, ";": 0.4},
    # -one (one, none, done, bone, stone)
    "one": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.4, "s": 0.5, "d": 0.3,
            "y": 0.3, "r": 0.3},
    # -ine (mine, thine, shine, fine, divine)
    "ine": {" ": 1.7, ",": 0.6, ".": 0.5, "\n": 0.4, "s": 0.5, "d": 0.3,
            "r": 0.3},
    # -age (age, page, cage, village)
    "age": {" ": 1.7, ",": 0.5, ".": 0.4, "\n": 0.4, "s": 0.5, "d": 0.3,
            "r": 0.3, "n": 0.3},
    # -ake (make, take, wake, shake, awake)
    "ake": {" ": 1.6, ",": 0.5, ".": 0.4, "\n": 0.4, "s": 0.5, "n": 0.4,
            "d": 0.3, "r": 0.3},
    # -ike (like, strike, alike, unlike)
    "ike": {" ": 1.6, ",": 0.5, ".": 0.4, "\n": 0.4, "s": 0.5, "d": 0.3,
            "n": 0.3},
    # -ook (look, book, took, cook, shook)
    "ook": {" ": 1.6, ",": 0.5, ".": 0.4, "\n": 0.4, "s": 0.5, "e": 0.4,
            "i": 0.3},
    # -ull covered above
    # -oul (soul, could, would, should, foul)
    "oul": {"d": 2.0, " ": 0.5, ",": 0.3, "s": 0.4, "e": 0.3},
    # -hou (thou, hour, house, should)
    "hou": {"l": 1.2, "r": 0.8, "s": 0.7, "g": 0.6, " ": 0.4, ",": 0.3,
            "t": 0.3, "n": 0.3},
    # -tho (though, those, thou)
    "tho": {"u": 1.8, "s": 1.0, " ": 0.5, ",": 0.3, "r": 0.3, "n": 0.3},
    # -sha (shall, shape, shade, share, shake, shame)
    "sha": {"l": 1.5, "m": 0.8, "k": 0.7, "r": 0.6, "p": 0.6, "d": 0.5,
            "n": 0.5, "t": 0.4, ",": 0.2},
    # -wha (what, whale, wharf)
    "wha": {"t": 2.5, "l": 0.5, "r": 0.3, " ": 0.3},
    # -whe (when, where, whet, whether)
    "whe": {"n": 2.0, "r": 1.8, "e": 1.0, "t": 0.7, " ": 0.2},
    # -whi (which, while, whither, white, whisper)
    "whi": {"c": 1.5, "l": 1.3, "t": 1.0, "s": 0.8, "p": 0.5, "r": 0.5,
            " ": 0.2, "n": 0.3, "z": 0.2},
    # -tha (than, that, thank, thanks)
    "tha": {"t": 2.5, "n": 2.0, "s": 0.5, "r": 0.3, "m": 0.3, " ": 0.2},
    # -the (the, then, there, these, them, they, thee)
    "the": {" ": 2.2, "r": 1.5, "n": 1.3, "s": 1.0, "m": 1.0, "y": 0.9,
            "e": 0.8, ",": 0.5, ".": 0.3, ";": 0.3, "\n": 0.3},
    # -thi (this, thing, think, thine)
    "thi": {"s": 1.8, "n": 1.6, "c": 0.5, "r": 0.4, "e": 0.5, " ": 0.2},
    # -hem (them, hem-)
    "hem": {" ": 2.0, ",": 0.7, ".": 0.5, "\n": 0.4, "s": 0.4, ";": 0.3,
            "!": 0.3},
    # -nds (lands, minds, hands, ends)
    "nds": {" ": 2.2, ",": 0.7, ".": 0.5, "\n": 0.5},
    # -ngs (kings, things, brings)
    # already above
    # -pon (upon, weapon, pond-like)
    "pon": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.4, "d": 0.5, "s": 0.4,
            "e": 0.3},
    # -per (proper, upper, paper, deeper)
    "per": {" ": 1.5, ",": 0.5, ".": 0.4, "\n": 0.4, "s": 0.5, "i": 0.4,
            "e": 0.4, "f": 0.3},
    # -ter (better, after, enter, matter, letter)
    "ter": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.5, "s": 0.6, "e": 0.4,
            "i": 0.3, "n": 0.3, "r": 0.3},
    # -ver (ever, never, over, river, cover)
    "ver": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.5, "s": 0.6, "y": 0.5,
            "e": 0.4, "t": 0.3, "i": 0.3, "n": 0.3},
    # -her (her, other, father, mother, brother, whether)
    "her": {" ": 1.8, "e": 1.3, ",": 0.6, ".": 0.4, "\n": 0.4, "s": 0.5,
            "i": 0.3, "o": 0.3, "n": 0.2},
    # -hes (these, ashes, wishes, rushes)
    "hes": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.4, "e": 0.4, "t": 0.3},
    # -ish (wish, fish, polish, English)
    "ish": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.4, "e": 0.4, "m": 0.3,
            "i": 0.3, "o": 0.3},
    # -ash (ash, flash, crash, wash)
    "ash": {" ": 1.6, ",": 0.5, "e": 0.5, ".": 0.3, "i": 0.3},
    # -ish above; -osh -ush (hush, rush, crush, brush, bush)
    "ush": {" ": 1.5, ",": 0.4, "e": 0.5, "i": 0.3},
    # -oce, -ice (once + ice, price, twice, voice, justice)
    "ice": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.4, "s": 0.5, "d": 0.3},
    # -ace (face, place, grace, peace-ace, race)
    "ace": {" ": 1.8, ",": 0.6, ".": 0.4, "\n": 0.4, "s": 0.5, "d": 0.3,
            "r": 0.3, "!": 0.3},
    # -uce (produce, induce)
    "uce": {" ": 1.3, "s": 0.5, "d": 0.5, ",": 0.3},
    # -ful (useful, beautiful, wonderful)
    "ful": {" ": 2.0, ",": 0.6, ".": 0.5, "\n": 0.4, "l": 0.7, "n": 0.3},
    # -ect (effect, protect, perfect, respect, object)
    "ect": {" ": 1.8, ",": 0.6, ".": 0.4, "\n": 0.4, "s": 0.5, "e": 0.4,
            "i": 0.3},
    # -act (act, fact, exact, attract, contract)
    "act": {" ": 1.6, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.5, "e": 0.3,
            "i": 0.3, "u": 0.3, "o": 0.3},
    # -ict (strict, conflict, district, predict)
    "ict": {" ": 1.3, ",": 0.4, "i": 0.4, "e": 0.3, "o": 0.2, "s": 0.4},
    # -uct (conduct, instruct, product, construct)
    "uct": {" ": 1.3, "i": 0.3, "s": 0.4, "e": 0.3},
    # -ude (include, rude, nude, pride-attitude)
    "ude": {" ": 1.4, ",": 0.5, "s": 0.4, "d": 0.2, "n": 0.3},
    # -ade (made, shade, trade, persuade)
    "ade": {" ": 1.6, ",": 0.5, ".": 0.3, "\n": 0.3, "s": 0.4, "d": 0.3},
    # -ide (wide, side, ride, inside, beside)
    "ide": {" ": 1.6, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.5, "d": 0.3,
            "r": 0.3, "n": 0.3},
    # -ude above
    # -ase (case, base, phrase, release)
    "ase": {" ": 1.6, ",": 0.5, "s": 0.4, "d": 0.3},
    # -ose (close, prose, rose, suppose, those)
    "ose": {" ": 1.7, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.4, "d": 0.3,
            "n": 0.3, "r": 0.3},
    # -ose above; -use (use, muse, abuse, refuse)
    "use": {" ": 1.6, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.4, "d": 0.3},
    # -ype (type, hype)
    "ype": {" ": 1.3, "s": 0.4},
    # -eak (speak, weak, break, peak)
    "eak": {" ": 1.6, ",": 0.4, "s": 0.4, "i": 0.3, "e": 0.3},
    # -oak -ead (head, read, dead, bread, ahead)
    "ead": {" ": 1.8, ",": 0.7, ".": 0.5, "\n": 0.5, "s": 0.5, "i": 0.3,
            "y": 0.3, "e": 0.3},
    # -eat (beat, heat, meat, great, treat, feat)
    "eat": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.4, "h": 0.6, "e": 0.3,
            "s": 0.5, "i": 0.3},
    # -eed (need, seed, weed, bleed, freed, exceed)
    "eed": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.4, "s": 0.5, "e": 0.4,
            "l": 0.3, "i": 0.3},
    # -eep (sleep, keep, weep, deep)
    "eep": {" ": 1.7, ",": 0.5, ".": 0.4, "\n": 0.4, "i": 0.3, "e": 0.3,
            "s": 0.4},
    # -ood (good, blood, food, flood, stood)
    "ood": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.4, "s": 0.4, "e": 0.3,
            "'": 0.3},
    # -oor (door, floor, poor, moor)
    "oor": {" ": 1.7, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.4},
    # -our (our, hour, four, power-our, flower-our)
    "our": {" ": 1.8, ",": 0.6, ".": 0.5, "\n": 0.4, "s": 0.6, "n": 0.4,
            "t": 0.3, "i": 0.3},
    # -eal (real, deal, seal, heal, meal, reveal)
    "eal": {" ": 1.6, ",": 0.5, ".": 0.4, "s": 0.5, "t": 0.4, "m": 0.3,
            "i": 0.3},
    # -ail (fail, hail, jail, sail, tail, rail, trail)
    "ail": {" ": 1.6, ",": 0.5, ".": 0.4, "s": 0.5, "o": 0.3, "e": 0.3},
    # -oul above
    # -ost (most, lost, cost, post, honest-, almost)
    "ost": {" ": 2.0, ",": 0.6, ".": 0.5, "\n": 0.4, "l": 0.3, "s": 0.4,
            "e": 0.3, "i": 0.3, "o": 0.3, "r": 0.3, "h": 0.3},
    # -ast (last, past, cast, vast, fast, master)
    "ast": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.4, "e": 0.6, "s": 0.4,
            "i": 0.3, "r": 0.3, "l": 0.3},
    # -ust (just, must, trust, lust, thrust, disgust, rust)
    "ust": {" ": 2.0, ",": 0.6, ".": 0.5, "\n": 0.4, "i": 0.3, "s": 0.4,
            "o": 0.3, "y": 0.3, "e": 0.3, "r": 0.3, "a": 0.3},
    # -orm (form, storm, warm-rm, uniform)
    "orm": {" ": 1.6, ",": 0.5, ".": 0.4, "s": 0.5, "e": 0.4, "a": 0.3,
            "i": 0.3},
    # -arm (arm, farm, harm, charm, alarm, warm)
    "arm": {" ": 1.6, ",": 0.5, ".": 0.4, "s": 0.4, "e": 0.4, "i": 0.3},
    # -ure already; -are (are, care, dare, share, bare, fare)
    "are": {" ": 1.8, ",": 0.6, ".": 0.4, "\n": 0.4, "s": 0.5, "d": 0.4,
            "r": 0.3, "n": 0.3},
    # -ord above; -ace above
    # -ict above
    # -ist (list, mist, twist, artist, exist)
    "ist": {" ": 1.7, ",": 0.5, ".": 0.4, "\n": 0.4, "s": 0.4, "i": 0.3,
            "e": 0.4, "r": 0.3},
    # -int (hint, print, point-nt, paint, print, joint)
    "int": {" ": 1.5, ",": 0.4, "s": 0.4, "e": 0.3, "i": 0.3, ".": 0.3},
    # -ort (sort, short, port, report)
    "ort": {" ": 1.7, ",": 0.5, ".": 0.4, "s": 0.5, "e": 0.3, "h": 0.4,
            "u": 0.3, "y": 0.3},
    # -urt (hurt, burnt, court)
    "urt": {" ": 1.3, ",": 0.4, "h": 0.4, "s": 0.3, "e": 0.3},
    # -ath above; -oth above
    # -ese (these, prose, curse-sese, please)
    "ese": {" ": 1.5, ",": 0.4, "n": 0.3, "!": 0.3},
    # -ise (wise, rise, arise, surprise, promise-ise)
    "ise": {" ": 1.6, ",": 0.5, ".": 0.4, "s": 0.4, "d": 0.3, "n": 0.3,
            "r": 0.3},
    # -use above
    # -ize (barely used in Shakespeare English — usually ise)
    # -ial (special, cordial, initial)
    "ial": {" ": 1.3, ",": 0.4, "l": 0.4, "s": 0.3, "t": 0.3},
    # -ian (christian, guardian, villain — mostly -ian ending)
    "ian": {" ": 1.3, ",": 0.4, "s": 0.4, "t": 0.3, "c": 0.2},
    # -ity (city, pity, beauty-ity)
    "ity": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.3, ";": 0.3},
    # -ety (safety, piety)
    "ety": {" ": 1.3, ",": 0.3, ".": 0.2, "s": 0.3},
    # -ly comes from digraph but reinforce: -ely (lovely, surely, truly-like
    # wait, truly is tru-ly; lovely is love+ly)
    "ely": {" ": 1.8, ",": 0.6, ".": 0.4, "\n": 0.3, ";": 0.3, "!": 0.3},
    # -ily (easily, happily, angrily)
    "ily": {" ": 1.8, ",": 0.6, ".": 0.4, "\n": 0.3, ";": 0.3},
    # -ory (story, glory, memory)
    "ory": {" ": 1.5, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.3},
    # -oral
    # -ady (lady, ready, body-like)
    "ady": {" ": 1.5, ",": 0.5, ".": 0.3, "'": 0.3, "s": 0.3},
    # -edy (ready, body, comedy, tragedy)
    "edy": {" ": 1.3, ",": 0.4, "s": 0.3},
    # -ody (body, everybody, melody)
    "ody": {" ": 1.4, ",": 0.4, ".": 0.3, "s": 0.3},
    # -oung (young)
    "oun": {"g": 1.5, "d": 1.5, "t": 1.0, "c": 0.7, " ": 0.3, "s": 0.4,
            "t": 0.4},
    # -ast above; -rd covered above
    # -ild (child, mild, wild)
    "ild": {" ": 1.5, ",": 0.4, "s": 0.4, "r": 0.5, "e": 0.3},
    # -eld (field, held, yield, world-ld)
    "eld": {" ": 1.5, ",": 0.4, "s": 0.4, "i": 0.3, "e": 0.3},
    # -old (old, bold, cold, told, world, hold)
    "old": {" ": 1.8, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.5, "e": 0.4,
            "i": 0.3},
    # -ull above; -unt (hunt, want-unt, grant-unt)
    "unt": {" ": 1.3, ",": 0.4, "s": 0.3, "e": 0.3, "y": 0.3},
    # -amp -emp (temp-, attempt)
    "emp": {"t": 1.5, "e": 0.5, "l": 0.5, "o": 0.4},
    # -omp (pomp, prompt, accompany)
    "omp": {"t": 1.0, "a": 0.5, "e": 0.3, "l": 0.3},
    # -ump (jump, pump, rumpus, trumpet)
    "ump": {" ": 1.0, "e": 0.4, "s": 0.3, "h": 0.3},
    # -imp (limp, simple, imp)
    "imp": {"l": 1.0, " ": 0.5, "o": 0.4, "a": 0.4},
    # -ump above
    # -amp (camp, damp, lamp, stamp)
    "amp": {" ": 1.0, "s": 0.4, "l": 0.4, "e": 0.3},
    # -ank (thank, rank, bank, drink)
    "ank": {" ": 1.3, ",": 0.4, "s": 0.4, "e": 0.4, "!": 0.3},
    # -ink (think, drink, link, sink, stink)
    "ink": {" ": 1.3, ",": 0.4, "s": 0.4, "i": 0.4, "l": 0.3, "e": 0.3},
    # -onk -unk (drunk, trunk)
    "unk": {" ": 1.0, ",": 0.3, "s": 0.3, "e": 0.3},
    # -ide above
    # -nce (prince, dance, chance, since, hence, once)
    "nce": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.4, "d": 0.3,
            "r": 0.3, "!": 0.3, ";": 0.3},
    # -rce (force, pierce, source, fierce)
    "rce": {" ": 1.4, ",": 0.4, "s": 0.3, "d": 0.3},
    # -lse (else, false, pulse)
    "lse": {" ": 1.4, ",": 0.4, ".": 0.3, "\n": 0.3},
    # -rse (verse, course, worse, hoarse, horse)
    "rse": {" ": 1.5, ",": 0.4, ".": 0.3, "s": 0.3, "d": 0.3, "l": 0.3},
    # -nse (sense, dense, tense, response, defense)
    "nse": {" ": 1.4, ",": 0.4, "s": 0.3, "d": 0.3},
    # -sse -pse -mse
    # -aby (baby)
    # -aze (gaze, daze, blaze, amaze)
    "aze": {" ": 1.2, ",": 0.3, "d": 0.4, "s": 0.3},
    # -eze (freeze, breeze)
    "eze": {"e": 0.5, " ": 0.5},
    # -sso, -zzo
}


# Global scale to tune layer strength without hand-editing every entry.
_GLOBAL_SCALE = 0.27


def _build_bias_vectors() -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    lowers = "abcdefghijklmnopqrstuvwxyz"
    for prefix, entries in _L3.items():
        vec = [0.0] * VOCAB_SIZE
        # Default negative for letters not listed — unusual continuations
        # after a common 3-letter prefix should be mildly penalized.
        neg = -3.0 * _GLOBAL_SCALE
        for target in lowers:
            if target not in entries:
                vec[VOCAB_INDEX[target]] = neg
        for nxt, bias in entries.items():
            if nxt in VOCAB_INDEX:
                scaled = bias * _GLOBAL_SCALE
                vec[VOCAB_INDEX[nxt]] = scaled
                if nxt.isalpha() and nxt.lower() == nxt:
                    up = nxt.upper()
                    if up in VOCAB_INDEX:
                        vec[VOCAB_INDEX[up]] = scaled * 0.3
        out[prefix] = vec
    return out


LETTER3_BIAS_VECTORS: dict[str, list[float]] = _build_bias_vectors()


def letter3_bias(word_buffer: str) -> list[float] | None:
    """Return a bias vector keyed on the last 3 letters of word_buffer,
    or None if not enough history or the trigram isn't listed.

    Apostrophes in the buffer are skipped (for Shakespearean
    contractions like 'tis, o'er).
    """
    if len(word_buffer) < 3:
        return None
    # Strip trailing apostrophe, keep only letters for the key.
    # Use the last 3 non-apostrophe characters.
    letters = [c for c in word_buffer if c != "'"]
    if len(letters) < 3:
        return None
    key = "".join(letters[-3:]).lower()
    return LETTER3_BIAS_VECTORS.get(key)
