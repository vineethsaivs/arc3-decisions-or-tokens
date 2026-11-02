"""Per-level segments from transcripts, with per-turn flags. Imported by the metric scripts."""
import re
from common import *

SEARCH_DEF = re.compile(r'\bdeque\b|\bheapq\b|heappush|def \w*(bfs|search|solve|astar|a_star|dijkstra|plan)\w*\s*\(', re.I)
SIM_DEF = re.compile(r'def \w*(sim|simulate|step|apply|move|predict|model|transition)\w*\s*\(', re.I)
DEF = re.compile(r'^def (\w+)\s*\(', re.M)
CALL = re.compile(r'\b([A-Za-z_]\w*)\s*\(')


def flags(t):
    u = t['user']
    codes = [x['code'] for x in t['resp']]
    t['gameover'] = 'GAME OVER occurred' in u
    t['dreset'] = 'You deliberately reset the current level' in u
    t['stopped'] = ('Executed actions (first' in u) or ('sequence was stopped before' in u)
    t['yield'] = 'turn_token_budget' in t['msg']
    t['stall'] = 'Stall check' in u
    t['newlevel'] = 'You have completed the previous level' in u
    t['searchdef'] = any(SEARCH_DEF.search(c) for c in codes)
    t['simdef'] = any(SIM_DEF.search(c) for c in codes)
    t['defs'] = [n for c in codes for n in DEF.findall(c)]
    t['rchars'] = sum(x['rchars'] for x in t['resp'])
    t['ncalls'] = sum(1 for x in t['resp'] if x['code'])
    t['errs'] = sum(1 for x in t['resp'] if 'Traceback' in x['result'] or re.search(r'^\w+Error\b', x['result'], re.M))


def segments():
    """yield run, game, level, turns_of_level, cleared(bool), t_start, t_end, all_turns"""
    for r, g, ts in games():
        for t in ts:
            flags(t)
        lv = [t['level'] for t in ts]
        i = 0
        while i < len(ts):
            L = lv[i]
            j = i
            while j < len(ts) and lv[j] == L:
                j += 1
            seg = ts[i:j]
            cleared = j < len(ts) and lv[j] is not None and L is not None and lv[j] > L
            t_end = ts[j]['t'] if j < len(ts) else ts[-1]['t'] + ts[-1]['dur']
            yield r, g, L, seg, cleared, ts[i]['t'], t_end, ts, i
            i = j
