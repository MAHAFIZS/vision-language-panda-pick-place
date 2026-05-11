from __future__ import annotations
import re
from typing import Dict, Any, Optional


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s)
    return s


COLOR_ALIASES = {
    "red": "red_box",
    "green": "green_box",
    "blue": "blue_box",
    "yellow": "yellow_box",
    "white": "box",
    "gray": "box",
    "grey": "box",
}


TARGET_ALIASES = {
    "bin": "bin_center",
    "basket": "bin_center",
    "bucket": "bin_center",
    "left": "zone_left",
    "right": "zone_right",
    "zone left": "zone_left",
    "zone right": "zone_right",
}


def _extract_object_phrase(t: str) -> Optional[str]:
    m = re.search(r"\b(pick|move|put|grab)\b\s+(.+?)(?:\s+\b(place|stack|on|to|at|into)\b|$)", t)
    if m:
        return m.group(2).strip()

    m = re.search(r"\bstack\b\s+(.+?)\s+\bon\b", t)
    if m:
        return m.group(1).strip()

    m = re.search(r"\bwhere\s+is\b\s+(.+)$", t)
    if m:
        return m.group(1).strip()

    return None


def _resolve_obj_token(obj_phrase: Optional[str]) -> Optional[str]:
    if not obj_phrase:
        return None

    p = obj_phrase.strip().lower()
    p = p.replace("the ", "").strip()
    p = p.replace("cube", "box").strip()
    p = p.replace("block", "box").strip()
    p = p.replace(" ", "_")

    if p in COLOR_ALIASES:
        return COLOR_ALIASES[p]

    if p in COLOR_ALIASES.values():
        return p

    for color, obj in COLOR_ALIASES.items():
        if color in p:
            return obj

    return p


def _find_color_object(t: str) -> Optional[str]:
    for color, obj in COLOR_ALIASES.items():
        if re.search(rf"\b{color}\b", t):
            return obj
    return None


def parse_command(text: str) -> Dict[str, Any]:
    t = _normalize(text)

    if t in {"q", "quit", "exit"}:
        return {"task": "quit"}

    if t in {"h", "help", "?"}:
        return {"task": "help"}

    if t in {"reset", "home"}:
        return {"task": "reset"}

    if t in {"open", "open gripper", "gripper open"} or "open gripper" in t:
        return {"task": "gripper", "mode": "open"}

    if t in {"close", "close gripper", "gripper close"} or "close gripper" in t:
        return {"task": "gripper", "mode": "close"}

    if t in {"list", "list objects", "objects", "what objects"}:
        return {"task": "list_objects"}

    if t.startswith("where is"):
        obj = _resolve_obj_token(_extract_object_phrase(t))
        return {"task": "where", "obj": obj or "box"}

    if t in {"sort", "sort all", "sort everything"}:
        return {"task": "sort_all"}

    if t in {"make a tower", "tower", "build a tower"}:
        return {"task": "tower"}

    m_stack = re.search(r"\bstack\b\s+(.+?)\s+\bon\b\s+(.+)$", t)
    if m_stack:
        obj = _resolve_obj_token(m_stack.group(1))
        base = _resolve_obj_token(m_stack.group(2))
        return {"task": "stack", "obj": obj or "box", "base": base or "box"}

    # Natural language pick/move/put/grab commands
    if any(w in t for w in ["pick", "move", "put", "grab"]):
        obj = _resolve_obj_token(_extract_object_phrase(t)) or _find_color_object(t) or "box"

        m_on = re.search(r"\b(place|put)\b\s+\bon\b\s+(.+)$", t)
        if m_on:
            base = _resolve_obj_token(m_on.group(2))
            return {"task": "place_on", "obj": obj, "base": base or "box"}

        for k, v in TARGET_ALIASES.items():
            if re.search(rf"\b{k}\b", t):
                return {"task": "pick_place", "obj": obj, "target": v}

        mx = re.search(r"\bx\s*(-?\d+(\.\d+)?)\b", t)
        my = re.search(r"\by\s*(-?\d+(\.\d+)?)\b", t)
        if mx and my:
            return {
                "task": "pick_place_xy",
                "obj": obj,
                "x": float(mx.group(1)),
                "y": float(my.group(1)),
            }

        return {"task": "pick_place", "obj": obj, "target": "bin_center"}

    return {"task": "unknown", "text": text}