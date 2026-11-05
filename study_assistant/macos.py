"""The few macOS features Qt doesn't expose: global shortcuts, screen capture, floating windows.

Everything goes through ctypes and system tools, so there's no extra dependency to bundle. On other
platforms the hotkey and float helpers do nothing and capture raises ScreenCaptureError.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_EVENT_CLASS_KEYBOARD = 0x6B657962  # 'keyb'
_EVENT_HOTKEY_PRESSED = 5
_HOTKEY_SIGNATURE = 0x43525458  # 'CRTX'
_PARAM_DIRECT_OBJECT = 0x2D2D2D2D  # '----'
_TYPE_HOTKEY_ID = 0x686B6964  # 'hkid'

SCREEN_SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"


class ScreenCaptureError(RuntimeError):
    """A readable reason the screen could not be captured."""


def _framework(name: str) -> ctypes.CDLL:
    return ctypes.CDLL(f"/System/Library/Frameworks/{name}.framework/{name}")


# ---------------------------------------------------------------------------------------------
# Global hotkey
# ---------------------------------------------------------------------------------------------

# Carbon modifier flags and US-layout virtual key codes (Carbon hotkeys use physical key positions).
_MODIFIERS = {"cmd": 0x0100, "shift": 0x0200, "option": 0x0800, "control": 0x1000}
_SYMBOLS = {"control": "⌃", "option": "⌥", "shift": "⇧", "cmd": "⌘"}  # Apple's display order
KEY_CODES: dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12,
    "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23,
    "=": 24, "9": 25, "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34,
    "p": 35, "return": 36, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44,
    "n": 45, "m": 46, ".": 47, "tab": 48, "space": 49, "`": 50,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101,
    "f10": 109, "f11": 103, "f12": 111,
}
_KEY_LABELS = {"space": "Space", "return": "↩", "tab": "⇥"}


@dataclass(frozen=True)
class Shortcut:
    """A global key combination, stored as text like "cmd+shift+space"."""

    key: str
    modifiers: frozenset[str] = frozenset()

    @classmethod
    def parse(cls, text: str) -> "Shortcut | None":
        """Read "cmd+shift+space". Returns None for "" or "off" (shortcut disabled)."""
        parts = [p.strip().casefold() for p in (text or "").split("+") if p.strip()]
        if not parts or parts == ["off"]:
            return None
        *names, key = parts
        aliases = {"command": "cmd", "ctrl": "control", "alt": "option", "opt": "option"}
        modifiers = frozenset(aliases.get(name, name) for name in names)
        if key not in KEY_CODES or not modifiers <= set(_MODIFIERS):
            raise ValueError(f"Unsupported shortcut: {text}")
        return cls(key, modifiers)

    def validate(self) -> None:
        """Global shortcuts need ⌘, ⌃, or ⌥, or they'd swallow ordinary typing in every app."""
        if not self.modifiers & {"cmd", "control", "option"} and not self.key.startswith("f"):
            raise ValueError("Use at least one of ⌘, ⌃ or ⌥ so the shortcut doesn't block normal typing.")

    @property
    def text(self) -> str:
        order = [m for m in ("cmd", "control", "option", "shift") if m in self.modifiers]
        return "+".join([*order, self.key])

    @property
    def label(self) -> str:
        symbols = "".join(_SYMBOLS[m] for m in ("control", "option", "shift", "cmd") if m in self.modifiers)
        return symbols + _KEY_LABELS.get(self.key, self.key.upper())

    @property
    def carbon(self) -> tuple[int, int]:
        mask = 0
        for modifier in self.modifiers:
            mask |= _MODIFIERS[modifier]
        return KEY_CODES[self.key], mask


def shortcut_label(text: str) -> str:
    try:
        shortcut = Shortcut.parse(text)
    except ValueError:
        return "OFF"
    return shortcut.label if shortcut else "OFF"


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("event_class", ctypes.c_uint32), ("event_kind", ctypes.c_uint32)]


class _EventHotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


class HotkeyManager:
    """Global shortcuts that fire on the main thread whichever app is in front (macOS).

    Carbon's RegisterEventHotKey needs no Accessibility or Input Monitoring permission. One event
    handler serves every shortcut and dispatches on the hotkey ID. Elsewhere this does nothing.
    """

    def __init__(self) -> None:
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._refs: list[ctypes.c_void_p] = []
        self._next_id = 1
        self._carbon = None
        if sys.platform != "darwin":
            return
        carbon = _framework("Carbon")
        carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
        carbon.InstallEventHandler.argtypes = [
            ctypes.c_void_p, _HANDLER, ctypes.c_ulong, ctypes.POINTER(_EventTypeSpec),
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ]
        carbon.RegisterEventHotKey.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, _EventHotKeyID, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
        ]
        carbon.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
        carbon.RemoveEventHandler.argtypes = [ctypes.c_void_p]
        carbon.GetEventParameter.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p,
        ]

        def handle(_call_ref, event, _user_data) -> int:
            hotkey = _EventHotKeyID()
            if carbon.GetEventParameter(event, _PARAM_DIRECT_OBJECT, _TYPE_HOTKEY_ID, None,
                                        ctypes.sizeof(hotkey), None, ctypes.byref(hotkey)) == 0:
                callback = self._callbacks.get(hotkey.id)
                if callback:
                    callback()
            return 0

        self._callback = _HANDLER(handle)  # Carbon holds this pointer, so it must outlive the handler
        self._handler_ref = ctypes.c_void_p()
        spec = _EventTypeSpec(_EVENT_CLASS_KEYBOARD, _EVENT_HOTKEY_PRESSED)
        if carbon.InstallEventHandler(carbon.GetApplicationEventTarget(), self._callback, 1,
                                      ctypes.byref(spec), None, ctypes.byref(self._handler_ref)) == 0:
            self._carbon = carbon

    def register(self, shortcut: Shortcut | None, on_press: Callable[[], None]) -> bool:
        """True when registered. False if off, unsupported here, or another app already owns it."""
        if shortcut is None or self._carbon is None:
            return False
        key, mask = shortcut.carbon
        ref = ctypes.c_void_p()
        hotkey_id = self._next_id
        self._next_id += 1
        status = self._carbon.RegisterEventHotKey(
            key, mask, _EventHotKeyID(_HOTKEY_SIGNATURE, hotkey_id),
            self._carbon.GetApplicationEventTarget(), 0, ctypes.byref(ref),
        )
        if status != 0:
            return False
        self._callbacks[hotkey_id] = on_press
        self._refs.append(ref)
        return True

    def clear(self) -> None:
        if self._carbon is not None:
            for ref in self._refs:
                self._carbon.UnregisterEventHotKey(ref)
        self._refs.clear()
        self._callbacks.clear()

    def close(self) -> None:
        self.clear()
        if self._carbon is not None:
            self._carbon.RemoveEventHandler(self._handler_ref)
            self._carbon = None


# ---------------------------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------------------------

def _core_graphics() -> ctypes.CDLL:
    graphics = _framework("CoreGraphics")
    graphics.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
    graphics.CGRequestScreenCaptureAccess.restype = ctypes.c_bool
    return graphics


def screen_capture_allowed() -> bool:
    return sys.platform != "darwin" or bool(_core_graphics().CGPreflightScreenCaptureAccess())


def request_screen_capture() -> None:
    """Show the system permission prompt. macOS only applies a grant after the app restarts."""
    if sys.platform == "darwin":
        _core_graphics().CGRequestScreenCaptureAccess()


def capture_region(x: int, y: int, width: int, height: int) -> bytes:
    """Return a PNG of one screen region. The file exists only for the length of this call."""
    if sys.platform != "darwin":
        raise ScreenCaptureError("Screen capture is only available on macOS.")
    with tempfile.TemporaryDirectory(prefix="cortex-screen-") as folder:
        path = Path(folder) / "screen.png"
        done = subprocess.run(
            ["/usr/sbin/screencapture", "-x", "-t", "png", "-R", f"{x},{y},{width},{height}", str(path)],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if done.returncode != 0 or not path.exists():
            raise ScreenCaptureError(done.stderr.strip() or "macOS did not return a screenshot.")
        return path.read_bytes()


def capture_interactive() -> bytes | None:
    """Let the user drag out a region (or click a window) with macOS's own picker.

    Returns None if they press Esc. Blocks until they finish, so call it off the UI thread.
    """
    if sys.platform != "darwin":
        raise ScreenCaptureError("Screen capture is only available on macOS.")
    with tempfile.TemporaryDirectory(prefix="cortex-screen-") as folder:
        path = Path(folder) / "area.png"
        done = subprocess.run(
            ["/usr/sbin/screencapture", "-i", "-x", "-t", "png", str(path)],
            capture_output=True, text=True, timeout=300, check=False,
        )
        if not path.exists():
            if done.returncode not in (0, 1):
                raise ScreenCaptureError(done.stderr.strip() or "macOS did not return a screenshot.")
            return None  # cancelled with Esc
        return path.read_bytes()


# ---------------------------------------------------------------------------------------------
# Floating window
# ---------------------------------------------------------------------------------------------

_LEVEL_STATUS = 25
_JOIN_ALL_SPACES = 1 << 0
_FULL_SCREEN_AUXILIARY = 1 << 8
_NON_ACTIVATING_PANEL = 1 << 7


def _send(receiver: int, selector: str, restype=ctypes.c_void_p, argtypes: tuple = (), *args):
    """objc_msgSend, typed per call as arm64 requires."""
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    send = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes))
    return send(receiver, objc.sel_registerName(selector.encode()), *args)


def float_above_everything(view_id: int) -> None:
    """Keep a window over other apps, on every Space and over full-screen apps.

    view_id is a Qt winId(), which on macOS is the NSView. The non-activating style lets the window
    take keystrokes without bringing the rest of the app, like the main window, to the front.
    """
    if sys.platform != "darwin" or not view_id:
        return
    window = _send(view_id, "window")
    if not window:
        return
    _send(window, "setLevel:", None, (ctypes.c_long,), _LEVEL_STATUS)
    _send(window, "setCollectionBehavior:", None, (ctypes.c_ulong,), _JOIN_ALL_SPACES | _FULL_SCREEN_AUXILIARY)
    _send(window, "setHidesOnDeactivate:", None, (ctypes.c_bool,), False)
    style = _send(window, "styleMask", ctypes.c_ulong)
    _send(window, "setStyleMask:", None, (ctypes.c_ulong,), style | _NON_ACTIVATING_PANEL)
