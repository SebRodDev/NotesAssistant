"""The few macOS features Qt doesn't expose: a global hotkey, screen capture, floating windows.

Everything goes through ctypes and system tools, so there's no extra dependency to bundle. On other
platforms the hotkey and float helpers do nothing and capture raises ScreenCaptureError.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

HOTKEY_LABEL = "⌃⌥Space"
_KEY_SPACE = 49
_MODIFIER_OPTION = 0x0800
_MODIFIER_CONTROL = 0x1000
_EVENT_CLASS_KEYBOARD = 0x6B657962  # 'keyb'
_EVENT_HOTKEY_PRESSED = 5
_HOTKEY_SIGNATURE = 0x43525458  # 'CRTX'

SCREEN_SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"


class ScreenCaptureError(RuntimeError):
    """A readable reason the screen could not be captured."""


def _framework(name: str) -> ctypes.CDLL:
    return ctypes.CDLL(f"/System/Library/Frameworks/{name}.framework/{name}")


# ---------------------------------------------------------------------------------------------
# Global hotkey
# ---------------------------------------------------------------------------------------------

class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("event_class", ctypes.c_uint32), ("event_kind", ctypes.c_uint32)]


class _EventHotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


class GlobalHotkey:
    """Calls back on the main thread when ⌃⌥Space is pressed, whichever app is in front.

    Carbon's RegisterEventHotKey is used because, unlike event taps, it needs no Accessibility or
    Input Monitoring permission.
    """

    def __init__(self, on_press: Callable[[], None]) -> None:
        self.registered = False
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
        self._carbon = carbon

        def handle(_call_ref, _event, _user_data) -> int:
            on_press()
            return 0

        self._callback = _HANDLER(handle)  # Carbon holds this pointer, so it must outlive the handler
        target = carbon.GetApplicationEventTarget()
        spec = _EventTypeSpec(_EVENT_CLASS_KEYBOARD, _EVENT_HOTKEY_PRESSED)
        self._handler_ref = ctypes.c_void_p()
        self._hotkey_ref = ctypes.c_void_p()
        if carbon.InstallEventHandler(target, self._callback, 1, ctypes.byref(spec), None,
                                      ctypes.byref(self._handler_ref)):
            return
        status = carbon.RegisterEventHotKey(
            _KEY_SPACE, _MODIFIER_CONTROL | _MODIFIER_OPTION, _EventHotKeyID(_HOTKEY_SIGNATURE, 1),
            target, 0, ctypes.byref(self._hotkey_ref),
        )
        self.registered = status == 0  # non-zero: another app already owns this shortcut

    def close(self) -> None:
        if self.registered:
            self._carbon.UnregisterEventHotKey(self._hotkey_ref)
            self._carbon.RemoveEventHandler(self._handler_ref)
            self.registered = False


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
