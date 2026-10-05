"""A minimal Twitch chat client (IRC over TLS): reads commands from the channel and posts replies.

The OAuth token is typed into the admin panel by the streamer and kept in secrets/chat.json (not in the web-served
data folder, ignored by git). It is never logged and never sent back to the page -- the status only says whether
a token is set.

Needs a Twitch account for the bot (can be the streamer's own) and a token with the chat:read and chat:edit scopes.
"""
from __future__ import annotations

import json
import queue
import re
import socket
import ssl
import threading
import time

from .goal import ROOT

CONFIG_PATH = ROOT / "secrets" / "chat.json"
HOST, PORT = "irc.chat.twitch.tv", 6697
SEND_EVERY_S = 1.6                      # Twitch drops messages above ~20 per 30 s for a normal account
PRIVMSG = re.compile(r"^(?:@\S+ )?:([^!\s]+)![^ ]+ PRIVMSG #\S+ :(.*)$")


def parse_privmsg(line: str):
    m = PRIVMSG.match(line.strip())
    return (m.group(1), m.group(2)) if m else None


class TwitchChat:
    def __init__(self, on_message, log, host: str = HOST, port: int = PORT, tls: bool = True):
        self.on_message, self.log = on_message, log
        self.host, self.port, self.tls = host, port, tls
        self.cfg = {"enabled": False, "channel": "", "nick": "", "token": ""}
        self.status, self.error = "off", ""
        self.out: queue.Queue = queue.Queue(maxsize=50)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock = None
        self._load()

    # ---------------------------------------------------------------- config (token never leaves this class)
    def _load(self):
        try:
            self.cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass

    def public(self) -> dict:
        return {"enabled": bool(self.cfg["enabled"]), "channel": self.cfg["channel"], "nick": self.cfg["nick"],
                "token_set": bool(self.cfg["token"]), "status": self.status, "error": self.error}

    def configure(self, channel=None, nick=None, token=None, enabled=None) -> dict:
        if channel is not None:
            self.cfg["channel"] = re.sub(r"[^a-z0-9_]", "", str(channel).lower().lstrip("#"))[:25]
        if nick is not None:
            self.cfg["nick"] = re.sub(r"[^a-z0-9_]", "", str(nick).lower())[:25]
        if token:                                              # empty = keep the stored one
            self.cfg["token"] = str(token).strip().replace("oauth:", "")[:80]
        if enabled is not None:
            self.cfg["enabled"] = bool(enabled)
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(json.dumps(self.cfg), encoding="utf-8")
        except OSError as e:
            self.error = f"could not save the settings: {e.__class__.__name__}"
        self.restart()
        return self.public()

    # ---------------------------------------------------------------- connection
    def restart(self):
        self.stop()
        if self.cfg["enabled"] and self.cfg["channel"] and self.cfg["nick"] and self.cfg["token"]:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="twitch-chat", daemon=True)
            self._thread.start()
        else:
            self.status = "off" if not self.cfg["enabled"] else "incomplete settings"

    def stop(self):
        self._stop.set()
        sock = self._sock
        if sock:
            try:
                sock.close()
            except OSError:
                pass
        t = self._thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=3)
        self._thread = None
        if not self.cfg["enabled"]:
            self.status = "off"

    def say(self, text: str):
        if self.status == "connected":
            try:
                self.out.put_nowait(text[:450])
            except queue.Full:
                pass

    def _run(self):
        backoff = 3
        while not self._stop.is_set():
            try:
                self._session()
                backoff = 3
            except Exception as e:
                self.status, self.error = "reconnecting", e.__class__.__name__      # never include the exception text: it may echo credentials
            if self._stop.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    def _session(self):
        self.status, self.error = "connecting", ""
        raw = socket.create_connection((self.host, self.port), timeout=15)
        sock = ssl.create_default_context().wrap_socket(raw, server_hostname=self.host) if self.tls else raw
        sock.settimeout(1.0)
        self._sock = sock

        def send(line: str):
            sock.sendall((line + "\r\n").encode("utf-8", "replace"))

        send("PASS oauth:" + self.cfg["token"])
        send("NICK " + self.cfg["nick"])
        send("JOIN #" + self.cfg["channel"])
        buf, last_send, last_rx = b"", 0.0, time.time()
        while not self._stop.is_set():
            try:
                data = sock.recv(4096)
                if not data:
                    raise ConnectionError("closed")
                buf += data
                last_rx = time.time()
            except socket.timeout:
                if time.time() - last_rx > 300:
                    raise ConnectionError("silent")
            while b"\r\n" in buf:
                line, buf = buf.split(b"\r\n", 1)
                text = line.decode("utf-8", "replace")
                if text.startswith("PING"):
                    send("PONG " + text[5:])
                elif "Login authentication failed" in text or "Improperly formatted auth" in text:
                    self.status, self.error = "login failed", "Twitch rejected the nick/token (needs chat:read + chat:edit)"
                    self._stop.set()
                    return
                elif " 366 " in text or "JOIN #" in text and self.status != "connected":
                    self.status = "connected"
                else:
                    msg = parse_privmsg(text)
                    if msg and msg[0].lower() != self.cfg["nick"]:
                        reply = None
                        try:
                            reply = self.on_message(msg[0], msg[1])
                        except Exception:
                            pass
                        if reply:
                            self.say(reply)
            if time.time() - last_send >= SEND_EVERY_S:
                try:
                    out = self.out.get_nowait()
                    send(f"PRIVMSG #{self.cfg['channel']} :{out}")
                    last_send = time.time()
                except queue.Empty:
                    pass
        try:
            sock.close()
        except OSError:
            pass
