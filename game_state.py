"""Robust game-state machine for the Missiles! autonomous pilot.

The launcher used to react ad-hoc to individual frames (clicking the red button
every 60 frames, guessing whether gameplay was active). That produced stale
state, double restarts, and getting stuck on menus.

This module centralises all of that into an explicit finite-state machine. It is
pure decision logic: it consumes per-frame *observations* and emits an *action*
for the launcher to execute (focus the window, click play/restart, run the AI,
recalibrate, wait). Keeping it side-effect free makes the transitions
unit-testable without a live game.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import config


class State(Enum):
    LAUNCHING = auto()      # game window not present / not focused
    CALIBRATING = auto()    # window focused, calibration required / being verified
    MENU = auto()           # main menu or score screen: needs a play/restart press
    STARTING = auto()       # play pressed, waiting for gameplay to appear
    PLAYING = auto()        # gameplay active, AI in control
    GAME_OVER = auto()      # death confirmed, will move to restart


class Action(Enum):
    FOCUS_WINDOW = auto()   # bring the game window to the foreground
    CALIBRATE = auto()      # (re)run calibration and verify it
    CLICK_PLAY = auto()     # press the menu play button
    CLICK_RESTART = auto()  # press the restart / red play button
    RUN_AI = auto()         # perceive + plan + drive the joystick for this frame
    RELEASE = auto()        # release the joystick and idle
    WAIT = auto()           # do nothing this tick (transition settling)


@dataclass
class Observation:
    """Signals derived from the current frame / window."""

    foreground: bool = False
    calibrated: bool = False
    player_present: bool = False
    gameover_signal: bool = False   # score screen / restart button visible
    menu_signal: bool = False       # menu play button visible
    now: float = 0.0


class GameStateMachine:
    """Explicit FSM that never gets stuck on stale coordinates or menus."""

    def __init__(self) -> None:
        self.state = State.LAUNCHING
        self._entered_at = 0.0
        self._player_streak = 0
        self._gameover_streak = 0
        self._last_click_at = -1e9
        self._last_calibrated_at = -1e9
        self._restart_attempts = 0

    # -- helpers ---------------------------------------------------------
    def _enter(self, state: State, now: float) -> None:
        if state != self.state:
            self.state = state
            self._entered_at = now
            if state != State.MENU and state != State.GAME_OVER:
                self._restart_attempts = 0

    def _time_in_state(self, now: float) -> float:
        return now - self._entered_at

    def _can_click(self, now: float) -> bool:
        return (now - self._last_click_at) >= config.STATE_RESTART_COOLDOWN

    # -- main tick -------------------------------------------------------
    def update(self, obs: Observation) -> Action:
        """Advance the FSM one tick and return the action to perform."""

        now = obs.now

        # A lost foreground always forces a re-focus, from any state, and
        # releases the joystick so we never hold input on the wrong window.
        if not obs.foreground:
            self._enter(State.LAUNCHING, now)
            return Action.FOCUS_WINDOW

        # Update rolling streaks used for hysteresis.
        self._player_streak = self._player_streak + 1 if obs.player_present else 0
        self._gameover_streak = self._gameover_streak + 1 if obs.gameover_signal else 0

        if self.state == State.LAUNCHING:
            # Foreground regained -> calibrate before anything else.
            self._enter(State.CALIBRATING, now)
            return Action.CALIBRATE

        if self.state == State.CALIBRATING:
            if not obs.calibrated:
                return Action.CALIBRATE
            self._last_calibrated_at = now
            # Route based on what the screen currently shows.
            if obs.player_present:
                self._enter(State.PLAYING, now)
                return Action.RUN_AI
            self._enter(State.MENU, now)
            return Action.WAIT

        if self.state == State.MENU:
            # Gameplay may already have begun between ticks.
            if self._player_streak >= config.STATE_PLAYING_CONFIRM_FRAMES:
                self._enter(State.PLAYING, now)
                return Action.RUN_AI
            if self._can_click(now):
                self._last_click_at = now
                self._restart_attempts += 1
                # Score screen uses the restart/red button; fresh menu uses play.
                if obs.gameover_signal:
                    self._enter(State.STARTING, now)
                    return Action.CLICK_RESTART
                self._enter(State.STARTING, now)
                return Action.CLICK_PLAY
            return Action.WAIT

        if self.state == State.STARTING:
            if self._player_streak >= config.STATE_PLAYING_CONFIRM_FRAMES:
                self._enter(State.PLAYING, now)
                return Action.RUN_AI
            # If the play press did not take, drop back to MENU to retry.
            if self._time_in_state(now) >= config.STATE_START_TIMEOUT:
                self._enter(State.MENU, now)
                return Action.WAIT
            # A game-over screen while "starting" means the press hit nothing
            # useful; go handle it as a menu/score screen.
            if self._gameover_streak >= config.STATE_GAMEOVER_CONFIRM_FRAMES:
                self._enter(State.MENU, now)
                return Action.WAIT
            return Action.RELEASE

        if self.state == State.PLAYING:
            # Confirmed death -> game over (hysteresis avoids flicker on
            # transient missed player detections).
            if self._gameover_streak >= config.STATE_GAMEOVER_CONFIRM_FRAMES:
                self._enter(State.GAME_OVER, now)
                return Action.RELEASE
            if obs.player_present:
                return Action.RUN_AI
            # Player missing but no explicit game-over: keep flying briefly on
            # last state, then release if it persists.
            if self._time_in_state(now) >= 0.0 and self._player_streak == 0:
                return Action.RUN_AI if self._gameover_streak == 0 else Action.RELEASE
            return Action.RUN_AI

        if self.state == State.GAME_OVER:
            # Verify death is stable, then restart. Re-verify calibration first
            # in case the death animation shifted the canvas.
            if now - self._last_calibrated_at >= config.STATE_RECALIBRATE_INTERVAL:
                self._enter(State.CALIBRATING, now)
                return Action.CALIBRATE
            if self._player_streak >= config.STATE_PLAYING_CONFIRM_FRAMES:
                # A new run somehow started; resume play.
                self._enter(State.PLAYING, now)
                return Action.RUN_AI
            if self._can_click(now):
                self._last_click_at = now
                self._restart_attempts += 1
                self._enter(State.STARTING, now)
                return Action.CLICK_RESTART
            return Action.WAIT

        return Action.WAIT

    # -- introspection ---------------------------------------------------
    def needs_recalibration(self, now: float) -> bool:
        return (now - self._last_calibrated_at) >= config.STATE_RECALIBRATE_INTERVAL

    def mark_calibrated(self, now: float) -> None:
        self._last_calibrated_at = now

    @property
    def restart_attempts(self) -> int:
        return self._restart_attempts

    def describe(self) -> str:
        return (
            f"state={self.state.name} player_streak={self._player_streak} "
            f"gameover_streak={self._gameover_streak} restarts={self._restart_attempts}"
        )
