import argparse
import json
import math
import os
import random
import threading
import time
import subprocess
import pickle
from typing import Dict, List, Optional, Tuple

import mido
from pyrekordbox.anlz import AnlzFile
from websockets.sync.client import connect

import suggest_transitions_bucketed as suggester
from rekordbox_usb_export_demo import RekordboxUSBExport

#MIDI constants for controller emulation:

DECK1_CHANNEL = 0
DECK2_CHANNEL = 1

DECK_PLAY_PAUSE_NOTE = 0x0B
DECK_CUE_NOTE = 0x0C

DECK_PLAY_STATE_NOTES = {0x0B, 0x47, 71}
DECK_CUE_STATE_NOTES = {0x0C, 0x48}

DECK1_PLAYPOS_MSB_CC = 0x50
DECK1_PLAYPOS_LSB_CC = 0x51
DECK1_LIVEBPM_MSB_CC = 0x56
DECK1_LIVEBPM_LSB_CC = 0x57
DECK1_TEMPORATE_MSB_CC = 0x62
DECK1_TEMPORATE_LSB_CC = 0x63
DECK1_BEAT_SYNC_CC = 0x7A

DECK2_PLAYPOS_MSB_CC = DECK1_PLAYPOS_MSB_CC
DECK2_PLAYPOS_LSB_CC = DECK1_PLAYPOS_LSB_CC
DECK2_LIVEBPM_MSB_CC = 0x60
DECK2_LIVEBPM_LSB_CC = 0x61
DECK2_TEMPORATE_MSB_CC = 0x64
DECK2_TEMPORATE_LSB_CC = 0x65
DECK2_BEAT_SYNC_CC = 0x7B

DECK_PLAYPOS_INPUT_MSB_CC = 0x7C
DECK_PLAYPOS_INPUT_LSB_CC = 0x7D
DECK_SYNC_MODE_INPUT_CC = 0x7E
DECK_QUANTIZE_SET_CC = 0x6A
MIXER_CONTROL_CHANNEL = 6

EXPECTED_CONNECTION_SYSEX = (0x00, 0x40, 0x05, 0x00, 0x00, 0x04, 0x05, 0x00, 0x50, 0x02)

IN_PORT_NAME = "DJ-AI Controller"
OUT_PORT_NAME = "DJ-AI Controller"

WEBSOCKET_PORT = 9001
WEBSOCKET_TOKEN = "toxic"
MODEL_PATH = "schedulerRandomForest.joblib"

BACKGROUND_POLL_SLEEP = 0.002
WAIT_SLEEP = 0.005
VERBOSE_PRINT_EVERY = 0.15
DEFAULT_TRIGGER_OFFSET_BEATS = 0.0

#Phrase decoding helpers:

#As a fallback from map_phrases_from_pssi(), map phrase IDs to high-variant labels
def decode_high_variant(phrase_id, k1, k2, k3):
    if phrase_id == 1:
        return "Intro"
    if phrase_id == 2:
        return "Up"
    if phrase_id == 3:
        return "Down"
    if phrase_id == 5:
        return "Chorus"
    if phrase_id == 6:
        return "Outro"
    return f"Phrase {phrase_id} (High mood)"

#Main mapping from IDs to phrase labels
def map_phrases_from_pssi(pssi_container):
    #Dictionaries for mappings
    low_labels = {1: "Intro", 2: "Verse", 3: "Verse", 4: "Verse", 5: "Verse", 6: "Verse", 7: "Verse", 8: "Bridge", 9: "Chorus", 10: "Outro"}
    mid_labels = low_labels.copy()
    bank_labels = {0: "Default/Cool", 1: "Cool", 2: "Natural", 3: "Hot", 4: "Subtle", 5: "Warm", 6: "Vivid", 7: "Club 1", 8: "Club 2"}

    mood = pssi_container.mood
    bank = pssi_container.bank
    end_beat = pssi_container.end_beat
    entries = list(pssi_container.entries)
    bank_label = bank_labels.get(bank, f"UnknownBank{bank}")
    beat_starts = [e.beat for e in entries]

    #Use attributes to determine phrase labels per Rekordbox DB documentation
    phrases = []
    for idx, entry in enumerate(entries):
        phrase_index = entry.index
        beat_start = entry.beat
        kind = entry.kind
        k1 = getattr(entry, "k1", 0)
        k2 = getattr(entry, "k2", 0)
        k3 = getattr(entry, "k3", 0)
        b_flag = getattr(entry, "b", 0)

        #Collect any extra phrase beat markers
        extra_beats = []
        if b_flag == 0:
            if getattr(entry, "beat_2", 0):
                extra_beats = [entry.beat_2]
        elif b_flag == 1:
            for attr in ("beat_2", "beat_3", "beat_4"):
                val = getattr(entry, attr, 0)
                if val:
                    extra_beats.append(val)

        #Store fill marker if one exists
        fill_flag = getattr(entry, "fill", 0)
        beat_fill = entry.beat_fill if fill_flag else None

        #Resolve final phrase label from mood and phrase kind
        if mood == 3:
            label = low_labels.get(kind, f"Phrase{kind}")
        elif mood == 2:
            label = mid_labels.get(kind, f"Phrase{kind}")
        elif mood == 1:
            label = decode_high_variant(kind, k1, k2, k3)
        else:
            label = f"UnknownMood{mood}-Phrase{kind}"

        #Build normalized phrase object
        beat_end = beat_starts[idx + 1] if idx < len(entries) - 1 else end_beat
        phrases.append({"index": phrase_index, "beat_start": beat_start, "beat_end": beat_end, "label": label, "extra_beats": extra_beats, "fill": bool(fill_flag), "beat_fill": beat_fill, "bank_label": bank_label})
    return phrases


#Helper for loading Rekordbox metadata and caching results
class RekordboxMetadataResolver:
    def __init__(self, usb_root: str) -> None:
        self.db = RekordboxUSBExport(usb_root)
        self._metadata_cache: Dict[str, Tuple[List[dict], list]] = {}

    #Return all usable track paths from the USB export
    def get_all_track_paths(self, existing_only: bool = True) -> List[str]:
        return self.db.get_track_paths(existing_only=existing_only)

    #Preload metadata for many tracks at once
    def preload_all_metadata(
        self,
        track_paths: Optional[List[str]] = None,
        show_progress: bool = False,
    ):
        phrases = {}
        beat_grids = {}

        if track_paths is None:
            track_paths = self.get_all_track_paths(existing_only=True)

        total = len(track_paths)
        start = time.perf_counter()

        #Loop all tracks and cache phrase / beatgrid data
        for i, path in enumerate(track_paths, 1):
            try:
                p, bg = self.get_metadata_for_track(path)
                key = os.path.basename(path)
                phrases[key] = p
                beat_grids[key] = bg
            except Exception as e:
                print(f"\n[Metadata preload] Failed: {os.path.basename(path)} -> {e}")

            #Print preload progress if enabled
            if show_progress and total > 0:
                elapsed = time.perf_counter() - start
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0

                print(
                    f"\r[Metadata preload] {i}/{total} "
                    f"({(i/total)*100:.1f}%) | ETA: {remaining:.1f}s",
                    end="",
                    flush=True,
                )

        if show_progress:
            print()

        return phrases, beat_grids

    #Normalize paths for cache lookups
    @staticmethod
    def _norm_path(path: str) -> str:
        return os.path.normcase(os.path.normpath(str(path)))

    #Resolve a track path back to its export DB entry
    def resolve_content(self, track_path: str):
        content = self.db.find_track_by_path(track_path)
        if content is not None:
            return content
        raise KeyError(f"Track not found in Rekordbox USB DB: {track_path}")

    #Load DAT and EXT metadata for one track
    def get_metadata_for_track(self, track_path: str) -> Tuple[List[dict], list]:
        norm = self._norm_path(track_path)
        if norm in self._metadata_cache:
            return self._metadata_cache[norm]

        #Resolve matching export entry and analysis paths
        content = self.resolve_content(track_path)
        paths = self.db.get_anlz_paths(content.ID)
        if not paths.get("DAT"):
            raise FileNotFoundError(f"No DAT analysis file for {track_path}")
        if not paths.get("EXT"):
            raise FileNotFoundError(f"No EXT analysis file for {track_path}")

        #Parse beat grid from DAT file
        anlz_dat = AnlzFile.parse_file(paths["DAT"])
        beat_grid = anlz_dat.get("beat_grid")
        beat_grid = [arr.tolist() if hasattr(arr, "tolist") else arr for arr in beat_grid]

        #Parse phrase info from EXT file
        anlz_ext = AnlzFile.parse_file(paths["EXT"])
        pssi = anlz_ext.get("PSSI")
        if pssi is None:
            raise ValueError(f"No PSSI phrase data in EXT for {track_path}")
        phrases = map_phrases_from_pssi(pssi)

        #Cache and return final metadata
        self._metadata_cache[norm] = (phrases, beat_grid)
        return phrases, beat_grid


#Main runtime controller for MIDI, Mixxx and transition execution
class AutoTransition:
    def __init__(
        self,
        top_k: int = 5,
        windows: Tuple[int, ...] = (32, 48, 64, 96, 128),
        blend: Tuple[float, float] = (0.2, 0.8),
        dedup_tol: int = 4,
        midi_in_port: str = IN_PORT_NAME,
        midi_out_port: str = OUT_PORT_NAME,
        websocket_port: int = WEBSOCKET_PORT,
        websocket_token: str = WEBSOCKET_TOKEN,
        model_path: str = MODEL_PATH,
        usb_root: str = "/media/jacob/711E-7C86",
        track_poll_interval: float = 0.25,
        lead_margin_beats: int = 4,
        max_future_beats: Optional[int] = None,
        virtual_midi: bool = True,
        wait_for_mixxx_sysex: float = 15.0,
        jit: bool = False,
    ) -> None:
        #Store main runtime config
        self.top_k = top_k
        self.windows = tuple(windows)
        self.blend = tuple(blend)
        self.dedup_tol = dedup_tol
        self.websocket_port = websocket_port
        self.websocket_token = websocket_token
        self.model_path = model_path
        self.track_poll_interval = track_poll_interval
        self.lead_margin_beats = lead_margin_beats
        self.max_future_beats = max_future_beats
        self.virtual_midi = virtual_midi
        self.usb_root = usb_root
        self.wait_for_mixxx_sysex = wait_for_mixxx_sysex
        self.in_port_name = midi_in_port
        self.out_port_name = midi_out_port
        self.jit = jit
        self.preload_state = None
        self.analysis_cache = {}
        
        #Load analysis cache if available
        self.cache_file = "analysis_cache.pkl"
        self.metadata_cache_file = "metadata_cache.pkl"
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.analysis_cache = pickle.load(f)
                print(f"[AutoTransition] Loaded cached analysis ({len(self.analysis_cache)} tracks)")
            except Exception as e:
                print(f"[AutoTransition] Failed to load cache: {e}")

        #Load metadata cache if available
        self.metadata = RekordboxMetadataResolver(usb_root)
        if os.path.exists(self.metadata_cache_file):
            try:
                with open(self.metadata_cache_file, "rb") as f:
                    self.metadata._metadata_cache = pickle.load(f)
                print(f"[AutoTransition] Loaded metadata cache ({len(self.metadata._metadata_cache)} tracks)")
            except Exception as e:
                print(f"[AutoTransition] Failed to load metadata cache: {e}")

        #Initialize transition state
        self.current_pair: Optional[Tuple[str, str]] = None
        self.transition_active = False
        self.transition_thread: Optional[threading.Thread] = None
        self.block_reprediction_until_track_change = False

        #Initialize states of each deck
        self.deck_state = {
            1: {"play_pos_msb": 0, "play_pos_lsb": 0, "play_pos_norm": 0.0, "sync_mode": 0, "is_playing": False, "bpm": 0, "tempo_raw": 0},
            2: {"play_pos_msb": 0, "play_pos_lsb": 0, "play_pos_norm": 0.0, "sync_mode": 0, "is_playing": False, "bpm": 0, "tempo_raw": 0},
        }
        self.loaded_tracks: Dict[int, Optional[str]] = {1: None, 2: None}
        self.last_track_poll = 0.0
        self.mixxx_connected = False
        self.last_sysex_time = 0.0
        
        #Initialize controller IO objects and threads
        self.inport = None
        self.outport = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread = None

    #Start MIDI input/output and polling thread
    def _start_controller_io(self) -> None:
        if self.inport is not None and self.outport is not None:
            return

        if not self.virtual_midi:
            self._ensure_existing_ports(self.in_port_name, self.out_port_name)

        self.inport, self.outport = self._open_midi_ports(
            self.in_port_name,
            self.out_port_name,
            self.virtual_midi,
        )

        if self._poll_thread is None or not self._poll_thread.is_alive():
            self._stop_event.clear()
            self._poll_thread = threading.Thread(target=self._poll_midi_forever, daemon=True)
            self._poll_thread.start()

    #Combine 7-bit MIDI MSB and LSB values -> 14-bit integer
    @staticmethod
    def _decode_14bit(msb: int, lsb: int) -> int:
        return ((msb & 0x7F) << 7) | (lsb & 0x7F)

    #Resolve a MIDI port name by matching fragments
    @staticmethod
    def _resolve_port_name(name_fragment: str, ports: List[str]) -> Optional[str]:
        needle = name_fragment.lower()
        matches = [p for p in ports if needle in p.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"[AutoTransition] Multiple MIDI port matches for '{name_fragment}':")
            for m in matches:
                print(f"  {m}")
        return None

    #Ensure required MIDI ports exist
    @classmethod
    def _ensure_existing_ports(cls, midi_in_port: str, midi_out_port: str) -> None:
        input_names = mido.get_input_names()
        output_names = mido.get_output_names()
        in_match = cls._resolve_port_name(midi_in_port, input_names)
        out_match = cls._resolve_port_name(midi_out_port, output_names)
        if in_match and out_match:
            print("[AutoTransition] Found existing MIDI ports:")
            print(f"  IN : {in_match}")
            print(f"  OUT: {out_match}")
            return
        raise SystemExit(
            "Required MIDI ports were not found.\n"
            f"Requested input fragment: {midi_in_port}\n"
            f"Requested output fragment: {midi_out_port}\n\n"
            "Detected input names:\n"
            + "\n".join(f"  {n}" for n in input_names)
            + "\n\nDetected output names:\n"
            + "\n".join(f"  {n}" for n in output_names)
        )

    #Open either virtual or existing MIDI ports
    @classmethod
    def _open_midi_ports(cls, midi_in_port: str, midi_out_port: str, virtual_midi: bool):
        if virtual_midi:
            try:
                inport = mido.open_input(midi_in_port, virtual=True)
                outport = mido.open_output(midi_out_port, virtual=True)
                print(f"[AutoTransition] Created virtual MIDI ports: IN='{midi_in_port}' OUT='{midi_out_port}'")
                print("[AutoTransition] Start this script first, then open Mixxx and select the DJ-AI Controller mapping")
                return inport, outport
            except Exception as e:
                print(f"[AutoTransition] Virtual MIDI failed ({e}); trying existing ports")

        input_names = mido.get_input_names()
        output_names = mido.get_output_names()
        in_name = midi_in_port if midi_in_port in input_names else cls._resolve_port_name(midi_in_port, input_names)
        out_name = midi_out_port if midi_out_port in output_names else cls._resolve_port_name(midi_out_port, output_names)

        if not in_name or not out_name:
            raise RuntimeError(
                "Could not resolve MIDI ports.\n"
                f"Requested IN fragment: {midi_in_port}\n"
                f"Requested OUT fragment: {midi_out_port}"
            )

        print("[AutoTransition] Using MIDI ports:")
        print(f"  IN : {in_name}")
        print(f"  OUT: {out_name}")
        return mido.open_input(in_name), mido.open_output(out_name)

    #Build a signature so cache invalidation can detect library changes
    def _get_library_signature(self, track_paths: List[str]) -> Tuple:
        export_stats = []
        for candidate in (self.metadata.db.export_pdb, self.metadata.db.export_ext_pdb):
            try:
                stat = candidate.stat()
                export_stats.append((candidate.name, int(stat.st_mtime_ns), int(stat.st_size)))
            except Exception:
                export_stats.append((candidate.name, None, None))

        track_stats = []
        for path in sorted(track_paths):
            try:
                st = os.stat(path)
                track_stats.append((path, int(st.st_mtime_ns), int(st.st_size)))
            except FileNotFoundError:
                track_stats.append((path, None, None))
        return (tuple(export_stats), tuple(track_stats))

    #Preload and cache metadata + audio features before runtime starts
    def _ensure_preloaded_analysis(self) -> None:
        if self.jit:
            print('[AutoTransition] JIT enabled; skipping preload before Mixxx connection.')
            return

        track_paths = self.metadata.get_all_track_paths(existing_only=True)
        signature = self._get_library_signature(track_paths)
        if self.preload_state == signature:
            print('[AutoTransition] Rekordbox library unchanged; using existing metadata/audio cache.')
            return

        if self.preload_state is not None:
            print('[AutoTransition] Rekordbox library change detected; updating cache incrementally...')
        else:
            print('[AutoTransition] Preloading Rekordbox metadata/audio analysis before Mixxx connection...')

        #Identify tracks still missing metadata, existing tracks are assumed not to change
        missing_metadata = [
            path for path in track_paths
            if self.metadata._norm_path(path) not in self.metadata._metadata_cache
        ]

        if missing_metadata:
            print(f"[AutoTransition] Loading metadata for {len(missing_metadata)} new tracks...")
            phrases, beat_grids = self.metadata.preload_all_metadata(
                track_paths=missing_metadata,
                show_progress=True
            )
        else:
            print("[AutoTransition] Metadata cache already up to date.")
            phrases = {}
            beat_grids = {}

        #Build a beatgrid map for all cached tracks
        full_beat_grids = {}

        for path in track_paths:
            norm = self.metadata._norm_path(path)
            if norm in self.metadata._metadata_cache:
                _, bg = self.metadata._metadata_cache[norm]
                full_beat_grids[os.path.basename(path)] = bg

        #Identify tracks still missing audio analysis
        missing_audio = [
            path for path in track_paths
            if path not in self.analysis_cache
        ]

        #Precompute audio analysis for remaining tracks
        suggester.preload_analysis(
            missing_audio,
            full_beat_grids,
            cache=self.analysis_cache,
            show_progress=True
        )
        
        #Persist updated caches to disk
        self.preload_state = signature
        print(f'[AutoTransition] Preload complete: metadata={len(beat_grids)} audio={len(self.analysis_cache)}')
        
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.analysis_cache, f)
        print(f"[AutoTransition] Saved analysis cache ({len(self.analysis_cache)} tracks)")
        with open(self.metadata_cache_file, "wb") as f:
            pickle.dump(self.metadata._metadata_cache, f)
        print(f"[AutoTransition] Saved metadata cache ({len(self.metadata._metadata_cache)} tracks)")

    #Wait for Mixxx to attach the controller mapping via SysEx
    def wait_for_mixxx_attachment(self, timeout: Optional[float] = None) -> bool:
        if timeout is None:
            timeout = self.wait_for_mixxx_sysex
        if timeout <= 0:
            return True

        print(f"[AutoTransition] Waiting up to {timeout:.1f}s for Mixxx controller SysEx on '{self.in_port_name}'…")
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._process_incoming_midi()
            if self.mixxx_connected:
                print("[AutoTransition] Mixxx controller connection confirmed via SysEx")
                return True
            time.sleep(0.05)

        print("[AutoTransition] No Mixxx SysEx seen yet; continuing anyway")
        return False

    #Send a note message with a short note-off after it
    def _send_midi_note(self, channel: int, note: int) -> None:
        on = mido.Message("note_on", channel=channel, note=note, velocity=127)
        off = mido.Message("note_on", channel=channel, note=note, velocity=0)
        self.outport.send(on)
        time.sleep(0.01)
        self.outport.send(off)

    #Send a clamped MIDI CC message
    def _send_midi_cc(self, channel: int, control: int, value: int) -> None:
        value = max(0, min(127, int(value)))
        self.outport.send(mido.Message("control_change", channel=channel, control=control, value=value))

    #Trigger play / pause on a deck
    def send_play_pause(self, deck: int) -> None:
        self._send_midi_note(DECK1_CHANNEL if deck == 1 else DECK2_CHANNEL, DECK_PLAY_PAUSE_NOTE)

    #Trigger cue on a deck
    def send_cue(self, deck: int) -> None:
        self._send_midi_note(DECK1_CHANNEL if deck == 1 else DECK2_CHANNEL, DECK_CUE_NOTE)

    #Update deck sync mode through controller input
    def update_sync_mode(self, deck: int, mode: int) -> None:
        ch = DECK1_CHANNEL if deck == 1 else DECK2_CHANNEL
        self._send_midi_cc(ch, DECK_SYNC_MODE_INPUT_CC, int(mode))
        self.deck_state[deck]["sync_mode"] = int(mode)

    #Toggle quantize on a deck
    def set_quantize(self, deck: int, enabled: bool) -> None:
        ch = DECK1_CHANNEL if deck == 1 else DECK2_CHANNEL
        self._send_midi_cc(ch, DECK_QUANTIZE_SET_CC, 127 if enabled else 0)

    #Set deck play position using 14-bit controller input
    def set_play_position_midi(self, deck: int, value: float) -> None:
        value = max(0.0, min(1.0, float(value)))
        scaled = math.floor(value * 16383.0)
        msb = (scaled >> 7) & 0x7F
        lsb = scaled & 0x7F
        ch = DECK1_CHANNEL if deck == 1 else DECK2_CHANNEL
        self._send_midi_cc(ch, DECK_PLAYPOS_INPUT_MSB_CC, msb)
        self._send_midi_cc(ch, DECK_PLAYPOS_INPUT_LSB_CC, lsb)

    #Handle incoming controller-attach SysEx messages
    def _handle_sysex(self, msg) -> None:
        data = tuple(getattr(msg, "data", ()))
        if data == EXPECTED_CONNECTION_SYSEX:
            with self._lock:
                self.mixxx_connected = True
                self.last_sysex_time = time.time()

    #Read all pending MIDI telemetry and update runtime state
    def _process_incoming_midi(self) -> None:
        if self.inport is None:
            return
        for msg in self.inport.iter_pending():
            if msg.type == "sysex":
                self._handle_sysex(msg)
                continue

            channel = getattr(msg, "channel", None)
            if channel is None:
                continue

            with self._lock:
                if msg.type == "control_change":
                    ctrl = int(msg.control)
                    val = int(msg.value)

                    #Update deck telemetry from control change messages
                    if channel in (DECK1_CHANNEL, DECK2_CHANNEL):
                        deck = 1 if channel == DECK1_CHANNEL else 2

                        if ctrl == DECK1_PLAYPOS_MSB_CC:
                            self.deck_state[deck]["play_pos_msb"] = val
                        elif ctrl == DECK1_PLAYPOS_LSB_CC:
                            self.deck_state[deck]["play_pos_lsb"] = val
                        elif deck == 1 and ctrl == DECK1_BEAT_SYNC_CC:
                            self.deck_state[1]["sync_mode"] = val
                        elif deck == 2 and ctrl == DECK2_BEAT_SYNC_CC:
                            self.deck_state[2]["sync_mode"] = val
                        elif deck == 1 and ctrl in (DECK1_LIVEBPM_MSB_CC, DECK1_LIVEBPM_LSB_CC):
                            if ctrl == DECK1_LIVEBPM_MSB_CC:
                                self.deck_state[1]["bpm_msb"] = val
                            else:
                                self.deck_state[1]["bpm_lsb"] = val
                            self.deck_state[1]["bpm"] = self._decode_14bit(self.deck_state[1].get("bpm_msb", 0), self.deck_state[1].get("bpm_lsb", 0))
                        elif deck == 2 and ctrl in (DECK2_LIVEBPM_MSB_CC, DECK2_LIVEBPM_LSB_CC):
                            if ctrl == DECK2_LIVEBPM_MSB_CC:
                                self.deck_state[2]["bpm_msb"] = val
                            else:
                                self.deck_state[2]["bpm_lsb"] = val
                            self.deck_state[2]["bpm"] = self._decode_14bit(self.deck_state[2].get("bpm_msb", 0), self.deck_state[2].get("bpm_lsb", 0))
                        elif deck == 1 and ctrl in (DECK1_TEMPORATE_MSB_CC, DECK1_TEMPORATE_LSB_CC):
                            if ctrl == DECK1_TEMPORATE_MSB_CC:
                                self.deck_state[1]["tempo_msb"] = val
                            else:
                                self.deck_state[1]["tempo_lsb"] = val
                            self.deck_state[1]["tempo_raw"] = self._decode_14bit(self.deck_state[1].get("tempo_msb", 0), self.deck_state[1].get("tempo_lsb", 0))
                        elif deck == 2 and ctrl in (DECK2_TEMPORATE_MSB_CC, DECK2_TEMPORATE_LSB_CC):
                            if ctrl == DECK2_TEMPORATE_MSB_CC:
                                self.deck_state[2]["tempo_msb"] = val
                            else:
                                self.deck_state[2]["tempo_lsb"] = val
                            self.deck_state[2]["tempo_raw"] = self._decode_14bit(self.deck_state[2].get("tempo_msb", 0), self.deck_state[2].get("tempo_lsb", 0))

                        #Rebuild normalized play position after telemetry update
                        msb = int(self.deck_state[deck]["play_pos_msb"])
                        lsb = int(self.deck_state[deck]["play_pos_lsb"])
                        self.deck_state[deck]["play_pos_norm"] = self._decode_14bit(msb, lsb) / 16383.0

                #Update play state from note messages
                elif msg.type == "note_on":
                    if channel in (DECK1_CHANNEL, DECK2_CHANNEL):
                        deck = 1 if channel == DECK1_CHANNEL else 2
                        if msg.note in DECK_PLAY_STATE_NOTES:
                            self.deck_state[deck]["is_playing"] = msg.velocity > 0

    #Background loop for MIDI polling
    def _poll_midi_forever(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._process_incoming_midi()
            except Exception as e:
                print(f"[AutoTransition] MIDI polling error: {e}")
            time.sleep(BACKGROUND_POLL_SLEEP)

    #Send one websocket command to Mixxx and return decoded response
    def _ws_command(self, payload: Dict) -> Dict:
        url = f"ws://localhost:{self.websocket_port}"
        with connect(url) as ws:
            ws.send(json.dumps(payload))
            msg = ws.recv()
            try:
                return json.loads(msg)
            except json.JSONDecodeError:
                return {"raw": msg}

    #Set a cue point on a deck through websocket control
    def set_deck_cue(self, deck: int, beat_num: int) -> None:
        resp = self._ws_command({"token": self.websocket_token, "command": "setDeckCue", "deck": deck, "beatNum": int(beat_num)})
        print(f"[AutoTransition] setDeckCue response: {resp}")

    #Prime incoming deck position before the transition starts
    def prime_incoming_position_then_cue(self, deck: int, beat_b: int, total_beats_incoming: int) -> None:
        total_beats_incoming = max(1, int(total_beats_incoming))
        norm = max(0.0, min(1.0, float(beat_b) / float(total_beats_incoming)))
        print(f"[AutoTransition] Now setting websocket cue for deck {deck} to beat {beat_b}")
        self.set_deck_cue(deck, beat_b)
        time.sleep(0.1)

        print(
            f"[AutoTransition] Priming deck {deck}: set normalized play position to {norm:.6f} "
            f"for beatB={beat_b}/{total_beats_incoming}"
        )
        self.set_play_position_midi(deck, norm)
        time.sleep(0.1)

        #Just in case position is still off and quantize might jump to next beat, pressing cue again moves it to the last full one which will be valid.
        self.send_cue(deck);
        time.sleep(0.1)
        
    #Move an idle deck back to the start and set a fresh cue
    def prepare_upcoming_deck(self, deck: int) -> None:
        print(f"[AutoTransition] Preparing deck {deck}: seek to start and set cue at beat 0")
        self.set_play_position_midi(deck, 0.1)
        time.sleep(0.05)
        self.set_play_position_midi(deck, 0.0)
        time.sleep(0.05)
        self.set_deck_cue(deck, 0)
        time.sleep(0.05)

    #Query Mixxx for the track loaded on a deck
    def get_loaded_track(self, deck: int) -> Optional[str]:
        resp = self._ws_command({"token": self.websocket_token, "command": "getLoadedTrack", "deck": deck})
        path = resp.get("path") or resp.get("location")
        return str(path) if path else None

    poll_loaded_tracks_error = False #don't want errors to loop on a Mixxx disconnect

    #Poll both deck slots for track changes
    def poll_loaded_tracks(self, force: bool = False) -> bool:
        now = time.time()
        if not force and (now - self.last_track_poll) < self.track_poll_interval:
            return False
        self.last_track_poll = now
        changed = False

        for deck in (1, 2):
            try:
                new_track = self.get_loaded_track(deck)
                self.poll_loaded_tracks_error = False
            except Exception as e:
                if not self.poll_loaded_tracks_error:
                    print(f"[AutoTransition] getLoadedTrack failed for deck {deck}: {e}")
                new_track = None
                self.poll_loaded_tracks_error = True

            #Update transition state if deck content changed
            if new_track != self.loaded_tracks[deck]:
                old_name = os.path.basename(self.loaded_tracks[deck]) if self.loaded_tracks[deck] else None
                new_name = os.path.basename(new_track) if new_track else None
                print(f"[AutoTransition] Deck {deck} track changed: {old_name} -> {new_name}")
                self.loaded_tracks[deck] = new_track
                self.current_pair = None
                self.block_reprediction_until_track_change = False
                changed = True

        return changed

    #Build transition suggestions for the current deck pair
    def _suggest_candidates(self, track_a: str, track_b: str):
        try:
            phrases_a, beat_grid_a = self.metadata.get_metadata_for_track(track_a)
            phrases_b, beat_grid_b = self.metadata.get_metadata_for_track(track_b)

            phrases = {os.path.basename(track_a): phrases_a, os.path.basename(track_b): phrases_b}
            beat_grids = {os.path.basename(track_a): beat_grid_a, os.path.basename(track_b): beat_grid_b}

            tops, candidates = suggester.suggest(
                track_a,
                track_b,
                phrases,
                beat_grids,
                windows=self.windows,
                top_k=self.top_k,
                model_path=self.model_path,
                blend=self.blend,
                dedup_tol=self.dedup_tol,
                preloaded_cache=self.analysis_cache,
            )
            return tops, candidates
        except Exception as e:
            print(f"[AutoTransition] Local suggestion failed for {os.path.basename(track_a)} -> {os.path.basename(track_b)}: {e}")
            return [[], [], []], []

    #Get max reachable beat from returned candidate windows
    def _get_total_beats(self, tops: List[List[Dict]]) -> int:
        max_seen = 0
        for bucket in tops:
            for cand in bucket:
                max_seen = max(max_seen, int(cand.get("beatA", 0)) + int(cand.get("windowBeats", 0)))
        return max_seen

    #Resolve usable beat count for a track from its beat grid
    def _get_track_total_beats(self, track_path: str) -> int:
        try:
            _phrases, beat_grid = self.metadata.get_metadata_for_track(track_path)
        except Exception as e:
            print(f"[AutoTransition] Failed to resolve total beats for {os.path.basename(track_path)}: {e}")
            return 1

        if isinstance(beat_grid, list) and beat_grid:
            final = beat_grid[-1]
            if isinstance(final, (list, tuple)):
                return max(1, len(final))
        return max(1, len(beat_grid))

    #Choose one currently valid candidate based on deck position and bucket weighting
    def _select_runtime_candidate(self, playing_deck: int, tops: List[List[Dict]]) -> Optional[Dict]:
        with self._lock:
            current_pos_norm = float(self.deck_state[playing_deck]["play_pos_norm"])

        #Bucket preference weights
        preferred_bucket_order = [0, 1, 2]
        preferred_bucket_weights = {0: 0.70, 1: 0.25, 2: 0.05}
        min_score = 0.70

        available_bucket_indices: List[int] = []
        available_weights: List[float] = []
        bucket_candidates: Dict[int, List[Dict]] = {}

        #Filter candidates that are still valid at the current playhead
        for bucket_idx in preferred_bucket_order:
            if bucket_idx >= len(tops):
                continue

            valid_bucket: List[Dict] = []
            for cand in tops[bucket_idx][: self.top_k]:
                cand_rel = float(cand.get("relPosA", 0.0))
                if cand_rel < current_pos_norm:
                    continue

                if float(cand.get("score", 0.0)) <= min_score:
                    continue

                #Prevent "safe outro bias"
                if cand_rel >= 0.89:
                    continue

                valid_bucket.append(cand)

            if valid_bucket:
                bucket_candidates[bucket_idx] = valid_bucket
                available_bucket_indices.append(bucket_idx)
                available_weights.append(preferred_bucket_weights[bucket_idx])

        if not available_bucket_indices:
            return None

        #Randomly pick a bucket, then a candidate within it
        selected_bucket_idx = random.choices(available_bucket_indices, weights=available_weights, k=1)[0]
        return random.choice(bucket_candidates[selected_bucket_idx])

    #Wait until a deck reaches a target beat position
    def _wait_until_target_beat(
        self,
        deck: int,
        target_beat: float,
        total_beats_hint: int,
        timeout: float = 250.0,
        verbose: bool = True,
        trigger_offset_beats: float = DEFAULT_TRIGGER_OFFSET_BEATS,
    ) -> bool:
        start_time = time.perf_counter()
        total_beats_hint = max(1, total_beats_hint)
        adjusted_target = max(0.0, float(target_beat) - float(trigger_offset_beats))
        target_norm = adjusted_target / float(total_beats_hint)
        last_print = 0.0

        #Keep polling deck position until target or timeout
        while time.perf_counter() - start_time < timeout:
            with self._lock:
                pos = float(self.deck_state[deck]["play_pos_norm"])
            now = time.perf_counter()
            if verbose and now - last_print >= VERBOSE_PRINT_EVERY:
                print(
                    f"[AutoTransition] Deck {deck} pos={pos:.5f} "
                    f"target={target_norm:.5f} (beat={target_beat}, offset={trigger_offset_beats})"
                )
                last_print = now
            if pos >= target_norm:
                print(
                    f"[AutoTransition] Deck {deck} reached target beat {target_beat} "
                    f"(adjusted target norm {target_norm:.5f})"
                )
                return True
            time.sleep(WAIT_SLEEP)

        print(f"[AutoTransition] Timeout waiting for deck {deck} to reach beat {target_beat}")
        return False

    #Execute one full transition from outgoing to incoming deck
    def _execute_transition(
        self,
        playing_deck: int,
        incoming_deck: int,
        cand: Dict,
        total_beats_playing: int,
        total_beats_incoming: int,
        trigger_offset_beats: float = DEFAULT_TRIGGER_OFFSET_BEATS,
    ) -> None:
        beat_a = int(cand["beatA"])
        beat_b = int(cand["beatB"])
        window_beats = int(cand["windowBeats"])

        print(
            f"[AutoTransition] Attempting transition: playing={playing_deck} incoming={incoming_deck} "
            f"beatA={beat_a} beatB={beat_b} window={window_beats} "
            f"totalPlaying={total_beats_playing} totalIncoming={total_beats_incoming} "
            f"primeThenCue=True triggerOffset={trigger_offset_beats}"
        )

        try:
            #Do quantization on decks first so we don't jump to inbetween beats.
            print(f"[AutoTransition] Enabling quantize on decks {playing_deck} and {incoming_deck}")
            self.set_quantize(playing_deck, True)
            self.set_quantize(incoming_deck, True)

            #Prime incoming deck before playback begins
            self.prime_incoming_position_then_cue(incoming_deck, beat_b, total_beats_incoming)

            print(f"[AutoTransition] Sending cue to deck {incoming_deck}")
            self.send_cue(incoming_deck)

            #Set outgoing as leader and incoming as follower
            print(f"[AutoTransition] Setting sync: deck {playing_deck}=leader(2), deck {incoming_deck}=follower(1)")
            self.update_sync_mode(playing_deck, 2)
            self.update_sync_mode(incoming_deck, 1)

            #Wait until outgoing deck reaches the chosen start beat
            print(f"[AutoTransition] Waiting for deck {playing_deck} to reach beatA={beat_a}")
            if not self._wait_until_target_beat(
                playing_deck,
                beat_a,
                total_beats_playing,
                timeout=250.0,
                verbose=True,
                trigger_offset_beats=trigger_offset_beats,
            ):
                print("[AutoTransition] Aborting transition because beatA was not reached")
                return

            #Start incoming deck at the prepared cue
            print(f"[AutoTransition] Starting incoming deck {incoming_deck} at beatB={beat_b}")
            self.send_play_pause(incoming_deck)

            #Let both decks overlap for the chosen window
            print(f"[AutoTransition] Letting tracks overlap naturally for window={window_beats} beats (no crossfader automation)")
            end_beat = beat_a + window_beats
            print(f"[AutoTransition] Waiting for deck {playing_deck} to reach end beat {end_beat}")
            if not self._wait_until_target_beat(
                playing_deck,
                end_beat,
                total_beats_playing,
                timeout=250.0,
                verbose=True,
                trigger_offset_beats=0.0,
            ):
                print("[AutoTransition] Aborting transition before stop because end beat was not reached")
                return

            #Stop outgoing deck once overlap window is done
            print(f"[AutoTransition] Stopping outgoing deck {playing_deck}")
            self.send_play_pause(playing_deck)
            print("[AutoTransition] Transition attempt complete")
        finally:
            None

    #Wrapper so transition thread can always reset runtime flags
    def _transition_wrapper(
        self,
        playing_deck: int,
        paused_deck: int,
        cand: Dict,
        total_beats_playing: int,
        total_beats_incoming: int,
        trigger_offset_beats: float = DEFAULT_TRIGGER_OFFSET_BEATS,
    ) -> None:
        try:
            self._execute_transition(
                playing_deck,
                paused_deck,
                cand,
                total_beats_playing,
                total_beats_incoming,
                trigger_offset_beats=trigger_offset_beats,
            )
        finally:
            self.transition_active = False
            self.block_reprediction_until_track_change = True

    #Main runtime loop for polling, suggestion and transition launching
    def run(self) -> None:
        print("[AutoTransition] Starting main loop… press Ctrl+C to exit.")
        self._ensure_preloaded_analysis()
        self._start_controller_io()
        self.wait_for_mixxx_attachment()
        self.poll_loaded_tracks(force=True)

        try:
            while True:
                self._process_incoming_midi()
                self.poll_loaded_tracks()

                if self.transition_active:
                    time.sleep(0.01)
                    continue

                playing_deck = None
                paused_deck = None
                multiple_playing = False

                #Determine which deck is currently active
                for d in (1, 2):
                    if self.deck_state[d]["is_playing"]:
                        if playing_deck is None:
                            playing_deck = d
                        else:
                            multiple_playing = True
                            break
                    else:
                        paused_deck = d

                if multiple_playing:
                    time.sleep(0.01)
                    continue

                if self.block_reprediction_until_track_change:
                    time.sleep(0.01)
                    continue

                #Predict and launch transition when one deck is playing and the other is ready
                if playing_deck and paused_deck and self.loaded_tracks[1] and self.loaded_tracks[2]:
                    pair = (self.loaded_tracks[playing_deck], self.loaded_tracks[paused_deck])
                    if pair != self.current_pair:
                        print(f"[AutoTransition] New track pair detected: {os.path.basename(pair[0])} -> {os.path.basename(pair[1])}")
                        self.current_pair = pair
                        self.prepare_upcoming_deck(paused_deck)
                        tops, _all_cands = self._suggest_candidates(pair[0], pair[1])

                        pool = [c for bucket in tops for c in bucket]
                        if not pool:
                            print(f"[AutoTransition] No candidate transitions found for {os.path.basename(pair[0])} -> {os.path.basename(pair[1])}")
                            time.sleep(0.05)
                            continue

                        total_beats_playing = self._get_track_total_beats(pair[0])
                        total_beats_incoming = self._get_track_total_beats(pair[1])
                        cand = self._select_runtime_candidate(playing_deck=playing_deck, tops=tops)
                        if cand is None:
                            print(f"[AutoTransition] Suggestions exist but none are still viable at the current play position for {os.path.basename(pair[0])}")
                            time.sleep(0.05)
                            continue

                        print(
                            f"[AutoTransition] Selected window: beatA={cand['beatA']}, beatB={cand['beatB']}, "
                            f"window={cand['windowBeats']}, score={cand.get('score', cand.get('prob', 0))}, "
                            f"totalPlaying={total_beats_playing}, totalIncoming={total_beats_incoming}"
                        )
                        self.transition_active = True
                        self.transition_thread = threading.Thread(
                            target=self._transition_wrapper,
                            args=(playing_deck, paused_deck, cand, total_beats_playing, total_beats_incoming, DEFAULT_TRIGGER_OFFSET_BEATS),
                            daemon=True,
                        )
                        self.transition_thread.start()

                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n[AutoTransition] Exiting…")
        finally:
            self._stop_event.set()
            self._poll_thread.join(timeout=1.0)
            try:
                self.inport.close()
            finally:
                self.outport.close()


#CLI parsing helpers:

def parse_csv_ints(s: str) -> Tuple[int, ...]: #get row of integers from argument
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def parse_csv_floats(s: str) -> Tuple[float, ...]: #get row of floats from argument
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def parse_optional_int(s: str) -> Optional[int]: #get row of optional integers from argument
    s = s.strip().lower()
    if s in {"", "none", "null"}:
        return None
    return int(s)


#Standalone entrypoint for running the controller demo
def main() -> None:
    ap = argparse.ArgumentParser(description="Automated DJ transition demo for Mixxx (Linux virtual controller version)")
    ap.add_argument("--topk", type=int, default=3, help="Top K suggestions per bucket")
    ap.add_argument("--windows", default="32,48,64,96,128", help="Candidate windows in beats")
    ap.add_argument("--blend", default="0.2,0.8", help="Heuristic/RF blend weights")
    ap.add_argument("--dedup-tol", type=int, default=4, help="Dedup tolerance in beats")
    ap.add_argument("--inport", default=IN_PORT_NAME, help="MIDI input port name")
    ap.add_argument("--outport", default=OUT_PORT_NAME, help="MIDI output port name")
    ap.add_argument("--wsport", type=int, default=WEBSOCKET_PORT, help="Mixxx websocket port")
    ap.add_argument("--wstoken", default=WEBSOCKET_TOKEN, help="Mixxx websocket token")
    ap.add_argument("--model", default=MODEL_PATH, help="Path to trained RF model")
    ap.add_argument("--usb-root", default="/media/jacob/711E-7C86", help="Mounted Rekordbox USB root")
    ap.add_argument("--track-poll-interval", type=float, default=0.25, help="Seconds between getLoadedTrack polls")
    ap.add_argument("--lead-margin-beats", type=int, default=4, help="Minimum beats ahead required for a candidate to still be viable")
    ap.add_argument("--max-future-beats", default="none", help="Optional max beats ahead for viable candidates")
    ap.add_argument("--virtual-midi", action="store_true", default=True, help="Use mido-created virtual MIDI ports")
    ap.add_argument("--no-virtual-midi", dest="virtual_midi", action="store_false", help="Use existing MIDI ports instead")
    ap.add_argument("--mixxx-sysex-timeout", type=float, default=25.0, help="Seconds to wait for Mixxx keepalive SysEx before continuing")
    ap.add_argument("--jit", action="store_true", help="Disable default full-library preload and analyze on demand instead")
    args = ap.parse_args()

    #Build runtime controller from CLI args
    engine = AutoTransition(
        top_k=args.topk,
        windows=parse_csv_ints(args.windows),
        blend=parse_csv_floats(args.blend),
        dedup_tol=args.dedup_tol,
        midi_in_port=args.inport,
        midi_out_port=args.outport,
        websocket_port=args.wsport,
        websocket_token=args.wstoken,
        model_path=args.model,
        usb_root=args.usb_root,
        track_poll_interval=args.track_poll_interval,
        lead_margin_beats=args.lead_margin_beats,
        max_future_beats=parse_optional_int(args.max_future_beats),
        virtual_midi=args.virtual_midi,
        wait_for_mixxx_sysex=args.mixxx_sysex_timeout,
        jit=args.jit,
    )
    engine.run()


if __name__ == "__main__":
    main()