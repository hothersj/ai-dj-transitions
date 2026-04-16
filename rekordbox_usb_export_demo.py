from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Dict, Iterator, List, Optional, Union

#Regexes
_AUDIO_EXT_RE = r"(?:mp3|wav|flac|aiff|aif|m4a|aac|ogg|opus|alac)"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ANLZ_PATH_RE = re.compile(r"/PIONEER/USBANLZ/[^\x00]*?ANLZ0000\.DAT", re.IGNORECASE)
_CONTENT_PATH_RE = re.compile(rf"/Contents/.*?\.{_AUDIO_EXT_RE}", re.IGNORECASE)


#Each USB DB track has these relevant fields:
@dataclass(slots=True)
class USBTrack:
    ID: int
    Title: str
    FolderPath: str
    FileNameL: str #target 'canonical' name, can sometimes be truncated for filesystem compatibility with long paths
    AnalysisDataPath: str


#Class for the Rekordbox DB itself.
#This parser class extracts track paths, titles and analysis ANLZ files used for phrase data.
class RekordboxUSBExport:
    def __init__(self, usb_root: Union[str, os.PathLike[str]]) -> None:
        self.usb_root = Path(usb_root).expanduser().resolve()
        self.pioneer_dir = self.usb_root / "PIONEER"
        self.rekordbox_dir = self.pioneer_dir / "rekordbox"
        self.export_pdb = self.rekordbox_dir / "export.pdb"
        self.export_ext_pdb = self.rekordbox_dir / "exportExt.pdb"

        if not self.export_pdb.exists():
            raise FileNotFoundError(f"export.pdb not found at {self.export_pdb}")

        self._tracks: List[USBTrack] = self._parse_export_pdb()
        self._by_id: Dict[int, USBTrack] = {t.ID: t for t in self._tracks}
        self._by_folder: Dict[str, USBTrack] = {t.FolderPath: t for t in self._tracks}

    #EXPOSED LIB API CALLS:

    def get_content(self) -> List[USBTrack]:
        return list(self._tracks)

    def iter_content(self) -> Iterator[USBTrack]:
        yield from self._tracks

    def get_track(self, track_id: int) -> USBTrack:
        return self._by_id[track_id]

    def find_track_by_path(self, path: Union[str, os.PathLike[str]]) -> Optional[USBTrack]:
        p = Path(path)
        #1) Exact path match
        s = str(p)
        if s in self._by_folder:
            return self._by_folder[s]

        #2) Same relative USB path from root
        try:
            rel = "/" + str(p.resolve().relative_to(self.usb_root)).replace(chr(92), "/")
            full = str((self.usb_root / rel.lstrip("/")).resolve())
            if full in self._by_folder:
                return self._by_folder[full]
        except Exception:
            pass

        #3) Exact basename fallback
        base = p.name.lower()
        matches = [t for t in self._tracks if t.FileNameL.lower() == base]
        if len(matches) == 1:
            return matches[0]

        #4) Fallback to a substring in case of truncation in filesystem
        query_norm = re.sub(r"[^a-z0-9]", "", p.name.lower())
        substring_matches: List[USBTrack] = []
        for t in self._tracks:
            track_name = t.FileNameL or Path(t.FolderPath).name
            track_norm = re.sub(r"[^a-z0-9]", "", track_name.lower())
            if not track_norm or not query_norm:
                continue
            if query_norm in track_norm or track_norm in query_norm:
                substring_matches.append(t)

        #If we have one match, then can simply return it.
        if len(substring_matches) == 1:
            print(f"[USB DB] Unique substring match: {p.name} -> {substring_matches[0].FileNameL}")
            return substring_matches[0]

        #Otherwise, print ambiguous matches.
        if len(substring_matches) > 1:
            print(f"[USB DB] Ambiguous substring matches for {p.name}:")
            for t in substring_matches:
                print(f"  - {t.FileNameL} :: {t.FolderPath}")

        return None

    def find_tracks_by_basename(self, basename: str) -> List[USBTrack]:
        base = Path(basename).name.lower()
        return [t for t in self._tracks if t.FileNameL.lower() == base]

    def get_track_paths(self, existing_only: bool = False) -> List[str]:
        paths = [t.FolderPath for t in self._tracks]
        if existing_only:
            paths = [path for path in paths if Path(path).exists()]
        return paths

    def iter_track_paths(self, existing_only: bool = False) -> Iterator[str]:
        for path in self.get_track_paths(existing_only=existing_only):
            yield path

    def get_anlz_paths(self, track_or_id: Union[int, USBTrack]) -> Dict[str, Optional[str]]:
        track = self.get_track(track_or_id) if isinstance(track_or_id, int) else track_or_id
        dat_path = Path(track.AnalysisDataPath)
        base = dat_path.with_suffix("")

        candidates = {
            "DAT": dat_path,
            "EXT": base.with_suffix(".EXT"),
            "2EX": base.with_suffix(".2EX"),
        }
        return {
            key: str(path) if path.exists() else None
            for key, path in candidates.items()
        }

    #INTERNAL HELPERS:

    def _extract_ascii_strings(self, data: bytes, min_len: int = 4) -> List[tuple[int, str]]:
        strings: List[tuple[int, str]] = []
        current: List[str] = []
        start: Optional[int] = None

        for i, b in enumerate(data):
            if 32 <= b < 127:
                if start is None:
                    start = i
                current.append(chr(b))
            else:
                if start is not None and len(current) >= min_len:
                    strings.append((start, "".join(current)))
                start = None
                current = []

        if start is not None and len(current) >= min_len:
            strings.append((start, "".join(current)))

        return strings

    #Removes a prefix
    def _clean_path(self, raw: str, expected_prefix: str) -> str:
        idx = raw.find(expected_prefix)
        if idx == -1:
            return raw
        return raw[idx:]

    #Clean title for printing as often song filenames are verbose.
    def _clean_title(self, raw: str) -> str:
        #Drop leading punctuation or length-marker artifacts.
        cleaned = re.sub(r"^[^A-Za-z0-9]+", "", raw).strip()
        return cleaned

    #Where the magic happens, export.pdb is parsed here.
    def _parse_export_pdb(self) -> List[USBTrack]:
        #First extract indexes of ANLZ path strings
        data = self.export_pdb.read_bytes()
        strings = self._extract_ascii_strings(data)
        anlz_idxs = [i for i, (_, s) in enumerate(strings) if "/PIONEER/USBANLZ/" in s]

        tracks: List[USBTrack] = []
        seen_paths: set[str] = set()

        for pos, idx in enumerate(anlz_idxs):
            next_idx = anlz_idxs[pos + 1] if pos + 1 < len(anlz_idxs) else len(strings)
            chunk_parts = [s for _, s in strings[idx:next_idx]]
            chunk = " ".join(chunk_parts)

            #Try matching to expected path prefixes
            raw_anlz = strings[idx][1]
            anlz_rel_match = _ANLZ_PATH_RE.search(self._clean_path(raw_anlz, "/PIONEER/USBANLZ/"))
            if not anlz_rel_match:
                anlz_rel_match = _ANLZ_PATH_RE.search(chunk) #fallback to full chunk search
            if not anlz_rel_match:
                continue

            #Now we found a match, ensure cleaning of any extra predecessing and successing bytes
            anlz_rel = self._clean_path(anlz_rel_match.group(0), "/PIONEER/USBANLZ/")
            content_match = _CONTENT_PATH_RE.search(chunk)
            if not content_match:
                continue

            #Store results for ease
            content_rel = content_match.group(0)
            folder_path = str((self.usb_root / content_rel.lstrip("/")).resolve())
            analysis_path = str((self.usb_root / anlz_rel.lstrip("/")).resolve())
            basename = Path(content_rel).name

            if folder_path in seen_paths:
                continue
            seen_paths.add(folder_path)

            title = Path(content_rel).stem
            for _, s in strings[idx + 1:next_idx]:
                if _DATE_RE.fullmatch(s):
                    continue
                if "/PIONEER/USBANLZ/" in s or "/Contents/" in s:
                    continue
                cand = self._clean_title(s)
                if cand:
                    title = cand
                    break

            track = USBTrack(
                ID=len(tracks) + 1,
                Title=title,
                FolderPath=folder_path,
                FileNameL=basename,
                AnalysisDataPath=analysis_path,
            )
            tracks.append(track) #append retrieved track object to collection

        return tracks


def open_usb_export(usb_root: Union[str, os.PathLike[str]]) -> RekordboxUSBExport:
    return RekordboxUSBExport(usb_root)


#If ran as main, this library will just print retrieved DB contents.
if __name__ == "__main__":
    import argparse
    from pyrekordbox.anlz import AnlzFile

    ap = argparse.ArgumentParser(description="Inspect a Rekordbox USB export")
    ap.add_argument("usb_root", help="USB mount root containing PIONEER/")
    ap.add_argument("--limit", type=int, default=10, help="Number of tracks to print")
    args = ap.parse_args()

    db = RekordboxUSBExport(args.usb_root)
    tracks = db.get_content()
    print(f"Tracks found: {len(tracks)}")

    for track in tracks[: args.limit]:
        print("-" * 80)
        print("ID:", track.ID)
        print("Title:", track.Title)
        print("FileNameL:", track.FileNameL)
        print("FolderPath:", track.FolderPath)
        paths = db.get_anlz_paths(track)
        print("ANLZ:", paths)

        dat = paths.get("DAT")
        ext = paths.get("EXT")
        if dat:
            anlz_dat = AnlzFile.parse_file(dat)
            beat_grid = anlz_dat.get("beat_grid")
            print("DAT beat_grid:", "present" if beat_grid else "missing")
        if ext:
            anlz_ext = AnlzFile.parse_file(ext)
            pssi = anlz_ext.get("PSSI")
            print("EXT PSSI:", "present" if pssi else "missing")
