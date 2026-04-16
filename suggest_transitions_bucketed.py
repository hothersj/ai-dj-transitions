import os, json, argparse, joblib, sys, time
import numpy as np
import librosa
import pandas as pd

ANALYSIS_CACHE = {}

#Slightly dampen plain Outro->Intro so it doesn't dominate
SOFT_OUTRO_INTRO_PENALTY = -0.25
#Equalize Down→(next is Outro) to behave like Outro (so it can compete with Outro→Intro)
EQUALIZE_DOWN_AS_OUTRO = True
EQUALIZE_REQUIRE_B_INTRO = False #false means doesn't need intro
#Reward when B's phrase length roughly equals A's Down+Outro combined length
DOWN_OUTRO_MATCH_B_ENABLE = True
DOWN_OUTRO_MATCH_B_TOL   = 8 #beats tolerance for what counts as "equal length" in window matching
DOWN_OUTRO_MATCH_B_BONUS = 0.35 #bonus score if |lenB - (lenDown+lenOutro)| <= TOL
#Tolerance if chosen window is approximately the combined length
DOWN_OUTRO_MATCH_WINDOW_TOL   = 8
DOWN_OUTRO_MATCH_WINDOW_BONUS = 0.15 #bonus score for down outro match case
DOWN_OUTRO_MATCH_REQUIRE_B_IN = {"Intro", "Up"} #Only apply when B is Intro/Up (typical landing phrases). Set to None to allow any.
#Use combined A envelope when next phrase is Outro
ALLOW_A_COMBINED_OUTRO = True
#B may only span Intro and/or Up, in order, within the window
ENFORCE_B_INTRO_UP_ONLY = True
#Apply B-end-on-phrase-boundary rule only for Down->Outro
ALIGN_B_END_GLOBAL = False #False = disable global usage of the rule
ALIGN_B_END_FOR_DOWN_OUTRO = True #also allow when A is Down and next is Outro
B_END_ALIGN_TOL = 0 #amount of tolerance for align in beats
#Allowed combinations for Chorus->B
A_COMBO_OUTRO_LABELS = {("Chorus","Outro"), ("Up","Outro")}
A_OUTRO_TAIL_TOL = 4 #tolerance in beats for the tail of outro
#Enable combo logic for A when the current phrase is shorter than the window
ENABLE_A_COMBINED_IF_SHORT = True

#Heuristic weights by phrase:
LABEL_SCORE_A = {"Outro": 3.0, "Down": 2.8, "Up": 1.75, "Chorus": 1.25, "Intro": 0.5, "Bridge": 1.75}
LABEL_SCORE_B = {"Intro": 3.0, "Up": 2.8, "Down": 1.5, "Chorus": 1.25, "Outro": 0.2, "Bridge": 1.25}
WINDOW_WEIGHT = {32: 0.8, 48: 0.5, 64: 1.0, 96: 0.5, 128: 0.7} #weights for window sizes in beats

#Audio feature params these match the settings used for generating training data:
HOP_LENGTH   = 512
MFCC_COEFFS  = 13
ROLLOFF_PCT  = 0.85

#Allowed heuristic range:
HEURISTIC_MIN = 0.0
HEURISTIC_MAX = 10.0

#Hardcoded expected columns to match trained RF.
EXPECTED_COLUMNS = [
    #Spectral features from A
    "A_centroid_0", "A_bandwidth_0", "A_rolloff_0", "A_rms_0", "A_zcr_0",
    "A_mfcc_0", "A_mfcc_1", "A_mfcc_2", "A_mfcc_3", "A_mfcc_4",
    "A_mfcc_5", "A_mfcc_6", "A_mfcc_7", "A_mfcc_8", "A_mfcc_9",
    "A_mfcc_10", "A_mfcc_11", "A_mfcc_12",

    #Spectral features from B
    "B_centroid_0", "B_bandwidth_0", "B_rolloff_0", "B_rms_0", "B_zcr_0",
    "B_mfcc_0", "B_mfcc_1", "B_mfcc_2", "B_mfcc_3", "B_mfcc_4",
    "B_mfcc_5", "B_mfcc_6", "B_mfcc_7", "B_mfcc_8", "B_mfcc_9",
    "B_mfcc_10", "B_mfcc_11", "B_mfcc_12",

    #Meta
    "beatA", "beatB", "windowBeats",
    #Extra
    "totalBeatsA","totalBeatsB","relPosA","relPosB",
    "beatsToEndA","beatsToEndB","relBeatsToEndA","relBeatsToEndB",
    
    #All other A and B features
    "A_edge_onset_mean_start", "A_edge_onset_peak_start", "A_edge_onset_slope_start", "A_edge_onset_mean_end", "A_edge_onset_peak_end", "A_edge_onset_slope_end", "A_edge_flux_mean_start", "A_edge_flux_peak_start",
    "A_edge_flux_mean_end", "A_edge_flux_peak_end", "A_edge_perc_mean_start", "A_edge_perc_mean_end", "A_edge_chroma_dmean_start", "A_edge_chroma_dmax_start", "A_edge_chroma_dmean_end", "A_edge_chroma_dmax_end",
    "B_edge_onset_mean_start", "B_edge_onset_peak_start", "B_edge_onset_slope_start", "B_edge_onset_mean_end", "B_edge_onset_peak_end", "B_edge_onset_slope_end", "B_edge_flux_mean_start", "B_edge_flux_peak_start",
    "B_edge_flux_mean_end", "B_edge_flux_peak_end", "B_edge_perc_mean_start", "B_edge_perc_mean_end", "B_edge_chroma_dmean_start", "B_edge_chroma_dmax_start", "B_edge_chroma_dmean_end", "B_edge_chroma_dmax_end",
    
]


#Helpers for paths and track retrieval:

def _norm_path(path):
    return os.path.normcase(os.path.normpath(str(path)))

def _cache_has_track(cache, path):
    if path in cache:
        return True
    n = _norm_path(path)
    return any(_norm_path(k) == n for k in cache.keys())

def _cache_get_track(cache, path):
    if path in cache:
        return cache[path]
    n = _norm_path(path)
    for k, v in cache.items():
        if _norm_path(k) == n:
            return v
    raise KeyError(path)

#Helpers for edge-aware features:

#Get edges for a specific subregion
def _slice_edges(f0, f1, frames_per_beat, edge_beats=2):
    n = max(1, int(round(edge_beats * frames_per_beat)))
    s = slice(f0, min(f0 + n, f1)) #start band
    e = slice(max(f0, f1 - n), f1) #end band
    return s, e

#Retrieve mean, peak and slope for x.
def _mean_peak_slope(x):
    if x.size == 0:
        return 0.0, 0.0, 0.0
    t = np.arange(x.size, dtype=float)
    slope = float(np.polyfit(t, x.astype(float), 1)[0]) if x.size >= 2 else 0.0 #slope via gradient
    return float(x.mean()), float(x.max()), slope

#Retrieve change of harmonic content (different frequency bands) over time
def _chroma_delta_stats(C):
    #C: (12, T) chroma frames in the band
    if C.size == 0 or C.shape[1] < 2:
        return 0.0, 0.0
    D = np.linalg.norm(np.diff(C, axis=1), axis=0)
    return float(D.mean()), float(D.max())

#Other time-series arrays for edge features
def build_edge_sources(y, sr, S=None, hop_length=HOP_LENGTH):
    if S is None:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length) #high values for sudden drum/note entries
    flux = np.maximum(0.0, np.diff(S, axis=1)).sum(axis=0) #spectral flux (novelty)
    H, P = librosa.decompose.hpss(S) #harmonic and percussive
    perc_env = P.sum(axis=0)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length) #frequency bands
    return {"onset_env": onset_env, "flux": flux, "perc_env": perc_env, "chroma": chroma}

#Return scalar features for edge of window:
def edge_features(edge_src, f0, f1, frames_per_beat, edge_beats=2):
    s_band, e_band = _slice_edges(f0, f1, frames_per_beat, edge_beats=edge_beats)

    onset = edge_src.get("onset_env", None)
    flux  = edge_src.get("flux", None)
    perc  = edge_src.get("perc_env", None)
    chroma = edge_src.get("chroma", None)

    def _stats_1d(arr, band):
        if arr is None: return (0.0, 0.0, 0.0)
        x = arr[band]
        return _mean_peak_slope(x)

    #onset
    onset_mean_s, onset_peak_s, onset_slope_s = _stats_1d(onset, s_band)
    onset_mean_e, onset_peak_e, onset_slope_e = _stats_1d(onset, e_band)

    #flux (novelty)
    flux_mean_s, flux_peak_s, _ = _stats_1d(flux, s_band)
    flux_mean_e, flux_peak_e, _ = _stats_1d(flux, e_band)

    #percussive envelope
    perc_mean_s, _, _ = _stats_1d(perc, s_band)
    perc_mean_e, _, _ = _stats_1d(perc, e_band)

    #chroma deltas
    if chroma is not None:
        C_s = chroma[:, s_band]
        C_e = chroma[:, e_band]
        cdm_s, cdx_s = _chroma_delta_stats(C_s)
        cdm_e, cdx_e = _chroma_delta_stats(C_e)
    else:
        cdm_s = cdx_s = cdm_e = cdx_e = 0.0

    return {
        #onset
        "onset_mean_start": onset_mean_s, "onset_peak_start": onset_peak_s, "onset_slope_start": onset_slope_s,
        "onset_mean_end":   onset_mean_e, "onset_peak_end":   onset_peak_e, "onset_slope_end":   onset_slope_e,
        #flux
        "flux_mean_start":  flux_mean_s,  "flux_peak_start":  flux_peak_s,
        "flux_mean_end":    flux_mean_e,  "flux_peak_end":    flux_peak_e,
        #percussive
        "perc_mean_start":  perc_mean_s,  "perc_mean_end":    perc_mean_e,
        #chroma change
        "chroma_dmean_start": cdm_s, "chroma_dmax_start": cdx_s,
        "chroma_dmean_end":   cdm_e, "chroma_dmax_end":   cdx_e,
    }


#Heuristic helpers:

#Score an outro that is combined with another phrase
def combined_outro_score(
    sA, nxtA, startA, L,
    b_score, align_score, phraseB_len, fill_bonus, fill_bonus_b, Lw,
    trackA_total_beats, label_score_A, tail_tol
):
    #Allow a short unlabeled tail past the labeled Outro end
    envelope_end_A = min(nxtA["beat_end"] + tail_tol, trackA_total_beats)
    if startA + L > envelope_end_A:
        return None  #window doesn't fit the combined envelope

    #Use Outro weight for this combo so it can compete fairly
    a_weight = label_score_A["Outro"]

    #Recompute length bonus with the *combined* A envelope (no short-phrase penalty)
    envelope_len_A = envelope_end_A - sA["beat_start"]
    length_bonus_combined = 0.2 if (envelope_len_A >= L and phraseB_len >= L) else -0.3

    return (
        a_weight + b_score +
        align_score + length_bonus_combined +
        fill_bonus + fill_bonus_b + Lw
    )

#Return list of segments over the given 0-based range
def segment_cover(segments0, start0, end0):
    cov = []
    for s in segments0:
        if end0 <= s["beat_start"]: #window ends before this segment begins
            break
        if start0 < s["beat_end"]: #overlaps
            cov.append(s)
    return cov

#True if candidate window ends on a B phrase boundary
def ends_on_B_phrase_boundary(startB, L, segB, tol=0):
    endB = startB + L
    last = None
    for s in segB:
        if endB <= s["beat_start"]:
            break
        if startB < s["beat_end"]:
            last = s
    if last is None:
        return False
    return abs(endB - last["beat_end"]) <= tol

#Ensure boundaries are normalized to start at beat 0
def normalize_segments_zero_based(segments):
    if not segments:
        return [], 0

    #Use earliest boundary across the track as zero
    min_b = min(min(s["beat_start"], s["beat_end"]) for s in segments)

    norm = []
    for s in segments:
        s2 = dict(s)
        s2["beat_start"] = s["beat_start"] - min_b
        s2["beat_end"]   = s["beat_end"]   - min_b
        if "beat_fill" in s and s["beat_fill"] is not None:
            s2["beat_fill"] = s["beat_fill"] - min_b
        norm.append(s2)

    return norm, min_b

#Treat unmarked segment after an outro as extension of outro so heuristic calculations still operate normally.
#If last segment ends before track end, extend outro to end of track as this would just be a Rekordbox labelling issue, no other segments would come after the outro anyway.
MERGE_TRAILING_TAIL_INTO_LAST = True
MERGE_TAIL_TOL = 8  #in beats
def close_trailing_gap_as_outro(segments, total_beats):
    if not segments:
        return segments
    last = segments[-1]
    gap = total_beats - last["beat_end"]
    if MERGE_TRAILING_TAIL_INTO_LAST and 0 < gap <= MERGE_TAIL_TOL:
        last = dict(last)
        last["beat_end"] = total_beats
        segments = segments[:-1] + [last]
    return segments


#Track analysis methods:

#Retrieve relavant features via librosa
def extract_features(y, sr, hop_length=HOP_LENGTH, n_mfcc=MFCC_COEFFS):
    return {
        'centroid':  librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length),
        'bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length),
        'rolloff':   librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length,
                                                     roll_percent=ROLLOFF_PCT),
        'rms':       librosa.feature.rms(y=y, frame_length=hop_length*2, hop_length=hop_length),
        'zcr':       librosa.feature.zero_crossing_rate(y=y, frame_length=hop_length*2,
                                                         hop_length=hop_length),
        'mfcc':      librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length,
                                          n_mfcc=n_mfcc),
    }

#Get mean across frames in window
def mean_over_window(mat, start_frame, end_frame):
    if end_frame <= start_frame:
        return np.zeros(mat.shape[0], dtype=float)
    return np.mean(mat[:, start_frame:end_frame], axis=1)

#Extract all relevant features for a specific track, given by path
def analyze_track(path, beat_grids):
    if not os.path.exists(path):
        print("Path not found: " + path)
        raise FileNotFoundError(path)
    
    y, sr = librosa.load(path, sr=None, mono=True)
    feats = extract_features(y, sr)
    edge_src = build_edge_sources(y, sr, S=None, hop_length=HOP_LENGTH)
    fn = os.path.basename(path)
    if fn not in beat_grids:
        raise KeyError(f"Beat grid not found for {fn}")
    beats = beat_grids[fn][-1]
    return sr, beats, feats, edge_src

#Returns a function for getting beats->frames
def beats_to_frames(beats, sr):
    #return lambda b: int(beats[b] * sr / HOP_LENGTH)
    def _f(b):
        if b < 0:
            return int(beats[0] * sr / HOP_LENGTH)
        elif b == len(beats):
            return int(beats[b-1] * sr / HOP_LENGTH)
        elif b > len(beats):
            raise IndexError(f"Beat index out of range: {b} (0..{len(beats)-1})")
        else:
            return int(beats[b] * sr / HOP_LENGTH)
    return _f

#Get vector of feat means over segment
def segment_vector(feats, f0, f1):
    vecs = {}
    for name, mat in feats.items():
        v = mean_over_window(mat, f0, f1).tolist()
        vecs[name] = v
    return vecs

#Returns next segment from index
def next_segment(segments, idx):
    nxt = idx + 1
    return segments[nxt] if 0 <= nxt < len(segments) else None

#Flatten features to 1D
def flatten_feats(prefix, feats):
    flat = {}
    for key, values in feats.items():
        for i, v in enumerate(values):
            flat[f"{prefix}{key}_{i}"] = v
    return flat

#Build dataframe for candidate window across the tracks
def build_row_A_B(trackA, beatA0, win, trackB, beatB0, cache):
    srA, beatsA, featsA, edgeA = cache[trackA]
    srB, beatsB, featsB, edgeB = cache[trackB]
    fA = beats_to_frames(beatsA, srA)
    fB = beats_to_frames(beatsB, srB)
    fA0, fA1 = fA(beatA0), fA(beatA0 + win)
    fB0, fB1 = fB(beatB0), fB(beatB0 + win)
    A = segment_vector(featsA, fA0, fA1)
    B = segment_vector(featsB, fB0, fB1)
    
    totalA = len(beatsA)
    totalB = len(beatsB)
    
    #Edge-aware stuff
    frames_per_beat_A = (np.median(np.diff(beatsA)) * srA / HOP_LENGTH) if len(beatsA) > 1 else 1.0
    frames_per_beat_B = (np.median(np.diff(beatsB)) * srB / HOP_LENGTH) if len(beatsB) > 1 else 1.0
    edgesA = edge_features(edgeA, fA0, fA1, frames_per_beat_A, edge_beats=2)
    edgesB = edge_features(edgeB, fB0, fB1, frames_per_beat_B, edge_beats=2)
    
    row = {}
    row.update(flatten_feats("A_", A))
    row.update(flatten_feats("B_", B))
    row.update({ f"A_edge_{k}": float(v) for k,v in edgesA.items() }) #edge-aware
    row.update({ f"B_edge_{k}": float(v) for k,v in edgesB.items() }) #edge-aware
    row.update({
        "beatA": int(beatA0),
        "beatB": int(beatB0),
        "windowBeats": int(win),
        
        "totalBeatsA": int(totalA),
        "totalBeatsB": int(totalB),
        "relPosA": float(beatA0/totalA) if totalA else 0.0,
        "relPosB": float(beatB0/totalB) if totalB else 0.0,
        "beatsToEndA": int(max(0, totalA - (beatA0 + win))),
        "beatsToEndB": int(max(0, totalB - (beatB0 + win))),
        "relBeatsToEndA": float(max(0, totalA - (beatA0 + win))/totalA) if totalA else 0.0,
        "relBeatsToEndB": float(max(0, totalB - (beatB0 + win))/totalB) if totalB else 0.0,
    })

    #Reindex to expected columns (fills missing with 0; also guards unexpected keys)
    df = pd.DataFrame([row]).reindex(columns=EXPECTED_COLUMNS, fill_value=0.0)
    return df

#Get total beats from sgement list
def total_beats(segments):
    return segments[-1]["beat_end"] if segments else 0


#Pre-emptive track analysis methods including Rekordbox DB:

#Display progress of job (used for analysis of tracks)
def _render_progress(prefix, completed, total, started_at):
    total = max(1, int(total))
    completed = max(0, min(int(completed), total))
    width = 28
    ratio = completed / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(0.0, time.perf_counter() - started_at)
    rate = completed / elapsed if elapsed > 0 and completed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else 0.0
    return (
        f"\r{prefix} [{bar}] {completed}/{total} "
        f"({ratio * 100:5.1f}%) | elapsed {elapsed:6.1f}s | ETA {remaining:6.1f}s"
    )

#Preload analysis data and return (updated) cache
def preload_analysis(track_paths, beat_grids, max_workers=None, cache=None, show_progress=True):
    target_cache = ANALYSIS_CACHE if cache is None else cache

    unique_paths = []
    seen = set()
    for path in track_paths:
        if not path or path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    missing = [path for path in unique_paths if not _cache_has_track(target_cache, path)]
    if not missing:
        if show_progress:
            print('[Analysis preload] All requested tracks already cached.')
        return target_cache

    started_at = time.perf_counter()
    completed = 0
    if show_progress:
        print(f"[Analysis preload] Analyzing {len(missing)} track(s) sequentially.")
        sys.stdout.write(_render_progress('[Analysis preload]', completed, len(missing), started_at))
        sys.stdout.flush()

    for path in missing:
        try:
            target_cache[path] = analyze_track(path, beat_grids)
        except Exception as e:
            print(f"\n[Analysis preload] Failed for {os.path.basename(path)}: {e}")
        finally:
            completed += 1
            if show_progress:
                sys.stdout.write(_render_progress('[Analysis preload]', completed, len(missing), started_at))
                sys.stdout.flush()

    if show_progress:
        sys.stdout.write('\n')
        sys.stdout.flush()
    return target_cache

#Main suggestion method, can be used as a library call
def suggest(trackA, trackB, phrases, beat_grids, windows=(32,48,64,96,128), top_k=3,
            model_path="schedulerRandomForest2.joblib", blend=(0.2,0.8), dedup_tol=4, preloaded_cache=None):

    #Extract track keys and validate phrase availability
    keyA = os.path.basename(trackA)
    keyB = os.path.basename(trackB)
    
    if keyA not in phrases or keyB not in phrases:
        raise KeyError("Missing phrase annotations for one or both tracks: " + keyA + ", " + keyB)
    
    #Load and normalize phrase segments
    segA = phrases[keyA]
    segB = phrases[keyB]
    segA, A_offset = normalize_segments_zero_based(segA)
    segB, B_offset = normalize_segments_zero_based(segB)

    #Ensure audio analysis is cached
    cache = ANALYSIS_CACHE if preloaded_cache is None else preloaded_cache
    missing_tracks = [p for p in (trackA, trackB) if not _cache_has_track(cache, p)]
    if missing_tracks:
        preload_analysis(missing_tracks, beat_grids, cache=cache, show_progress=False)

    #Retrieve cached analysis data
    trackA_data = _cache_get_track(cache, trackA)
    trackB_data = _cache_get_track(cache, trackB)
    trackA_total_beats = len(trackA_data[1])
    trackB_total_beats = len(trackB_data[1])

    #Close trailing gaps and compute total lengths
    segA = close_trailing_gap_as_outro(segA, trackA_total_beats)
    totalA = total_beats(segA)
    totalB = total_beats(segB)

    #Load trained Random Forest model if available
    RF = None
    if model_path and os.path.exists(model_path):
        RF = joblib.load(model_path)
        EXPECTED_COLUMNS = list(RF.feature_names_in_) if hasattr(RF, "feature_names_in_") else EXPECTED_COLUMNS
    print("Loaded RF.")

    #Generate candidate transitions across all window sizes
    cands = []
    for L in windows:
        Lw = WINDOW_WEIGHT.get(L, 0.6)

        for sA in segA:
            startA = sA["beat_start"]
            if startA + L > totalA:
                continue

            #Compute A-side heuristic components
            a_score = LABEL_SCORE_A.get(sA["label"], 1.0)
            fill_bonus = 0.4 if (sA.get("fill") and sA.get("beat_fill") is not None and (sA["beat_end"] - sA["beat_fill"]) <= 4) else 0.0

            for sB in segB:
                startB = sB["beat_start"]
                if startB + L > totalB:
                    continue

                #Compute B-side heuristic components
                b_score = LABEL_SCORE_B.get(sB["label"], 1.0)
                fill_bonus_b = 0.2 if (sB.get("fill") and sB.get("beat_fill") is not None and (sB["beat_fill"] - sB["beat_start"]) <= 4) else 0.0

                if startB == 0:
                    b_score *= 0.5
                
                #Handle special 16+16 phrase grouping case
                endB = startB + L
                coveredB = segment_cover(segB, startB, endB)
                if len(coveredB) == 2 and coveredB[0]["beat_start"] == startB and coveredB[-1]["beat_end"] == endB:
                    len1 = coveredB[0]["beat_end"] - coveredB[0]["beat_start"]
                    len2 = coveredB[1]["beat_end"] - coveredB[1]["beat_start"]
                    if abs(len1 - 16) <= 0 and abs(len2 - 16) <= 0:
                        pass

                #Alignment and phrase-length scoring
                align = (startA == sA["beat_start"]) and (startB == sB["beat_start"])
                align_score = 1.0 if align else -0.5

                phraseA_len = sA["beat_end"] - sA["beat_start"]
                phraseB_len = sB["beat_end"] - sB["beat_start"]
                length_bonus = 0.2 if (phraseA_len >= L and phraseB_len >= L) else -0.3
                
                #Base heuristic score
                heuristic_base = (
                    a_score + b_score +
                    align_score + length_bonus +
                    fill_bonus + fill_bonus_b + Lw
                )

                #Apply Outro damping
                simple_outro_intro_penalty = 0.0
                if sA["label"] == "Outro" and sB["label"] == "Intro":
                    simple_outro_intro_penalty = SOFT_OUTRO_INTRO_PENALTY
                elif sA["label"] == "Outro":
                    simple_outro_intro_penalty = SOFT_OUTRO_INTRO_PENALTY * 0.8

                heuristic = heuristic_base + simple_outro_intro_penalty

                #Apply clash penalties for high-energy overlaps
                clash_penalty = 0.0
                if sA["label"] == "Chorus" and sB["label"] == "Up":
                    clash_penalty -= 1.00
                elif sA["label"] == "Chorus" and sB["label"] == "Chorus":
                    clash_penalty -= 1.00
                elif sA["label"] == "Up" and sB["label"] == "Chorus":
                    clash_penalty -= 1.00

                heuristic += clash_penalty
                
                #Apply combined A envelope when short phrases require extension
                if ENABLE_A_COMBINED_IF_SHORT:
                    nxtA = next_segment(segA, sA["index"])
                    if nxtA and (sA["beat_end"] - sA["beat_start"]) < L:
                        if nxtA["label"] == "Outro":
                            envelope_end_A = min(nxtA["beat_end"] + A_OUTRO_TAIL_TOL, trackA_total_beats)
                        else:
                            envelope_end_A = nxtA["beat_end"]

                        if startA + L <= envelope_end_A and (startA + L) == envelope_end_A:
                            target_label = "Outro" if nxtA["label"] == "Outro" else nxtA["label"]
                            a_weight = LABEL_SCORE_A.get(target_label, LABEL_SCORE_A.get(sA["label"], 1.0))

                            envelope_len_A = envelope_end_A - sA["beat_start"]
                            length_bonus_combined = 0.2 if (envelope_len_A >= L and phraseB_len >= L) else -0.3

                            heuristic_combined = (
                                a_weight + b_score +
                                align_score + length_bonus_combined +
                                fill_bonus + fill_bonus_b + Lw
                            )
                            heuristic = max(heuristic, heuristic_combined)
                
                #Apply Chorus/Up -> Outro combined envelope logic
                if ALLOW_A_COMBINED_OUTRO:
                    nxtA = next_segment(segA, sA["index"])
                    if nxtA and (sA["label"], nxtA["label"]) in A_COMBO_OUTRO_LABELS:
                        h2 = combined_outro_score(
                            sA, nxtA, startA, L,
                            b_score, align_score, phraseB_len, fill_bonus, fill_bonus_b, Lw,
                            trackA_total_beats, LABEL_SCORE_A, A_OUTRO_TAIL_TOL
                        )
                        if h2 is not None:
                            heuristic = max(heuristic, h2)
                
                #Handle Down -> Outro pattern with constraints and bonuses
                if sA["label"] == "Down":
                    nxtA = next_segment(segA, sA["index"])
                    if nxtA and nxtA["label"] == "Outro":

                        if EQUALIZE_DOWN_AS_OUTRO:
                            if (not EQUALIZE_REQUIRE_B_INTRO) or (sB["label"] == "Intro"):
                                delta = LABEL_SCORE_A["Outro"] - LABEL_SCORE_A["Down"]
                                heuristic_equalized = heuristic_base + delta
                                heuristic = max(heuristic, heuristic_equalized)

                        if ENFORCE_B_INTRO_UP_ONLY:
                            endB = startB + L
                            covered = segment_cover(segB, startB, endB)

                            if len(covered) > 2:
                                continue

                            labelsB = [s["label"] for s in covered]
                            if not set(labelsB).issubset({"Intro","Up"}):
                                continue

                            if len(covered) == 2 and not (
                                covered[0]["label"] == "Intro" and covered[1]["label"] == "Up"
                            ):
                                continue
                        
                        if ALIGN_B_END_FOR_DOWN_OUTRO and not ALIGN_B_END_GLOBAL:
                            if not ends_on_B_phrase_boundary(startB, L, segB, B_END_ALIGN_TOL):
                                continue
                        
                        if DOWN_OUTRO_MATCH_B_ENABLE:
                            combined_len = nxtA["beat_end"] - sA["beat_start"]
                            len_diff = abs(phraseB_len - combined_len)

                            ok_b_label = True
                            if DOWN_OUTRO_MATCH_REQUIRE_B_IN is not None:
                                ok_b_label = (sB["label"] in DOWN_OUTRO_MATCH_REQUIRE_B_IN)

                            if ok_b_label and len_diff <= DOWN_OUTRO_MATCH_B_TOL:
                                heuristic += DOWN_OUTRO_MATCH_B_BONUS

                                if abs(L - combined_len) <= DOWN_OUTRO_MATCH_WINDOW_TOL:
                                    heuristic += DOWN_OUTRO_MATCH_WINDOW_BONUS
                
                #Apply global B-end alignment constraint
                if ALIGN_B_END_GLOBAL and not ends_on_B_phrase_boundary(startB, L, segB, B_END_ALIGN_TOL):
                    continue
                
                #Compute Random Forest score
                rf_proba = None
                if RF is not None:
                    X = build_row_A_B(trackA, startA, L, trackB, startB, cache)
                    try:
                        proba = RF.predict_proba(X)
                        rf_proba = float(proba[0][-1])
                    except Exception as e:
                        print(f"[warn] RF prediction failed: {e}")
                        rf_proba = None

                #Boost heuristic if RF is highly confident
                if rf_proba >= 0.95 and heuristic < 6:
                    heuristic += 1.5

                #Blend heuristic and RF score
                h_norm = max(0.0, min(1.0, (heuristic - HEURISTIC_MIN) / (HEURISTIC_MAX - HEURISTIC_MIN)))
                h_w, r_w = blend
                score = heuristic if RF is None else (h_w * h_norm + r_w * rf_proba)

                #Store candidate
                cands.append({
                    "trackA": keyA, "trackB": keyB,
                    "beatA": int(startA), "beatB": int(startB),
                    "windowBeats": int(L),
                    "heuristic": float(round(heuristic, 4)),
                    "rf_proba": None if rf_proba is None else float(round(rf_proba, 4)),
                    "score": float(round(score, 4)),
                    "labels": (sA["label"], sB["label"]),
                    "segments": (sA["index"], sB["index"]),
                    "phraseA_len": int(phraseA_len), "phraseB_len": int(phraseB_len),
                    "relPosA": float(startA / totalA)
                })

    #Bucket candidates by relative position in track A
    bucket1Cands = [c for c in cands if c["relPosA"] >= 0.75]
    bucket2Cands = [c for c in cands if c["relPosA"] >= 0.5 and c["relPosA"] < 0.75]
    bucket3Cands = [c for c in cands if c["relPosA"] < 0.5]
    print("Buckets collected.")

    #Sort, deduplicate, and select top candidates per bucket
    def rank_bucket(bucket):
        bucket.sort(key=lambda x: (-x["score"], -x["heuristic"], x["windowBeats"]))
        top = []
        for c in bucket:
            if not any(
                c["windowBeats"] == t["windowBeats"] and
                abs(c["beatA"] - t["beatA"]) <= dedup_tol and
                abs(c["beatB"] - t["beatB"]) <= dedup_tol
                for t in top
            ):
                top.append(c)
            if len(top) >= top_k:
                break
        return top

    bucket1Top = rank_bucket(bucket1Cands)
    bucket2Top = rank_bucket(bucket2Cands)
    bucket3Top = rank_bucket(bucket3Cands)
    
    print("Returning buckets.")
    return [bucket1Top, bucket2Top, bucket3Top], cands