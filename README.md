# AI-Driven DJ Transition Demo

## Introduction

This repository contains a simple demonstration of using machine-learning and
audio analysis to automatically identify optimal windows for musical transitions
between DJ tracks. The system analyzes the structure and timbre of two songs
and predicts suitable overlapping sections where a human can perform an
EQ-mix or cross-fade within the transition.

The model powering the demo is a Random Forest classifier trained on over
1000 transition examples. These consist of human-labelled transitions
augmented with machine-generated invalid ones. Due to the dataset being focused
on a subset of popular house tracks, the model performs best within similar
house music and may not generalise well across other genres.

The goal of this project is to demonstrate how AI can assist in identifying
good-sounding transition points in audio.

---

## How It Works

1. **Rekordbox Analysis**
   `rekordbox_usb_export_demo.py` parses a Rekordbox USB export and extracts:

   * Track paths
   * Beat grids
   * Phrase structure (Intro, Chorus, Outro, etc.)

2. **Audio Feature Extraction**
   Using `librosa`, the system computes:

   * MFCCs
   * Spectral features (centroid, bandwidth, rolloff)
   * Energy, onset, flux, and chroma-based features

3. **Candidate Generation**
   `suggest_transitions_bucketed.py`:

   * Generates transition windows (32–128 beats)
   * Uses phrase-aware heuristics (e.g. Outro → Intro)
   * Buckets candidates by track position

4. **Model Scoring**
   A trained Random Forest (`schedulerRandomForest.joblib`) scores each candidate.
   This is blended with heuristic scoring for final ranking.

5. **Automated Execution (Mixxx Integration)**
   `unified_auto_transition_new.py`:

   * Connects to Mixxx via WebSocket
   * Tracks loaded songs and playback state
   * Selects optimal transitions
   * Automatically schedules and performs transitions via MIDI

   The system handles **when and how to transition structurally**, leaving
   EQ and creative control to the user.

---

## Limitations

* **Genre-specific**
  Trained primarily on house music

* **No EQ decisions**
  Only handles timing and structure of transitions

* **Experimental**
  Intended as a demo / proof of concept

---

## Setup

### 1. Clone the Repository

```bash
git clone --recursive <your-repo>
cd <your-repo>
```

---

### 2. Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Rekordbox USB Requirement

You must export your music library using Rekordbox to a USB device.

The USB **must be mounted in a location accessible to both this script and Mixxx**.

Required contents:

* `export.pdb`
* `PIONEER/USBANLZ/` analysis files

Example usage:

```bash
--usb-root /media/username/YOUR_USB
```

---

### 4. Mixxx (Custom Build Required)

This project depends on the custom Mixxx build included in this repository:

`mixxx_ws/`

This fork provides:

* Programmatic track loading
* WebSocket control interface
* Integration with external systems

#### Build Steps (Linux)

```bash
cd mixxx_ws

sudo apt install libwebsocketpp-dev

tools/debian_buildenv.sh setup

mkdir build
cd build
cmake ..
cmake --build .
```

---

### 5. Run the System

⚠️ **Important: Start the Python script BEFORE opening Mixxx**

The script:

* Creates the virtual MIDI devices
* Waits for Mixxx to connect on startup
* Hooks into Mixxx via WebSocket

#### Step-by-step:

1. Start the automation script:

```bash
python unified_auto_transition_new.py \
    --usb-root /media/username/YOUR_USB \
    --model schedulerRandomForest.joblib \
    --virtual-midi
```

2. Then launch Mixxx (from `mixxx_ws/build`)

3. In Mixxx, select the MIDI device:

```
DJ-AI Controller
```

---

## How to Use

Once everything is running:

1. Load a track into **both decks**
2. Press play on **one deck**

The system will then:

* Detect the currently playing deck
* Treat the other deck as the next track
* Begin preparing an optimal transition

When a transition is executed:

* The system will continue monitoring deck state
* Any time the **track combination changes**, it will:

  * Recompute transition candidates
  * Prepare a new transition automatically

This creates a continuous workflow where transitions are always being
prepared dynamically based on what is currently loaded and playing.

---

## Behaviour

The system:

* Suggests and performs transitions automatically
* Selects optimal transition windows based on structure + audio features
* Executes timing via MIDI and WebSocket control

A human DJ can then focus on:

* EQ mixing
* Effects
* Creative decisions

---

## CLI Options

```bash
python unified_auto_transition_new.py --help
```

Key options:

* `--topk` → number of candidates per bucket
* `--windows` → beat window sizes
* `--blend` → heuristic vs model weighting
* `--lead-margin-beats` → minimum future margin

---

## Examples

In order to test generalisation whilst staying within the House genre, I made a Rekordbox Library with unseen songs not present in the model's training data.
Below I will list names of tracks that the model proved well generalisating to, but will not be distributing:

- Steve Angello, Laidback Luke - Show Me Love (feat. Robin S) Extended Mix
- *More to be added soon!*

---

## Training Data

**Note on Training Data**  
> The model is pre-trained and distributed without any audio data, as the original
> training set consisted of copyrighted tracks. However, the training process uses
> the exact same feature extraction pipeline as inference. Each transition example
> was converted into structured audio features (e.g. spectral, rhythmic, and phrase-based
> attributes) and labelled accordingly. This ensures consistency between training
> and real-time prediction, with the model operating purely on derived features
> rather than raw audio.

## Acknowledgements

* Uses a custom Mixxx fork with WebSocket support developed for this project:
  https://github.com/hothersj/mixxx_ws

* Built using:

  * `librosa` for audio analysis
  * `scikit-learn` for ML
  * `pyrekordbox` for Rekordbox parsing

---

This project is a demonstration of AI-assisted DJ workflows.
