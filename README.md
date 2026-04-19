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
If you want to see the list of tracks that the model was trained on, please see *Training Data*.
Below I will list names of a couple of unseen tracks that the model proved well generalising for validation:

- Steve Angello, Laidback Luke - Show Me Love (feat. Robin S) Extended Mix *->* Calvin Harris, Disciples, Chris Lake - How Deep Is Your Love - Chris Lake Remix
- Karen Harding, LuvBug - Say Something - Luvbug Remix *->* Au/Ra, CamelPhat - Panic Room - Club Mix

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

*The track list for training, was as follows:*
   - Dirty Cash (Money Talks) - Extended Version — Live In Lover, LX Feat.
   - Feel 4 Me - Edit — The Trip
   - Nite Life - M-High Extended Remix — Kim English, M-High
   - Some Time — Roeléh
   - Runnin Around — Clark James
   - Hold On - Extended Mix — Nic Fanciulli, Marc E. Bassy
   - If You - Extended Mix — Kubinowski, Mafia Mike
   - Stay With Me — X CLUB.
   - Fresh - House of Prayers Poolside Edit — Crazibiza, House of Prayers
   - R U Dreaming - Harry Romero Raw Dog Remix — Damian Lazarus, Mathew Jonson
   - Feelin’ — Avalon Child
   - Would U Like — Soul Mass Transit System
   - Everybody Ain’t Nobody - Original Mix — House of Prayers, Crazibiza
   - Da Funk (feat. Joni) — Mochakk, Joni
   - Gagarin — Sugarman
   - Fast Times - Extended — My Friend, Steven Weston
   - Lost In Music - Extended — Joshwa
   - After Moods - Extended — Julius Strieder
   - Sax Thing - Extended Mix — Capri
   - This Dance — Martin Occo
   - Inside (At Night) - Instrumental Club — Tommy Vee, Keller, Mauro Ferrucci
   - Just Be Good To Me - Extended Mix — Baccus
   - Fighting Love - Extended Mix — Mark Knight, Mark Dedross
   - Music Is Passion — Chinonegro
   - Pump (1991 Remix) — Chris Lorenzo
   - Now U Do — DJ Seinfeld, Confidence Man
   - I Want You - Extended Mix — Butch, Nic Fanciulli
   - Do You Really Like It? — DJ Pied Piper & The Masters of Ceremonies
   - Know You Better - Extended Mix — Luke Sica
   - Call The Shots — Jade Cox
   - Deep Inside - Original — Ivan Masa, HOMIEZ
   - In The Morning — Schiela
   - Ready For The Floor - Mella Dee Dub — Hot Chip, Mella Dee
   - A Never Ending — Pancrazio
   - Bette Davis Eyes - Boys Shorts Remix — Kim Carnes, Boys Shorts
   - Devil’s Destination — PARAMIDA
   - It’s Time — Cassy
   - Rave Digger - Extended Breaks Mix — Genix, MC Lethal
   - Pump It Up - Extended Mix — Kristin Velvet
   - Permanence — Mitch Freeman
   - MØND — Huxley
   - First Summer — Tuccillo
   - Time For Peace — Tiger Stripes
   - Cada Vez — FONTI
   - Yo Te Deseo - David Gtronic & Reboot Remix — Miguel Lobo, Miluhska
   - Rumbero - Original Mix — Simone Liberali
   - El House — Hector Couto, Alejandro Paz
   - Circus — Leyeo, Easttown, Thierry Tomas
   - From The Stars — Mochakk, The Rah Band
   - Share My Love With You - Extended Mix — The Dirty Rabbit
   - Do To Me — Tim Sanders
   - Tell Me What You Want - Extended Mix — Oden & Fatzo, Poppy Baskcomb
   - House Party — Ozzie Guven, Chris Gialanze
   - I Can’t Stop - Joseph Capriati Extended Groove — Sandy Rivera, Joseph Capriati
   - How Does It Make You Feel — Young Marco
   - Always On My Mind (Let U Go) — Novaj, Louis Seigner
   - Midnight Forever — Delta Heavy
   - 8AM — AUTOFLOWER, DJ Afterthought
   - Going Down — Dennis Cruz
   - New World Expression — X CLUB.
   - Hideaway - Extended Mix — Kiesza
   - She’s Gone Dance On — Audiojack
   - Body Groove - Mix Version — Architechs, Nay Nay
   - Sinnerman - 2022 Extended Version — Swell
   - A Fresh Energy — Gaskin
   - House Every Weekend — David Zowie
   - Wish You Were Mine — Dizzman
   - The Cure & The Cause - Dennis Ferrer Remix — Fish Go Deep, Tracey K
   - Delilah (pull me out of this) — Fred again.., Delilah Montagu
   - You & Me — Disclosure, Eliza Doolittle
   - Feel My Needs - Extended Mix — WEISS, James Hype
   - Something To Me — Josh Baker
   - Damage — Sammy Virji
   - Do It Right - Club Mix — Martin Solveig, Tkay Maidza
   - You Don’t Know Me - Extended Mix — Nick G
   - Wearing My Rolex - Radio Edit — Wiley
   - Places - Club Mix — Martin Solveig, Ina Wroldsen
   - Still Sleepless - Extended Mix — D.O.D, Carla Monroe
   - Why’s this dealer? — Niko B
   - Flowers - Sunship Remix — Sweet Female Attitude
   - Katy on a Mission — Katy B
   - White Noise — Disclosure, AlunaGeorge
   - Lauren (I Can’t Stay Forever) — Oden & Fatzo
   - Too Much For Me - Extended Mix — SELKER
   - A Little Bit Of Luck - XDBR Extended Remix — DJ Luck & MC Neat
   - The Days - NOTION Extended Remix — Chrystal, NOTION
   - Rinse & Repeat - Extended Mix — Riton, Kah-Lo
   - Dance With Me - Extended Mix — Dizzee Rascal
   - My My My - Original Club Mix — Armand Van Helden
   - What I Might Do - Club Mix — Ben Pearce
   - Gecko (Overdrive) - Extended — Oliver Heldens, Becky Hill
   - Show Me (Edit) — Jethro Heston, Huxley, B
   - Total Unison — D Stone
   - B.O.T.A. (Baddest Of Them All) — Eliza Rose, Interplanetary Criminal

## Acknowledgements

* Uses a custom Mixxx fork with WebSocket support developed for this project:
  https://github.com/hothersj/mixxx_ws

* Built using:

  * `librosa` for audio analysis
  * `scikit-learn` for ML
  * `pyrekordbox` for Rekordbox parsing

---

This project is a demonstration of AI-assisted DJ workflows.
