# Forecasting a turbulent channel-flow signal: recurrent networks, a Transformer and a linear baseline

These five notebooks forecast one velocity signal from a direct numerical simulation of turbulent channel flow: the streamwise velocity at one wall-normal position, sampled every 0.01 time units for 106,000 steps across 106 simulation files. Stage 1 extracts that signal from the raw files into one small HDF5 file and a set of TFRecords, keeping the file index of every sample so that no training window straddles two files. Stage 2 cuts it into windows of 300 past values and the next 10, splits them in time, and trains a dense network, SimpleRNN, GRU and LSTM against persistence; Stage 3 poses the same task to a small Transformer encoder. Stage 4 asks what a careful engineer would ask first: on exactly the same windows and test period, how do these networks compare with a linear autoregressive model fitted by least squares? Stage 5 asks where the linear model stops winning (output windows of 10 to 300 steps) and whether the Transformer beats persistence once it gets the training budget the recurrent networks had. Every training is repeated with three seeds.

Portfolio page: https://vamsigudipati.github.io/cfd-signal-forecasting-rnn-transformer.html

## Notebooks

| Notebook | Stage | Results | Models | Animations |
| --- | --- | --- | --- | --- |
| `notebooks/01_extract_cfd_signal.ipynb` (with `notebooks/cfd_data_tools.py`) | Stage 1 — Extracting one near-wall velocity signal from 106 simulation files | `results/01_extract_cfd_signal/` (106 TFRecord shards and their metadata) | — | — |
| `notebooks/02_recurrent_forecasting.ipynb` | Stage 2 — Forecasting the near-wall signal with recurrent networks | `results/02_recurrent_forecasting/exercises/` | `models/02_recurrent_forecasting/checkpoints/` | `animations/02_*` |
| `notebooks/03_transformer_forecasting.ipynb` | Stage 3 — Forecasting the near-wall signal with a small Transformer | `results/03_transformer_forecasting/exercises/` | `models/03_transformer_forecasting/checkpoints/` | `animations/03_*` |
| `notebooks/04_horizon_and_baselines.ipynb` | Stage 4 — Horizon and baselines: what the networks buy over a linear model | `results/04_horizon_and_baselines/` | `models/04_horizon_and_baselines/` (seed 0 of each network) | — |
| `notebooks/05_horizon_and_tuned_transformer.ipynb` | Stage 5 — Horizon and a tuned Transformer: where the linear model stops winning | `results/05_horizon_and_tuned_transformer/` | `models/05_horizon_and_tuned_transformer/` (seed 0: GRU at each horizon, both Transformer settings) | — |

Each notebook is also exported as HTML next to it, with all outputs. The `exercises/` folders hold the tables (CSV) and figures (PNG) of the experiments and of the three-seed repeats (`seeds_runs.csv`, `seeds_summary.csv`). `animations/` holds the training videos and their final frames.

The outputs in the notebooks come from my single run of each notebook. For this repository I changed only text afterwards: a header cell and a paths cell in each notebook, the path lines (data, models, outputs), comments, labels and printed text. The notebooks were not executed again in this layout; the paths cell is therefore the one cell without an output.

## Data

`data/utmp_y50_raw_timeseries.h5` (0.95 MB, in git) is the processed signal that Stage 1 writes and Stages 2–5 read; my run of Stage 1 reproduced it byte for byte (SHA-1 `f0a7789027c49e6465af62148cfbc1ea2e6aa22e`). Stages 2–5 need nothing else.

Stage 1 alone needs the 106 raw simulation files (`P125_83pi_30001.stp` … `30106.stp`, HDF5, 53 MB each, 5.64 GB in total), which it reads from `release_assets/raw_stp/`. They are not distributed with this repository; Stage 1 is included to show how the signal was extracted and checked, and its executed outputs are kept.

`release_assets/` is ignored by git. Every model in `models/` was trained by my own runs; no pretrained model is used.

## Setting up and running

I ran everything on a laptop (Apple M3 Max, 48 GB) with Python 3.12, TensorFlow 2.18.1, Keras 3.6 and the `tensorflow-metal` plug-in for the Apple GPU. The frozen environment is in `requirements.txt`; the videos use the ffmpeg binary that `imageio-ffmpeg` installs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name py312-tf-metal-studio
cd notebooks
# Stage 1 needs the raw simulation files in ../release_assets/raw_stp/, which are not distributed
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 02_recurrent_forecasting.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 03_transformer_forecasting.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 04_horizon_and_baselines.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 05_horizon_and_tuned_transformer.ipynb
```

All paths inside the notebooks are relative to `notebooks/`, so run them from there. Every model is trained when its notebook runs. On the laptop, Stage 1 took 6 seconds, Stage 2 32 minutes, Stage 3 75 minutes, Stage 4 21 minutes and Stage 5 25 minutes, one at a time. Stage 5 reads Stage 4's device timings and result table, so run Stage 4 first.

**Why this device.** Before training, I timed one training step of each network inside `model.fit` (batch 64, window 300, median of 30 steps after 10 warm-up steps), on the CPU and on the Apple GPU (`results/04_horizon_and_baselines/device_timing_*.csv`). In milliseconds per step: dense network 0.35 vs 5.97, SimpleRNN 8.6 vs 15,532, GRU 35.5 vs 17.7, LSTM 23.3 vs 18.2, two stacked GRUs 81 vs 26, Transformer 127 vs 53. So the dense network and SimpleRNN train on the CPU and the others on the GPU, each placed with `tf.device`; Stage 5 reuses these timings (same environment) and keeps the GRU and the Transformer on the GPU. One caution I found afterwards: for the same weights, the Apple GPU's GRU kernel gives a different output from Keras's reference GRU. My Stage 2 GRU scores 0.0530 on the GPU and 0.469 with the GPU hidden (the LSTM gives 0.0536 on both), so the GRU models here reproduce their numbers only on an Apple GPU, and the GRU timing compares two different computations. With the same seed and settings, a GPU training did repeat: the Stage 4 Transformer's seeds 1 and 2 reproduce Stage 3's seed runs to 7 digits.

**Why three seeds.** A single training gives one draw of the initial weights and of the batch order. Every training here takes under 15 minutes (the longest in Stage 5 is 204 s), so each is repeated with the seeds 0, 1 and 2 (seed 0 is the base run), and I report the mean, the standard deviation and the range. A difference smaller than that spread is not an effect: for example, Stage 3's Transformer beats persistence in the base run (0.0969 against 0.1028) but not in another seed, 0.1017 ± 0.0047 over the three.

## What the stages show

- Stage 1 turns 5.64 GB of simulation output into a 0.95 MB signal with its file boundaries preserved. Every window of Stages 2–4 stays inside one file, but consecutive windows overlap, so the first 30 windows after each split boundary reuse inputs from the part before it (never its targets).
- In Stage 2 the recurrent networks halve the error of persistence: GRU 0.0525 ± 0.0006, LSTM 0.0548 ± 0.0022, SimpleRNN 0.0578 ± 0.0038 against 0.1028 (normalised mean absolute error over the 10 future steps), with the dense network barely better than persistence. The networks there end in one output, so each predicts a single value for all 10 steps; the GRU edge over the LSTM is smaller than the seed spread.
- In Stage 3 the Transformer, trained with its notebook's settings (learning rate 1e-4, 50 epochs, never stopping early), does not beat persistence over three seeds, while the recurrent networks of its Experiment 4, predicting all 10 steps, reach about 0.036.
- Stage 4 shows what the extra complexity buys. On the same windows, a linear autoregressive model fitted by least squares in 42 ms reaches 0.0004, 77 to 258 times lower than any network (SimpleRNN 0.0306, GRU 0.0307, LSTM 0.0354, Transformer 0.1031); fitted on only the last 2 values it already reaches 0.035. The signal is smooth over 10 samples, so at these horizons the networks are not needed; whether they pay off at longer horizons is Stage 5's question.
- Stage 5 answers it on one set of windows scored at 10, 30, 100 and 300 steps ahead: the GRU never beats the linear model beyond its seed range (error ratios 73×, 4.8×, 1.3× and 1.07×), and at 300 steps both approach the error of predicting the mean. The Transformer, given the recurrent networks' training budget, beats persistence (0.064 against 0.094, beyond the seed range) but stays far from the GRU (0.029): the Stage 3 gap was partly its 1e-4 learning rate, not only its architecture.
