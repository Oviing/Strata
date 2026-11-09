# DiffusionGemma on a Mac (MLX, experimental)

This page is for people with an Apple-silicon Mac who want to try Google's DiffusionGemma-26B-A4B next to their Strata.
It runs through a small separate engine on MLX (Apple's machine-learning library). It does not use Strata's Metal engine.

> **Status: not measured by us.** We have not run this model on a Mac yet. Every number on this page is either
> marked as Google's report, marked as a third party's, or marked as a setup estimate. Read [Not tested yet](#not-tested-yet).

> **On this page:** [What it is](#what-it-is) · [Who it is for](#who-it-is-for) · [Install and run](#install-and-run) ·
> [The app](#the-app) · [The API](#the-api) · [Limits](#limits-and-known-issues) · [Figures](#figures-from-others) ·
> [Not tested yet](#not-tested-yet) · [How to report results](#how-to-report-results)

## What it is

DiffusionGemma-26B-A4B is a Google model under the Apache 2.0 licence. It has 26 billion parameters. Its mixture-of-experts
design uses about 4 billion of them at each step.

Most models write one token (a piece of a word) at a time. DiffusionGemma can write a **block** of 256 tokens at once.
The block is called a canvas:

1. It starts the canvas with random tokens.
2. It rewrites the whole canvas, step after step, for up to 48 steps.
3. When the canvas stops changing, the block is kept. The next block starts after it.

The report says the sampler stops early and averages about 12 steps per block (report section 3 and Algorithm 1, page 10).
After each block is kept, the model reads it and adds it to its memory before the next block. The prompt is read the same way.

The same weights can also write one token at a time, as ordinary models do (report section 3.5). Strata calls this
**One token at a time**. In the API and the Terminal it is the mode `ar`.

Strata runs the model through a sidecar: a small Python program on MLX that speaks the same protocol as Strata's own
engine. The server, the Terminal, the app and `strata connect` use their normal paths.

## Who it is for

- **Apple-silicon Macs only** (M1 or newer, macOS 14 or newer). Setup refuses on any other computer. `strata launch --plan`
  still works anywhere and changes nothing.
- **Text only.** Pictures are not sent to this model.
- **One request at a time.** A second request waits for the first.
- **16 GB of memory is tight.** The default there is the 3-bit copy, `mlx-q3`:
  - It keeps the experts at 3 bits, and the attention and embeddings at higher precision. The picture part is left out.
  - Setup estimates 10 to 11 GB in memory. We have not measured this.
  - Your Mac makes it from Google's BF16 model. That download is about 52 GB (51.7 GB in setup's table). The plan asks for
    about 73 GB of free disk: the download, the 11 GB copy made from it, and 10 GB spare.
  - Setup asks before it deletes the big download afterwards. `--yes` answers yes.
- **`mlx-q4`** is the 4-bit copy by mlx-community. It is a 16.6 GB download with no conversion, and it is the default from
  32 GB of memory. Setup lists 24 GB of memory for it. On a 16 GB Mac we expect it to be too large, from its size alone.
  We have not tested that.

## Install and run

`dgemma-strata` is a second copy of the Terminal's `strata` command. Put it in the same folder as `strata`.
It keeps its settings, saved setups, runtime and model files in `~/.dgemma-strata`, so an existing Strata install is not
changed. Set `DGEMMA_STRATA_HOME` to use another folder.

**The port.** The launcher sets a default port of 8090 (through `STRATA_DEFAULT_PORT`), so both servers can run at once. The
Terminal's own default stays 8080. The commands below name `--port 8090` anyway, so they work the same way.

```sh
# 1. Check the plan. It works on any computer and changes nothing.
dgemma-strata launch --plan --family diffusiongemma --model mlx-q3 --port 8090

# 2. Set up and serve without questions. Mac only. Downloads about 52 GB.
dgemma-strata launch --family diffusiongemma --model mlx-q3 --port 8090 --yes --no-tui

# 3. Check one app. The connect default address is port 8080, so pass the address.
dgemma-strata connect <app> --url http://127.0.0.1:8090 --check

# Show what is serving, and stop it.
dgemma-strata launch --status
dgemma-strata launch --stop
```

To use model files you already have, add `--model-path PATH` to point at an MLX folder. To use one-token mode, add
`--mode ar`. The default is `--mode diffusion`.

| Flag | For DiffusionGemma |
| --- | --- |
| `--model mlx-q3` or `mlx-q4` | The two sizes above |
| `--mode diffusion` or `ar` | A block at a time (default), or one token at a time |
| `--context` | 8K to 128K. Default 32K. Past 128K is not tested on a Mac |
| `--model-path PATH` | An MLX folder you already have |
| `--data-dir DIR` | Where the model files go |
| `--kv`, `--parallel`, `--speed`, `--drafting`, `--mtp`, `--tune` | Not used. The launch says so and leaves them out |

**Optional: more memory for the GPU on 16 GB.** macOS may let the GPU use only part of the memory by default. A third-party
guide suggests raising that share with:

```sh
sudo sysctl iogpu.wired_limit_mb=<megabytes>
```

Strata never runs this command. It needs an administrator password. The value is a third party's suggestion, not ours,
and we have not tested it or checked what it does to the rest of macOS. Use it only if you understand the trade-off.

## The app

- **Model settings: Mode (DiffusionGemma).** The choices are "Default (Diffusion)", "Diffusion" and "One token at a time".
  The app's own text says: "We have not measured which is faster or better on your Mac." The mode applies the next
  time the model loads.
- **Live canvas.** While Diffusion writes, the chat shows the whole current block in faded text, after the text already
  written. The block is replaced at each step. Only the text that has been kept stays in the reply.
- **No choices for this model:** no graphics-card picker, no tuning, and no picture reading.
- **Not tested.** The app can show and load this model. We have not run the packaged app with a real model on a Mac.
  The sidecar's Python needs `venv` and `pip` from the packaged app, and that is not checked either.

## The API

The model uses the same chat endpoint as the other models, `POST /v1/chat/completions`. These fields go in the request,
next to `temperature`:

| Field | Default | What it does |
| --- | --- | --- |
| `temperature` | 0.8 | Sets the schedule of the denoising steps. The schedule runs from `0.8 × s` down to `0.4 × s`, where `s = temperature / 0.8` |
| `max_denoising_steps` | 48 | The most steps for one block |
| `entropy_bound` | 0.1 | The entropy budget that decides which positions are kept at each step (report Algorithm 1) |
| `stop_entropy` | 0.005 | Stops a block early when the mean entropy is at or below this and the top choices match the previous step |
| `seed` | 0 (random) | A fixed seed repeats a result. Zero or no value gives a new random seed each time |
| `mode` | `diffusion` | `diffusion` for blocks, or `ar` for one token at a time |
| `enable_thinking` | off | Thinking on or off. There are no levels |

In diffusion mode, `top_k`, `top_p` and the penalties are ignored. In `ar` mode the usual sampling keys apply.

**Live canvas over the API.** Ask for it with a Strata field on the request:

```json
{"model": "diffusiongemma-26b-a4b-it", "messages": [{"role": "user", "content": "Write a haiku."}],
 "stream": true, "strata": {"drafts": true}}
```

Each streamed chunk then has `delta.strata_draft`. That field holds the whole current canvas as text, not a change to
the last one. The text that has been kept still arrives in `delta.content` and `delta.reasoning_content`. Clients that do
not ask for drafts get the same output as before.

**Thinking** is switched with `enable_thinking`. The server puts the model's thinking marker in the system prompt when
it is on.

**Tool calls.** The model writes its own tool-call text. The server turns it into the usual tool calls on the OpenAI,
Anthropic and Responses routes. We have checked this with fixtures, not with the real model.

## Limits and known issues

From the report (section 10):

- **Stuttering.** A token can repeat.
- **Short answers.** The report notes that answers can be short.
- **Missing closing thought tag.** The report saw this in some prompts with pictures. Strata sends no pictures to this
  model, so we do not expect it here. Not tested.

Strata's own limits:

- No pictures.
- One request at a time.
- Context from 8K to 128K. Past 128K is not tested on a Mac.
- The report's speed is for a data-centre GPU. On 16 GB, a diffusion step touches about 84 of the 128 experts in each
  MoE layer (report page 17). The report does not show the speed of that on a 16 GB Mac. One-token mode may be faster
  there. Neither mode is measured by us.
- **The first block can take a while.** Diffusion mode shows no text until the first block of 256 tokens is done. The
  server window then says "the prompt is read ... writing the first tokens". Before this release the sidecar did the
  work over the 262,144-token vocabulary in numpy on the CPU, at every step; one user report saw no text after five
  minutes on a short prompt. It now does that work on the GPU in MLX, as mlx-vlm does. How long a block takes on a Mac
  is not measured by us.
- **One token at a time (`ar`) crashed before this release** on the first request. It now reads each next token
  from the model's decoder over a block of one token, after the text so far. We have not checked that this is the
  one-token mode of the report (section 3.5), or how good its answers are.
- **The 3-bit copy may be larger than setup's estimate.** One user's conversion log said "4.790 bits per weight", not
  about 3. Setup's 10-11 GB is an estimate; check the size of the folder after the conversion.

## Figures from others

No figure below was measured by Strata.

| Figure | Who measured it | Hardware and settings | Where |
| --- | --- | --- | --- |
| About 1,500 tokens/s in diffusion mode, about 20 tokens per forward pass | Google (the report) | One H100, FP8, batch 1 | Report section 3 |
| AIME 84.2 in one-token mode, 69.1 in the report's diffusion column | Google (the report) | One H100, FP8, batch 1 | Report Table 3 |
| LiveCodeBench 71.4 in one-token mode, 69.1 in the report's diffusion column | Google (the report) | One H100, FP8, batch 1 | Report Table 3 |
| 9.8 GB community build: 12.4 GB peak memory, about 1.5 tokens/s | A third party | A 16 GB Mac mini | A community report, not checked by us |

## Not tested yet

We have not:

- run the MLX engine on a Mac, or made a model copy there;
- measured memory, speed, steps or tokens per forward pass in either mode;
- run the app with a real model, or checked the packaged app's Python for the sidecar;
- checked whether 16 GB is enough for `mlx-q3` in practice, or what the optional memory step does;
- checked thinking and tool calls with the real model. The repository's tests for the sampler, the protocol and the parser
  use fake models and fixtures.

## How to report results

Follow [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md). For this model, please include:

- your Mac's chip, memory, macOS version and free disk;
- the size (`mlx-q3` or `mlx-q4`), the mode, the context and the Strata version;
- peak memory, tokens per second, and the steps, canvases and tokens-per-forward figures the monitor shows;
- the same prompts in both modes, on the same day, with the same build;
- what you did not test.

Label the result the way the guide asks:

> **Measured on:** Mac model and chip, memory, macOS version, Strata version, model size and mode, context, prompt.

Check [CONTRIBUTING.md](../CONTRIBUTING.md) first. Search the open issues, and start the title with its area, for example
`[Benchmark] DiffusionGemma mlx-q3 on a 16 GB Mac`.
