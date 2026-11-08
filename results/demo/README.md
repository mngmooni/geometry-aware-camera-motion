# Camera Motion — Synchronized Three-Model Demonstrations

Two clips use the existing, previously rendered videos for the **same held-out shot condition**, synchronized frame-by-frame and shown in three columns. Left: relative pose FM; middle: body-frame increment FM; right: SO(3) Riemannian FM.

- `E1a_three_model_synchronized.mp4`: four orbit-mode examples, same condition, non-paired stochastic samples.
- `E1b_three_model_synchronized.mp4`: GT camera pan/roll versus each model’s generated example. Views are from the original source video, with GT on top and generated example below. Same condition and route, independent/non-paired samples.

These demos are **qualitative**. They are not a viewer study, proof of cinematic quality, nor a numerical evaluator. Full m2 metrics are in the benchmark. No model retraining, rerendering or changes to trajectory paths were performed for this composite; the source MP4s were resized, labeled and combined only.
