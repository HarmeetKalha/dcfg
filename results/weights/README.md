# Trainable-only weights (best validation epoch)

Each file holds ONLY the trained tensors (GenD's LayerNorms + head; for LUMEN2/uncond also sensor incl. BatchNorm stats, gate,
LN generator, and `const` for uncond). The frozen CLIP ViT-L/14 weights come from the public GenD checkpoint
`yermandy/GenD_CLIP_L_14` (HF revision 891ce014a0308386c4d7d25b3dcf436a22db5504). Load with
`dcfg.lumen2.evaluate_lumen2.load_lumen2(path)` (LUMEN2 / uncond, auto-detected) or `dcfg.gend.evaluate_gend --trainable` (GenD-aug).

| file | model | best epoch | val mean AUC | bytes |
|---|---|---|---|---|
| gend_aug_best_trainable.pt | GenD-aug | 16 | 0.9199 | 452,715 |
| lumen2_best_trainable.pt | LUMEN2 | 6 | 0.9305 | 7,449,899 |
| lumen2_uncond_best_trainable.pt | LUMEN-uncond | 6 | 0.9268 | 7,451,443 |

sha256: see `SHA256SUMS` (`sha256sum -c SHA256SUMS`).
