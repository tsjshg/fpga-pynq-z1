#!/usr/bin/env python3
"""SmolLM2-135M の重みを三値 (BitNet b1.58 と同じ符号化) に落とす。

  γ = mean(|W|)            … 丸めの尺度
  W̃ = clip(round(W/γ), -1, 1)   … -1 / 0 / +1

BitNet の論文はテンソル1個につき γ を1つ置く。ここでは
**出力チャネルごとに γ を持つ版**も測る。回路側の負担はゼロで
（int32 の累算が出たあとに CPU で掛けるだけ）、精度は上がるはず。

注意: BitNet b1.58 は **三値で学習したモデル**。
      普通に学習した重みを後から三値に丸めるのはまったく別の話で、
      品質は落ちる。どれくらい落ちるかをここで実測する。
"""
import numpy as np
from st_read import SafeTensors


def ternarize(W, per_row=True):
    """W (out, in) -> (三値 int8, 尺度 γ)。γ は per_row なら (out,1)、でなければ スカラー。"""
    A = np.abs(W)
    g = A.mean(axis=1, keepdims=True) if per_row else np.array([[A.mean()]])
    g = np.maximum(g, 1e-12)
    T = np.clip(np.rint(W / g), -1, 1).astype(np.int8)
    return T, g.astype(np.float32)


def report(W, T, g):
    """丸めでどれだけ形が変わったかを測る。"""
    R = g * T                                   # 復元した重み
    err = np.linalg.norm(W - R) / np.linalg.norm(W)
    cos = float((W * R).sum() / (np.linalg.norm(W) * np.linalg.norm(R) + 1e-30))
    z = float((T == 0).mean())
    return err, cos, z


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    st = SafeTensors(hf_hub_download("HuggingFaceTB/SmolLM2-135M", "model.safetensors"))

    print("層0 の各行列で、テンソル1個に γ 1つ（BitNet 論文どおり）と")
    print("出力チャネルごとに γ を持つ版を比べる。")
    print(f"\n{'行列':22s} {'形':14s} {'ゼロ率':>7s} {'相対誤差(全体γ)':>16s} {'相対誤差(行ごとγ)':>18s}")
    print("-"*84)
    names = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
             "mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
    for n in names:
        W = st.get(f"model.layers.0.{n}.weight").astype(np.float32)
        Tt, gt = ternarize(W, per_row=False); et, _, zt = report(W, Tt, gt)
        Tr, gr = ternarize(W, per_row=True);  er, _, zr = report(W, Tr, gr)
        print(f"{n:22s} {str(W.shape):14s} {zr:6.1%} {et:15.4f} {er:17.4f}")

    W = st.get("model.embed_tokens.weight").astype(np.float32)
    Tt, gt = ternarize(W, per_row=False); et, _, _ = report(W, Tt, gt)
    Tr, gr = ternarize(W, per_row=True);  er, _, zr = report(W, Tr, gr)
    print(f"{'embed_tokens(=lm_head)':22s} {str(W.shape):14s} {zr:6.1%} {et:15.4f} {er:17.4f}")

    # 全層の集計
    print("\n全 30 層・行ごと γ での集計:")
    tot = z_tot = 0; errs = []
    for l in range(30):
        for n in names:
            W = st.get(f"model.layers.{l}.{n}.weight").astype(np.float32)
            T, g = ternarize(W, per_row=True)
            e, c, z = report(W, T, g)
            errs.append(e); tot += T.size; z_tot += int((T == 0).sum())
    print(f"  行列 {len(errs)} 個 / 重み {tot/1e6:.1f} M")
    print(f"  ゼロになった重み: {z_tot/tot:.1%}")
    print(f"  相対誤差: 最小 {min(errs):.4f} / 中央 {np.median(errs):.4f} / 最大 {max(errs):.4f}")
    st.close()
