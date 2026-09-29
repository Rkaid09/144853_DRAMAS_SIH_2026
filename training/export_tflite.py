import os, sys, json
import numpy as np
import tensorflow as tf
from model import build_model

MODEL   = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".keras")
           else os.path.join("artifacts", "prahari_dscnn.keras"))
OUTDIR  = "artifacts"
DATA    = "data"
BUDGET  = 256 * 1024
FREE    = 60 * 1024

FOLD = "--keep-input-bn" not in sys.argv

FAST_HEAD = "--keep-mean" not in sys.argv


def affine_from_bn(bn):
    gamma, beta, mean, var = [np.asarray(w, dtype=np.float64)
                              for w in bn.get_weights()]
    if gamma.size != 1:
        sys.exit(f"input_bn has {gamma.size} channels, expected 1 - "
                 f"the scalar fold in this script does not apply")
    inv = 1.0 / np.sqrt(var + bn.epsilon)
    A = float(np.ravel(gamma * inv)[0])
    B = float(np.ravel(beta - mean * gamma * inv)[0])
    return A, B


def rebuild_for_export(model, A, B, Xcheck, folded, fast_head):
    stripped = build_model(input_bn=not folded, fast_head=fast_head)
    for layer in stripped.layers:
        if not layer.weights:
            continue
        try:
            src = model.get_layer(layer.name)
        except ValueError:
            sys.exit(f"layer '{layer.name}' is not in the trained model - "
                     f"model.py and the checkpoint have diverged. Retrain, "
                     f"or export with --keep-input-bn.")
        w = src.get_weights()
        if layer.name == "score" and w[0].ndim != layer.get_weights()[0].ndim:
            w = [w[0].reshape(1, 1, *w[0].shape)] + w[1:]
        layer.set_weights(w)

    ref = model.predict(Xcheck, verbose=0).ravel()
    xin = (A * Xcheck + B).astype("float32") if folded else Xcheck
    got = stripped.predict(xin, verbose=0).ravel()
    err = float(np.max(np.abs(ref - got)))
    print(f"equivalence     : max |export(x) - model(x)| = {err:.3e} "
          f"over {len(ref)} clips")
    if err > 1e-4:
        sys.exit("EXPORT IS NOT EQUIVALENT - refusing to write anything. "
                 "Re-run with --keep-input-bn --keep-mean and tell someone.")
    return stripped


def rep_dataset(X, n=500):
    idx = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    def gen():
        for i in idx:
            yield [X[i:i + 1].astype(np.float32)]
    return gen


def human(n):
    return f"{n:,} B ({n/1024:.1f} KB)"


def main():
    if not os.path.exists(MODEL):
        sys.exit(f"no model at {MODEL}")
    os.makedirs(OUTDIR, exist_ok=True)

    import glob as _glob
    _feat = os.path.join(os.environ.get("PRAHARI_DATA", r"E:\PRAHARI_DATA"), "features")
    _shards = sorted(_glob.glob(os.path.join(_feat, "*", "shard_0000.npy")))
    if _shards:
        _rng = np.random.default_rng(0)
        _parts = []
        for _f in _shards:
            _a = np.load(_f, mmap_mode="r")
            _k = min(len(_a), max(1, 4000 // len(_shards)))
            _parts.append(np.asarray(_a[np.sort(_rng.choice(len(_a), _k, replace=False))]))
        Xtr = np.concatenate(_parts).astype("float32")[..., None]
        print(f"calibration: {len(Xtr)} windows from {len(_shards)} prep_corpus sources")
    else:
        _xtr = np.load(f"{DATA}/X_train.npy", mmap_mode="r")
        _idx = np.sort(np.random.default_rng(0).choice(len(_xtr), size=min(4000, len(_xtr)), replace=False))
        Xtr = np.asarray(_xtr[_idx]).astype("float32")[..., None]
    Xte = np.load(f"{DATA}/X_test.npy").astype("float32")[..., None]
    yte = np.load(f"{DATA}/y_test.npy").astype("int32")
    model = tf.keras.models.load_model(MODEL)

    A, B, folded = 1.0, 0.0, False
    if FOLD and any(l.name == "input_bn" for l in model.layers):
        A, B = affine_from_bn(model.get_layer("input_bn"))
        print(f"input BN        : y = {A:.9f}*x + {B:.9f}  -> folded")
        folded = True
    elif FOLD:
        print("input BN        : not present in this checkpoint - nothing to fold")
    else:
        print("input BN        : kept in the graph (--keep-input-bn)")
    print("head            : " + ("AveragePooling2D, ESP-NN kernel"
                                  if FAST_HEAD else
                                  "GlobalAveragePooling2D -> MEAN (--keep-mean)"))

    if folded or FAST_HEAD:
        model = rebuild_for_export(model, A, B, Xte[:128], folded, FAST_HEAD)
    if folded:
        Xtr = (A * Xtr + B).astype("float32")
        Xte = (A * Xte + B).astype("float32")

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_dataset(Xtr)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()

    path = os.path.join(OUTDIR, "prahari_int8.tflite")
    open(path, "wb").write(blob)
    print(f"model file      : {human(len(blob))}   -> {path}")

    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    try:
        from collections import Counter
        ops = [interp._get_op_details(k)["op_name"]
               for k in range(interp._interpreter.NumNodes())]
        ops = [o for o in ops if o != "DELEGATE"]
        c = Counter(ops)
        print(f"\noperators       : {len(ops)} nodes, {len(c)} distinct "
              f"-> MicroMutableOpResolver<{len(c)}>")
        for name, n in sorted(c.items()):
            flag = "   <-- RUNTIME SHAPE, TFLM WILL REJECT THIS" if name in (
                "SHAPE", "STRIDED_SLICE", "PACK", "CONCATENATION") else ""
            print(f"  {n:>2} x {name}{flag}")
        print()
    except Exception as e:
        print(f"(could not list operators: {e})")
    print(f"input           : {inp['shape']} {inp['dtype'].__name__} "
          f"scale={inp['quantization'][0]:.7f} zero_point={inp['quantization'][1]}")

    s_in_q, z_in_q = inp["quantization"]
    eff_scale = s_in_q / A
    eff_zp    = z_in_q + B / s_in_q
    eff_fill  = int(np.clip(round(eff_zp), -128, 127))
    aff = {"folded": folded, "A": A, "B": B,
           "s_in": float(s_in_q), "z_in": int(z_in_q),
           "eff_scale": float(eff_scale), "eff_zp": float(eff_zp),
           "eff_fill": eff_fill}
    json.dump(aff, open(os.path.join(OUTDIR, "input_affine.json"), "w"), indent=2)
    print(f"device constants: scale {eff_scale:.9f}  zero_point {eff_zp:.6f}  "
          f"fill {eff_fill}   -> artifacts/input_affine.json")
    print(f"output          : {out['shape']} {out['dtype'].__name__} "
          f"scale={out['quantization'][0]:.7f} zero_point={out['quantization'][1]}")

    td = interp.get_tensor_details()
    def nbytes(t):
        if len(t["shape"]) == 0:
            return 0
        return int(np.prod(t["shape"]) * np.dtype(t["dtype"]).itemsize)

    sizes = {t["index"]: nbytes(t) for t in td}
    biggest = sorted(td, key=lambda t: -sizes[t["index"]])[:6]
    print("\nlargest tensors")
    for t in biggest:
        print(f"  {human(sizes[t['index']]):>22}  {str(t['shape']):<22} {t['name'][:44]}")

    peak, worst = 0, None
    try:
        n_ops = interp._interpreter.NumNodes()
        for i in range(n_ops):
            d = interp._get_op_details(i)
            live = sum(sizes.get(x, 0) for x in list(d["inputs"]) + list(d["outputs"]))
            if live > peak:
                peak, worst = live, d.get("op_name", f"op{i}")
    except Exception as e:
        print(f"\n(could not walk ops: {e}; falling back to 2x largest tensor)")
        peak = 2 * max(sizes.values())
        worst = "estimate"

    print(f"\nARENA LOWER BOUND : {human(peak)}   (driven by '{worst}')")
    print(f"device headroom   : {human(FREE)}")
    if peak < FREE * 0.6:
        print("VERDICT           : fits comfortably")
    elif peak < FREE:
        print("VERDICT           : fits, but tight - little room for anything else")
    else:
        print("VERDICT           : DOES NOT FIT - architecture must change")
        print("                    options: stride freq harder, fewer channels,")
        print("                    or streaming inference (caches state, not the")
        print("                    whole time axis)")

    print("\nquantisation cost")
    interp.resize_tensor_input(inp["index"], [1] + list(Xte.shape[1:]))
    interp.allocate_tensors()
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]
    q = np.empty(len(Xte), dtype=np.float32)
    for i in range(len(Xte)):
        x = np.clip(np.round(Xte[i:i+1] / s_in) + z_in, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        q[i] = (float(np.ravel(interp.get_tensor(out["index"]))[0]) - z_out) * s_out
    f = model.predict(Xte, verbose=0).ravel()

    a1 = tf.keras.metrics.AUC(); a1.update_state(yte, f)
    a2 = tf.keras.metrics.AUC(); a2.update_state(yte, q)
    print(f"  AUC float32     : {a1.result().numpy():.4f}")
    print(f"  AUC int8        : {a2.result().numpy():.4f}"
          f"   ({(a2.result().numpy()-a1.result().numpy())*100:+.2f} pts)")
    pos = yte == 1
    for t in (0.80, 0.90, 0.95):
        print(f"  t={t:.2f}  detect {(q[pos]>=t).mean()*100:5.1f}%   "
              f"FAR {(q[~pos]>=t).mean()*100:5.2f}%")

    carr = os.path.join(OUTDIR, "prahari_model.h")
    with open(carr, "w") as fh:
        fh.write("// Generated by export_tflite.py - do not edit\n")
        fh.write(f"// {len(blob)} bytes, int8 quantised\n")
        fh.write("#pragma once\n#include <cstdint>\n\n")
        fh.write("alignas(16) const unsigned char prahari_model_tflite[] = {\n")
        for i in range(0, len(blob), 12):
            fh.write("  " + ", ".join(f"0x{b:02x}" for b in blob[i:i+12]) + ",\n")
        fh.write("};\n")
        fh.write(f"const unsigned int prahari_model_tflite_len = {len(blob)};\n")
    print(f"\nC array         : {carr}")

    import shutil
    dst = os.path.join("..", "firmware", "core", "prahari_model.h")
    shutil.copyfile(carr, dst)
    print(f"copied to       : {dst}")
    print("\nNEXT, IN THIS ORDER:")
    print("  1. python gen_test_clip.py     regenerate the self-test expectations")
    print("     (the board compares against scores baked in at build time -")
    print("      a new model WILL fail the old ones, and that is correct)")
    print("  2. python arena_calc.py artifacts/prahari_int8.tflite")
    print("  3. check ARENA_BYTES in src/main.cpp still covers it")


if __name__ == "__main__":
    main()
