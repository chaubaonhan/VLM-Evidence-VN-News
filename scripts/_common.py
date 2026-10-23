"""Shared helpers for the command-line scripts."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from vlmstudy import core as V  # noqa: E402

MODELS = {
    # name used for the output folder: (wrapper kind, Hugging Face checkpoint)
    "qwen3vl8b": ("qwen3vl", "Qwen/Qwen3-VL-8B-Instruct"),
    "internvl35_8b": ("internvl", "OpenGVLab/InternVL3_5-8B-HF"),
}


def load_corpus(data_root, text_col=None, ref_npz=None):
    """Load the ten topic folders and assign the fixed 80/10/10 split (stratified, seed 42)."""
    return V.assign_split(V.load_corpus(data_root, text_col=text_col), ref_npz=ref_npz)


def ids_for(df, which):
    return df.global_id.tolist() if which == "all" else df.global_id[df.split == which].tolist()


class _Saved:
    """Stand-in for a loaded VLM so stored results can be analysed without a GPU."""

    def __init__(self, name):
        self.name = name


def saved_studies(df, out_root, names=tuple(MODELS)):
    """Study objects that only read results already written under out_root/<model>/."""
    from PIL import Image

    studies = []
    for name in names:
        if not os.path.isdir(os.path.join(out_root, name, "core")):
            continue
        st = V.Study.__new__(V.Study)
        st.vlm, st.df, st.out = _Saved(name), df.set_index("global_id", drop=False), os.path.join(out_root, name)
        st._img = lambda g, st=st: Image.open(st.df.at[g, "image_path"]).convert("RGB")
        studies.append(st)
    return studies
