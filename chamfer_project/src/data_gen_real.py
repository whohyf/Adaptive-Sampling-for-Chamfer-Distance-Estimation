"""Real dataset generators matching Bakshi et al. 2023 Table 1 / Figures 3-4.

Paper datasets reproduced with actual public data:

  (a) ShapeNet  → ShapeNet Part HDF5 (Stanford, no registration, ~346 MB)
                  make_shapenet_part_pair()
                  Requires: pip install h5py

  (b) Text      → GloVe 300d word vectors (via gensim, ~376 MB)
                  + 20 Newsgroups documents (sklearn, ~20 MB)
                  Documents embedded as mean of GloVe word vectors.
                  make_glove_20news_pair()
                  Requires: pip install gensim scikit-learn

  (c) Outlier   → Inherently synthetic (planted outlier construction).
                  Use make_outlier_pair() from data_gen.py unchanged.

Each make_* returns (A, B) float64, or (None, None) on failure.
"""

import os
import re
import zipfile
import numpy as np

_CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data_cache")
)
os.makedirs(_CACHE_DIR, exist_ok=True)

# ShapeNet Part 16 categories (alphabetical = label index)
SHAPENET_PART_CATS = [
    "airplane", "bag", "cap", "car", "chair",
    "earphone", "guitar", "knife", "lamp", "laptop",
    "motorbike", "mug", "pistol", "rocket", "skateboard", "table",
]
SHAPENET_PART_URL = (
    "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"
)


# ============================================================
# ShapeNet Part — 3-D point clouds
# ============================================================

def download_shapenet_part(verbose=True):
    """Download and extract ShapeNet Part HDF5 from Stanford (~346 MB).

    No registration required.
    Returns path to the extracted hdf5_data/ directory, or None on failure.
    """
    zip_path    = os.path.join(_CACHE_DIR, "shapenet_part_seg_hdf5_data.zip")
    extract_dir = os.path.join(_CACHE_DIR, "shapenet_part_hdf5_data")
    hdf5_dir    = os.path.join(extract_dir, "hdf5_data")

    if os.path.isdir(hdf5_dir):
        return hdf5_dir

    if not os.path.exists(zip_path):
        if verbose:
            print(f"Downloading ShapeNet Part HDF5 from Stanford (~346 MB) ...")
            print(f"  {SHAPENET_PART_URL}")
        try:
            import requests
            resp = requests.get(SHAPENET_PART_URL, stream=True, timeout=600)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done  = 0
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(65536):
                    f.write(chunk)
                    done += len(chunk)
                    if verbose and total > 0 and done % (20 * 1024 * 1024) < 65536:
                        print(f"  {100*done/total:.0f}%  ({done/1e6:.0f}/{total/1e6:.0f} MB)",
                              flush=True)
            if verbose:
                print(f"  Saved {zip_path}")
        except Exception as e:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if verbose:
                print(f"  Download failed: {e}")
            return None

    if verbose:
        print("Extracting ShapeNet Part HDF5 ...")
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
    except Exception as e:
        if verbose:
            print(f"  Extraction failed: {e}")
        return None

    # The zip may place files directly or inside a subdirectory
    if os.path.isdir(hdf5_dir):
        return hdf5_dir
    # Fallback: look for any directory that contains .h5 files
    import glob
    for root, dirs, files in os.walk(extract_dir):
        if any(f.endswith(".h5") for f in files):
            return root
    return None


def _load_shapenet_part_cache(hdf5_dir, cat_id, verbose=False):
    """Load and cache all 2048-pt point clouds for one ShapeNet Part category.

    Returns (N, 2048, 3) float32 array.
    Requires h5py.
    """
    import glob

    cache_path = os.path.join(_CACHE_DIR, f"shapenet_part_cat{cat_id}.npz")
    if os.path.exists(cache_path):
        return np.load(cache_path)["pts"]

    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install h5py")

    h5_files = sorted(glob.glob(os.path.join(hdf5_dir, "*.h5")))
    all_pts = []
    for path in h5_files:
        with h5py.File(path, "r") as hf:
            # Try standard PointNet key names
            data  = hf.get("data")  or hf.get("points")
            label = hf.get("label") or hf.get("labels")
            if data is None or label is None:
                continue
            pts  = data[:].astype(np.float32)    # (N, 2048, 3)
            lbl  = label[:].flatten().astype(int) # (N,)
            mask = lbl == cat_id
            if mask.any():
                all_pts.append(pts[mask])

    if not all_pts:
        raise ValueError(f"No shapes found for ShapeNet Part category id={cat_id}. "
                         f"Checked {len(h5_files)} h5 files in {hdf5_dir}.")

    result = np.concatenate(all_pts, axis=0)  # (N_shapes, 2048, 3)
    np.savez_compressed(cache_path, pts=result)
    cat_name = SHAPENET_PART_CATS[cat_id] if cat_id < len(SHAPENET_PART_CATS) else str(cat_id)
    if verbose:
        print(f"  {cat_name} (id={cat_id}): {len(result)} shapes, "
              f"{len(result)*2048:,} vertices total")
    return result


def make_shapenet_part_pair(n=8000, high_cv=True, rng=None, verbose=False):
    """Real ShapeNet Part point clouds (Stanford HDF5, no registration).

    Mirrors make_shapenet_pair() in data_gen.py:
      high_cv=True  : B and 97% of A from 'chair' (cat_id=4);
                      3% of A from 'airplane' (cat_id=0) → elevates CV.
      high_cv=False : A and B both from 'chair' (low CV).

    n=8000 matches Bakshi Table 1 column (a): |A| = 8 000.
    Shapes are randomly sampled from the category's 2048-pt point clouds;
    each trial uses a different rng seed for variety.

    Returns (None, None) if download or h5py is unavailable.
    """
    rng = np.random.default_rng(rng)

    hdf5_dir = download_shapenet_part(verbose=verbose)
    if hdf5_dir is None:
        return None, None

    try:
        shapes_chair    = _load_shapenet_part_cache(hdf5_dir, cat_id=4, verbose=verbose)
        shapes_airplane = _load_shapenet_part_cache(hdf5_dir, cat_id=0, verbose=verbose)
    except (ImportError, ValueError) as e:
        if verbose:
            print(f"  Failed to load ShapeNet Part: {e}")
        return None, None

    def _sample_pool(shapes, k):
        """Sample k points uniformly from the pooled vertex cloud."""
        # shapes: (N_shapes, 2048, 3)
        flat = shapes.reshape(-1, 3)
        idx  = rng.choice(len(flat), size=k, replace=len(flat) < k)
        return flat[idx].astype(np.float64)

    B = _sample_pool(shapes_chair, n)
    if high_cv:
        n_near = int(0.97 * n)
        n_far  = n - n_near
        A = np.vstack([_sample_pool(shapes_chair,    n_near),
                       _sample_pool(shapes_airplane, n_far)])
    else:
        A = _sample_pool(shapes_chair, n)

    return A, B


# ============================================================
# GloVe 300d + 20 Newsgroups — text document embeddings
# ============================================================

def _load_glove_gensim(d=300, verbose=False):
    """Download GloVe word vectors via gensim (~376 MB for d=300).

    Returns a gensim KeyedVectors object, or None if gensim unavailable.
    """
    model_name = f"glove-wiki-gigaword-{d}"
    try:
        import gensim.downloader as api
    except ImportError:
        if verbose:
            print("  gensim not installed: pip install gensim")
        return None

    if verbose:
        print(f"  Loading GloVe {d}d word vectors via gensim "
              f"(~{376 if d==300 else d*1.25:.0f} MB, one-time download) ...")
    try:
        model = api.load(model_name)
        if verbose:
            print(f"  Loaded {len(model)} word vectors, d={d}")
        return model
    except Exception as e:
        if verbose:
            print(f"  gensim load failed: {e}")
        return None


def _embed_docs_glove(texts, glove_model, d):
    """Embed a list of documents as mean of GloVe word vectors.

    Returns (N, d) float64 array. Documents with no known words get zero vector.
    """
    vecs = np.zeros((len(texts), d), dtype=np.float64)
    for i, text in enumerate(texts):
        tokens = re.findall(r"[a-z]+", text.lower())
        word_vecs = []
        for t in tokens:
            try:
                word_vecs.append(glove_model[t])
            except KeyError:
                pass
        if word_vecs:
            vecs[i] = np.mean(word_vecs, axis=0)
    return vecs


def _get_glove_20news_embeddings(d=300, verbose=False):
    """Build and cache GloVe-embedded 20 Newsgroups document vectors.

    Returns (vecs, labels, cat_names) or (None, None, None) on failure.
    """
    cache_path = os.path.join(_CACHE_DIR, f"glove_20news_d{d}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return (data["vecs"].astype(np.float64),
                data["labels"],
                list(data["cat_names"]))

    glove = _load_glove_gensim(d=d, verbose=verbose)
    if glove is None:
        return None, None, None

    if verbose:
        print("  Fetching 20 Newsgroups corpus ...")
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError:
        if verbose:
            print("  scikit-learn not installed: pip install scikit-learn")
        return None, None, None

    news = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))

    if verbose:
        print(f"  Embedding {len(news.data)} documents with GloVe {d}d vectors ...")
    vecs = _embed_docs_glove(news.data, glove, d)

    # L2-normalise (matches word2vec convention)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms

    np.savez_compressed(cache_path,
                        vecs=vecs.astype(np.float32),
                        labels=np.array(news.target),
                        cat_names=np.array(news.target_names))
    if verbose:
        print(f"  Cached {len(vecs)} document embeddings → {cache_path}")
    return vecs.astype(np.float64), np.array(news.target), list(news.target_names)


def make_glove_20news_pair(n_a=2500, n_b=1800, d=300, high_cv=True,
                           rng=None, verbose=False):
    """Real text embeddings: GloVe 300d + 20 Newsgroups.

    Mirrors make_word2vec_like_text_pair() in data_gen.py with actual
    GloVe word vectors instead of the synthetic word2vec simulation.
    d=300 matches Bakshi Table 1 column (b).

    B             : documents from sports/recreation newsgroups.
    A (high_cv)   : 70% sports + 30% science (distant topics → high CV).
    A (low_cv)    : 100% sports (same pool as B → low CV).

    Returns (None, None) if gensim / sklearn unavailable.
    """
    rng = np.random.default_rng(rng)
    vecs, labels, cat_names = _get_glove_20news_embeddings(d=d, verbose=verbose)
    if vecs is None:
        return None, None

    sports_cats  = ["rec.sport.hockey", "rec.sport.baseball",
                    "rec.autos",        "rec.motorcycles"]
    science_cats = ["sci.space", "sci.med", "sci.electronics", "sci.crypt"]

    def _ci(cats):
        return [cat_names.index(c) for c in cats if c in cat_names]

    pool_sports  = vecs[np.isin(labels, _ci(sports_cats))]
    pool_science = vecs[np.isin(labels, _ci(science_cats))]

    def _sample(pool, k):
        return pool[rng.choice(len(pool), size=k, replace=len(pool) < k)]

    B = _sample(pool_sports, n_b)
    if high_cv:
        n_close = int(0.70 * n_a)
        A = np.vstack([_sample(pool_sports,  n_close),
                       _sample(pool_science, n_a - n_close)])
    else:
        A = _sample(pool_sports, n_a)

    return A.astype(np.float64), B.astype(np.float64)
