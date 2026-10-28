"""MNIST 训练数据完整性实验；只处理本机课程数据，不访问外部模型。"""
from __future__ import annotations

import os
# 限制小矩阵计算线程，降低不同机器上的调度开销与结果波动。
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import csv
import hashlib
import json
import platform
import struct
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

VERSION = 'mnist-label-poisoning-v2'
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / 'model_security_project' / 'data' / 'MNIST' / 'raw'
SEEDS = [17, 29, 43, 71, 101]
SPLIT_SEED = 20261025


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_idx(path: Path) -> np.ndarray:
    """核验 IDX 魔数、维度和文件长度；遇到坏数据直接失败，不回退到随机数据。"""
    with path.open('rb') as stream:
        zero, dtype, ndim = struct.unpack('>HBB', stream.read(4))
        if zero != 0 or dtype != 8 or ndim not in (1, 3):
            raise ValueError(f'Unexpected IDX header: {path.name}')
        shape = struct.unpack('>' + 'I' * ndim, stream.read(4 * ndim))
        payload = np.frombuffer(stream.read(), dtype=np.uint8)
    if payload.size != int(np.prod(shape)):
        raise ValueError(f'Incorrect IDX length: {path.name}')
    return payload.reshape(shape).copy()


def load_data(folder: Path, train_size: int = 10000, val_size: int = 2000):
    names = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
             't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
    images, labels, test_images, test_labels = [read_idx(folder / name) for name in names]
    assert images.shape == (60000, 28, 28) and test_images.shape == (10000, 28, 28)
    assert len(labels) == 60000 and len(test_labels) == 10000
    indices = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(indices, train_size=train_size,
                                         random_state=SPLIT_SEED, stratify=labels)
    val_idx, _ = train_test_split(rest_idx, train_size=val_size,
                                 random_state=SPLIT_SEED + 1, stratify=labels[rest_idx])
    assert not np.intersect1d(train_idx, val_idx).size

    def prepare(x):
        # 固定2×2均值池化，不从验证集或测试集拟合预处理参数。
        return (x.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)) / 255).astype(np.float32).reshape(-1, 196)

    return dict(X=prepare(images[train_idx]), y=labels[train_idx].astype(int),
                Xval=prepare(images[val_idx]), yval=labels[val_idx].astype(int),
                Xtest=prepare(test_images), ytest=test_labels.astype(int),
                train_idx=train_idx, val_idx=val_idx,
                source_hashes={name: sha256(folder / name) for name in names})


@dataclass(frozen=True)
class Setting:
    attack: str
    rate: float
    seed: int
    source: int = 3
    target: int = 8
    model: str = 'mlp'
    epochs: int = 60

    @property
    def key(self):
        return f'{self.model}_{self.attack}_{self.source}to{self.target}_{self.rate:g}_s{self.seed}_e{self.epochs}'


def corrupt(X, y, setting):
    rng = np.random.default_rng(setting.seed)
    pool = np.arange(len(y)) if setting.attack == 'indiscriminate' else np.flatnonzero(y == setting.source)
    order = rng.permutation(pool)
    n = int(np.floor(setting.rate * len(pool)))
    chosen = order[:n]
    Xp, yp = X.copy(), y.copy()
    if setting.attack == 'indiscriminate':
        # 共享候选顺序与错误标签，同seed的不同预算构成嵌套投毒集合。
        alternatives = (y[order] + rng.integers(1, 10, len(order))) % 10
        yp[chosen] = alternatives[:n]
    elif setting.attack == 'targeted':
        yp[chosen] = setting.target
    else:
        raise ValueError(setting.attack)
    assert np.count_nonzero(yp != y) == n
    assert np.array_equal(Xp, X)
    return Xp, yp, chosen, len(pool)


def evaluate(model, X, y, source, target):
    pred = model.predict(X)
    source_mask, other_mask = y == source, y != source
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average='macro')),
                asr=float(np.mean(pred[source_mask] == target)),
                source_recall=float(np.mean(pred[source_mask] == source)),
                other_accuracy=float(np.mean(pred[other_mask] == y[other_mask])),
                source_count=int(source_mask.sum()),
                target_errors=int(np.sum(pred[source_mask] == target))), pred


def fit_model(Xp, yp, data, setting):
    history = []
    if setting.model == 'logistic':
        model = LogisticRegression(C=1, solver='lbfgs', max_iter=1500, random_state=setting.seed)
        model.fit(Xp, yp)
        assert model.n_iter_.max() < 1500
        return model, history
    model = MLPClassifier(hidden_layer_sizes=(96,), activation='relu', solver='adam',
                          alpha=1e-4, batch_size=128, learning_rate_init=.001,
                          shuffle=True, random_state=setting.seed, early_stopping=False)
    for epoch in range(1, setting.epochs + 1):
        # partial_fit每次完整遍历一次训练集；所有配置固定轮数，测试结果不参与选模。
        model.partial_fit(Xp, yp, classes=np.arange(10))
        if epoch == 1 or epoch % 5 == 0 or epoch == setting.epochs:
            p = model.predict(data['Xval'])
            mask = data['yval'] == setting.source
            history.append(dict(epoch=epoch, train_loss=float(model.loss_),
                                val_accuracy=float(np.mean(p == data['yval'])),
                                val_asr=float(np.mean(p[mask] == setting.target))))
    return model, history


def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plan(epochs):
    plan = []
    for seed in SEEDS:
        for rate in [0, .07, .1, .2, .4, .6]:
            plan.append(Setting('indiscriminate', rate, seed, epochs=epochs))
        for rate in [.1, .3, .5, .7]:
            plan.append(Setting('targeted', rate, seed, epochs=epochs))
        plan.append(Setting('targeted', .5, seed, source=0, target=1, epochs=epochs))
        for rate in [0, .5]:
            plan.append(Setting('targeted', rate, seed, model='logistic', epochs=epochs))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    parser.add_argument('--smoke', action='store_true', help='仅执行短流程检查，不用作正式结果')
    args = parser.parse_args()
    out = args.out.resolve()
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    data = load_data(args.data_dir, train_size=2000 if args.smoke else 10000)
    epochs = 3 if args.smoke else 60
    plan = [Setting('indiscriminate', 0, 17, epochs=epochs),
            Setting('targeted', .5, 17, epochs=epochs)] if args.smoke else make_plan(epochs)
    fingerprint = hashlib.sha256(json.dumps(dict(version=VERSION, hashes=data['source_hashes'],
                                  train=data['train_idx'].tolist(), val=data['val_idx'].tolist(),
                                  script=sha256(Path(__file__)), python=platform.python_version(),
                                  numpy=np.__version__, sklearn=sklearn.__version__), sort_keys=True).encode()).hexdigest()
    start = time.perf_counter()
    rows = []
    log_path = out / 'execution.log'

    def log(text):
        print(text, flush=True)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(text+'\n')

    log(f'RUN {datetime.now(timezone.utc).isoformat()} version={VERSION} smoke={args.smoke}')
    log(f'train={len(data["y"])} validation={len(data["yval"])} test={len(data["ytest"])} epochs={epochs}')
    log(f'train_counts={np.bincount(data["y"]).tolist()} test_counts={np.bincount(data["ytest"]).tolist()}')
    with threadpool_limits(limits=1):
        for i, setting in enumerate(plan, 1):
            run_file = out / 'runs' / (setting.key+'.json')
            cached = json.loads(run_file.read_text(encoding='utf-8')) if run_file.exists() else None
            if cached and cached.get('fingerprint') == fingerprint:
                row = cached['metrics']
                log(f'[{i}/{len(plan)}] cached {setting.key}')
            else:
                run_start = time.perf_counter()
                Xp, yp, chosen, pool_size = corrupt(data['X'], data['y'], setting)
                model, history = fit_model(Xp, yp, data, setting)
                metrics, pred = evaluate(model, data['Xtest'], data['ytest'], setting.source, setting.target)
                row = dict(key=setting.key, **asdict(setting), n_poison=len(chosen), pool_size=pool_size,
                           pool_rate=len(chosen)/pool_size, global_rate=len(chosen)/len(yp),
                           seconds=time.perf_counter()-run_start, **metrics)
                np.savez_compressed(out / 'runs' / (setting.key+'.npz'),
                                    pred=pred, ytest=data['ytest'],
                                    poisoned_local_indices=chosen, original_labels=data['y'][chosen],
                                    poisoned_labels=yp[chosen],
                                    confusion=confusion_matrix(data['ytest'], pred, labels=np.arange(10)))
                run_file.write_text(json.dumps(dict(fingerprint=fingerprint, metrics=row, history=history),
                                               indent=2, ensure_ascii=False), encoding='utf-8')
                log(f'[{i}/{len(plan)}] {setting.key} n={len(chosen)} '
                    f'acc={row["accuracy"]:.4%} ASR={row["asr"]:.4%} time={row["seconds"]:.1f}s')
            rows.append(row)
            save_csv(out / 'runs.csv', rows)
    np.savez_compressed(out / 'split_indices.npz', train=data['train_idx'], validation=data['val_idx'])
    metadata = dict(version=VERSION, fingerprint=fingerprint, smoke=args.smoke,
                    python=platform.python_version(), python_executable=os.sys.executable,
                    numpy=np.__version__, sklearn=sklearn.__version__, platform=platform.platform(),
                    seeds=SEEDS, split_seed=SPLIT_SEED, train=len(data['y']), validation=len(data['yval']),
                    test=len(data['ytest']), train_counts=np.bincount(data['y']).tolist(),
                    test_counts=np.bincount(data['ytest']).tolist(), data_hashes=data['source_hashes'],
                    epochs=epochs, n_runs=len(rows), elapsed_seconds=time.perf_counter()-start,
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    model=dict(hidden_layer_sizes=[96], activation='relu', alpha=.0001,
                               solver='adam', batch_size=128, learning_rate_init=.001),
                    protocol='Fixed split, 5 paired poison/model seeds, no test-based tuning, mean and sample SD.')
    (out / 'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    log(f'COMPLETE {len(rows)} fitted models, elapsed={metadata["elapsed_seconds"]:.1f}s')


if __name__ == '__main__':
    main()
