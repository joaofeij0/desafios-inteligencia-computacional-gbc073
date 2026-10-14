"""
Desafio 3 — GBC073. Treino configurável e instrumentado.

Integrantes: 
Ana Alice Cordeiro de Souza - 12211BCC028
Anna Karolyna Pereira Santos - 12221BCC046
João Vitor Feijó Asevedo - 12311BCC061

Problema: classificação da categoria do produto a partir dos atributos da
transação (dataset tabular de e-commerce, ~1M linhas).

Uso típico:
  python treino.py --dados amazon_ecommerce_1M.csv --otim adamw --lr 3e-3 --agenda warmup_cosseno --semente 0

Cada execução:
  * grava logs/<id>.csv  (sinais a cada --log_cada passos)
  * acrescenta uma linha em experimentos.csv
"""
import argparse, csv, json, math, os, time, hashlib, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

AQUI = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Dados: leitura, pré-processamento e cache
# --------------------------------------------------------------------------
def preparar_dados(caminho, alvo, descartar, max_card, semente_divisao=1234):
    """Lê o CSV, separa colunas categóricas/numéricas, divide 70/15/15 e salva cache.
    A divisão usa semente FIXA (independe da semente do treino), igual para todos."""
    chave = hashlib.md5(json.dumps([os.path.abspath(caminho), alvo, sorted(descartar),
                                    max_card, semente_divisao]).encode()).hexdigest()[:10]
    cache = os.path.join(os.path.dirname(os.path.abspath(caminho)), f".cache_{chave}.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)

    df = pd.read_csv(caminho)
    df = df.drop(columns=[c for c in descartar if c in df.columns])
    y_txt = df.pop(alvo).astype(str)
    classes = sorted(y_txt.unique())
    y = y_txt.map({c: i for i, c in enumerate(classes)}).to_numpy(np.int64)

    cat_cols, num_cols, ignoradas = [], [], []
    for c in df.columns:
        s = df[c]
        nome = c.strip().lower().replace("-", "_")
        if nome == "id" or nome.endswith((" id", "_id")) or nome.startswith(("id ", "id_")):
            ignoradas.append(c)  # identificadores não são atributos
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            num_cols.append(c)
        else:
            # tenta data/hora → vira componentes numéricos
            if pd.api.types.is_string_dtype(s) or s.dtype == object:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dt = pd.to_datetime(s, errors="coerce")
                if dt.notna().mean() > 0.95:
                    df[c + "_mes"] = dt.dt.month.fillna(0)
                    df[c + "_dsem"] = dt.dt.dayofweek.fillna(0)
                    df[c + "_hora"] = dt.dt.hour.fillna(0)
                    num_cols += [c + "_mes", c + "_dsem", c + "_hora"]
                    ignoradas.append(c)
                    continue
            n = s.nunique()
            if n <= max_card:
                cat_cols.append(c)
            else:
                ignoradas.append(c)  # IDs e textos livres de alta cardinalidade

    num_cols = [c for c in num_cols if df[c].nunique() > 1]  # descarta constantes (ex.: hora da data)
    for c in list(num_cols):
        v = df[c].astype(float)
        if v.min() >= 0 and v.max() > 10 * (v.median() + 1):  # escala muito assimétrica
            df[c + "_log"] = np.log1p(v)
            num_cols.append(c + "_log")

    rng = np.random.default_rng(semente_divisao)
    idx = rng.permutation(len(df))
    n_tr, n_va = int(0.70 * len(df)), int(0.15 * len(df))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    # numéricas: preenche faltantes com mediana do treino e padroniza com estatísticas do treino
    Xn = df[num_cols].astype(np.float64).to_numpy() if num_cols else np.zeros((len(df), 0))
    med = np.nanmedian(Xn[tr], axis=0) if num_cols else np.zeros(0)
    Xn = np.where(np.isnan(Xn), med, Xn)
    mu, sd = Xn[tr].mean(0), Xn[tr].std(0) + 1e-8
    Xn = ((Xn - mu) / sd).astype(np.float32)

    # categóricas: índice 0 = desconhecido/raro
    Xc, cards = [], []
    for c in cat_cols:
        vals = df[c].astype(str)
        vocab = {v: i + 1 for i, v in enumerate(sorted(vals.iloc[tr].unique()))}
        Xc.append(vals.map(vocab).fillna(0).to_numpy(np.int64))
        cards.append(len(vocab) + 1)
    Xc = np.stack(Xc, 1) if Xc else np.zeros((len(df), 0), np.int64)

    pack = dict(
        num_cols=num_cols, cat_cols=cat_cols, ignoradas=ignoradas, cards=cards, classes=classes,
        **{f"{p}_{s}": torch.from_numpy(a[i]) for s, i in (("tr", tr), ("va", va), ("te", te))
           for p, a in (("xn", Xn), ("xc", Xc), ("y", y))},
    )
    torch.save(pack, cache)
    return pack


# --------------------------------------------------------------------------
# Modelo: embeddings categóricos + MLP residual com LayerNorm
# --------------------------------------------------------------------------
class Bloco(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 2 * d)
        self.fc2 = nn.Linear(2 * d, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.norm(x)))))


class ModeloTabular(nn.Module):
    def __init__(self, n_num, cards, n_classes, d=256, n_blocos=3, d_emb=16, p=0.1):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, min(d_emb, max(4, int(round(1.6 * c ** 0.56)))))
                                  for c in cards])
        d_in = n_num + sum(e.embedding_dim for e in self.emb)
        self.entrada = nn.Linear(d_in, d)
        self.blocos = nn.ModuleList([Bloco(d, p) for _ in range(n_blocos)])
        self.norm_final = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, xn, xc):
        partes = [xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)]
        h = self.entrada(torch.cat(partes, 1))
        for b in self.blocos:
            h = b(h)
        return self.head(self.norm_final(h))


def camadas_monitoradas(model):
    """Primeira, uma do meio e última camada (pesos)."""
    meio = len(model.blocos) // 2
    return ["entrada.weight", f"blocos.{meio}.fc1.weight", "head.weight"]


# --------------------------------------------------------------------------
# Muon (implementação própria: funciona em qualquer versão do PyTorch)
# --------------------------------------------------------------------------
@torch.no_grad()
def ortogonaliza(G, passos=5):
    """Newton–Schulz quíntico (Jordan, 2024): aproxima U V^T de G = U Σ V^T."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float() / (G.norm() + 1e-7)
    T = X.size(0) > X.size(1)
    if T:
        X = X.T
    for _ in range(passos):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if T else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momento (Nesterov) → ortogonalização → passo. Escala max(1, linhas/colunas)^0.5.
    Weight decay desacoplado. Só para matrizes 2-D das camadas ocultas."""

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, nesterov=True, ns_passos=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                      nesterov=nesterov, ns_passos=ns_passos))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                d = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                o = ortogonaliza(d, g["ns_passos"]) * max(1.0, p.size(0) / p.size(1)) ** 0.5
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(o, alpha=-g["lr"])


def criar_otimizadores(model, a):
    nomeados = list(model.named_parameters())
    if a.otim == "muon":
        muon_p = [p for n, p in nomeados if p.ndim == 2 and not n.startswith(("head", "emb"))]
        ids = {id(p) for p in muon_p}
        resto = [p for n, p in nomeados if id(p) not in ids]
        return [Muon(muon_p, lr=a.lr, momentum=0.95, weight_decay=a.wd),
                torch.optim.AdamW(resto, lr=a.lr_adamw, weight_decay=a.wd)]
    ps = [p for _, p in nomeados]
    if a.otim == "sgd":
        return [torch.optim.SGD(ps, lr=a.lr, momentum=0.9, weight_decay=a.wd)]
    if a.otim == "sgd_nesterov":
        return [torch.optim.SGD(ps, lr=a.lr, momentum=0.9, nesterov=True, weight_decay=a.wd)]
    if a.otim == "adam":
        return [torch.optim.Adam(ps, lr=a.lr)]  # Adam "puro": sem weight decay
    if a.otim == "adamw":
        return [torch.optim.AdamW(ps, lr=a.lr, weight_decay=a.wd)]
    if a.otim == "nadam":
        return [torch.optim.NAdam(ps, lr=a.lr, decoupled_weight_decay=True, weight_decay=a.wd)]
    raise ValueError(a.otim)


# --------------------------------------------------------------------------
# Agendas (multiplicador de η aplicado igualmente a todos os otimizadores)
# --------------------------------------------------------------------------
def fator_agenda(nome, T, warmup_frac=0.05, wsd_decai=0.2):
    W = max(1, int(warmup_frac * T))
    D = max(1, int(wsd_decai * T))

    def f(t):
        if nome == "constante":
            return 1.0
        if nome == "cosseno":
            return 0.5 * (1 + math.cos(math.pi * min(t, T) / T))
        if nome == "warmup_cosseno":
            if t < W:
                return (t + 1) / W
            return 0.5 * (1 + math.cos(math.pi * min(t - W, T - W) / (T - W)))
        if nome == "wsd":  # warmup → estável → decaimento linear nos últimos 20%
            if t < W:
                return (t + 1) / W
            if t < T - D:
                return 1.0
            return max(0.0, (T - t) / D)
        if nome == "onecycle":  # 30% subindo de η/25, depois cosseno até η/1e4
            S = int(0.3 * T)
            if t < S:
                return 1 / 25 + (1 - 1 / 25) * 0.5 * (1 - math.cos(math.pi * t / S))
            return 1e-4 + (1 - 1e-4) * 0.5 * (1 + math.cos(math.pi * min(t - S, T - S) / (T - S)))
        raise ValueError(nome)
    return f


# --------------------------------------------------------------------------
# Avaliação
# --------------------------------------------------------------------------
@torch.no_grad()
def avaliar(model, xn, xc, y, lote=16384):
    model.eval()
    perda, certos = 0.0, 0
    for i in range(0, len(y), lote):
        out = model(xn[i:i + lote], xc[i:i + lote])
        perda += F.cross_entropy(out, y[i:i + lote], reduction="sum").item()
        certos += (out.argmax(1) == y[i:i + lote]).sum().item()
    model.train()
    return perda / len(y), certos / len(y)


# --------------------------------------------------------------------------
# Treino
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True)
    ap.add_argument("--alvo", default="category")
    ap.add_argument("--descartar", nargs="*", default=["subcategory"],
                    help="colunas a remover (subcategory determina a categoria: vazamento)")
    ap.add_argument("--max_card", type=int, default=200, help="categóricas acima disso são ignoradas")
    ap.add_argument("--otim", default="adamw",
                    choices=["sgd", "sgd_nesterov", "adam", "adamw", "nadam", "muon"])
    ap.add_argument("--lr", type=float, default=1e-3, help="η (no Muon: η das matrizes)")
    ap.add_argument("--lr_adamw", type=float, default=1e-3, help="η da parte AdamW do Muon")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--agenda", default="warmup_cosseno",
                    choices=["constante", "cosseno", "warmup_cosseno", "wsd", "onecycle"])
    ap.add_argument("--passos", type=int, default=3000)
    ap.add_argument("--lote", type=int, default=1024)
    ap.add_argument("--clip", type=float, default=0.0, help="0 = sem clipping (norma é registrada igual)")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--log_cada", type=int, default=25)
    ap.add_argument("--val_cada", type=int, default=250)
    ap.add_argument("--largura", type=int, default=256)
    ap.add_argument("--blocos", type=int, default=3)
    ap.add_argument("--etapa", default="", help="rótulo livre: baseline, lr, agenda, confirmacao")
    ap.add_argument("--teste", action="store_true", help="avalia no TESTE no fim (use uma vez só!)")
    ap.add_argument("--saida", default=AQUI)
    a = ap.parse_args()

    torch.manual_seed(a.semente); np.random.seed(a.semente)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = preparar_dados(a.dados, a.alvo, a.descartar, a.max_card)
    xn_tr, xc_tr, y_tr = (D[k].to(dev) for k in ("xn_tr", "xc_tr", "y_tr"))
    xn_va, xc_va, y_va = (D[k].to(dev) for k in ("xn_va", "xc_va", "y_va"))

    model = ModeloTabular(xn_tr.shape[1], D["cards"], len(D["classes"]),
                          d=a.largura, n_blocos=a.blocos).to(dev)
    CAMADAS = camadas_monitoradas(model)
    params = dict(model.named_parameters())
    opts = criar_otimizadores(model, a)
    fator = fator_agenda(a.agenda, a.passos)
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, fator) for o in opts]

    rid = (f"{a.etapa + '_' if a.etapa else ''}{a.otim}_lr{a.lr:g}"
           f"{'_la' + format(a.lr_adamw, 'g') if a.otim == 'muon' else ''}_{a.agenda}"
           f"_p{a.passos}_s{a.semente}")
    os.makedirs(os.path.join(a.saida, "logs"), exist_ok=True)
    log_path = os.path.join(a.saida, "logs", rid + ".csv")
    log = []

    g = torch.Generator(device=dev).manual_seed(a.semente)
    n_tr = len(y_tr)
    t0 = time.time(); divergiu = False; perda0 = None
    melhor_val_acc, val_loss, val_acc = -1.0, float("nan"), float("nan")

    for t in range(a.passos):
        i = torch.randint(0, n_tr, (a.lote,), device=dev, generator=g)
        loss = F.cross_entropy(model(xn_tr[i], xc_tr[i]), y_tr[i])
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()

        registra = (t % a.log_cada == 0) or (t == a.passos - 1)
        if registra:
            with torch.no_grad():
                gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                                       if p.grad is not None)).item()
                theta = torch.sqrt(sum((p ** 2).sum() for p in model.parameters())).item()
                g_cam = {n: params[n].grad.norm().item() for n in CAMADAS}
                th_cam = {n: params[n].norm().item() for n in CAMADAS}
                antes = {n: params[n].detach().clone() for n in CAMADAS}
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        for o in opts:
            o.step()
        lrs = [o.param_groups[0]["lr"] for o in opts]
        for s in scheds:
            s.step()

        lv = loss.item() if registra else None
        if registra:
            with torch.no_grad():
                rho = {n: ((params[n] - antes[n]).norm() / (antes[n].norm() + 1e-12)).item()
                       for n in CAMADAS}
            if perda0 is None:
                perda0 = lv
            linha = dict(passo=t, perda=lv, grad=gnorm, theta=theta, lr=lrs[0],
                         lr_adamw=lrs[1] if len(lrs) > 1 else "",
                         **{f"grad/{n}": v for n, v in g_cam.items()},
                         **{f"theta/{n}": v for n, v in th_cam.items()},
                         **{f"rho/{n}": v for n, v in rho.items()},
                         val_perda="", val_acc="")
            if (not math.isfinite(lv)) or lv > 10 * max(perda0, 1e-3):
                divergiu = True
        if t % a.val_cada == 0 or t == a.passos - 1 or divergiu:
            val_loss, val_acc = avaliar(model, xn_va, xc_va, y_va)
            melhor_val_acc = max(melhor_val_acc, val_acc)
            if registra:
                linha.update(val_perda=val_loss, val_acc=val_acc)
            else:
                log.append(dict(passo=t, val_perda=val_loss, val_acc=val_acc))
        if registra:
            log.append(linha)
        if divergiu:
            print(f"[{rid}] DIVERGIU no passo {t} (perda={lv})")
            break
        if registra and t % (a.log_cada * 20) == 0:
            print(f"[{rid}] passo {t:5d} perda {lv:.4f} |g| {gnorm:.3g} |θ| {theta:.1f} "
                  f"η {lrs[0]:.2e} val_acc {val_acc:.4f}", flush=True)

    tempo = time.time() - t0
    pd.DataFrame(log).to_csv(log_path, index=False)

    teste_acc = ""
    if a.teste and not divergiu:
        xn_te, xc_te, y_te = (D[k].to(dev) for k in ("xn_te", "xc_te", "y_te"))
        _, teste_acc = avaliar(model, xn_te, xc_te, y_te)

    exp = os.path.join(a.saida, "experimentos.csv")
    linha = dict(id=rid, etapa=a.etapa, otim=a.otim, lr=a.lr,
                 lr_adamw=a.lr_adamw if a.otim == "muon" else "", wd=a.wd, agenda=a.agenda,
                 semente=a.semente, passos=a.passos, lote=a.lote, clip=a.clip,
                 largura=a.largura, blocos=a.blocos,
                 val_acc_final=("" if divergiu else val_acc),
                 val_perda_final=("" if divergiu else val_loss),
                 val_acc_melhor=melhor_val_acc, divergiu=divergiu, tempo_s=round(tempo, 1),
                 teste_acc=teste_acc, dispositivo=dev, log=os.path.relpath(log_path, a.saida),
                 comando="python treino.py " + " ".join(
                     f"--{k} {json.dumps(v) if isinstance(v, str) and ' ' in v else v}"
                     for k, v in vars(a).items()
                     if k not in ("saida", "teste", "descartar") and v != "") +
                 (" --descartar " + " ".join(json.dumps(c) for c in a.descartar) if a.descartar else ""))
    novo = not os.path.exists(exp)
    with open(exp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linha))
        if novo:
            w.writeheader()
        w.writerow(linha)
    print(f"[{rid}] fim: val_acc={val_acc:.4f} divergiu={divergiu} tempo={tempo:.0f}s")
    print(f"  numéricas={D['num_cols']}\n  categóricas={D['cat_cols']}\n"
          f"  ignoradas={D['ignoradas']}\n  classes={len(D['classes'])}")


if __name__ == "__main__":
    main()
