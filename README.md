# Mucha, która liczy - część „model” (Python / PyTorch)

Sieć neuronowa o strukturze połączeń z mózgu muszki owocówki (connectome), uczona dodawania i odejmowania mod 64
na 6 „nogach” (bit = noga). Ten katalog zawiera **część uczenia maszynowego**; wizualizacja w C++/OpenGL jest osobno
i łączy się z tą częścią wyłącznie przez **format JSON epizodu** (patrz niżej).

> **Stan na 8 listopada 2026:** generator, podziały, grafy kontrolne, modele odniesienia, pętla treningowa i narzędzia analizy **działają**. **Rate (`rate_best`)** na małym grafie: `best.pt` test_pairs **0,757 ± 0,145** (5 ziaren; 0,819 ± 0,049 dla czterech, które wystartowały),
> GRU ≈ 0,98. Poprawę daje głównie weight decay na `gain` (`gain` rośnie w treningu z 8 do ok. 13-14,6). Wyniki główne GRU nie zależą od podziału danych (E26); `init_only` uczą LSTM (0,78) i GRU 128 (0,80).
> Wyniki: [`docs/eksperymenty.md`](docs/eksperymenty.md) (E0-E30). Na punkt kontrolny: [`docs/skrypt_punkt_kontrolny.md`](docs/skrypt_punkt_kontrolny.md) i slajdy. Następny etap: [`docs/plan_connectome.md`](docs/plan_connectome.md).

---

## 1. Szybki start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # torch: wersję pod Wasz sprzęt wybierzcie na pytorch.org

python -m pytest                    # wszystkie testy (generator, grafy, modele, podsumowania)
python -m experiments.xor_demo     # mini-eksperyment: liniowy ~0.693 vs MLP rozwiązuje XOR

python -m train.train --config experiments/configs/gru.yaml          # pierwszy trening (GRU)
python -m train.train --config experiments/configs/rate_small.yaml   # model Rate na małym grafie
```

Wszystkie polecenia uruchamiamy **z katalogu głównego repo** (`python -m ...`).

---

## 2. Co jest w repo

```
fly-arith/
├── README.md                  ten plik
├── requirements.txt
├── task/
│   └── generator.py           ZADANIE: bity, mod 64, przeniesienia, ekran LED, podziały, epizody, eksport JSON (numpy)
├── graphs/
│   └── spec.py                GraphSpec + mały losowy graf warstwowy + 4 grafy kontrolne (numpy)
├── models/
│   ├── baselines.py           Linear, MLP, Elman RNN, GRU, LSTM, mały Transformer
│   ├── rate.py                model Rate na zamrożonym grafie (rzadka macierz, BPTT)
│   └── __init__.py            fabryka modeli, liczenie parametrów, dopasowanie rozmiarów (krok 16)
├── train/
│   └── train.py               pętla treningowa (BPTT, Adam, logi CSV, checkpointy, ewaluacja per podział)
├── experiments/
│   ├── configs/*.yaml         linear, mlp, elman, gru, lstm, transformer, rate_small
│   ├── xor_demo.py            demonstracja: model liniowy vs MLP na XOR
│   ├── diagnose_rate.py       diagnostyka Rate: aktywność tanh, gradienty, zapamiętanie jednego batcha
│   ├── copy_test.py           test liniowy dla Rate (kopiowanie bitów X; wymaga kodowania binary)
│   └── run_seeds.sh           ten sam config na ziarnach 0..4
├── scripts/
│   ├── export_episode.py      zapis epizodu do JSON (dla odtwarzacza w C++)
│   ├── predict_episode.py     model + epizod JSON -> JSON z wypełnionymi predykcjami
│   ├── count_params.py        dopasowanie liczby parametrów modeli do modelu Rate
│   ├── summarize_runs.py      zbiorcza tabela wyników: średnia ± odchylenie po ziarnach (+ per bit)
│   ├── error_analysis.py      błędy a przeniesienia/operacja/długość łańcucha (RQ5)
│   ├── bit_order.py           w którym kroku treningu każdy bit przekracza próg dokładności (RQ5)
│   └── calibrate_gain.py      dobór gain Rate bez treningu: nasycenie, czułość na instrukcję vs historię, promień spektralny
├── tests/
│   ├── test_generator.py      13 testów (numpy)  - wszystkie 4096 par, przeniesienia, podziały, szum, tryby nóg
│   ├── test_graph.py          8 testów (numpy)   - role neuronów, normalizacja, grafy kontrolne
│   ├── test_models.py         testy modeli (PyTorch) - kształty, gradienty, przyczynowość, dopasowanie parametrów, input_scale
│   ├── test_summarize.py      testy skryptu zbiorczego (3)
│   └── test_analysis.py       testy analizy błędów i kolejności bitów (5)
└── docs/
    ├── hipotezy.md            RQ1-RQ7 + hipotezy (SZKIC - zatwierdźcie przed pierwszym eksperymentem)
    ├── decyzje.md             decyzje projektowe D1-D16 i punkty [OTWARTE]
    ├── plan_connectome.md     plan przejścia z małego grafu na prawdziwy connectome (bramka, dane, ryzyka)
    ├── eksperymenty.md        DZIENNIK EKSPERYMENTÓW: co uruchomiono, jakie polecenia, jakie liczby, jakie wnioski
    └── przyklad_epizodu.json  przykładowy epizod (16 kroków) w formacie dla kolegi
```

---

## 3. Zadanie w skrócie

- **Stan** = liczba 0..63 na 6 nogach (bit `i` = noga `i`, wartość 2^i). **Instrukcja** `+X` / `-X`, nowy stan `(stan ± X) mod 64`.
- **Ekran LED 8×8 = 64 piksele.** Domyślnie `onehot_signed`: piksel o indeksie X zapalony, wartość +1 (dodawanie) albo −1 (odejmowanie).
  Alternatywa `binary` (X binarnie na pikselach 0-5, piksel 6 = odejmowanie).
- **Epizod** = 16 instrukcji. W każdym kroku model widzi ekran i położenie nóg, a ma wypisać 6 bitów nowego stanu.
- **Tryby położenia nóg** (`task.legs_mode`): `teacher` (prawdziwy stan w każdym kroku; domyślny) i `init_only` (stan tylko w kroku 0, potem sieć musi go pamiętać).
- Każdy bit wyniku = **XOR** bitu stanu, bitu X i przeniesienia/pożyczki z niższych bitów. Dlatego model liniowy jest
  bezradny: **zmierzone tu (regresja logistyczna w numpy, 2000 epizodów): strata ≈ 0,69 dla każdego bitu** (ln 2 = 0,693).

### Podziały danych (RQ4)

| Podział | Co wolno | Po co |
|---|---|---|
| `train` | pary (stan, X) niewyłączone; stan docelowy nie jest wyłączonym stanem | trening (epizody losowane na bieżąco) |
| `val` | 5% niewidzianych par | wybór checkpointu i hiperparametrów |
| `test_pairs` | 10% niewidzianych par | uogólnianie na nowe pary |
| `test_x` | 6 wyłączonych wartości X | uogólnianie na nowe X |
| `test_states` | 6 wyłączonych stanów | uogólnianie na nowe stany |

Szczegóły (maska `measure`, wypełniacze łańcucha) w `docs/decyzje.md` (D3, D4) i w docstringu klasy `Splits`.

---

## 4. Jak uruchomić model

### 4.1 Pojedynczy trening

```bash
python -m train.train --config experiments/configs/gru.yaml
```

Nadpisania z linii poleceń (dowolny klucz z YAML, notacja z kropką):

```bash
python -m train.train --config experiments/configs/gru.yaml --set seed=3 model.hidden=96 train.steps=10000
python -m train.train --config experiments/configs/gru.yaml --set task.legs_mode=init_only
python -m train.train --config experiments/configs/gru.yaml --set task.screen_noise=0.3
python -m train.train --config experiments/configs/rate_small.yaml --set model.graph.control=no_signs
python -m train.train --config experiments/configs/gru.yaml --set train.lr_schedule=cosine   # spadek lr do 0
python -m train.train --config experiments/configs/rate_small.yaml --set model.input_scale=4  # większa skala początkowych wag wejścia
```

**Uwaga o kodowaniu ekranu:** `onehot_signed` jest szybszy w nauce, ale **nie nadaje się do testu nowych X** (`test_x` ≈ zgadywanie, zob. `docs/eksperymenty.md`, E5).
Do podziału `test_x` używajcie `--set task.screen_encoding=binary`.

Na ekranie: co `eval_every` kroków strata i dokładność całej liczby na `val` oraz dokładność na `test_pairs/test_x/test_states`.
**Punkt odniesienia: strata 0,693 = zgadywanie.** Jeśli model na `val` stoi na 0,69 - nic się nie nauczył.

### 4.2 Dostępne modele (`model.name`)

| name | Opis | Pamięć między krokami |
|---|---|---|
| `linear` | jedna warstwa liniowa na kroku; **oracle zgadywania** (spodziewane ≈0,693) | nie |
| `mlp` | MLP (`hidden`, `depth`) na kroku | nie |
| `elman`, `gru`, `lstm` | RNN/GRU/LSTM (`hidden`) + głowica liniowa | tak |
| `transformer` | mały Transformer, maska przyczynowa po krokach (`hidden`=d_model, `n_layers`, `n_heads`) | przez uwagę |
| `rate` | sieć rekurencyjna na zamrożonym grafie; uczone: wejście, czucie, odczyt, α, progi, wzmocnienie | tak (stan neuronów) |

Wspólny interfejs: `model(screens (B,T,64), legs (B,T,6)) -> logity (B,T,6)`. Nowy model = nowa klasa + wpis w `models/__init__.py`.

### 4.3 Model Rate i grafy kontrolne

`rate_small.yaml` używa małego losowego grafu warstwowego (~3 600 neuronów, ~95 000 krawędzi). Graf kontrolny wybieracie
przez `model.graph.control`: `none | degree_preserving | shuffled_weights | no_signs | random_signs` (interpretacja w `docs/decyzje.md`, D9).
Seed grafu: `model.graph.seed`, seed kontroli: `model.graph.control_seed`.

### 4.4 Pięć ziaren jednego modelu (krok 21)

```bash
bash experiments/run_seeds.sh gru
bash experiments/run_seeds.sh rate_small --set model.graph.control=degree_preserving
```

### 4.5 Co powstaje: `runs/<name>/seed<N>/`

`config.yaml` (pełny config przebiegu), `metrics.csv` (strata/dokładności per podział w czasie), `best.pt` (najlepszy po `val`),
`last.pt`, `results.json` (końcowe metryki + liczba parametrów + czas).

### 4.6 Dopasowanie rozmiarów modeli (krok 16)

```bash
python -m scripts.count_params --config experiments/configs/rate_small.yaml
```
Wypisuje budżet parametrów modelu Rate i `hidden` dla MLP/Elman/GRU/LSTM/Transformera. Wpiszcie wyniki do configów.
Uwaga: `hidden` w dołączonych configach policzyłem analitycznie pod domyślny graf `rate_small` (24 391 parametrów, ±10%); zweryfikujcie tym skryptem.

### 4.7 Odpowiedź modelu dla kolegi (wizualizacja)

```bash
python -m scripts.export_episode --out epizod.json --seed 1 --split test_pairs
python -m scripts.predict_episode --run runs/gru/seed0 --episode epizod.json --out odpowiedz.json
```

---

## 5. Format epizodu JSON (kontrakt z wizualizacją)

Przykład: `docs/przyklad_epizodu.json` (jeden krok na linię).

```json
{
 "format_version": 1,
 "meta": {"split": "train", "modulus": 64, "n_bits": 6, "bit_order": "bit i = noga i, wartość 2^i", "task_config": {...}},
 "steps": [
  {"t": 0, "instruction": "+25", "op": "+", "x": 25,
   "screen": [64 liczb, wiersz po wierszu, 8x8],
   "state_before": 23, "legs_before": [1,1,1,0,1,0],
   "target_state": 48, "target_bits": [0,0,0,0,1,1],
   "predicted_bits": null, "correct": null}
 ]
}
```
`predicted_bits` i `correct` wypełnia `scripts/predict_episode.py`. Jeśli kolega potrzebuje innych pól (np. aktywności neuronów),
dodajemy je w `episode_to_dict` i podbijamy `format_version`.

---

## 6. Status względem listy 30 kroków

| Krok | Co | Status |
|---|---|---|
| 1 | Przegląd literatury | **do zrobienia** (Wy) |
| 2 | Pytania badawcze i hipotezy | szkic w `docs/hipotezy.md` - **zatwierdźcie przed eksperymentami** |
| 3 | Generator zadania | **gotowe, przetestowane** |
| 4 | Podziały danych | **gotowe, przetestowane** |
| 5 | Testy jednostkowe na 4096 parach | **gotowe** (obie operacje, przeniesienia, XOR) |
| 6 | Odtwarzacz epizodów w C++ | kolega (kontrakt: JSON z pkt 5) |
| 7 | Wczytanie connectome + znaki | **do zrobienia** (`GraphSpec` gotowy na dane; `build_graph` rzuca `NotImplementedError` dla connectome) |
| 8 | Wybór neuronów wejścia/wyjścia/czucia | **do zrobienia** (role są polami `GraphSpec`) |
| 9 | Mały losowy graf | **gotowe, przetestowane** |
| 10 | Pętla treningowa | **gotowa, działa** (GRU na GPU) |
| 11 | Trening Rate na małym grafie | `rate_best` 5 ziaren: 4 z 5 w zakresie 0,78-0,87, jedno późny start, jedno zepsute skokiem w ostatnim kroku; poprawę daje decay na `gain` (E28-E29); do zrobienia: ocena `best.pt` |
| 12-15 | Elman, GRU, LSTM, MLP, Transformer | **zrobione** (5 ziaren, oba kodowania, + linear jako kontrola) |
| 16 | Dopasowanie liczby parametrów | skrypt gotowy, nieuruchomiony (`hidden` w configach policzone analitycznie; GRU 24 126 vs Rate 24 391 parametrów) |
| 17 | Wizualizacja chmury neuronów | kolega |
| 18 | Grafy kontrolne | **gotowe, przetestowane** (numpy) |
| 19 | Trening na pełnym connectome | **do zrobienia** (po krokach 7-8; uwaga na pamięć, zob. `models/rate.py`) |
| 20 | Rate na grafach kontrolnych | konfiguracja gotowa (`model.graph.control`); **czeka na działający Rate** |
| 21 | Wiele ziaren | `run_seeds.sh` + `summarize_runs.py`; GRU zrobione (5 ziaren) |
| 22 | Ewaluacja na niewidzianych danych | wbudowana w trening (E9); analiza błędów a bity/przeniesienia uruchomiona (E15, E19) |
| 23 | Długość sekwencji i szum | GRU zrobione (E11, E13, E20): długość bez wpływu w trybie teacher; szum 0,2 na `binary` 0,74, na `onehot_signed` 0,22; `init_only` niezbieżne |
| 24 | Ablacje | częściowo (`screen_encoding`, `legs_mode`, `init_gain`); reszta **do zrobienia** |
| 25 | Analiza wnętrza sieci | **do zrobienia** |
| 26 | Serwer WebSocket | **do zrobienia** (`predict_episode.py` to baza) |
| 27 | Wydajność rysowania | kolega |
| 28 | Zamrożenie modelu, epizody demo | **do zrobienia** (narzędzia gotowe) |
| 29 | README i skrypty tabel | README i `summarize_runs.py` (tabele z `runs/`) są; skrypt odtwarzający wszystkie biegi **do zrobienia** |
| 30 | Test przez osobę spoza zespołu | **do zrobienia** |

### Co zostało sprawdzone, a co nie

- **Uruchomione i działające (u Was):** testy; `xor_demo`; wszystkie modele odniesienia po 5 ziaren; ablacje GRU; `diagnose_rate`, `copy_test`, `summarize_runs`, `bit_order`, `error_analysis`; trening Rate z `reset_each_step`.
- **Najważniejsze wyniki:** na `binary` wszystkie rodziny uogólniają na nowe pary, X i stany (0,97-1,00; Transformer 1,000); na `onehot_signed` nowe X = zgadywanie (artefakt kodowania);
  bit 5 (najstarszy) jest najtrudniejszy, bity 0-4 uczą się jednocześnie; `Rate` bezstanowy uczy się częściowo (0,47), ze stanem się zatrzaskuje.
- **Uruchomione:** `train.wd_exclude` (E25).
- **Jeszcze nie ma:** wyników na prawdziwym connectome, grafów kontrolnych w treningu, analizy RQ6, trybu z pamięcią dla Rate, wielu ziaren dla Rate.

---

## 7. Następne kroki: komendy do terminala

Wszystkie z katalogu głównego repo, w aktywowanym `.venv`.

**A. Mechanizm `gain` bez treningu (kilka sekund)**
```bash
for r in rate_reset_10k rate_wd005_excl rate_wd_gain_only rate_reset_wd; do python -m scripts.eval_checkpoints "runs/$r/seed0" --ckpt last.pt | tail -1; done
```
Porównuje `gain` po treningu: bez decay (`rate_reset_10k`), z decay bez `gain` (`rate_wd005_excl`), tylko `gain` (`rate_wd_gain_only`), wszystkie (`rate_reset_wd`).

**B. Opcjonalnie: stabilniejsze 5 ziaren (ok. 21 min na ziarno przy wolnym GPU)**
```bash
bash experiments/run_seeds.sh rate_best name=rate_best_cos10k train.steps=10000 train.lr_schedule=cosine
python -m scripts.eval_checkpoints "runs/rate_best_cos10k/seed*" --ckpt best.pt
```

**C. Connectome** - po punkcie kontrolnym, zgodnie z `docs/plan_connectome.md`.

**D. Commit**
```bash
git add -A && git commit -m "punkt kontrolny: rate_best best.pt, slajdy i skrypt"
```
