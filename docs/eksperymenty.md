# Dziennik eksperymentów

Stan na **2026-11-04** (po drugiej serii eksperymentów). Wszystkie liczby pochodzą z logów wklejonych z konsoli (uruchomienia lokalne na GPU, `device=cuda`);
nic nie zostało uzupełnione z pamięci. Jeśli czegoś tu nie ma, to znaczy, że tego jeszcze nie uruchamialiśmy.

**Konwencje.** *exact* = dokładność całej liczby (wszystkie 6 bitów poprawne), liczona tylko na krokach z maską `measure`.
Zgadywanie: strata ln 2 = 0,693; exact ≈ 1/64 ≈ 0,016. Wartości „końcowe” to ostatni krok treningu, nie `best.pt`.
Wszystkie biegi używają tego samego podziału danych (`task.split_seed=0`; wyłączone X: 2, 16, 19, 31, 38, 50;
wyłączone stany: 30, 37, 40, 45, 53, 60) i tych samych zbiorów ewaluacyjnych (512 epizodów na podział).

---

## E0. Środowisko i testy
- `python -m pytest`: 33 testy przechodzą (generator 13, grafy 8, modele 12). Środowisko: `.venv` projektu, PyTorch z CUDA.
- Ostrzeżenia (niegroźne) usunięte w tej wersji: `float(loss)` → `loss.item()`, `check_invariants=False` dla macierzy rzadkiej.
- Po tej aktualizacji dochodzą testy `test_summarize` (3) i `test_rate_input_scale...` (1); nowe pliki jeszcze nieuruchamiane na GPU.

## E1. Demo XOR (`python -m experiments.xor_demo`)
Model liniowy: strata 0,6931, dokładność 0,50. MLP(2-8-1): strata ≈ 0, dokładność 1,00.
**Wniosek:** potwierdza lekcję z planu - XOR wymaga nieliniowości.

## E2. Model liniowy na zadaniu (kontrola w numpy, bez treningu w PyTorchu)
Regresja logistyczna na kroku (ekran + nogi → bity), 2000 epizodów: strata ≈ 0,69 dla każdego bitu (0,688–0,691).
**Wniosek:** zadanie jest dla modeli liniowych nierozwiązywalne (każdy bit to XOR trzech wielkości).
*Nie uruchamialiśmy jeszcze `linear.yaml` ani `mlp.yaml` w PyTorchu.*

## E3. GRU, kodowanie `onehot_signed`
Polecenie: `python -m train.train --config experiments/configs/gru.yaml` (hidden 60, 24 126 parametrów, 5000 kroków; potem 20 000).
- **Seed 0:** w kroku 500 strata treningowa 0,63 (plateau), w 750 już 0,16 - nagłe „załapanie” reguły. Po 5000 krokach val exact 0,968,
  test_pairs 0,980, test_states 0,964, **test_x 0,025**. Po 20 000 krokach: val 0,985, test_pairs 0,977, test_x 0,024, test_states 0,976.
  Dłuższy trening nie poprawia uogólniania: strata treningowa → 0,0000 przy stracie `val` ≈ 0,009 (zapamiętywanie tabeli par).

## E4. GRU, kodowanie `binary`
Polecenie: `... --set name=gru_binary task.screen_encoding=binary train.steps=20000`.
- **Seed 0, 5000 kroków:** test_x skacze z 0,025 do 0,877 (val 0,885) - **potwierdzenie diagnozy**, że zero na `test_x` w `onehot_signed`
  było artefaktem kodowania (piksele wyłączonych X nigdy nie świecą w treningu, więc ich wagi wejściowe nie dostają gradientu).
- Uczy się wolniej niż `onehot_signed`; dłuższy trening dochodzi do poziomu pary ≈ 0,97-0,99.

## E5. GRU, 5 ziaren (wartości z ostatniego kroku)
Polecenia: `bash experiments/run_seeds.sh gru train.steps=8000` oraz
`bash experiments/run_seeds.sh gru name=gru_binary task.screen_encoding=binary train.steps=20000`.

| kodowanie | kroki | val | test_pairs | test_x | test_states |
|---|---|---|---|---|---|
| `onehot_signed` | 8 000 | 0,972 ± 0,008 | 0,978 ± 0,003 | **0,016 ± 0,009** | 0,974 ± 0,005 |
| `binary` | 20 000 | 0,981 ± 0,017 | 0,989 ± 0,009 | **0,985 ± 0,014** | 0,973 ± 0,014 |

Wartości na ziarno (val / test_pairs / test_x / test_states):

| seed | `onehot_signed` 8k | `binary` 20k |
|---|---|---|
| 0 | 0,975 / 0,980 / 0,028 / 0,975 | 0,952 / 0,973 / 0,961 / 0,951 |
| 1 | 0,982 / 0,978 / 0,023 / 0,983 | 0,994 / 0,997 / 0,997 / 0,990 |
| 2 | 0,976 / 0,981 / 0,008 / 0,971 | 0,989 / 0,989 / 0,992 / 0,978 |
| 3 | 0,968 / 0,980 / 0,014 / 0,970 | 0,989 / 0,994 / 0,990 / 0,975 |
| 4 | 0,960 / 0,973 / 0,009 / 0,971 | 0,980 / 0,993 / 0,983 / 0,970 |

**Wnioski i zastrzeżenia:**
- `test_x` w `onehot_signed` jest na poziomie zgadywania we wszystkich ziarnach → to właściwość kodowania, nie modelu. Test nowych X raportujemy na `binary`.
- `binary` ma wyższy sufit na parach (4 z 5 ziaren ≈ 0,99; `onehot_signed` plateau ≈ 0,978), ale uczy się wolniej i z większym rozrzutem
  między ziarnami (seed 4 stał ok. 2 000 kroków na ~0,42-0,47 przed przełamaniem). Różnica jest rzędu 1 punktu procentowego.
- Budżety kroków są różne (8k vs 20k), więc to porównanie plateau, nie porównanie przy równym czasie treningu.
- Rozrzut obejmuje inicjalizację i losowanie batchy, ale **nie** wybór podziału danych (jeden `split_seed`).
- Wartości z ostatniego kroku; `val` ma tylko 336 kroków, więc `exact` waha się o ok. ±0,5 pp między kolejnymi ewaluacjami.

## E6. Model `Rate` na małym grafie warstwowym (pierwsze próby) (3 616 neuronów, 95 188 krawędzi, 24 391 parametrów)
Config: `rate_small.yaml` (tanh, `inner_steps=8`, `init_gain=8`, `onehot_signed`, 3000 kroków).
- **Domyślny bieg:** strata treningowa 0,6914-0,6928 przez cały trening, val loss 0,692-0,696, exact = szum (0,007-0,04). **Brak uczenia.**
- **`init_gain=14` i `init_gain=20` (1500 kroków):** krzywe praktycznie identyczne z domyślnym biegiem (to samo do 3-4 cyfr).
- **`act=relu`:** NaN już od kroku 250 (aktywność nieograniczona + 128 kolejnych aktualizacji w BPTT).

**Wstępna interpretacja:** skoro wzmocnienie nie zmienia przebiegu, a strata jest minimalnie poniżej ln 2, model prawdopodobnie uczy się tylko
średnich częstości bitów. Przyczynę wyjaśniają E7 i E8. To **nie jest test hipotezy H1a** (graf nie jest connectome, a model technicznie nie działa).

---

## E7. Diagnostyka Rate na GPU (`diagnose_rate.py`, `copy_test.py`; `rate_small`, `onehot_signed`, gain 8)
- **Aktywność:** średnie |z| przed tanh = 3,65; tylko 13% neuronów w zakresie liniowym (|z| < 0,3), **79% nasyconych** (|z| > 2).
  (Z `input_scale=4`: 8% liniowo, 80% nasyconych.) **Moja wcześniejsza hipoteza, że sieć pracuje prawie liniowo, była błędna** - jest odwrotnie.
- **Gradienty:** niezerowe, rzędu 1e-2 (`v` 0,45; `W_in` 1,2e-2; `gain` 8e-4), więc technicznie gradient płynie.
- **Zapamiętanie jednego stałego batcha** (16 epizodów × 16 kroków, 400 kroków Adama): strata 0,693 → 0,686, exact 0,016-0,023. Sieć z 24 tys. parametrów
  **nie potrafi zapamiętać 256 przykładów**, więc zawodzi sama ścieżka ekran → wyjście, a nie trudność XOR-a.
- **Test kopiowania (cel liniowy: bity X z ekranu `binary`), 600 kroków:** strata 0,692 przez cały czas, także z `input_scale=4`. Nawet zadanie liniowe nie jest uczone.
**Wniosek:** ścieżka sygnału jest martwa (sieć jest nasycona i nie reaguje na wejście), nie brakuje jej nieliniowości.

## E8. Symulacja dynamiki Rate w numpy (bez uczenia; losowe wagi wejścia jak domyślny init, 16 epizodów, 16 kroków × 8 kroków wewnętrznych)
Miary: *input-sens* = średnia zmiana aktywności neuronów ruchowych po zmianie instrukcji w OSTATNIM kroku, *history-sens* = po zmianie instrukcji w PIERWSZYM kroku.

| tryb | gain | nasycone | mean\|h\| ruch. | input-sens | history-sens |
|---|---|---|---|---|---|
| stan przenoszony | 2 | 0% | 0,26 | 0,0006 | 0,076 |
| stan przenoszony | 3 | 5% | 0,87 | 0,0000 | 0,32 |
| stan przenoszony | 4 | 46% | 0,94 | 0,0000 | 0,12 |
| stan przenoszony | 6 | 80% | 0,98 | 0,0000 | 0,12 |
| stan przenoszony | 8 | 89% | 0,99 | 0,0000 | 0,12 |
| **reset przed instrukcją** | 4 | 0% | 0,034 | 0,019 | 0 |
| **reset przed instrukcją** | 6 | 0% | 0,21 | 0,11 | 0 |
| **reset przed instrukcją** | 8 | 12% | 0,49 | 0,22 | 0 |

Dla gain 2-3 z `input_scale` do 10 (stan przenoszony) input-sens pozostaje ≤ 0,004 przy history-sens 0,07-0,58.
**Wniosek:** przy przenoszeniu stanu sieć (pętle zwrotne + normalizacja wag do sumy 1) przy gain ≥ 3 nasyca się i „zatrzaskuje” w stanie zależnym od historii,
nowa instrukcja prawie niczego nie zmienia (to wyjaśnia, dlaczego gain 8/14/20 dają identyczne krzywe); przy gain 2 aktywność jest, ale wpływ wejścia to ok. 1% wpływu historii.
Nie ma dobrego okna wzmocnienia dla trybu z pamięcią. Przy zerowaniu stanu przed każdą instrukcją (`model.reset_each_step=true`) okno jest szerokie (gain 6-8).
*Zastrzeżenia:* symulacja bez uczenia, jeden graf, jeden batch, prosta miara czułości; zgodność z E7 jest tylko jakościowa (nasycenie 80% w obu).

## E9. Modele odniesienia, 5 ziaren (ostatni krok, exact, średnia ± sd; wygenerowane przez `scripts.summarize_runs`)
`onehot_signed`, 8 000 kroków (linear: 3 000):

| model | parametry | val | test_pairs | test_x | test_states |
|---|---|---|---|---|---|
| linear | 426 | 0,014 ± 0,005 | 0,010 ± 0,003 | 0,005 ± 0,004 | 0,017 ± 0,005 |
| mlp | 24 406 | 0,906 ± 0,016 | 0,921 ± 0,017 | 0,023 ± 0,013 | 0,934 ± 0,019 |
| elman | 24 406 | 0,991 ± 0,007 | 0,998 ± 0,002 | 0,028 ± 0,005 | 0,998 ± 0,002 |
| gru | 24 126 | 0,972 ± 0,008 | 0,978 ± 0,003 | 0,016 ± 0,009 | 0,974 ± 0,006 |
| lstm | 24 706 | 0,953 ± 0,008 | 0,961 ± 0,006 | 0,011 ± 0,002 | 0,959 ± 0,011 |
| transformer | 26 610 | 0,996 ± 0,007 | 0,996 ± 0,007 | 0,004 ± 0,005 | 0,996 ± 0,007 |

`binary`, 20 000 kroków:

| model | val | test_pairs | test_x | test_states |
|---|---|---|---|---|
| mlp_binary | 0,975 ± 0,011 | 0,987 ± 0,008 | 0,982 ± 0,010 | 0,984 ± 0,005 |
| elman_binary | 0,978 ± 0,012 | 0,986 ± 0,007 | 0,986 ± 0,004 | 0,985 ± 0,007 |
| gru_binary | 0,981 ± 0,017 | 0,989 ± 0,009 | 0,985 ± 0,014 | 0,973 ± 0,014 |
| lstm_binary | 0,982 ± 0,006 | 0,991 ± 0,004 | 0,989 ± 0,003 | 0,991 ± 0,003 |
| transformer_binary | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 ± 0,000 |

**Wnioski i zastrzeżenia:**
- Model liniowy: poziom zgadywania (kontrola: zadanie nie jest liniowe). Wszystkie pozostałe rodziny uczą się zadania.
- `binary`: **wszystkie** rodziny uogólniają na nowe X i nowe stany (0,97-1,00); transformer osiąga 1,000 we wszystkich 5 ziarnach.
- `onehot_signed`: nowe X = poziom zgadywania dla wszystkich modeli (artefakt kodowania, E4); na parach i stanach najlepsze Elman i Transformer, najsłabszy MLP.
- Uwaga: w trybie `teacher` pamięć nie jest potrzebna, więc różnice między rodzinami to głównie różnice optymalizacji/architektury, nie „zdolności pamięciowe”.
  Transformer nie ma tu przewagi dzięki uwadze po czasie, tylko prawdopodobnie dzięki lepszej optymalizacji (pre-LN, głębokość 2) - nie sprawdzaliśmy.
- Liczby parametrów są zbliżone (24-27 tys.), ale nie identyczne (transformer +9%).
- Budżety kroków różne między kodowaniami (8k vs 20k); modele z `onehot_signed` mogły osiągnąć plateau wcześniej.

## E10. Który bit jest najtrudniejszy (z tabel per bit, ostatni krok)
We wszystkich modelach i podziałach (poza nowymi X w `onehot_signed`) bity 0-4 mają dokładność ≈ 1,000, a **bit 5 (najstarszy)** jest najsłabszy
(np. gru val 0,972; gru_binary test_pairs 0,990; mlp val 0,920; lstm 0,955). To zgodne z hipotezą H5a (wyższy bit trudniejszy),
ale kolejność uczenia bitów 0-4 w czasie i zależność od przeniesień nie były jeszcze badane: do tego służą `scripts/bit_order.py` i `scripts/error_analysis.py` (dodane, jeszcze nieuruchomione).

## E11. Ablacje na GRU (`onehot_signed`, 8 000 kroków, 5 ziaren)
| wariant | val | test_pairs | test_states |
|---|---|---|---|
| bazowy (`gru`) | 0,972 ± 0,008 | 0,978 ± 0,003 | 0,974 ± 0,006 |
| `seq_len=32` | 0,972 ± 0,007 | 0,984 ± 0,005 | 0,978 ± 0,008 |
| `legs_mode=init_only` | 0,171 ± 0,055 | 0,193 ± 0,064 | 0,194 ± 0,066 |
| `screen_noise=0.3` | 0,073 ± 0,005 | 0,081 ± 0,002 | 0,090 ± 0,005 |

- **Długość epizodu (teacher):** brak pogorszenia (0,984 vs 0,978). Z 32 krokami uczy się szybciej w liczbie kroków optymalizatora (więcej celów na batch: 32 × 128 zamiast 16 × 128).
- **`init_only` (model musi pamiętać stan):** słabo (0,19), ale **trening nie zbiegł**: strata treningowa nadal malała w kroku 8000 (0,28-0,37), a seed 3 poprawiał się szybko pod koniec.
  Bity są uczone nierówno (val: bity 0,1,2 ≈ 0,85-0,94, bity 3-4 ≈ 0,46-0,55, bit 5 ≈ 0,82), bez monotonii po pozycji bitu. Wymaga dłuższego treningu, zanim cokolwiek powiemy o RQ7.
- **Szum 0,3:** załamanie (0,08). Prosty dekoder (największy |piksel|) odczytuje poprawnie instrukcję w ok. **75%** przypadków przy σ = 0,3 (98,6% przy σ = 0,2; 29,5% przy σ = 0,5; symulacja numpy, 200 tys. próbek),
  więc sufit dokładności jest daleko powyżej 0,08: model po 8 000 krokach jeszcze nie nauczył się odszumiać (strata 0,63-0,64, poniżej ln 2, ale daleko od 0).
  σ = 0,3 to za duży poziom na pierwszą ablację; lepsze są σ ∈ {0,1; 0,2} i dłuższy trening.

## E12. Promień spektralny i reżim „echo-state” (symulacja numpy, bez uczenia; ten sam układ co E8)
Promień spektralny macierzy `W` (znaki × moduły wag, wagi wchodzące znormalizowane do sumy 1) małego grafu: **ρ = 0,533** (dokładnie z `eig`; estymata `graphs.spectral_radius` daje to samo).
Gain krytyczny ≈ 1/ρ = 1,88, co zgadza się z przejściem do podtrzymywanej aktywności zaobserwowanym w E8 (gain 2-3).

| stan przenoszony, gain (ρ_eff = gain·ρ) | nasycone | mean\|h\| ruch. | input-sens | history-sens |
|---|---|---|---|---|
| 1,50 (0,8) | 0% | 0,0022 | 0,0002 | 0,0000 |
| 1,69 (0,9) | 0% | 0,0048 | 0,0004 | 0,0000 |
| 1,88 (1,0) | 0% | 0,0151 | 0,0006 | 0,0028 |
| 2,25 (1,2) | 0% (ale mean\|h\| 0,67) | 0,6736 | 0,0001 | 0,2525 |

Przy ρ_eff = 0,9 z większą skalą wejścia (×5, ×20, ×60) sieć jest wejściowo-sterowana i bez nasycenia (history-sens ≈ 0, pamięć wygasa), ale aktywność motoryczna jest mała (0,017-0,033).
Przy ρ_eff = 0,95 pojawia się niewielka pamięć (history-sens 0,0005-0,0008 vs input-sens 0,002-0,006).
**Wnioski:** (1) dobór gain na oko jest niepotrzebny: promień spektralny wskazuje, gdzie sieć przechodzi do aktywności podtrzymywanej; (2) reżim subkrytyczny omija zatrzask, ale ma słabą nieliniowość w głębi sieci
(sygnał jest uśredniany i wygaszany przez znormalizowane wagi o różnych znakach), więc nie jest jeszcze użytecznym rozwiązaniem dla trybu z pamięcią; (3) dla trybu bezstanowego (`reset_each_step`) dobre okno
to gain 6-8 (ρ_eff 3-4), czyli znacznie powyżej 1, bo reset nie pozwala aktywności się zatrzasnąć.
*Zastrzeżenia:* jeden graf, jeden batch, aktywność neuronów ruchowych zamiast logitów, bez uczenia; nie wiemy, jak to się przenosi na connectome.
Narzędzie do powtórzenia na prawdziwych danych: `scripts/calibrate_gain.py` (dodane, nieuruchomione).

## E13. Poprawione ablacje GRU (`onehot_signed`, 5 ziaren; wartości z ostatniego kroku wg logów, mean ± sd)
| wariant | kroki | val | test_pairs | test_states |
|---|---|---|---|---|
| `legs_mode=init_only` (30k) | 30 000 | 0,452 ± 0,098 | 0,448 ± 0,065 | 0,410 ± 0,103 |
| `screen_noise=0.2` | 12 000 | 0,212 ± 0,009 | 0,217 ± 0,006 | 0,235 ± 0,010 |

- **`init_only` jest uczący się, ale niezbieżny i silnie zależny od ziarna:** po 30k krokach seedy 1 i 2 dochodzą do 0,51-0,59 i wciąż rosną, seedy 0, 3 i 4 stoją na plateau ok. 0,37-0,41.
  Wcześniejszy wynik 0,19 po 8k krokach był więc niedotrenowaniem, ale 30k też jeszcze nie wystarcza. Nie ma podstaw, żeby mówić o „granicy” modelu.
- **`screen_noise=0.2`:** plateau ok. 0,21-0,22 od ok. kroku 6 000 (strata ~0,48), mimo że prosty dekoder odczytuje instrukcję w ok. 98,6% przypadków (E11), więc sufit jest > 0,9.
  Model uczy się częściowo, ale nie odszumia pikseli „one-hot”. Hipoteza (niesprawdzona): przy `binary` odszumianie jest łatwiejsze (6 pikseli z progiem zamiast wyboru 1 z 64) - do sprawdzenia.

## E14. Rate bezstanowy (`model.reset_each_step=true`) - poprawka zadziałała
Wszystko `rate_small`, gain 8, tanh, `onehot_signed`, 1 ziarno.
- **Aktywność (`diagnose_rate`):** mean|z| = 0,31; **74% wejść do tanh w zakresie liniowym (|z| < 0,3), tylko 2,9% nasyconych** (poprzednio 79% nasyconych). Gradienty niezerowe (rzędu 1e-2).
- **Zapamiętanie jednego batcha:** strata 0,633 → 0,276, exact 0,06 → 0,62 po 400 krokach (poprzednio 0,693 → 0,686, exact ≈ 0,02). Ścieżka sygnału działa.
- **Test kopiowania (`copy_test`, kodowanie binary):** strata 0,569 (krok 50) → 0,0043 (krok 600). Zadanie liniowe jest uczone.
- **Trening 3000 kroków:** val loss 0,694 → 0,245; exact (val / test_pairs / test_states): krok 1500 0,152 / 0,126 / 0,158; 2000 0,322 / 0,296 / 0,306; 2500 0,492 / 0,392 / 0,406; **3000 0,443 / 0,471 / 0,465**;
  test_x 0,034 (artefakt `onehot_signed`, E4). Strata treningowa 0,089, a val loss ≈ 0,245 od kroku ~2500, czyli zaczyna się luka train-val.
**Wniosek:** `Rate` bezstanowy **się uczy**: ścieżka sygnału działa, a wynik na niewidzianych parach rośnie z treningiem (0,47 po 3000 krokach).
**Zastrzeżenia:** jedno ziarno, 3000 kroków, wciąż rośnie; GRU na tym samym kodowaniu miał ≈ 0,8 już po 750 krokach i ≈ 0,98 końcowo, więc Rate jest wyraźnie słabszy.
Nie wiemy, czy to ograniczenie losowej struktury grafu, niedotrenowanie, czy zła skala/gain. Zbiór `val` ma 256 epizodów, więc `exact` waha się o kilka punktów.
Uwaga: dla stanu przenoszonego hipoteza o „reżimie liniowym” była błędna (E7), natomiast w trybie bezstanowym aktywność jest w przeważającej części blisko liniowa, a sieć mimo to uczy się zadania XOR-owego.

## E15. RQ5: kolejność uczenia bitów i błędy a przeniesienia (GRU)
**Kolejność uczenia (`bit_order`, GRU `onehot_signed`, 5 ziaren, `val`, próg 0,9):** bity 0-4 przekraczają 0,9 w tym samym kroku (mediana 750; 5/5 ziaren), bit 5 w kroku 1000 (5/5).
Rozdzielczość to 250 kroków (`eval_every`), więc nie widzimy różnic między bitami 0-4: uczą się praktycznie jednocześnie (nagłe „załapanie”), a bit 5 (najstarszy) pozostaje w tyle o co najmniej jedną ewaluację.
**H5a (bit 0 najszybciej, wyższe później) nie jest potwierdzona na tej rozdzielczości;** potwierdzone jest tylko, że bit 5 jest ostatni.

**Błędy a przeniesienia (`error_analysis`, GRU `binary`, seed 0 - najsłabsze ziarno, `last.pt`, `test_pairs`, 2048 epizodów, ok. 29 500 kroków):**
- Bity 0-3: 0,0% błędów. Bit 4: 0,0% przy carry=0, 0,2% przy carry=1. **Prawie wszystkie błędy są w bicie 5.**
- Bit 5 wg (operacja, przeniesienie wejściowe): `+` carry=0 3,8%; `+` carry=1 0,6%; `-` carry=0 1,6%; `-` carry=1 4,6%.
- Bit 5 wg długości łańcucha przeniesienia L: L=0 2,7%; L=1 0,1%; L=2 2,7%; L=3 3,6%; L=4 5,5%; L=5 3,2%. Bit 4: błędy tylko dla L=2-4 (0,2-0,4%).
**Wniosek:** H5b („błędy tam, gdzie przeniesienie propaguje się przez ≥ 3 bity”) jest **tylko częściowo** potwierdzona: błędy rosną dla L ≥ 2 w bicie 4, ale w bicie 5 występują też bez przeniesienia (L=0: 2,7%)
i nie rosną monotonicznie z L. Asymetria (`+`/carry=0 i `-`/carry=1 gorsze) sugeruje inny czynnik niż długość łańcucha (np. przekroczenie zakresu 0-63, czyli „zawinięcie” mod 64, albo kombinacja bitów s5, x5);
do sprawdzenia nowym rozbiciem `--detail-bits 5` (wg operacji, bitu stanu, bitu X i przeniesienia). To jedno ziarno jednego modelu.

## E16. `calibrate_gain` na GPU (pierwsze uruchomienie, `reset_each_step=true`, `rate_small`, ρ = 0,533)
| gain (ρ_eff) | nasycenie | mean\|logit\| | input-sens | history-sens |
|---|---|---|---|---|
| 4 (2,13) | 0,0% | 0,036 | 0,019 | 0 |
| 6 (3,20) | 0,0% | 0,211 | 0,109 | 0 |
| 8 (4,26) | 2,5% | 0,440 | 0,226 | 0 |
| 10 (5,33) | 9,6% | 0,572 | 0,315 | 0 |
| 12 (6,40) | 16,4% | 0,635 | 0,362 | 0 |
Zgodne jakościowo z symulacją numpy (E8): bez zatrzasku, czułość na instrukcję rośnie z gain, nasycenie pojawia się od gain ≈ 8. Skrypt działa.

## E17. Rate bezstanowy, 10 000 kroków, `cosine` (`rate_small`, 1 ziarno; wartości z logów)
| wariant | val loss końcowy | test_pairs końcowy (szczyt) | test_x | test_states | strata treningowa |
|---|---|---|---|---|---|
| gain 8, `onehot_signed` | 0,392 (min 0,217 w kroku 2750) | 0,514 (**0,561** w kroku 3750) | 0,031 | 0,533 | 0,0035 |
| gain 6, `onehot_signed` | 0,590 | 0,331 (0,344) | 0,030 | 0,341 | 0,017 |
| gain 10, `onehot_signed` | 0,539 | 0,463 (0,498 w kroku 4250) | 0,012 | 0,461 | 0,0010 |
| gain 8, `binary` | 0,221 | 0,272 (0,290) | **0,376** | 0,343 | 0,121 |
Wcześniej (E14, stała lr, 3000 kroków, gain 8, `onehot_signed`): test_pairs 0,471.
- **`onehot_signed`: silne przeuczenie.** Strata treningowa schodzi do ~0,001-0,017, a val loss rośnie po kroku ~2750-3000 (z ok. 0,22-0,28 do 0,39-0,59); dokładność na niewidzianych parach zatrzymuje się na 0,46-0,56
  (najlepszy gain to 8; gain 6 jest wyraźnie gorszy; w gain 6 widać też epizod niestabilności w kroku 1500: val loss 0,81). Rozrzut ewaluacji ±0,03, więc różnice między gain 6, 8 i 10 są większe niż szum.
- **`binary`: bez przeuczenia** (strata treningowa 0,121, val 0,221), ale niedotrenowanie (0,27-0,38), przy czym `test_x` (0,376) jest wyższy niż `test_pairs` (0,272), czyli reguła na bitach uogólnia, ale uczy się wolno.
- **Luka względem modeli odniesienia pozostaje duża:** GRU ≈ 0,98, MLP 0,92 (`onehot`) / 0,99 (`binary`), Rate najlepiej ≈ 0,56 (szczyt, nie wartość końcowa).
**Interpretacja (ostrożna):** `Rate` na losowym grafie uczy się raczej zapamiętywać pary niż regułę arytmetyczną; zamrożone, losowe, uśredniające połączenia dają słabą reprezentację, a uczone są tylko wejście, odczyt, progi i stałe czasowe.
Nie wiemy, czy to wada tej konkretnej losowej struktury, czy ogólna cecha takiej klasy modeli (tzw. rezerwuar z uczonym odczytem). To jest właśnie pytanie RQ1/RQ2 i dlatego porównanie z grafami kontrolnymi na connectome ma sens dopiero po tym, jak będziemy znać sufit małego grafu.

## E18. RQ5 z gęstą ewaluacją: kolejność uczenia bitów (`gru_fine`, 5 ziaren, `eval_every=25`, `val`, próg 0,9)
| bit | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| mediana kroku | 575 | 575 | 625 | 675 | 725 | **850** |
Wszystkie ziarna (5/5) przekroczyły próg dla każdego bitu. **Kolejność jest monotoniczna: im wyższy bit, tym później** (kroki co ok. 50, bit 5 po bicie 4 o ok. 125).
**Korekta E15:** wcześniejszy wniosek „bity 0-4 uczą się jednocześnie” wynikał z rozdzielczości 250 kroków. **H5a jest potwierdzona** (dla GRU, `onehot_signed`).

## E19. RQ5: błędy bitu 5 wg (operacja, s5, x5, c5), GRU `binary`, `test_pairs`, 2048 epizodów, `last.pt`
Błąd w % (n ≈ 1400-2450 na komórkę); c = przeniesienie/pożyczka wejściowa do bitu 5:
| op | s5 | x5 | seed 0: c=0 | seed 0: c=1 | seed 1: c=0 | seed 1: c=1 |
|---|---|---|---|---|---|---|
| + | 0 | 0 | 1,1 | 0,4 | 0,2 | 0,0 |
| + | 0 | 1 | 0,2 | 0,0 | 0,0 | 0,0 |
| + | 1 | 0 | 5,9 | 0,5 | 0,2 | 0,0 |
| + | 1 | 1 | **8,0** | 1,6 | **4,2** | 0,1 |
| - | 0 | 0 | 0,2 | 4,7 | 0,0 | 0,0 |
| - | 0 | 1 | 0,1 | 4,9 | 0,0 | 0,3 |
| - | 1 | 0 | 3,3 | 0,9 | 0,0 | 0,0 |
| - | 1 | 1 | 2,7 | **8,0** | 0,0 | 0,7 |
Bit 4 (seed 0): błędy tylko 0,8% dla (+, s4=0, x4=0, c=1) i 0,7% dla (−, 1, 1, 1). Bity 0-3: 0,0%. Seed 1 jest znacznie lepszy (bit 5: 1,1% dla `+`/c=0, ok. 0,2% dla `-`).
**Wnioski:** (1) błędy skupiają się w kilku komórkach i zależą od ziarna (seed 1 myli się prawie tylko w jednej); (2) w obu ziarnach najgorszą komórką dla dodawania jest (s5=1, x5=1, c5=0) - 8,0% i 4,2% -
oba najstarsze bity zapalone, czyli suma na pewno przekracza zakres (zawinięcie mod 64); (3) nie wszystkie wysokie błędy to zawinięcie (np. `+`/(1,0,0) 5,9% w seed 0), więc to nie jest pełne wyjaśnienie.
**H5b** (długość łańcucha przeniesienia) jest **słabsza niż hipoteza o specyficznych kombinacjach bitów** (zob. E15: L nie rosło monotonicznie). Do sprawdzenia: czy powtarza się to w innych ziarnach i modelach, i czy to efekt rozkładu danych treningowych
(5 z 6 wyłączonych stanów to stany ≥ 32).

## E20. Szum na kodowaniu `binary` i `init_only` dłużej
- **`gru_binary_noise02` (σ = 0,2, 20 000 kroków, 5 ziaren; wartości z ostatniego kroku wg logów):** val 0,730 ± 0,010, test_pairs **0,741 ± 0,008**, test_x 0,717 ± 0,009, test_states 0,704 ± 0,014.
  Dla porównania `onehot_signed` z σ = 0,2 (E13) miał 0,217. **Potwierdzona hipoteza z E13:** szum jest o wiele łatwiejszy do zniesienia na `binary`.
  Sufit prostego odczytu instrukcji dla `binary`: ok. 95,7% (σ = 0,2) i 70,9% (σ = 0,3; symulacja numpy) - model jest poniżej sufitu (0,74 vs ≈ 0,95), więc nie jest to jeszcze granica.
- **`gru_initonly_60k` (seed 0, `cosine`):** val 0,347, test_pairs 0,382, test_x 0,019, test_states 0,352. Od kroku ok. 35 000 wynik stoi w miejscu (0,38). Seed 0 bez `cosine` miał po 30k 0,390, więc
  **dłuższy trening nie pomaga** temu ziarnu; wcześniej seedy 1 i 2 dochodziły do 0,51-0,53. Przyczyna nieznana; hipoteza: błędy kumulują się, bo pojedynczy błąd stanu psuje wszystkie następne kroki.
  Sprawdzenie: nowe rozbicie dokładności wg kroku t w `error_analysis` (dodane, nieuruchomione).

## E21. Rate z weight decay (`rate_small`, `reset_each_step`, gain 8, `onehot_signed`, stała lr, 6000 kroków, 1 ziarno)
| weight_decay (AdamW) | test_pairs końcowy (szczyt) | test_states | val loss końcowy (min) | strata treningowa |
|---|---|---|---|---|
| 0 (E17: 10k kroków, cosine) | 0,514 (0,561) | 0,533 | 0,392 (0,217) | 0,0035 |
| **0,05** | **0,762** (0,776 w kroku 5100) | 0,723 | **0,162** (0,159) | 0,011 |
| 0,2 | 0,141 (0,175) | 0,134 | 0,359 | 0,348 |
- **Weight decay 0,05 usuwa przeuczenie:** val loss maleje do ok. kroku 4800 i się stabilizuje (0,16-0,17), test_pairs 0,74-0,78 od kroku ok. 4800 (było 0,51-0,56). To największa poprawa Rate dotąd, ale nadal poniżej GRU (≈ 0,98).
- **0,2 jest za silne:** model nie wychodzi z plateau (strata treningowa 0,35), test_pairs ≈ 0,14.
- **Zastrzeżenia:** (1) jedno ziarno; (2) porównanie z wd = 0 miesza inny harmonogram lr i liczbę kroków; (3) AdamW zmniejsza WSZYSTKIE parametry, także `gain`, `bias`, `alpha_logit`, `b_out`: przy wd = 0,05 i lr = 3e-3 to czynnik ok. e^(-0,9) ≈ 0,4 na 6000 kroków
  (przy wd = 0,2 ok. e^(-3,6) ≈ 0,03) dla parametru bez przeciwdziałania gradientu. To może tłumaczyć porażkę przy 0,2 i część poprawy przy 0,05 (mniejszy `gain` = mniejsze nasycenie), więc **nie wiemy, ile daje regularyzacja W_in, a ile zmniejszenie gain**.
  Rozstrzygnięcie: nowa opcja `train.wd_exclude` (decay tylko na W_in, W_sens, v; dodana, nieuruchomiona na GPU).

## E22. Rate na `binary`, dłużej (`rate_small`, `reset_each_step`, stała lr, 1 ziarno)
| wariant | test_pairs końcowy (szczyt) | test_x końcowy (szczyt) | val loss końcowy (min) |
|---|---|---|---|
| gain 8, 30 000 kroków | 0,269 (0,290) | 0,305 (0,404 w kroku 6000) | 1,32 (0,206 w kroku 5750) |
| gain 12, 20 000 kroków | 0,333 (0,345) | 0,298 (0,344) | 1,17 (0,254) |
**`binary` też się przeucza przy dłuższym treningu** (val loss rośnie z ok. 0,2-0,25 do ponad 1, strata treningowa → 0,002-0,03); wcześniejsza „odporność na przeuczenie” z E17 wynikała z zatrzymania przez harmonogram `cosine`.
Gain 12 jest nieco lepszy niż 8, ale ciągle ≈ 0,33. `binary` nie pomaga Rate (0,27-0,34 vs 0,76 dla `onehot_signed` z weight decay).

## E23. RQ5: błędy bitu 5 wg (operacja, s5, x5, c5), GRU `binary`, wszystkie 5 ziaren (`test_pairs`, 2048 epizodów, `last.pt`)
Błąd w %, wybrane komórki (reszta komórek: zwykle ≤ 1%):
| komórka (op, s5, x5, c5) | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | średnia |
|---|---|---|---|---|---|---|
| `+`, 1, 1, 0 | 8,0 | 4,2 | 4,2 | 2,6 | 2,2 | **4,2** |
| `-`, 1, 1, 1 | 8,0 | 0,7 | 2,7 | 1,1 | 3,2 | **3,1** |
| `+`, 1, 0, 0 | 5,9 | 0,2 | 0,0 | 0,3 | 0,6 | 1,4 |
Poza tym: seed 0 dodatkowo 4,7-4,9% dla `-`, s5=0, c=1; seed 2 dodatkowo 4,0% dla `-`, 1, 1, 0 i 2,2% dla `-`, 1, 0, 1; seed 4 2,5% dla `+`, 0, 1, 0. Bity 0-3: 0,0% we wszystkich ziarnach; bit 4: ≤ 0,3%.
**Wnioski:** (1) komórki z **oboma najstarszymi bitami zapalonymi (s5 = x5 = 1)** są najtrudniejsze w każdym ziarnie, szczególnie dodawanie bez przeniesienia i odejmowanie z pożyczką; (2) poza nimi wzór zależy od ziarna;
(3) błąd NIE rośnie monotonicznie z długością łańcucha przeniesienia (np. seed 4: 0,4-0,9% niezależnie od L); (4) dokładność w czasie jest płaska (exact 0,99-1,00 dla t ≥ 1), natomiast **krok t = 0 jest o 1-4 pp gorszy** (0,96-0,98).
Wciąż nie wiemy dlaczego akurat s5 = x5 = 1; w tych komórkach suma/różnica zawsze przekracza zakres 0-63 (zawinięcie mod 64) lub jest blisko niego, ale nie wszystkie wysokie błędy to zawinięcie. Hipoteza o rozkładzie danych (5 z 6 wyłączonych stanów ≥ 32)
sprawdza się dopiero przy innym `split_seed`.

## E24. RQ7: dlaczego `init_only` nie zbiega (`error_analysis`, split `train`, 2048 epizodów, GRU `onehot_signed`)
| przebieg | exact wg t: t=0 → t=15 | błąd bitu 3 | błąd bitu 4 | błąd bitu 5 | bity 0-2 |
|---|---|---|---|---|---|
| `gru_initonly_30k`, seed 1 | 0,76 → 0,60 | 6-7% | **27-30%** | 5-6% | 0-0,2% |
| `gru_initonly_60k` (cosine), seed 0 | 0,60 → 0,49 | **43-45%** | 8% | 3-4% | 0-0,1% |
**Wnioski:** (1) gubienie stanu w czasie jest **łagodne** (spadek o 11-16 pp przez 16 kroków); (2) dominującym problemem jest **jeden środkowy bit, którego model w ogóle się nie nauczył** (bit 4 w seed 1: błąd 27-30%; bit 3 w seed 0: błąd 43-45%),
a błąd jest taki sam dla `+` i `-`, carry=0 i carry=1, czyli to nie problem przeniesienia; (3) nawet krok t = 0, w którym stan nóg jest widoczny, jest słaby (0,60-0,76 na zbiorze treningowym), a w trybie `teacher` GRU ma tam > 0,99.
Moja wcześniejsza hipoteza („błędy kumulują się z krokiem”) jest więc tylko częściowo potwierdzona, a dominuje inny czynnik: wolne uczenie bitów przy konieczności zapisu stanu w pamięci rekurencyjnej, z różnym „zablokowanym” bitem w różnych ziarnach.

## E25. Rate: weight decay Z WYKLUCZENIEM `gain`, `bias`, `alpha_logit`, `b_out` (`train.wd_exclude`; 6000 kroków, stała lr, `onehot_signed`, gain 8, 1 ziarno)
| wariant | test_pairs końcowy (szczyt) | test_states | val loss końcowy (min) | strata treningowa |
|---|---|---|---|---|
| wd 0,05 na wszystkie parametry (E21) | **0,762** (0,776) | 0,723 | **0,162** (0,159) | 0,011 |
| wd 0,02, bez decay na gain/bias/alpha/b_out | 0,477 (0,498) | 0,459 | 0,390 (0,234) | 0,004 |
| wd 0,05, bez decay na gain/bias/alpha/b_out | 0,550 (0,585) | 0,565 | 0,399 (0,248) | 0,005 |
| wd 0,1, bez decay na gain/bias/alpha/b_out | 0,635 (0,635 w kroku 6000) | 0,573 | 0,305 (0,211) | 0,010 |
- **Poprawa z E21 NIE wynika z regularyzacji W_in, W_sens i v.** Gdy te trzy zestawy wag są jedynymi regularyzowanymi, wynik jest gorszy (0,48-0,64) niż z decay na wszystkich parametrach (0,76), a val loss znów rośnie po kroku ok. 2900.
- **Z wykluczeniem wynik rośnie z siłą decay** (0,48 → 0,55 → 0,64 dla 0,02 → 0,05 → 0,1), więc silniejszy decay na W_in/W_sens/v też pomaga, ale nie dorównuje wariantowi z decay na wszystkich parametrach (nie sprawdziliśmy wd > 0,1 z wykluczeniem).
- **Wniosek:** istotna część poprawy pochodzi z decay na `gain`, `bias`, `alpha_logit` i/lub `b_out`. **Nie wiemy, na który z nich.** Hipotezy: (a) zmniejszanie `gain` (mniej nasycenia), (b) ograniczenie wolnych parametrów per neuron (`bias` i `alpha_logit` to po ok. 3600 parametrów każdy, razem ok. 7200, czyli dużo możliwości zapamiętywania par), (c) `b_out`.
- **Zastrzeżenia:** jedno ziarno; ewaluacja `val`/`test` ±0,03; w wd 0,02 widać pojedynczy skok straty w kroku 2500 (train 0,80), po którym trening wraca (niestabilność zdarza się też w innych biegach Rate).

## E26. GRU `binary` z innym podziałem danych (`task.split_seed=1`, 20 000 kroków, 5 ziaren; wyłączone X: 2, 9, 27, 30, 46, 58; stany: 15, 16, 24, 25, 41, 50)
| podział | val | test_pairs | test_x | test_states |
|---|---|---|---|---|
| `split_seed=0` (E5) | 0,981 ± 0,017 | 0,989 ± 0,009 | 0,985 ± 0,014 | 0,973 ± 0,014 |
| `split_seed=1` | 0,983 ± 0,012 | 0,990 ± 0,007 | 0,987 ± 0,009 | 0,983 ± 0,009 |
**Wyniki główne nie zależą od podziału** (różnice w granicach rozrzutu).
**Błędy bitu 5 (seed 0-2, `test_pairs`):** błędy są małe (maks. komórka 2,4%) i **rozproszone**, a komórka s5 = x5 = 1, która w `split_seed=0` była najtrudniejsza, tu praktycznie nie błądzi:
`+`/(1,1,c=0) 0,0 / 1,1 / 0,0% (w split 0: 8,0 / 4,2 / 4,2%), `-`/(1,1,c=1) 0,1 / 0,6 / 0,3% (w split 0: 8,0 / 0,7 / 2,7%). Najgorsze komórki to teraz `-`/(0,0,c=1) 2,2 / 2,4 / 0,2% i `+`/(1,0,c=1) 1,9 / 0,5 / 0,0%.
Bity 0-3: 0-0,2%, bit 4: ≤ 0,3%. Krok t = 0 nadal gorszy o 1-3 pp (0,97-0,98 vs 0,99-1,00).
**Wniosek:** wzór „najtrudniejsze s5 = x5 = 1” z E23 był **cechą podziału `split_seed=0`** (tam 5 z 6 wyłączonych stanów to stany ≥ 32), a nie ogólną własnością zadania. Stabilne w obu podziałach: błędy tylko w bicie 5 (i śladowo w 4) oraz słabszy krok t = 0.

## E27. `init_only`: większa pamięć i LSTM (30 000 kroków, 1 ziarno każdy; `onehot_signed`, teacher wyłączony)
| model | parametry | val | test_pairs końcowy (szczyt) | test_states |
|---|---|---|---|---|
| GRU hidden 60 (5 ziaren, E13) | 24 126 | 0,452 ± 0,098 | 0,448 ± 0,065 (0,39-0,53) | 0,410 ± 0,103 |
| **GRU hidden 128** | 77 574 | 0,770 | **0,795** (0,819) | 0,665 |
| **LSTM hidden 50** | 24 706 | 0,763 | **0,784** | 0,632 |
- **Pamięć nie jest barierą, tylko pojemność/architektura.** LSTM o tej samej liczbie parametrów co GRU 60 osiąga 0,78 (GRU 60: ok. 0,45), a GRU 128 osiąga 0,80. Oba wyszły z plateau (LSTM po kroku ok. 12 500, GRU 128 po ok. 14 000) i ciągle rosną.
- Nadal poniżej trybu `teacher` (GRU 0,98), i `test_states` (0,63-0,67) odstaje od `test_pairs` (0,78-0,80).
- **Zastrzeżenia:** jedno ziarno na model; GRU 128 ma ponad 3 razy więcej parametrów niż Rate, więc to nie jest porównanie parametr-do-parametru, natomiast LSTM tak.

## E28. `rate_best`, 5 ziaren (stan zerowany, AdamW 0,05 na wszystkie parametry, 6000 kroków, `onehot_signed`; test_pairs)
| ziarno | ostatni krok | mediana 5 ostatnich ewaluacji | szczyt | uwagi | czas |
|---|---|---|---|---|---|
| 0 | 0,785 | 0,780 | 0,787 | skoki straty w krokach 4800 i 5600 | 2037 s |
| 1 | 0,875 | 0,866 | 0,875 | skok w kroku 4900 (strata treningowa 1,43), wraca w następnej ewaluacji | 1815 s |
| 2 | 0,811 | 0,811 | 0,811 | skoki w krokach 1500 i 3400 | 1816 s |
| 3 | 0,495 | 0,463 | 0,509 | **plateau ≈ 0,1 do kroku ok. 5100**, start nauki dopiero tuż przed końcem; skok w kroku 5900 | 749 s |
| 4 | **0,170** | 0,836 | 0,851 | **skok straty dokładnie w ostatnim kroku** (w kroku 5900 było 0,845) | 747 s |
**Zbiorczo:** ostatni krok **0,627 ± 0,294**, mediana ostatnich 5 ewaluacji 0,751 ± 0,164, szczyt 0,767 ± 0,148; cztery ziarna bez ziarna 3 (mediana): 0,823 ± 0,037. (`summarize_runs` pokazuje dla ostatniego kroku 0,627 ± 0,294.)
- **Wartość „z ostatniego kroku” jest tu mylącą miarą:** trening Rate ma powtarzalne skoki straty (jeden w ostatnim kroku ziarna 4 zniszczył jego wynik), a ziarno 3 wystartowało zbyt późno. Typowa wartość dla czterech ziaren to 0,78-0,87.
- **Czas:** 6000 kroków trwa ok. 12,5 min, gdy GPU jest wolne (ziarna 3 i 4: 747-749 s), a 30-34 min, gdy jest współdzielone (ziarna 0-2).
- **Wniosek na sufit małego grafu:** Rate z weight decay 0,05 osiąga dla większości ziaren 0,78-0,87 na niewidzianych parach (GRU 0,98, MLP 0,92-0,99), przy dużej niestabilności treningu. Do raportu: wartości z `best.pt` (`scripts/eval_checkpoints`) lub z treningu ze spadkiem lr (`cosine`), nie z ostatniego kroku.

## E29. Który parametr zawdzięcza poprawę weight decay (`rate_best`, 1 ziarno, 6000 kroków; decay tylko na wskazane parametry)
| decay tylko na | test_pairs końcowy (szczyt) | val loss końcowy (min) | strata treningowa |
|---|---|---|---|
| W_in, W_sens, v (E25, wd 0,05) | 0,550 (0,585) | 0,399 (0,248) | 0,005 |
| **gain** | **0,704** (0,712) | **0,157** (0,157, na końcu) | 0,004 |
| bias, alpha_logit | 0,518 (0,573) | 0,443 (0,245) | 0,007 |
| gain, bias, alpha_logit, b_out | 0,736 (0,755) | 0,207 (0,183) | 0,006 |
| wszystkie (E21 / ziarno 0 `rate_best`) | 0,762 / 0,785 | 0,162 / 0,136 | 0,011 / 0,010 |
- **Weight decay na `gain` daje prawie całą poprawę:** samo `gain` zdejmuje przeuczenie (val loss maleje do końca treningu, 0,157) i daje 0,70; `bias` + `alpha_logit` bez `gain` nie pomagają (0,52, val loss rośnie).
  Różnice między „tylko gain” (0,70), „gain + bias + alpha + b_out” (0,74) i „wszystkie” (0,76-0,79) mieszczą się w rozrzucie między ziarnami (E28), więc ich nie rozróżniamy.
- **Interpretacja:** przeuczenie Rate wiąże się z dużym `gain` (silne nasycenie i nieliniowość sprzyjają zapamiętywaniu par); decay powoli zmniejsza gain w trakcie treningu. Nie mierzyliśmy wartości `gain` po treningu (dodane logowanie `gain` i `final_gain`, jeszcze nieuruchomione).
  Samo niższe początkowe wzmocnienie bez decay (E17: gain 6) nie dało tego efektu (0,33), więc ważna jest prawdopodobnie ścieżka „zaczynamy wysoko, wzmocnienie maleje”; to hipoteza do sprawdzenia.
- **Zastrzeżenia:** 1 ziarno na wariant; skoki straty (np. `gain only` w kroku 2500, `bias_alpha` w kroku 5000) mogą zniekształcać pojedyncze punkty.

## E30. `rate_best`: ocena checkpointów i wartość `gain` po treningu (`scripts/eval_checkpoints`)
| ziarno | `best.pt`: krok | val | test_pairs | test_states | `gain` | `last.pt`: test_pairs |
|---|---|---|---|---|---|---|
| 0 | 5500 | 0,774 | 0,777 | 0,750 | 14,62 | 0,785 |
| 1 | 6000 | 0,895 | 0,875 | 0,860 | 13,44 | 0,875 |
| 2 | 4800 | 0,837 | 0,779 | 0,755 | 14,04 | 0,811 |
| 3 | 5800 | 0,586 | 0,509 | 0,559 | 12,91 | 0,495 |
| 4 | 5900 | 0,857 | 0,845 | 0,833 | 14,47 | **0,170** |
| **średnia ± sd** | | 0,790 ± 0,122 | **0,757 ± 0,145** | 0,751 ± 0,118 | | 0,627 ± 0,294 |
- **Wartość do raportu (`best.pt`, wybrany po `val`):** test_pairs **0,757 ± 0,145**; bez ziarna 3 (późny start): 0,819 ± 0,049. `test_x` 0,018 ± 0,008 to znany artefakt kodowania `onehot_signed`.
- **`last.pt` zawyża rozrzut przez skok w ostatnim kroku ziarna 4 (0,170).** `best.pt` ten problem usuwa (0,845).
- **Korekta hipotezy z E29: `gain` po treningu WZRÓSŁ** z 8 do 12,9-14,6 (we wszystkich ziarnach), mimo weight decay na `gain`. Decay nie „zmniejsza gain poniżej wartości początkowej”.
  Nowa hipoteza: bez decay `gain` rośnie jeszcze bardziej (niezmierzone), duże wzmocnienie sprzyja nasyceniu i zapamiętywaniu par, a decay ogranicza ten wzrost. Sprawdzenie bez treningu: `gain` w checkpointach wcześniejszych biegów (bez decay, tylko `gain`, z wykluczeniem).

## Co zostało do zrobienia (eksperymentalnie)
1. **Rate:** (zrobione: ocena `best.pt`, E30) opcjonalnie powtórzenie 5 ziaren z `cosine` i 10 000 kroków (stabilny koniec treningu, ziarno 3 zdąży wystartować).
2. **Mechanizm:** porównać `gain` w checkpointach starszych biegów (wd = 0 vs z decay na `gain` vs z wykluczeniem), bez nowego treningu.
3. **RQ7 (opcjonalnie):** `init_only` z kilkoma ziarnami dla LSTM i GRU 128; szum dla innych modeli niż GRU.
4. Zweryfikować `scripts.count_params` (transformer d=36: 26 610 parametrów, +9% względem Rate).
5. Connectome i grafy kontrolne (RQ1, RQ2) - po punkcie kontrolnym (zob. `docs/plan_connectome.md`).
