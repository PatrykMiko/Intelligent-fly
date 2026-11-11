# Decyzje projektowe (model / zadanie)

Każda decyzja ma ID, powód i sposób zmiany. Rzeczy oznaczone **[OTWARTE]** warto uzgodnić z prowadzącym lub
z kolegą odpowiedzialnym za wizualizację, zanim ruszą długie eksperymenty.

## Zadanie
**D1. Kodowanie ekranu: `onehot_signed`.** Piksel o indeksie X zapalony, wartość +1 dla `+X`, −1 dla `-X`.
Mieści się w 64 pikselach i niesie znak operacji. Alternatywa `binary` (X na 6 pikselach + piksel znaku) jest w kodzie
(`task.screen_encoding`) jako ablacja. *Uwaga:* dla X=0 `+0` i `-0` dają ten sam wynik, więc różnica na ekranie jest nieszkodliwa.
**Po eksperymentach (2026-11-04):** `onehot_signed` NIE nadaje się do podziału `test_x` (piksele wyłączonych X nigdy nie są uczone → wynik na poziomie zgadywania, dla wszystkich 5 ziaren GRU). Podział `test_x` raportujemy na `binary`; `onehot_signed` zostaje jako szybszy punkt odniesienia dla par i stanów. W tabelach podawać oba kodowania.

**D2. Wejście modelu w kroku: ekran + prawdziwe położenie nóg (`legs_mode: teacher`).**
Zgodne z planem („przy każdej instrukcji widzi ekran oraz aktualne położenie nóg”). Zadanie na poziomie kroku to wtedy
funkcja (stan, instrukcja) → nowy stan; pamięć nie jest konieczna. Tryb `init_only` (stan nóg tylko w kroku 0) wymusza
pamiętanie stanu i jest użyty w RQ7. **[OTWARTE]** który tryb jest „głównym” w raporcie - proponuję `teacher` jako główny,
`init_only` jako osobny wynik.

**D3. Dane to epizody po 16 instrukcji, losowane „na żywo” z dozwolonego zbioru ruchów (nie ma stałego zbioru treningowego).**
Stan początkowy losowy. Łańcuch stanów jest spójny (stan po kroku t to stan przed krokiem t+1).

**D4. Podziały (RQ4).** `train` / `val` (5% wolnych par) / `test_pairs` (10% wolnych par) / `test_x` (6 wyłączonych X) /
`test_states` (6 wyłączonych stanów). Para = (stan, X), dotyczy obu operacji. W treningu stan docelowy też nie jest
wyłączonym stanem. W podziałach ewaluacyjnych liczą się tylko kroki spełniające kryterium (`measure`); reszta to wypełniacz
łączący łańcuch. *Znana subtelność:* `-X` ≡ `+(64−X)` jako funkcja wyniku, więc wyłączone X dla odejmowania ma „kuzyna” w treningu
dla dodawania; ekrany są jednak różne (znak), więc to nie jest wyciek wejścia, ale warto o tym wspomnieć w raporcie.

## Model `Rate`
**D5. Zamrożone: topologia, znaki, moduły wag. Uczone: wejście, czucie, odczyt, stałe czasowe, progi, jedno globalne wzmocnienie.**
Dosłowna interpretacja RQ1 („uczone tylko parametry wejścia, wyjścia i dynamiki”).

**D6. Znak z neuronu presynaptycznego (prawo Dale'a); moduły wag znormalizowane tak, że suma wag wchodzących = 1.**
Zgodnie z planem (znaki z neuroprzekaźników, normalizacja wag). Normalizacja jest też ablacją (krok 24) - kod to `GraphSpec.normalize()`.

**D7. Dynamika: `h ← (1−α)h + α·tanh(g·W h + u)`, 8 kroków wewnętrznych na instrukcję, stan h przenoszony między instrukcjami.**
`tanh` zamiast ReLU dla stabilności BPTT. 8 kroków wewnętrznych ≥ liczba „skoków” w małym grafie (4), by sygnał z ekranu
dotarł do neuronów ruchowych. **[OTWARTE]** dla pełnego connectome liczbę kroków wewnętrznych trzeba będzie dobrać do
długości najkrótszych ścieżek wejście → wyjście.

**D8. `init_gain = 8.0`.** Wybrane z numerycznego sprawdzenia dynamiki bez uczenia (numpy) na domyślnym małym grafie:
przy gain ≤ 4 sygnał wygasał przed neuronami ruchowymi (|h| ≈ 0,02), przy 8 miał |h| ≈ 0,35 bez nasycenia, przy 14 |h| ≈ 0,78.
To tylko punkt startowy - gain jest uczony. Dla innego grafu (connectome) sprawdzenie trzeba powtórzyć.
**Po eksperymentach (2026-11-04):** `init_gain` 8, 14 i 20 dało praktycznie identyczne krzywe `rate_small` (brak uczenia), więc samo wzmocnienie nie jest przyczyną problemu; sprawdzenie sygnału numpy dotyczyło tylko dotarcia sygnału do neuronów ruchowych, nie tego, czy sieć wychodzi z reżimu liniowego. Zob. D14.

## Grafy kontrolne (RQ2)
**D9. Interpretacja czterech kontroli z listy kroków [OTWARTE - potwierdzić]:**
- `degree_preserving` - permutacja celów krawędzi (zachowane stopnie wejściowe i wyjściowe oraz znaki; możliwe pętle własne i krawędzie wielokrotne),
- `shuffled_weights` - topologia bez zmian, moduły i znaki rozlosowane między krawędziami,
- `no_signs` - wszystko pobudzające,
- `random_signs` - znaki losowane per neuron presynaptyczny z tą samą frakcją hamujących.
Słowo „przetasowany” z listy mogło oznaczać coś innego; zmiana to kilka linijek w `graphs/spec.py`.

## Porównania (RQ3)
**D10. Dopasowanie liczby parametrów:** `scripts/count_params.py` dobiera `hidden` dla MLP/Elman/GRU/LSTM/Transformera do liczby
*trenowanych* parametrów modelu `Rate` z danego configu. Model liniowy nie jest dopasowywany (to oracle „zgadywania”).
Wartości `hidden` w `experiments/configs/*.yaml` policzyłem analitycznie (wzorami na liczbę parametrów) pod domyślny graf `rate_small` (24 391 parametrów Rate): MLP 122, Elman 122, GRU 60, LSTM 50, Transformer d=36 (±10%). To wartości wstępne i niezweryfikowane uruchomieniem - sprawdźcie je `scripts/count_params.py` i policzcie ponownie po wczytaniu connectome.

## Trening i ewaluacja
**D11. Strata: BCE per bit, średnia po bitach i krokach; Adam lr 3e-3, clip normy gradientu 1.0, pełne BPTT po epizodzie.**
**D12. Metryki: strata, dokładność per bit, dokładność całej liczby (wszystkie 6 bitów) - na krokach z maską `measure`.**
Zbiory ewaluacyjne są stałe i wspólne dla wszystkich ziaren (`split_seed + 12345`). Wybór checkpointu `best.pt` po `val`,
nigdy po testach.
**D13. Ziarno (`seed`) zmienia inicjalizację modelu i losowanie batchy; podziały danych (`task.split_seed`) i graf (`model.graph.seed`)
są niezależne.** Chcąc mierzyć wpływ konkretnego grafu, zmieniajcie `model.graph.seed`.

## Czego jeszcze nie ma (zob. README, tabela statusu)
Wczytanie prawdziwego connectome (krok 7-8), wybór neuronów wejścia/wyjścia/czucia z prawdziwych danych, analizy RQ6, serwer WebSocket.


## Dodatki po pierwszych eksperymentach (2026-11-04)
**D14. Rate nie uczył się - przyczyna ustalona, poprawka do sprawdzenia.** Diagnoza (E7, E8): przy przenoszeniu stanu między instrukcjami sieć (pętle zwrotne, wagi znormalizowane do sumy 1)
dla gain ≥ 3 nasyca się (79% neuronów |z| > 2 na GPU) i „zatrzaskuje” w stanie zależnym od historii; bieżąca instrukcja prawie nie wpływa na wyjście,
dlatego nie działa nawet test liniowy (kopiowanie bitów X) ani zapamiętanie jednego batcha. Dla gain 2 aktywność jest, ale wpływ wejścia to ok. 1% wpływu historii.
Hipoteza o „reżimie liniowym” z poprzedniej wersji dokumentu była błędna. `act=relu` daje NaN (aktywność nieograniczona przy 8 × 16 aktualizacjach w BPTT).
**Poprawka 1 (uruchomiona 2026-11-05, zadziałała, zob. E14):** `model.reset_each_step=true` zeruje stan sieci przed każdą instrukcją. Dla `legs_mode=teacher` pamięć nie jest potrzebna (MLP rozwiązuje zadanie), więc to uczciwy test,
czy sama struktura grafu liczy funkcję (stan, instrukcja) → nowy stan. W symulacji numpy okno wzmocnienia to ok. 6-8. **Ograniczenie:** to nie jest już model „z pamięcią”;
dla `init_only` (RQ7) trzeba będzie osobnego rozwiązania (np. słabsze pętle zwrotne, sprzężenie stanu z czuciem nóg), którego jeszcze nie mamy.
Dodany też parametr `model.input_scale` (nie pomógł w E7).

**D15. Protokół raportowania wyników.** Tabele z ostatniego kroku treningu (nie `best.pt`), średnia ± odchylenie z ≥ 5 ziaren (`scripts/summarize_runs.py`),
osobno dla każdego kodowania ekranu; `val` służy tylko do wyboru checkpointu. `val` ma 336 kroków, więc do wyników końcowych użyć `train.eval_episodes=2048`
(uwaga: zmienia to zbiory ewaluacyjne, więc nie mieszać z dotychczasowymi biegami 512).

**D16. Ograniczenia obecnych wyników.** Jeden `split_seed` (rozrzut nie obejmuje wyboru wyłączonych X i stanów), jeden graf, stała lr (opcjonalnie `train.lr_schedule=cosine`),
różne budżety kroków między kodowaniami (8k vs 20k).

**D17. Wyniki baseline'ów a decyzje o kodowaniu.** Na `binary` wszystkie modele odniesienia uogólniają na nowe X i stany, więc `binary` staje się kodowaniem głównym dla RQ4;
`onehot_signed` zostaje dla szybkich testów par i stanów. Ostatnie kroki z 5 ziaren, `scripts/summarize_runs.py`.

**D18. Poziom szumu w RQ7.** σ = 0,3 przy `onehot_signed` jest zbyt silny na pierwszą ablację (sufit odczytu instrukcji ok. 75%, GRU po 8k krokach 0,08). Proponowane σ ∈ {0,1; 0,2} i dłuższy trening
(sufit ok. 100% i 98,6%); szum dla `binary` jeszcze nie sprawdzany.

**D19. Dobór gain przez promień spektralny.** Zamiast ręcznego `init_gain` na connectome używać `graphs.spectral_radius(graph)` (estymata Gelfanda na liście krawędzi, działa dla 25,6 mln krawędzi)
i `scripts/calibrate_gain.py` (nasycenie, czułość na bieżącą instrukcję vs historię). Na małym grafie ρ = 0,533, gain krytyczny ≈ 1,9 (E12). W trybie bezstanowym dobre okno gain to ρ_eff ≈ 3-4.
Plan przejścia na connectome: `docs/plan_connectome.md`.

**D20. Rate bezstanowy działa, ale jest słabszy od modeli odniesienia.** Po `reset_each_step=true`: 2,9% neuronów nasyconych, 74% w zakresie liniowym, zapamiętanie jednego batcha i test kopiowania przechodzą,
trening schodzi do val loss 0,245 (test_pairs 0,47 po 3000 krokach, 1 ziarno). Do ustalenia: czy luka względem GRU to niedotrenowanie, skala/gain, kodowanie wejścia (`binary`), czy struktura losowego grafu.
Kolejne próby: dłuższy trening z harmonogramem `cosine`, `binary`, gain 6/10, `scripts/calibrate_gain`. Dopiero po tym ma sens rozdzielanie wpływu struktury (grafy kontrolne).

**D21. Rozdzielczość pomiarów RQ5.** `eval_every=250` jest za gruby do porównywania kolejności uczenia bitów (wszystkie bity 0-4 „załapują” między dwiema ewaluacjami). Do RQ5 uruchamiać krótkie biegi z `train.eval_every=25`.

**D22. Protokół dla modeli, które się przeuczają (Rate).** Dla Rate raportować oprócz wartości z ostatniego kroku także szczyt `test_pairs` oraz krok, w którym val loss jest najmniejszy (z `metrics.csv`);
sam ostatni krok zaniża wynik. Wybór punktu zatrzymania robić na `val`, nie na `test`. Nowa opcja `train.weight_decay` (AdamW, domyślnie 0 = bez zmian).

**D23. Wniosek z E17 dla planu connectome.** Ponieważ Rate na losowym grafie przeucza się i daje ≈ 0,5, nie będzie wiadomo, czy wynik na connectome jest „dobry”, bez punktu odniesienia z grafu losowego o tych samych parametrach i tej samej procedurze
(grafy kontrolne `degree_preserving`, `shuffled_weights` oraz małe losowe grafy z różnymi `graph.seed`). Zapisać sufit małego grafu przed przejściem na connectome.

**D24. Weight decay w Rate.** `train.weight_decay=0.05` daje test_pairs 0,76 (E21), 0,2 jest za silne. AdamW zmniejsza także `gain`, `bias`, `alpha_logit`, `b_out`, co może być niezamierzone; do rozdzielenia efektów dodano `train.wd_exclude`
(np. `"train.wd_exclude=[gain,bias,alpha_logit,b_out]"`). Dopóki to nie zostanie sprawdzone, wynik 0,76 opisywać jako „Rate z weight decay 0,05 (decay na wszystkie parametry)”.

**D25. Kodowanie `binary` nie jest remedium dla Rate** (E22): przeucza się tak samo, wyniki 0,27-0,34. `onehot_signed` z weight decay pozostaje wariantem głównym dla Rate; `binary` zostaje do testów nowych X.

**D26. Wariant Rate do punktu kontrolnego: `experiments/configs/rate_best.yaml`** (stan zerowany, AdamW 0,05 na wszystkie parametry, 6000 kroków, `onehot_signed`). Rozstrzygnięcie D24: decay z wykluczeniem `gain`/`bias`/`alpha_logit`/`b_out` jest **gorszy** (0,48-0,64 vs 0,76),
więc nie wolno opisywać poprawy jako „regularyzacji wag wejścia i odczytu”. Poprawny opis: „Rate z weight decay 0,05 (decay na wszystkie parametry; mechanizm nieustalony)”.
Sufit małego grafu raportować jako średnią ± odchylenie z 5 ziaren tego wariantu.

**D27. Podział danych.** `split_seed=0` i `split_seed=1` dają zgodne wyniki główne dla GRU `binary` (E26), więc `split_seed=0` zostaje podziałem głównym. Analizy typu „która komórka (op, s5, x5, c5) jest trudna” NIE uogólniają się między podziałami i nie powinny trafiać do raportu jako własność zadania.

**D28. `init_only` (RQ7).** Do raportu: wynik dla GRU 60 (ok. 0,45), LSTM 50 (0,78), GRU 128 (0,80), z zastrzeżeniem 1 ziarna dla dwóch ostatnich. Nie wiadomo, czy LSTM/GRU 128 dojdą do `teacher` (0,98) przy dłuższym treningu.

**D29. Protokół raportowania dla niestabilnego treningu (Rate).** Wartość z ostatniego kroku jest nierzetelna, gdy trening ma skoki straty (E28: ziarno 4 spadło z 0,845 do 0,170 w ostatnim kroku). Dla Rate raportować: (1) wynik `best.pt` wybranego po `val` (`scripts/eval_checkpoints`),
(2) medianę ostatnich 5 ewaluacji z `metrics.csv`, (3) liczbę ziaren, które wystartowały (jedno z pięciu miało plateau do kroku ok. 5100). Rozważyć trening z `lr_schedule=cosine` i 10 000 krokami, żeby koniec treningu był stabilny.

**D30. Mechanizm weight decay (aktualizacja D24/D26).** Poprawa pochodzi głównie z decay na `gain` (E29). Opis: „Rate z weight decay 0,05 (decay na wszystkie parametry); główny wkład daje decay na wzmocnienie `gain`; wartość końcowa `gain` niezmierzona”. Dodano logowanie `gain` w `train.train` (`final_gain` w `results.json`) i w `eval_checkpoints`.

**D31. Liczba do raportu dla Rate (mały graf).** `best.pt` wybrany po `val`: test_pairs 0,757 ± 0,145 (5 ziaren); 0,819 ± 0,049 dla 4 ziaren, które wystartowały; jedno ziarno wystartowało późno (0,509). Opis mechanizmu: „decay na `gain` poprawia wynik; `gain` rośnie w treningu do ok. 13-14,6 (z 8)”.
