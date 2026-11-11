# Plan: od małego grafu do prawdziwego connectome

Stan na 2026-11-05. Dokument odpowiada na pytanie „co po tym, jak Rate zacznie się uczyć na małym grafie?”. To plan, nie wynik:
niczego z kroków 3-6 jeszcze nie zrobiliśmy, a o samych danych wiemy tylko tyle, ile jest w opisach publicznych (zob. „Dane”).

## 0. Bramka: kiedy wolno ruszyć connectome
Connectome ruszamy dopiero, gdy na MAŁYM grafie (`rate_small`) spełnione są wszystkie punkty:
1. `copy_test` schodzi do ~0 (ścieżka sygnału działa).
2. `rate_small` z `reset_each_step=true` na `train` schodzi wyraźnie poniżej 0,693, a na `test_pairs` ma dokładność całej liczby istotnie powyżej zgadywania
   (docelowo > 0,9 jak modele odniesienia, ale nie jest to warunek konieczny: wystarczy, że wynik rośnie z treningiem i nie jest artefaktem).
3. `scripts/calibrate_gain` dla wybranego gain pokazuje małe nasycenie i niezerową czułość na bieżącą instrukcję.
4. Diagnostyka (`diagnose_rate`) pokazuje, że sieć potrafi zapamiętać jeden stały batch.
Powód: gdy Rate nie działa na prostym grafie, na connectome nie da się rozdzielić „zła dynamika” od „zła struktura / zły dobór neuronów”.

**Stan bramki na 2026-11-05:** (1) `copy_test` - **spełnione** (strata 0,0043 po 600 krokach); (4) zapamiętanie jednego batcha - **spełnione** (exact 0,62 po 400 krokach, wciąż rośnie);
(2) `rate_small` bezstanowy poniżej 0,693 - **spełnione** (val loss 0,245, test_pairs 0,47 po 3000 krokach, 1 ziarno), ale daleko od modeli odniesienia (≈ 0,98); (3) `calibrate_gain` - **uruchomione** (E16, działa, gain 8 daje 2,5% nasycenia). Aktualizacja po 10k krokach (E17): Rate przeucza się na `onehot_signed` (szczyt test_pairs 0,56, końcowo 0,51), na `binary` uczy się wolno (0,27-0,38); luka względem GRU ≈ 0,98 jest duża.
Aktualizacja (E21, E25): z weight decay 0,05 na wszystkie parametry Rate osiąga test_pairs 0,76 (1 ziarno), ale z wykluczeniem `gain`/`bias`/`alpha_logit`/`b_out` z decay tylko 0,48-0,64, więc poprawa nie pochodzi z regularyzacji wag wejścia; mechanizm nieustalony (D26). Aktualizacja (E28-E29): `rate_best` ma 4 z 5 ziaren w zakresie 0,78-0,87 (jedno późno wystartowało, jedno zepsuł skok straty w ostatnim kroku); poprawę daje głównie decay na `gain`. Ocena `best.pt` (E30): test_pairs 0,757 ± 0,145 (5 ziaren), 0,819 ± 0,049 dla czterech, które wystartowały. Bramka zaliczona; punkt kontrolny można zamknąć tą liczbą.
Czyli bramka jest zaliczona technicznie, ale przed connectome warto ustalić sufit małego grafu (weight decay, wcześniejsze zatrzymanie, dłuższy trening na `binary`, kilka ziaren), żeby wiedzieć, jak interpretować wynik na connectome.

## 1. Uwaga o grafach kontrolnych na małym grafie
`degree_preserving`, `shuffled_weights` itd. na losowym grafie warstwowym dają grafy prawie równoważne oryginałowi (oryginał też jest losowy),
więc różnica między nimi nie powie nic o strukturze mózgu. Na małym grafie służą tylko do sprawdzenia, że kod działa (trening nie wybucha).
Prawdziwe RQ2 ma sens dopiero na connectome.

## 2. Dane (krok 7 listy)
- **Co wiemy z publicznych opisów:** MaleCNS v1.0 to connectome całego ośrodkowego układu nerwowego dorosłego samca: mózg, płaty wzrokowe i brzuszny sznur nerwowy,
  166 700 neuronów i 25,6 mln połączeń. W plan wpisaliście zbiór `fernandofernandes/fly-connectome-malecns-166k`; **nie udało mi się potwierdzić jego istnienia ani formatu** -
  sprawdźcie to jako pierwszy krok. Znalazłem inne paczki z tymi samymi danymi (np. `QuixiAI/MaleCNS` na Hugging Face: oficjalne pliki „flat-connectome” z bucketu Janelia, liczby synaps bez zmian, safetensors).
- **Prior art (do kroku 1 listy: przegląd literatury):** ten sam schemat „zamrożony connectome + uczone wejście/odczyt” pojawił się w demonstracjach typu *fly-llm-hf* / *FLM*
  (zamrożona sieć MaleCNS jako rezerwuar echo-state dla małego modelu językowego, uczony odczyt) oraz w symulacjach spikingowych (np. `flybrain`, „Fly Rig”). Warto je przeczytać pod kątem
  skalowania wag, doboru wejść/wyjść i reżimu dynamiki. To są projekty demonstracyjne, nie recenzowana literatura; do raportu potrzebne są też prace z listy materiałów (Shiu i in. 2024, Lappalainen i in. 2024).
- **Do ustalenia na prawdziwych danych (bez zgadywania):** czy są adnotacje klas neuronów (które neurony ruchowe obsługują które nogi), znaki neuroprzekaźników (GABA, glutaminian, histamina),
  pozycje komórek (do wizualizacji kolegi) i jak oznaczone są neurony czuciowe. Plan zakłada fotoreceptory jako wejście - w tym zbiorze mogą nie być jako osobne neurony (to zależy od zakresu rekonstrukcji); sprawdźcie.

## 3. Implementacja (kroki 7-8)
1. `graphs/connectome.py` zwracający `GraphSpec` (znaki z neuroprzekaźnika presynaptycznego, moduły wag z liczby synaps, `normalize()`); `models.build_graph(kind: connectome)`.
2. **Wybór ról** (największa decyzja projektowa): neurony wejścia (ekran), czucia położenia nóg, ruchowe dla każdej z 6 nóg. Zapisać kryteria w `docs/decyzje.md` (D-nowe) i trzymać je w configu.
   Policzyć, ile neuronów ruchowych przypada na nogę (H5c: nogi mogą się różnić).
3. **Pośredni rozmiar:** zanim pełne 166 tys. neuronów, wziąć podgraf (np. neurony w odległości ≤ k skoków od wejść lub wyjść, ewentualnie kilka–kilkadziesiąt tysięcy neuronów).
   Wiele publicznych demonstracji używa podgrafów, np. kilkutysięcznych korytarzy.
4. **Wydajność:** 25,6 mln krawędzi × batch × 8 kroków wewnętrznych × 16 instrukcji. `RateModel` buduje macierz rzadką w każdym forward (zob. docstring) - przed pełnym grafem zbudować ją raz i trzymać na GPU,
   rozważyć mniejszy batch, `inner_steps`, obcięte BPTT i mieszaną precyzję. Zmierzyć pamięć na małym podgrafie przed skalowaniem.
5. **Dynamika na prawdziwym grafie.** Gain z małego grafu NIE przenosi się: trzeba powtórzyć `scripts/calibrate_gain` na connectome. Promień spektralny (`graphs.spectral_radius`) jest dobrym punktem odniesienia
   (na małym grafie ρ = 0,533, a przejście do podtrzymywanej aktywności widać przy gain ≈ 1/ρ = 1,9). Prawdziwy connectome ma silne pętle zwrotne, więc spodziewajcie się powtórki problemu nasycenia (E7-E8).
   `inner_steps` powinno być ≥ liczby skoków wejście → wyjście; sprawdzić najkrótsze ścieżki (krok „wizualizacja połączeń” i tak ich potrzebuje).

## 4. Kolejność eksperymentów na connectome
1. Rate na connectome, `reset_each_step=true`, `binary` i `onehot_signed`, ≥ 5 ziaren, z kalibracją gain - odpowiedź na RQ1 (czy struktura wystarcza).
2. Grafy kontrolne (RQ2): `degree_preserving`, `shuffled_weights`, `no_signs`, `random_signs` - te same hiperparametry i dopasowane parametry.
   Silna hipoteza (struktura pomaga) wymaga różnicy większej niż rozrzut między ziarnami; sam odczyt i wejście z uczonymi parametrami mogą dawać podobne wyniki dla każdego grafu o tej gęstości.
3. Porównanie z modelami odniesienia przy tej samej liczbie trenowanych parametrów (RQ3): `scripts/count_params` zwróci inne rozmiary niż dla małego grafu, bo `W_in`, `bias`, `alpha` rosną z liczbą neuronów.
4. Analizy RQ5-RQ6 (które bity, które regiony, wyłączanie neuronów).
5. Dopiero na końcu tryb z pamięcią (`init_only`, RQ7) - nierozwiązany dla Rate (zob. niżej).

## 5. Znane ryzyka
- **Rate bez zerowania stanu:** w symulacji (E8) przy gain ≥ krytycznego sieć się zatrzaskuje; przy gain poniżej krytycznego (ρ_eff ≈ 0,9) jest wejściowo-sterowana i bez nasycenia (pamięć wygasa),
  ale aktywność w głębi sieci jest bardzo mała (rzędu 0,01-0,05), więc nieliniowość nie jest angażowana. Dobrego reżimu z pamięcią i nieliniowością jeszcze nie mamy.
- **Zerowanie stanu** rozwiązuje tryb `teacher`, ale tym samym RQ7 (pamięć) trzeba oprzeć na innym rozwiązaniu albo uczciwie zapisać ograniczenie.
- **Interpretacja:** jeśli Rate osiągnie wynik podobny do grafów kontrolnych, to wniosek brzmi „uczone wejście/wyjście wystarczają, struktura nie pomaga” - to też jest wynik, nie porażka. Zapiszcie to w hipotezach przed eksperymentem.
- **Czas:** 14-tygodniowy plan miał connectome w tygodniach 5-7; jesteście z tym mniej więcej w tygodniu 4 ze stratą, ale baseline'y i diagnostyka są już gotowe, więc część fazy 3 jest zrobiona.
