# Pytania badawcze i hipotezy (SZKIC)

> **Status: szkic do zatwierdzenia przez zespół PRZED pierwszym eksperymentem.** Po zatwierdzeniu
> zróbcie commit z datą i nie edytujcie treści hipotez - nowe obserwacje dopisujcie na dole w sekcji
> „Aktualizacje po wynikach”. To krok 2 z listy 30 kroków.
>
> Hipotezy napisane zostały bez żadnych wyników modeli; to oczekiwania wynikające z teorii
> (XOR nie jest liniowo separowalny, przeniesienia wymagają propagacji informacji między bitami itd.).
> Nikt tu jeszcze niczego nie zmierzył - jeśli któraś hipoteza Wam nie pasuje, zmieńcie ją teraz, nie po fakcie.

Miara sukcesu: **dokładność całej liczby** (wszystkie 6 bitów poprawne) i **dokładność per bit** na podziałach
`val`, `test_pairs`, `test_x`, `test_states`; strata odniesienia dla zgadywania to ln 2 ≈ 0,693.
Wszystkie wnioski na podstawie ≥ 5 ziaren; raportujemy średnią ± odchylenie.

## RQ1 - czy sama struktura connectome wystarcza?
*Czy sama struktura połączeń connectome, przy nauczonych tylko parametrach wejścia, wyjścia i dynamiki, wystarcza do nauczenia się dodawania i odejmowania mod 64?*
- **H1a:** Model `Rate` na connectome schodzi wyraźnie poniżej 0,693 (uczy się czegoś więcej niż zgadywanie), bo dynamika z nieliniowością i pętlami umożliwia interakcje wejście × stan.
- **H1b:** Pełne 6/6 bitów (dokładność liczby > 95% na `test_pairs`) jest mniej prawdopodobne niż częściowe: spodziewam się, że część bitów (niższe) nauczy się, a wyższe, zależne od długiego łańcucha przeniesień, będą słabsze.
- **Obalenie H1a:** strata na `val` pozostaje w granicach ±0,01 od 0,693 po pełnym treningu na ≥ 5 ziarnach.

## RQ2 - connectome vs grafy kontrolne
- **H2:** Prawdziwy connectome uczy się **nie gorzej** niż grafy kontrolne. Nie zakładam przewagi: duża losowa sieć rekurencyjna o podobnej gęstości może uczyć się równie dobrze, bo uczone parametry wejścia/wyjścia dużo „robią”. Silniejsza hipoteza (struktura pomaga) wymaga różnicy większej niż rozrzut między ziarnami.
- **H2b:** Usunięcie znaków (graf bez hamowania) pogarsza wyniki, bo bez hamowania aktywność łatwo nasyca się.
- **Obalenie silnej wersji:** różnica średnich connectome − kontrola mniejsza niż 1 odchylenie standardowe między ziarnami.

## RQ3 - connectome vs standardowe modele o tej samej liczbie parametrów
- **H3:** Przy tej samej liczbie *trenowanych* parametrów GRU/LSTM wypadną lepiej niż `Rate` na connectome (gdzie większość parametrów to stałe połączenia, a nie uczone), a MLP w trybie `teacher` będzie bardzo mocnym punktem odniesienia, bo zadanie na poziomie kroku nie wymaga pamięci.
- **Obalenie:** `Rate` na connectome osiąga wynik równy lub lepszy od GRU przy porównywalnej liczbie parametrów.

## RQ4 - uogólnianie
- **H4:** Kolejność trudności: `test_pairs` (najłatwiejsze) > `test_x` > `test_states`. Modele nauczą się reguły per bit, więc nowe pary powinny działać; niewidziane X i niewidziane stany to sprawdzian, czy sieć uogólnia regułę, czy zapamiętuje tabelę.
- **H4b:** Modele nierekurencyjne bez struktury (MLP) mogą zapamiętywać tabelę (stan, X) i na `test_x`/`test_states` spaść do poziomu zgadywania.

## RQ5 - które bity, przeniesienia, przypisanie bitów do nóg
- **H5a:** Bit 0 (bez przeniesienia: XOR dwóch bitów) uczy się najszybciej; im wyższy bit, tym później.
- **H5b:** Błędy koncentrują się tam, gdzie przeniesienie/pożyczka propaguje się przez ≥ 3 bity.
- **H5c:** Przypisanie bitów do nóg ma znaczenie dla `Rate`, bo nogi różnią się liczbą neuronów ruchowych i czuciowych w connectome (do sprawdzenia po wczytaniu prawdziwych danych).

## RQ6 - które regiony biorą udział
- **H6:** Wyłączenie neuronów o największej aktywności dla danego bitu psuje ten bit bardziej niż losowo wybrane neurony; efekt specyficzny dla bitu będzie widoczny dla bitów nauczonych, nie dla zgadywanych.

## RQ7 - długość sekwencji i szum
- **H7a:** W trybie `init_only` (model musi pamiętać stan) dokładność spada z długością epizodu; w `teacher` długość nie ma znaczenia.
- **H7b:** Szum na ekranie psuje najpierw modele uczące się zapamiętanej tabeli (typowo MLP), `Rate` i GRU degradują się łagodniej.

---
## Aktualizacje po wynikach
(puste - dopisujcie z datą; nie zmieniajcie hipotez powyżej)

### 2026-11-04 (GRU, 5 ziaren; Rate, 1 ziarno - szczegóły w `docs/eksperymenty.md`)
- **RQ4 / H4 (kolejność trudności test_pairs > test_x > test_states) - niespełniona dla `onehot_signed`:** test_x = 0,016 ± 0,009 (poziom zgadywania),
  test_states = 0,974 ± 0,005. Przyczyna techniczna: piksele wyłączonych X nigdy nie świecą w treningu, więc ich wagi wejściowe nie są uczone.
  To wynik o kodowaniu wejścia, nie o zdolnościach modelu. Dla `binary`: test_pairs 0,989 ± 0,009, test_x 0,985 ± 0,014, test_states 0,973 ± 0,014,
  czyli nowe X uogólniają się tak dobrze jak nowe pary, a kolejność trudności brzmi: nowe pary ≈ nowe X > nowe stany (różnice w granicach rozrzutu).
- **RQ3 / H3:** mamy dopiero punkt odniesienia GRU (zbiega do ≈ 0,98-0,99 na parach). Brakuje `linear`, `mlp`, `elman`, `lstm`, `transformer` i działającego `Rate`,
  więc H3 jest jeszcze nieoceniona.
- **RQ1 / H1a:** pierwsza próba na małym losowym grafie warstwowym (nie connectome): `Rate` nie schodzi poniżej ln 2 przez 3000 kroków,
  a zmiana `init_gain` (8, 14, 20) nic nie zmienia. To jeszcze nie jest test hipotezy - model technicznie nie działa; diagnoza w toku. H1a pozostaje nierozstrzygnięta.
- **H5, H7:** jeszcze niesprawdzane (brak analizy per bit i ablacji `legs_mode`, szumu, długości).


### 2026-11-04, druga seria (baseline'y 5 ziaren, ablacje GRU, diagnostyka Rate; szczegóły: `docs/eksperymenty.md` E7-E11)
- **RQ3 / H3:** na `binary` wszystkie rodziny osiągają 0,986-1,000 na parach (Transformer 1,000 ± 0,000, LSTM 0,991, GRU 0,989, MLP 0,987, Elman 0,986);
  na `onehot_signed` Elman 0,998, Transformer 0,996, GRU 0,978, LSTM 0,961, MLP 0,921. Część H3 („MLP będzie mocnym punktem odniesienia”) potwierdzona dla `binary`,
  nie dla `onehot_signed`. Część „GRU/LSTM wypadną lepiej niż Rate” jest nieoceniona: Rate technicznie nie działa.
- **RQ4 / H4b** („MLP zapamiętuje tabelę i spada do zgadywania na test_x/test_states”): **niepotwierdzona.** Na `binary` MLP uogólnia (test_x 0,982, test_states 0,984);
  na `onehot_signed` test_states 0,934 (niżej niż modele rekurencyjne, ale daleko od zgadywania), a test_x to zgadywanie u wszystkich modeli (artefakt kodowania).
- **RQ5 / H5a:** zgodne z końcowymi tabelami: bity 0-4 ≈ 1,000, najsłabszy jest bit 5 (najstarszy) we wszystkich modelach. Kolejność uczenia bitów w czasie i H5b (przeniesienia) jeszcze nie zbadane.
- **RQ7 / H7a:** w trybie `teacher` długość epizodu nie szkodzi (32 kroki: 0,984 vs 16 kroków: 0,978) - zgodnie z hipotezą. W trybie `init_only` GRU po 8k kroków ma 0,19 (trening niezbieżny), więc brak wniosku.
  **H7b** (szum): GRU przy σ = 0,3 załamuje się (0,08) mimo sufitu ok. 0,75 - wynik mówi o niedotrenowaniu, nie o odporności; inne modele nie testowane.
- **RQ1 / H1a:** nadal nierozstrzygnięta. Diagnoza `Rate`: sieć jest nasycona (79% neuronów |z| > 2) i „zatrzaskuje się” (wpływ historii ≫ wpływ bieżącej instrukcji), nie zapamiętuje nawet jednego batcha
  ani nie uczy się zadania liniowego (kopiowanie bitów X). To problem dynamiki modelu, nie hipotezy o connectome.

### 2026-11-05 (Rate bezstanowy, ablacje GRU, analiza bitów; `docs/eksperymenty.md` E13-E15)
- **RQ1 / H1a:** nadal nierozstrzygnięta (graf nie jest connectome), ale technicznie `Rate` **działa** w trybie bezstanowym: val loss 0,245, test_pairs 0,47 po 3000 krokach (1 ziarno, wciąż rośnie),
  przy czym GRU osiąga ≈ 0,98. Hipoteza H3 („GRU/LSTM lepsze od Rate”) jest więc na razie zgodna z danymi, ale porównanie jest nierówne (Rate: 3000 kroków, 1 ziarno).
- **RQ5 / H5a** (bit 0 najszybciej, wyższe później): niepotwierdzona na rozdzielczości 250 kroków - bity 0-4 przekraczają próg 0,9 jednocześnie (mediana kroku 750, 5/5 ziaren), bit 5 później (1000).
  **H5b** (błędy tam, gdzie łańcuch przeniesienia ≥ 3 bity): tylko częściowo; w GRU `binary` seed 0 prawie wszystkie błędy są w bicie 5 i występują też bez przeniesienia (2,7%), nie rosną monotonicznie z długością łańcucha.
- **RQ7 / H7a:** `init_only` po 30k krokach: 0,45 ± 0,07 (test_pairs), silnie zależne od ziarna (0,39-0,53), niezbieżne - H7a (pamięć wymusza pogorszenie wraz z długością) wymaga dłuższych biegów, na razie nie oceniamy.
  **H7b:** szum 0,2 na `onehot_signed`: plateau 0,22 mimo sufitu > 0,9; wynik dotyczy GRU i kodowania, nie wnioskujemy o innych modelach.

### 2026-11-05, druga tura wyników (E16-E20)
- **RQ5 / H5a: potwierdzona** na gęstej ewaluacji (GRU, `onehot_signed`, 5 ziaren): mediana kroku, w którym bit przekracza 0,9 na `val`: bity 0 i 1 - 575, bit 2 - 625, bit 3 - 675, bit 4 - 725, bit 5 - 850.
  To **koryguje** wpis z tego samego dnia („bity 0-4 jednocześnie”), który wynikał z rozdzielczości 250 kroków. **H5b** (długość łańcucha przeniesienia): niepotwierdzona; błędy GRU `binary` skupiają się w bicie 5 w kilku komórkach
  (op, s5, x5, c5), zależą od ziarna, a w obu zbadanych ziarnach najgorsza jest komórka dodawania z s5 = x5 = 1 i c5 = 0 (8,0% i 4,2%).
- **RQ1 / H1a i RQ3 / H3:** `Rate` bezstanowy na losowym grafie: szczyt test_pairs 0,56 (`onehot_signed`, gain 8), silne przeuczenie (strata treningowa ≈ 0 przy rosnącej stracie walidacyjnej), `binary` bez przeuczenia, ale 0,27-0,38.
  Wyraźnie poniżej GRU (≈ 0,98) i MLP (0,92-0,99): H3 („GRU/LSTM lepsze od Rate”) jest zgodna z danymi dla małego grafu. H1a (czy struktura connectome wystarcza) nadal nieoceniona: to jeszcze nie connectome.
- **RQ7 / H7b:** na `binary` GRU znosi szum σ = 0,2 znacznie lepiej niż na `onehot_signed` (0,741 ± 0,008 vs 0,217; sufit odczytu ok. 0,96 dla `binary`). H7a (`init_only`): seed 0 po 60k krokach z `cosine` stoi na 0,38,
  a seedy 1 i 2 dochodziły wcześniej do 0,51-0,53 - silna zależność od ziarna, wynik niezbieżny.

### 2026-11-05, czwarta tura (E21-E24)
- **RQ1 / H1a i RQ3 / H3:** `Rate` bezstanowy z weight decay 0,05 osiąga test_pairs 0,76 (1 ziarno, `onehot_signed`), wcześniej 0,51-0,56. Nadal poniżej GRU (≈ 0,98) i MLP (0,92-0,99), więc H3 jest zgodna z danymi dla małego grafu losowego.
  Nie wiadomo, ile poprawy daje regularyzacja, a ile zmniejszenie `gain` (AdamW zmniejsza też `gain`). H1a (struktura connectome) bez zmian: to jeszcze nie connectome.
- **RQ5 / H5b:** komórki z s5 = x5 = 1 są najtrudniejsze we wszystkich 5 ziarnach GRU `binary` (średnio 4,2% dla `+`/c=0 i 3,1% dla `-`/c=1), błąd nie rośnie z długością łańcucha przeniesienia. H5b (łańcuchy) odrzucona, hipoteza o zawinięciu mod 64 / s5 = x5 = 1 jest obiecująca, ale niepotwierdzona.
- **RQ7 / H7a:** diagnoza `init_only`: gubienie stanu w czasie jest łagodne (0,76 → 0,60 przez 16 kroków), głównym problemem jest jeden środkowy bit, którego model się nie nauczył (inny w różnych ziarnach). Nie potwierdza hipotezy o kumulacji błędów jako głównej przyczynie.

### 2026-11-05, piąta tura (E25-E27)
- **RQ5 / wzór s5 = x5 = 1 (E23) - wycofany jako ogólny:** przy `split_seed=1` ta komórka praktycznie nie błądzi, a błędy bitu 5 są rozproszone i małe. Poprzedni wniosek („oba najstarsze bity zapalone są najtrudniejsze”) był cechą podziału `split_seed=0`.
  Stabilne między podziałami: błędy tylko w bicie 5 (śladowo 4), brak zależności od długości łańcucha przeniesienia (H5b odrzucona), słabszy krok t = 0 (o 1-4 pp). H5a (kolejność bitów) potwierdzona wcześniej (E18).
- **RQ1 / H1a i RQ3 / H3:** poprawa Rate z weight decay (0,51-0,56 → 0,76) **nie pochodzi z regularyzacji wag W_in, W_sens, v**: z wykluczeniem `gain`, `bias`, `alpha_logit`, `b_out` z decay wynik to 0,48-0,64. Mechanizm (gain? bias/alpha? b_out?) nieustalony.
  H3 pozostaje zgodna z danymi (Rate ≤ 0,76 vs GRU ≈ 0,98) dla małego grafu losowego, 1 ziarno.
- **RQ7 / H7a:** hipoteza „pamięć w trybie `init_only` jest barierą” nie potwierdza się jako ogólna: LSTM (hidden 50, 24,7 tys. parametrów) osiąga 0,78 i GRU hidden 128 0,80 (1 ziarno każdy, 30k kroków, nadal rosną), a GRU hidden 60 ok. 0,45.
  Trudność `init_only` zależy więc od architektury i pojemności, nie od samego wymogu pamiętania stanu.

### 2026-11-06, szósta tura (E28-E29)
- **RQ1 / H1a i RQ3 / H3:** `rate_best` na małym grafie losowym: 4 z 5 ziaren osiąga 0,78-0,87 na niewidzianych parach (mediana ostatnich ewaluacji 0,823 ± 0,037 dla tych czterech; wszystkie pięć: 0,751 ± 0,164, ostatni krok 0,627 ± 0,294 z powodu skoków straty i późnego startu ziarna 3).
  Pozostaje poniżej GRU (≈ 0,98) i MLP (0,92-0,99), więc H3 jest zgodna z danymi. H1a (struktura connectome) nadal nieoceniona: to graf losowy.
- **Mechanizm (korekta E21, E25):** poprawa po weight decay wynika głównie z decay na `gain` (samo `gain`: 0,70, brak przeuczenia), a nie z regularyzacji wag wejścia/odczytu (0,55) ani `bias`/`alpha_logit` (0,52). Hipoteza: przeuczenie wiąże się z dużym wzmocnieniem; decay je zmniejsza w trakcie treningu (niezmierzone).

### 2026-11-08 (E30)
- **Korekta mechanizmu:** `gain` po treningu z weight decay wzrósł z 8 do 12,9-14,6 (nie zmalał). Hipoteza „decay zmniejsza gain” jest nieprawdziwa. Aktualna (niezmierzona): decay na `gain` ogranicza jego WZROST, a duży `gain` sprzyja przeuczeniu.
- **RQ1 / H1a i RQ3 / H3:** Rate (`rate_best`, mały graf losowy, `best.pt`): test_pairs 0,757 ± 0,145 (5 ziaren), 0,819 ± 0,049 dla czterech, które wystartowały. GRU ≈ 0,98, MLP 0,92-0,99. H3 zgodna z danymi; H1a dopiero na connectome.
