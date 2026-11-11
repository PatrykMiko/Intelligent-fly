# Skrypt na punkt kontrolny (część „model”)

Czas: ok. 7-8 minut. Mówimy prosto; liczby w nawiasach to te, które pokazujemy na slajdach (źródło: `docs/eksperymenty.md`).
**Najważniejsza zasada:** nie mówimy, że model działa na connectome. Mówimy, że sprawdziliśmy całą metodę na **małym grafie losowym** i że connectome to następny krok.

---

## Slajd 1. Co robimy
- Uczymy sieć neuronową dodawania i odejmowania mod 64. Docelowo ma to robić sieć o strukturze prawdziwego mózgu muszki (connectome).
- Moja część to model i eksperymenty (Python / PyTorch). Kolega robi wizualizację (OpenGL).
- Na tym punkcie kontrolnym pokazujemy: zadanie, jak je testowaliśmy, wyniki modeli porównawczych i **wyniki na małym grafie**. Connectome jest następny.

## Slajd 2. Zadanie
- Stan to liczba 0-63 zapisana na 6 nogach (noga podniesiona = bit 1).
- Instrukcja to `+X` albo `-X`, pokazana na ekranie 8×8. Nowy stan = `(stan ± X) mod 64`. Epizod to 16 instrukcji.
- Dlaczego to nie jest proste: każdy bit wyniku to XOR trzech rzeczy (bit stanu, bit X, przeniesienie z niższych bitów). Model, który tylko „dodaje wpływy”, zostaje na zgadywaniu. Sprawdziliśmy: model liniowy ma stratę 0,69, czyli rzut monetą.

## Slajd 3. Jak to testujemy
- Uczciwy test: model dostaje w testach rzeczy, których nie widział w treningu: **nowe pary** (stan, X), **nowe wartości X**, **nowe stany**.
- Każdy wynik to **5 ziaren losowych** (średnia ± odchylenie), żeby nie opierać się na jednym szczęśliwym biegu.
- Dwa sposoby pokazania instrukcji na ekranie: „jeden piksel = jedna liczba” (`onehot`) i „liczba zapisana binarnie” (`binary`).

## Slajd 4. Modele porównawcze: co umiemy
- Przetestowaliśmy 6 modeli o podobnej liczbie parametrów (ok. 24 tys.): liniowy, MLP, Elman, GRU, LSTM, Transformer.
- Na kodowaniu **binary** wszystkie poza liniowym uczą się zadania i uogólniają na nowe pary, nowe X i nowe stany (0,97-1,00). Transformer ma 1,000.
- Ważna lekcja: na kodowaniu **onehot** nowe wartości X są dla każdego modelu nierozwiązywalne (poziom zgadywania). Powód: „zapalony piksel” dla wartości, której nigdy nie było w treningu, nie niesie żadnej wiedzy. To błąd kodowania, nie modelu.
- Najmłodszy bit uczy się pierwszy, najstarszy (bit 5) ostatni.

## Slajd 5. Model na grafie
- **Co to jest:** sieć, w której struktura połączeń jest ustalona i nie jest uczona. Uczymy tylko: jak ekran trafia do sieci, jak odczytujemy nogi i parametry neuronów (próg, szybkość reakcji, wzmocnienie).
- **Na czym:** na małym grafie losowym (ok. 3600 neuronów, ok. 95 tys. połączeń), nie na connectome. To próba generalna całej metody.
- **Droga, której się nauczyliśmy:**
  1. Najpierw sieć nic nie uczyła się: neurony były „zalane” sygnałem (79% nasyconych) i sieć trzymała się starego stanu.
  2. Zerowanie stanu przed każdą instrukcją naprawiło to: sieć zaczęła się uczyć.
  3. Potem się przeuczała (zapamiętywała pary). Pomogła kara za zbyt duże parametry (weight decay), głównie na wzmocnieniu sieci.
- **Wynik:** 0,757 ± 0,145 na niewidzianych parach (5 ziaren). Cztery z pięciu ziaren: 0,82. Jedno ziarno wystartowało za późno (0,51). Dla porównania GRU ma ok. 0,98.

## Slajd 6. Najważniejsze wnioski
1. Zadanie jest rozwiązywalne: modele standardowe uczą się go prawie bezbłędnie.
2. **Sposób pokazania wejścia decyduje o uogólnianiu** (onehot kontra binary).
3. Sieć o ustalonej strukturze **się uczy**, ale jest słabsza od standardowych modeli i wrażliwa na ustawienia (wzmocnienie, regularyzacja, niestabilny trening).
4. Pamięć jest trudna: gdy stan nóg widać tylko na początku, GRU z 60 jednostkami stoi na ok. 0,45, a LSTM i większy GRU dochodzą do ok. 0,8.
- **Ograniczenia, które mówimy sami:** to graf losowy, nie connectome; sieć na grafie nie ma pamięci między instrukcjami (stan zerujemy); wyniki sieci na grafie są niestabilne między ziarnami.

## Slajd 7. Co dalej: connectome
- Zbiór MaleCNS: ok. 166 tys. neuronów, 25,6 mln połączeń.
- Kroki: wczytać dane, **wybrać neurony** wejścia, wyjścia (po 6 nóg) i czucia, dobrać wzmocnienie (przez promień spektralny, mamy już narzędzie), wytrenować na podgrafie, potem na całości.
- Pytania badawcze: czy sama struktura prawdziwego mózgu wystarcza (RQ1), czy jest lepsza od grafów kontrolnych o tej samej gęstości (RQ2), jak wypada na tle standardowych modeli (RQ3).
- Uczciwie: wynik „connectome jest taki sam jak graf losowy” też byłby wynikiem (znaczyłby, że uczone wejście i odczyt wystarczają).

---

## Pytania, które mogą paść, i uczciwe odpowiedzi
- **Czy to jest już connectome?** Nie. To graf losowy podobnej wielkości. Sprawdziliśmy na nim metodę, bo na connectome nie rozdzielilibyśmy błędów metody od błędów danych.
- **Dlaczego sieć na grafie jest słabsza od GRU?** Nie wiemy. W grafie uczymy tylko wejście, odczyt i parametry neuronów, a połączenia są ustalone; to może ograniczać. To jest właśnie pytanie badawcze RQ1/RQ3, które chcemy sprawdzić na connectome.
- **Co znaczy ±0,145?** Że wyniki między ziarnami bardzo się różnią: jedno ziarno wystartowało późno (0,51). Dla czterech pozostałych jest 0,82 ± 0,05.
- **Czemu pomaga weight decay?** Głównie decay na wzmocnieniu sieci. Zmierzyliśmy, że wzmocnienie rośnie w treningu z 8 do 13-14,6; hipoteza jest taka, że decay ogranicza ten wzrost. Mechanizmu jeszcze nie potwierdziliśmy (sprawdzenie na starszych checkpointach jest w planie).
- **Czy wyniki są wiarygodne?** Wyniki modeli porównawczych: tak (5 ziaren, dwa podziały danych). Wynik sieci na grafie: jedno ustawienie, 5 ziaren, duży rozrzut, więc podajemy go ostrożnie.
- **Co z pamięcią (stan nóg widoczny tylko na początku)?** Sieć na grafie jej nie ma. Standardowe modele radzą sobie tylko częściowo (GRU 60: 0,45; LSTM: 0,78). To zostaje otwarte.
