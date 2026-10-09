# Personality Type Predictor 🌊

Ein Machine-Learning-Abschlussprojekt: Ein Modell sagt anhand eines kurzen
Big-Five-Fragebogens (19 Fragen, Alter, Geschlecht, Schreibhand) den
**Persönlichkeitstyp** einer Person voraus. Das Modell wird über eine lokale
**Streamlit-App** bereitgestellt, in der man den Fragebogen ausfüllt und
sofort das Ergebnis sieht.

**Datensatz (Google Drive):**
https://drive.google.com/drive/folders/1KhwTPAG07EdaENW_XX9nVvKhC-DP1Ags?usp=sharing

## Overview

### Was das Projekt tut

Das Projekt simuliert den gesamten Machine-Learning-Prozess von den Rohdaten
bis zur Anwendung: Datenexploration, Bereinigung, Vorverarbeitungs-Pipeline,
Modellvergleich mit Hyperparameter-Tuning, Speichern des besten Modells und
eine Streamlit-App, die das gespeicherte Modell lädt und Vorhersagen liefert.

### Das Problem

Gesucht ist der **Persönlichkeitstyp** einer Person. Es handelt sich um eine
**überwachte Klassifikationsaufgabe mit vier Klassen**. Die Eingabe sind die
Antworten auf 19 Fragen des Big-Five-Tests (OCEAN) sowie Alter, Geschlecht und
Schreibhand. Die Ausgabe ist einer von vier Persönlichkeitstypen.

### Der Datensatz

- **Quelle:** Big-Five-Persönlichkeitstest (OCEAN), veröffentlicht auf Kaggle.
  Die bereitgestellte Version enthält statt der ursprünglichen 50 Fragen die
  19 informativsten sowie demografische Daten.
- **Eine Zeile** entspricht einer befragten Person (19.719 Zeilen, nach der
  Bereinigung 19.514).
- **Merkmale:** 19 Fragen mit Antworten von 1 (*Trifft nicht zu*) bis 5
  (*Trifft zu*): `N1`–`N10` (Neurotizismus), `E1, E3, E4, E5, E7, E9, E10`
  (Extraversion), `A4` (Verträglichkeit), `C4` (Gewissenhaftigkeit), dazu
  `age`, `gender` und `hand`.
- **Zielvariable `target`:** vier Persönlichkeitstypen (im Datensatz
  englisch benannt). Die Klassen sind unausgewogen.

| Typ | Beschreibung | Anteil |
| --- | --- | --- |
| `Moderate` (Moderat) | Ausgeglichen, keine extremen Merkmale | ca. 43 % |
| `Resilient` | Emotional stabil, ruhig unter Druck | ca. 31 % |
| `Overcontroller` (Überkontrolliert) | Ängstlich und introvertiert | ca. 14 % |
| `Undercontroller` (Unterkontrolliert) | Impulsiv, kümmert sich weniger um Regeln | ca. 12 % |

Die Daten selbst sind **nicht** Teil dieses Repositorys (Download siehe
Abschnitt *Setup*).

### Der Ansatz

1. **EDA** (`notebooks/01_eda.ipynb`): Datenqualität prüfen (fehlende Werte,
   ungültige Antworten, unplausible Alterswerte, Duplikate), Verteilungen und
   Zusammenhänge untersuchen, bereinigten Datensatz `data/clean.csv` erzeugen.
2. **Vorverarbeitungs-Pipeline** (`src/preprocessing.py`): Standardisierung der
   Fragen und des Alters, One-Hot-Encoding von Geschlecht und Schreibhand.
3. **Modellierung und Tuning** (`notebooks/02_modeling.ipynb`): stratifizierter
   Split (80 % Training, 20 % Test), Dummy-Baselines und drei Modelle
   (Logistic Regression, Random Forest, Hist Gradient Boosting), Tuning mit
   `GridSearchCV` bzw. `RandomizedSearchCV` und 5-facher stratifizierter
   Cross-Validation, Metrik `f1_macro`.
4. **Speichern:** Die **gesamte Pipeline** (Vorverarbeitung + Modell) wird mit
   `joblib` nach `models/personality_pipeline.joblib` gespeichert.
5. **Streamlit-App** (`app.py`): lädt die gespeicherte Pipeline und sagt den
   Persönlichkeitstyp aus den Eingaben des Benutzers voraus.

Alle zufälligen Schritte verwenden `random_state=42`, daher ist das Ergebnis
reproduzierbar.

### Das Ergebnis

**Bestes Modell: Hist Gradient Boosting** (flache Bäume mit
`max_depth=5`, `max_leaf_nodes=15`, `min_samples_leaf=20`,
`learning_rate=0.1`, `l2_regularization=0.1`). **Metrik: `f1_macro`.**

| Modell | `f1_macro` ohne Tuning (CV) | `f1_macro` getunt (CV) | `f1_macro` Test |
| --- | --- | --- | --- |
| **Hist Gradient Boosting** | 0,797 | **0,8045** | **0,805** |
| Random Forest | 0,743 | 0,784 | 0,792 |
| Logistic Regression | 0,767 | 0,781 | 0,788 |
| Dummy (Baseline, zufällig nach Klassenanteil) | 0,248 | – | – |

Auf dem Testdatensatz erreicht das beste Modell einen `f1_macro` von 0,805 bei
einer Accuracy von 86,9 %. Die Klassen *Resilient* (F1 0,98), *Moderate* und
*Overcontroller* (je 0,87) werden gut erkannt. Schwächste Klasse ist
*Undercontroller* (F1 0,50), die häufig mit *Moderate* verwechselt wird. Die
ausführliche Begründung der Modellwahl steht am Ende von
`notebooks/02_modeling.ipynb`.

### So verwendest du die App

Der Benutzer gibt Alter, Geschlecht und Schreibhand an, beantwortet die 19
Aussagen auf einer Skala von 1 bis 5 und klickt auf den Button. Die App zeigt
den vorhergesagten Persönlichkeitstyp mit Beschreibung, der Modell-Konfidenz
und den Wahrscheinlichkeiten aller vier Typen. Das Ganze findet in einer
animierten Ozean-Umgebung statt. 🐋🐢🐬

## Setup

_Wird in einem späteren Schritt ergänzt._
