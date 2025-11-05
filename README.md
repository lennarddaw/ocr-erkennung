# OCR Empfänger-Erkennung

## Inhaltsverzeichnis

- [Workflow-Diagramm](#workflow-diagramm)


## Workflow-Diagramm
```
┌─────────────────┐
│ Bild Upload     │
│ (Base64/HEIC)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HEIC → RGB      │ ──────────► für einheitliches Format heic, jpg oder png zu RGB
│ Konvertierung   │             Ohne Konvertierung: Crashes, falsche Farben
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bildgröße       │ ──────────► Angepasst auf max 2000px, wegen der Performance wird später
│ Optimierung     │             vergrößert wenn nötig. Qualitätseinbruch marginal,
└────────┬────────┘             aber bei 4x - 5x vergrößern dauert es deutlich länger
         │                      als es eh schon ist
         ▼
┌─────────────────┐
│ Quality Check   │ ──────────► Um kleinere Ungenauigkeiten direkt auszugleichen,
│ Brightness,     │             wenn Schwellenwert erreicht 
│ Contrast,       │             => nach einer der 3 Methoden
│ Sharpness       │             Vorher z.B. immer 2.0x Sharpness => Überschärfung
└────────┬────────┘
         │
         ├─── Quality Score: 0-39  → ultra_aggressive
         ├─── Quality Score: 40-69 → aggressive
         └─── Quality Score: 70+   → standard
         │
         ▼
┌─────────────────┐
│ Adaptive        │ ──────────► OCR kann bei <60 Helligkeit kaum Text erkennen.
│ Enhancement     │             Sharpness: 1.8-3.0x
│                 │             Contrast: 1.5-2.5x
└────────┬────────┘             Brightness: 0.7-1.4x
         │                      Stellt sicher dass der betroffene Faktor richtig enhanced wird.
         ▼                      Nicht zu viel und auch nicht zu wenig
┌─────────────────────────────────────────┐
│ Hybrid OCR                              │
│                                         │
│ ┌─────────────┐  ┌──────────────┐       │
│ │ EasyOCR     │  │ Tesseract    │       │ ──► In der Regel sind die Bilder nicht so gut => für schlechte Bilder
│ │ Strategy 1-2│  │ Strategy 3-5 │       │     EasyOCR mit dem Deep Learning + dann das ultra_aggressive Enhacement
│ └─────────────┘  └──────────────┘       │     Wenn EasyOCR genug erkannt hat direkt weiter
│                                         │     Wenn Easy OCR nicht genug erkennt versucht es dann Tesseract
└────────┬────────────────────────────────┘     70% der Fälle reicht easyocr_ultra_aggressive
         │
         ├─── Text 1: 145 chars (easyocr_ultra_aggressive)
         ├─── Text 2: 132 chars (easyocr_fallback)
         ├─── Text 3: 98 chars  (tesseract_psm6)
         ├─── Text 4: 87 chars  (tesseract_psm4)
         └─── Text 5: 76 chars  (easyocr_inverted)
         │
         ▼
┌─────────────────┐
│ Beste Strategie │ ──────────► 5 Strategien produzieren 5 verschiedene Texte.
│ Auswahl         │             Welcher ist am besten?
└────────┬────────┘             LÖSUNG: Nach Qualität (Anzahl erkannter Buchstaben) sortieren.
         │                      Längster = bester.
         ▼
┌─────────────────────────────────────────┐
│ Word Pool Extraction                    │
│                                         │
│ Input: "Dr. MAKS MUSTERMAN\nGmbH\n..."  │
│                                         │ ──► Spart Zeit, wenn wir dann nachher die Kombinationen erstellen
│ 1. Titel-Filter: "Dr." → entfernt       │     und gegen die Liste der Empfänger matchen, weil direkt Dr. oder GMBH
│ 2. Keyword-Filter: "GmbH" → entfernt    │     dort gar nicht vorkommen sondern nur Max Mustermann.
│ 3. Großbuchstaben: "MAKS", "MUSTERMAN"  │     OCR-Text enthält viel Rauschen, "Dr GmbH" als Namen matchen
│                                         │
│ → Word Pool: {"Maks", "Musterman"}      │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Smart Combinations (mit Priorität)      │
│                                         │
│ Priorität 4: 3-Wort-Kombinationen       │
│ Priorität 3: 2-Wort-Kombinationen       │ ──► Welche Kombination hat die größte Aussicht auf Erfolg? Daher an
│ Priorität 2: 2-Wort-Permutationen       │     erster Stelle lange Kombination mit 2 - 3 Strings 
│ Priorität 1: Einzelwörter               │     Ohne Sortierung würden alle Kombinationen gleichbehandelt
│                                         │     
│ → [("Maks Musterman", 3),               │
│    ("Musterman Maks", 2),               │
│    ("Maks", 1),                         │
│    ("Musterman", 1)]                    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Standort-Filter (Kombinationen)         │
│                                         │
│ Input: [("Maks Musterman", 3), ...]     │ ──► Jeder mögliche Kandidat, alle möglichen Kombinationen von vorher
│                                         │     werden gegen Liste mit Standorten gematcht => Fallen dann sofort raus
│ Fuzzy gegen KNOWN_LOCATIONS:            │     Läuft mit Fuzzy und Levenshtein, da OCR Text selten 100% korrekt
│ "Maks Musterman" ≠ Standort → behalten  │     Einfach & Effektiv weil Adresse häufig Namen enthält
│                                         │     Ohne Filter:
│                                         │     wird Adresse als Empfänger gematcht
│ → Gefiltert: [("Maks Musterman", 3),...]│
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Mehrschichtiges Matching                            │
│                                                     │
│ ┌────────────────────────────────────────────────┐  │
│ │ EBENE 1: Fuzzy auf kompletten Pool             │  │
│ │ "Maks Musterman" vs. KNOWN_RECIPIENTS          │  │ ──► Erstmal Fuzzy auf alle möglichen Empfänger, spart enorm
│ │ → 5 Scorer (token_set, token_sort, ...)        │  │     Zeit wenn direkt 100% Match dann auch kein Levenshtein
│ │ → Beste: 68% (token_sort_ratio)                │  │     Ein Scorer findet nicht alle Matches. Jeder der 5 Scorer
│ └────────────────────────────────────────────────┘  │     hat einen eigenen Use Case, wir wollen zumindest alle 
│                                                     │     mal gehört haben 
│ ┌────────────────────────────────────────────────┐  │
│ │ EBENE 2: Fuzzy auf Top-20 Kombinationen        │  │
│ │ "Maks Musterman" vs. KNOWN_RECIPIENTS          │  │ ──► Deshalb vorhin das Ranking der Kombinationen
│ │ → Nach Priorität sortiert                      │  │     jetzt gehen wir die bestmöglichen mit Fuzzing durch und
│ │ → Früher Abbruch bei Score ≥95                 │  │     reichen weiter wenn gut genug
│ │ → Beste: 72% (partial_token_set_ratio)         │  │     Ohne Limit würden alle Kombinationen getestet
│ └────────────────────────────────────────────────┘  │
│                                                     │
│ ┌────────────────────────────────────────────────┐  │
│ │ EBENE 3: Levenshtein auf Namens-Teilen         │  │
│ │                                                │  │
│ │ Cache-Lookup:                                  │  │ ──► Levenshtein stellt sicher, dass wir die Fehler von OCR
│ │ "maks" in NAME_PARTS_CACHE?                    │  │     ausbessern in den meisten Fällen nur 1 - 2 falsch
│ │ → Nicht direkt                                 │  │     platziert in vor oder Nachname, dann funktioniert auch
│ │                                                │  │     Fuzzing besser
│ │ Levenshtein auf Cache-Keys:                    │  │
│ │ "maks" vs "max" → Distance: 2                  │  │
│ │ Similarity: (4-2)/4 = 50%                      │  │
│ │                                                │  │
│ │ "musterman" vs "mustermann" → Distance: 1      │  │
│ │ Similarity: (10-1)/10 = 90%                    │  │
│ │                                                │  │
│ │ Cache-Hit: NAME_PARTS_CACHE["mustermann"]      │  │
│ │ → ["Max Mustermann", "Maria Mustermann"]       │  │
│ │                                                │  │
│ │ Word Pool: {"Maks", "Musterman"}               │  │
│ │ "Max Mustermann" enthält beide Teile           │  │
│ │ → Common Parts Match: 2/2 = 100%               │  │
│ │                                                │  │
│ │ → MATCH: "Max Mustermann" (85%)                │  │
│ └────────────────────────────────────────────────┘  │
│                                                     │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Threshold-Prüfung (Dynamisch)           │
│                                         │
│ Best Match: "Max Mustermann"            │
│ Score: 85%                              │ ──► Threshold dynamisch, weil Levenshtein genauer ist und ansonsten
│ Method: "levenshtein_parts"             │     Match rausfallen würde obwohl es richtig ist 
│                                         │     Mit fixem
│                                         │     Threshold 70% würden viele gute Levenshtein-Matches durchfallen
│ Adjusted Threshold:                     │
│ 70% - 5% = 65%                          │
│                                         │
│ 85% ≥ 65% → MATCH ACCEPTED              │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Response        │
│ {               │
│   "recipient":  │
│   "Max Muster-  │
│   mann",        │
│   "confidence": │
│   85,           │
│   "method":     │
│   "levenshtein_ │
│   parts",       │
│   "time": 7.2s  │
│ }               │
└─────────────────┘
```

---
