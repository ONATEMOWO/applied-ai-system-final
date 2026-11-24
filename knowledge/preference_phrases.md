# Preference phrases

> Project-authored vocabulary. Each section maps a handful of everyday words to a *soft*
> scoring target that the existing recommender already understands. The system only
> reacts to the exact phrases listed in `keywords`; it does not understand other wording.

## [KB-MOD-CALM] Calm

"Calm" and its near-synonyms describe low-energy, slower music.

```profile
kind: modifier
keywords: calm, calming, mellow, gentle, soothing, quiet, soft, laid back, peaceful
prefs: energy=0.30, tempo_bpm=75
summary: "Calm" is read as low energy (0.30) and a slow tempo (about 75 BPM).
```

## [KB-MOD-LOW-ENERGY] Low energy

Phrases that ask for less energy are read as a low energy target. This is only a soft
target; a hard cap is applied separately when the wording is explicit enough.

```profile
kind: modifier
keywords: low energy, not too much energy, without too much energy, not too energetic, not too intense, not too loud, laid back
prefs: energy=0.30
summary: Wording that asks for less energy is read as a low energy target (0.30).
```

## [KB-MOD-UPBEAT] Upbeat

Upbeat music is bright and lively: fairly high energy and high valence.

```profile
kind: modifier
keywords: upbeat, cheerful, bouncy, feel good, positive, sunny, joyful, lively
prefs: energy=0.80, valence=0.85
summary: "Upbeat" is read as high energy (0.80) and high valence (0.85).
```

## [KB-MOD-HIGH-ENERGY] High energy

High-energy wording is read as a very high energy target with a fast tempo.

```profile
kind: modifier
keywords: high energy, energetic, pumped, hype, powerful, fast paced, adrenaline
prefs: energy=0.90, tempo_bpm=135
summary: High-energy wording is read as energy 0.90 and a fast tempo (about 135 BPM).
```

## [KB-MOD-INSTRUMENTAL] Instrumental

Instrumentalness estimates how little singing or speech a track contains. Asking for
instrumental music is read as a high instrumentalness target. Saying "instrumental only" or
"no vocals" additionally applies a hard filter (see the explicit-request rules).

```profile
kind: modifier
keywords: instrumental, instrumentals, wordless, no vocals, without vocals, no lyrics, without lyrics
prefs: instrumentalness=0.85
summary: "Instrumental" is read as a high instrumentalness target (0.85).
```

## [KB-MOD-ACOUSTIC] Acoustic

Acoustic wording is read as a preference for songs with a high acousticness value.

```profile
kind: modifier
keywords: acoustic, unplugged, organic, stripped back
prefs: likes_acoustic=true
summary: "Acoustic" is read as a preference for high-acousticness songs.
```

## [KB-MOD-DANCEABLE] Danceable

Danceable wording is read as a high danceability target.

```profile
kind: modifier
keywords: danceable, groove, groovy, make me dance, dance floor
prefs: danceability=0.85
summary: "Danceable" is read as a high danceability target (0.85).
```

## [KB-MOD-GLOOMY] Gloomy and wistful

Wistful, rainy-day wording is read as lower valence and lower energy. This is a loose
interpretation of mood and may not match what every listener means.

```profile
kind: modifier
keywords: gloomy, rainy, heartbroken, tearful, somber, wistful, rainy day
prefs: valence=0.30, energy=0.35
summary: Wistful or rainy-day wording is read as low valence (0.30) and low energy (0.35).
```

## [KB-MOD-SLOW-TEMPO] Slow tempo

Slow-tempo wording is read as a tempo target of 70 BPM.

```profile
kind: modifier
keywords: slow tempo, slow songs, downtempo, slow paced
prefs: tempo_bpm=70
summary: Slow-tempo wording is read as a tempo target of 70 BPM.
```

## [KB-MOD-FAST-TEMPO] Fast tempo

Fast-tempo wording is read as a tempo target of 150 BPM.

```profile
kind: modifier
keywords: fast tempo, fast songs, uptempo, quick tempo
prefs: tempo_bpm=150
summary: Fast-tempo wording is read as a tempo target of 150 BPM.
```

## [KB-MOD-POPULAR] Popular

Popular wording is read as a high popularity target. Popularity in the catalog is an
illustrative number, not real streaming data.

```profile
kind: modifier
keywords: popular, mainstream, hits, chart toppers, well known
prefs: popularity=80
summary: "Popular" is read as a high popularity target (80, an illustrative catalog number).
```

## [KB-MOD-HIDDEN-GEMS] Hidden gems

Wording about lesser-known songs is read as a low popularity target.

```profile
kind: modifier
keywords: underrated, hidden gems, obscure, lesser known, undiscovered
prefs: popularity=30
summary: Lesser-known wording is read as a low popularity target (30).
```
