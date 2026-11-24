# Song features: what each attribute means

> Project-authored descriptions of the 15 catalog attributes. All feature values in
> `data/songs.csv` are illustrative numbers for fictional songs, not audio measurements.
> These sections add explanatory notes to a response; they do not change scoring.

## [KB-FEAT-ENERGY] Energy

Energy runs from 0 (very calm) to 1 (very intense). It has the largest numeric weight in the
scoring rule, so it usually decides the order between songs of the same genre.

```profile
kind: feature
feature: energy
keywords: energy, energetic, intensity, intense
summary: Energy (0 to 1) is the strongest numeric signal in scoring.
```

## [KB-FEAT-VALENCE] Valence

Valence runs from 0 (dark, sad) to 1 (bright, cheerful). It describes musical positivity, not
the quality of the song.

```profile
kind: feature
feature: valence
keywords: valence, positive, cheerful, bright, dark, sad
summary: Valence (0 to 1) describes how bright or dark a song feels.
```

## [KB-FEAT-DANCEABILITY] Danceability

Danceability runs from 0 to 1 and describes how suitable the rhythm is for dancing.

```profile
kind: feature
feature: danceability
keywords: danceability, danceable, dance, groove, beat
summary: Danceability (0 to 1) describes how suited the rhythm is for dancing.
```

## [KB-FEAT-TEMPO] Tempo

Tempo is measured in beats per minute (1 to 300). The scorer treats a 60 BPM gap as completely
different, so tempo targets matter most for clearly slow or clearly fast requests.

```profile
kind: feature
feature: tempo_bpm
keywords: tempo, bpm, beats, pace, paced, fast, slow
summary: Tempo is in beats per minute; a 60 BPM gap counts as completely different.
```

## [KB-FEAT-ACOUSTICNESS] Acousticness

Acousticness runs from 0 (electronic, synthetic) to 1 (acoustic instruments). The scorer only
uses it when the listener states a taste for acoustic or non-acoustic music.

```profile
kind: feature
feature: acousticness
keywords: acousticness, acoustic, unplugged, electronic, organic
summary: Acousticness (0 to 1) is only scored when an acoustic or non-acoustic taste is stated.
```

## [KB-FEAT-INSTRUMENTALNESS] Instrumentalness

Instrumentalness runs from 0 (mostly vocals) to 1 (almost no vocals). It is the feature behind
requests for instrumental music or music without singing.

```profile
kind: feature
feature: instrumentalness
keywords: instrumentalness, instrumental, vocals, lyrics, singing, wordless
summary: Instrumentalness (0 to 1) estimates how little singing a song has.
```

## [KB-FEAT-SPEECHINESS] Speechiness

Speechiness runs from 0 to 1 and rises for spoken-word-like tracks such as rap. In this
catalog only the hip hop songs score noticeably above 0.1.

```profile
kind: feature
feature: speechiness
keywords: speechiness, spoken, rap, speech, wordy
summary: Speechiness (0 to 1) rises for spoken or rap-like tracks.
```

## [KB-FEAT-LIVENESS] Liveness

Liveness runs from 0 to 1 and estimates how much a track sounds like a live performance.

```profile
kind: feature
feature: liveness
keywords: liveness, live, concert, audience
summary: Liveness (0 to 1) estimates how live a recording sounds.
```

## [KB-FEAT-POPULARITY] Popularity

Popularity is a 0 to 100 number attached to each fictional song. It is not real streaming
data and says nothing about how popular any real song is.

```profile
kind: feature
feature: popularity
keywords: popularity, popular, mainstream, underrated, obscure, hits
summary: Popularity (0 to 100) is an illustrative catalog number, not real streaming data.
```

## [KB-FEAT-DECADE] Release decade

Release decade is stored as a year such as 1990 or 2020. The catalog covers the 1970s to the
2020s and stores no exact release date, so requests about a specific year or "new releases"
cannot be honoured.

```profile
kind: feature
feature: decade
keywords: decade, decades, release, released, era, nineties, eighties, seventies
summary: Release decade (1970 to 2020) is stored per song; exact years and recency are not available.
```

## [KB-FEAT-GENRE-MOOD] Genre and mood labels

Genre and mood are single text labels per song and are matched exactly: a song either has the
requested label or it does not. Related labels (for example "chill" and "relaxed") do not
partially match.

```profile
kind: feature
feature: genre_mood
keywords: genre, genres, mood, moods, label, labels
summary: Genre and mood are exact-match labels; similar labels do not partially match.
```
