# Ranking and diversity strategies

> Project-authored notes about the three ranking modes and the variety rerank implemented in
> `src/recommender.py`. A matching `profile` block lets a request phrase select a strategy.

## [KB-STRAT-BALANCED] Balanced ranking

Balanced mode is the default: genre, mood and energy all count (weights 2.0, 1.5 and 2.0).

```profile
kind: strategy
keywords: balanced, all round, all around
mode: balanced
summary: Balanced mode (default) weighs genre 2.0, mood 1.5 and energy 2.0.
```

## [KB-STRAT-GENRE-FIRST] Genre-first ranking

Genre-first mode lets an exact genre match dominate (genre 4.0, mood 1.0, energy 1.0). It suits
requests where staying inside one genre matters most, but it can rank a poor mood match first.

```profile
kind: strategy
keywords: stick to the genre, same genre, only that genre, strict genre, genre matters most, exact genre
mode: genre-first
summary: Genre-first mode makes an exact genre match dominate (genre 4.0, mood 1.0, energy 1.0).
```

## [KB-STRAT-MOOD-ENERGY] Mood-and-energy ranking

Mood-energy mode lets mood and energy dominate and treats genre as a tie-breaker (genre 0.5,
mood 3.0, energy 3.0). It suits requests about a vibe rather than a genre.

```profile
kind: strategy
keywords: mood matters most, match the mood, match my mood, match the vibe, vibe over genre, feel matters
mode: mood-energy
summary: Mood-energy mode lets mood and energy dominate and uses genre as a tie-breaker.
```

## [KB-STRAT-DIVERSITY-ARTIST] Artist variety

The variety rerank subtracts points for each earlier pick by the same artist, so one artist
with several strong matches does not fill the list. Requests such as "different artists" turn
it on with an artist penalty of 0.75 per repeat.

```profile
kind: strategy
keywords: different artists, variety of artists, not the same artist, diverse artists, mix of artists, many artists, varied artists, no repeat artists
diversity: artist=0.75
summary: Artist variety subtracts 0.75 points for each earlier pick by the same artist.
```

## [KB-STRAT-DIVERSITY-GENRE] Genre variety

The same rerank can also subtract points for each earlier pick in the same genre. Requests such
as "different genres" turn it on with a genre penalty of 0.75 per repeat.

```profile
kind: strategy
keywords: different genres, variety of genres, mix of genres, various genres, genre variety, diverse genres
diversity: genre=0.75
summary: Genre variety subtracts 0.75 points for each earlier pick in the same genre.
```

## [KB-STRAT-VARIETY] General variety

A plain request for "variety" is read as a moderate variety rerank across both artists and genres.

```profile
kind: strategy
keywords: variety, diverse, varied, mix it up
diversity: artist=0.50, genre=0.50
summary: A general variety request applies a 0.50 penalty per repeated artist and per repeated genre.
```
