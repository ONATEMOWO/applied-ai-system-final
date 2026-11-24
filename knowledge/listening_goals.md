# Listening goals and preference profiles

> Project-authored guidance written for this assignment. These profiles are design
> assumptions about what common listening situations tend to need. They are **not**
> externally verified music research and are not derived from real listener data.

Each section has a stable ID. The `profile` block is the only machine-read part: its
`prefs` line becomes *soft* scoring targets for the existing recommender (never filters).

## [KB-GOAL-STUDY] Focus and study listening

Studying and reading usually go better with steady, low-distraction music: moderate-to-low
energy, a slow-to-medium tempo, mostly acoustic textures and few or no vocals. This profile
therefore targets high instrumentalness and moderate-low energy.

```profile
kind: goal
goal: study
keywords: study, studying, homework, concentrate, concentration, focus, focusing, reading, exam, revision, library, essay, goal_study
prefs: energy=0.40, instrumentalness=0.85, likes_acoustic=true, tempo_bpm=80, mood=focused
mode: balanced
summary: Study sessions usually favour steady, low-distraction, mostly instrumental music at moderate-to-low energy.
```

## [KB-GOAL-WORKOUT] Workout and exercise listening

Exercise usually benefits from high energy and a fast, driving tempo with a strong beat.
Acoustic, slow music is rarely what people reach for here. This profile targets very high
energy, a tempo near 140 BPM and a danceable beat.

```profile
kind: goal
goal: workout
keywords: workout, gym, exercise, running, cardio, lifting, training, jogging, sprint, pumped, goal_workout
prefs: energy=0.92, tempo_bpm=140, danceability=0.75, likes_acoustic=false, mood=energetic
mode: balanced
summary: Workouts usually favour very high energy, a fast driving tempo and a strong beat, with little acoustic texture.
```

## [KB-GOAL-PARTY] Party and social listening

Party music is typically upbeat, bright and very danceable. People at a party also tend to
notice repeated artists, so this profile is usually paired with an explicit variety request.
It targets high energy, high valence (a positive, bright feel) and high danceability.

```profile
kind: goal
goal: party
keywords: party, dance, dancing, club, celebrate, celebration, birthday, friends, gathering, goal_party
prefs: energy=0.85, valence=0.85, danceability=0.88, likes_acoustic=false, mood=happy
mode: balanced
summary: Party playlists usually favour high-energy, bright (high-valence), very danceable songs.
```

## [KB-GOAL-RELAX] Relaxation and unwinding

Relaxing usually calls for low energy, a slow tempo, acoustic warmth and a gentle, positive
or neutral mood rather than a dark one. This profile targets low energy, moderate valence and
an acoustic sound.

```profile
kind: goal
goal: relax
keywords: relax, relaxing, relaxation, unwind, peaceful, wind down, rest, spa, evening, goal_relax
prefs: energy=0.30, likes_acoustic=true, valence=0.60, tempo_bpm=75, mood=relaxed
mode: balanced
summary: Relaxation usually favours low energy, a slow tempo and acoustic warmth with a gentle, neutral-to-positive feel.
```

## [KB-GOAL-DISCOVERY] General discovery

When someone only wants "something new", there is no energy or mood target to infer, so the
profile does not invent one. It assumes two things and says so: that "new to me" is better
approximated by less mainstream songs (a popularity target of 40), and that variety across
artists and genres matters more than usual. Both are assumptions, not facts about the user.

```profile
kind: goal
goal: discovery
keywords: discover, discovery, explore, new, surprise, different, fresh, unfamiliar, goal_discovery
prefs: popularity=40
mode: balanced
diversity: artist=0.25, genre=0.20
summary: Discovery assumes less mainstream songs (popularity target 40) and extra variety; both are assumptions, not facts about the listener.
```
