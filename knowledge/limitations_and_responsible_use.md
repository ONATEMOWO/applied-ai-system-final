# Limitations and responsible recommendation behaviour

> Project-authored policy notes. Several of these are attached to every response so that
> limitations are stated rather than implied.

## [KB-POLICY-SYNTHETIC] Fictional catalog

Every song, artist and number in the catalog is fictional and illustrative. The assistant
recommends from these 30 songs only; it cannot recommend, play or link to real songs, and it
never invents a catalog entry.

```profile
kind: policy
keywords: real, spotify, apple music, youtube, streaming, fictional, catalog, playlist
summary: The catalog is 30 fictional songs; results are not real streaming recommendations.
```

## [KB-POLICY-INFERENCE] Inferred preferences are assumptions

Targets that come from a listening goal or a phrase are project assumptions, not facts about
the listener. They are labelled "inferred", they never act as filters, and anything the user
states explicitly overrides them. When a request is ambiguous, unsupported or contradictory
the assistant asks for clarification instead of guessing.

```profile
kind: policy
keywords: assumption, assumptions, inferred, infer, guess, ambiguous, clarify, clarification, contradictory
summary: Inferred targets are assumptions; explicit wording overrides them and unclear requests get a clarification.
```

## [KB-POLICY-COVERAGE] Catalog coverage and fewer results

The catalog is small and uneven: some genres and moods have a single song and no song has a
language, content-rating or exact release-date field. When too few songs satisfy a hard
constraint the assistant returns fewer songs than requested, or none, and says why.

```profile
kind: policy
keywords: coverage, limited, small, language, explicit, clean, fewer, no match, missing, imbalance, bias
summary: The catalog is small and uneven; the assistant returns fewer songs rather than relaxing a hard constraint.
```

## [KB-POLICY-FILTER-BUBBLE] Filter bubbles and variety

Recommending only the closest matches repeats the same artists and genres. The variety rerank
is the built-in counterweight; it is off unless the request or the listening goal asks for it,
so repeated artists and genres are possible by default.

```profile
kind: policy
keywords: filter bubble, repeat, repeats, narrow, exposure, fairness, variety
summary: Results favour the closest matches, which can repeat artists and genres; the variety rerank is the counterweight.
```

## [KB-POLICY-PRIVACY] Privacy

The assistant keeps no user profile and writes nothing to disk unless a trace file is
explicitly requested. A trace stores the request text, so it should not contain personal
information.

```profile
kind: policy
keywords: privacy, private, personal, data, store, stored, trace, log, logging
summary: Nothing is persisted by default; a trace file stores the request text only if explicitly requested.
```
