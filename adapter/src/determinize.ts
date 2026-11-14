import {buildTrainingGameOptions} from './config';
import {
  GameCards,
  ICorporationCard,
  IGame,
  IPreludeCard,
  IProjectCard,
  Phase,
  SelectCard,
  constants,
  newCorporationCard,
  newPrelude,
  newProjectCard,
  newStandardDraft,
} from './engine';
import {
  ActorObservation,
  DraftHistoryEntry,
  SampledWorld,
  Seat,
  SupportedMap,
  UnseenPool,
} from './types';

function compareStrings(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

interface StaticCardPools {
  readonly projectNames: ReadonlyArray<string>;
  readonly projectSet: ReadonlySet<string>;
  readonly corpNames: ReadonlyArray<string>;
  readonly corpSet: ReadonlySet<string>;
  readonly preludeNames: ReadonlyArray<string>;
  readonly preludeSet: ReadonlySet<string>;
}

const cachedCardPools = new Map<SupportedMap, StaticCardPools>();

function getStaticCardPools(map: SupportedMap): StaticCardPools {
  const existing = cachedCardPools.get(map);
  if (existing !== undefined) {
    return existing;
  }
  const gc = new GameCards(buildTrainingGameOptions(map));
  const projectNames = gc.getProjectCards().map((c) => c.name).sort(compareStrings);
  const corpNames = gc.getCorporationCards().map((c) => c.name).sort(compareStrings);
  const preludeNames = gc.getPreludeCards().map((c) => c.name).sort(compareStrings);
  const pools: StaticCardPools = {
    projectNames,
    projectSet: new Set(projectNames),
    corpNames,
    corpSet: new Set(corpNames),
    preludeNames,
    preludeSet: new Set(preludeNames),
  };
  cachedCardPools.set(map, pools);
  return pools;
}

function subtractCards(offered: ReadonlyArray<string>, chosen: ReadonlyArray<string>): Array<string> {
  const counts = new Map<string, number>();
  for (const name of chosen) {
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  const remaining: Array<string> = [];
  for (const name of offered) {
    const c = counts.get(name) ?? 0;
    if (c > 0) {
      counts.set(name, c - 1);
    } else {
      remaining.push(name);
    }
  }
  return remaining;
}

function extractCardNames(val: string | ReadonlyArray<string> | undefined): Array<string> {
  if (val === undefined) {
    return [];
  }
  return typeof val === 'string' ? [val] : val.slice();
}

export function computeUnseenPool(obs: ActorObservation): UnseenPool {
  const actor = obs.actor;
  const opp: Seat = actor === 0 ? 1 : 0;
  const pools = getStaticCardPools(obs.public.map);

  const actorColor = obs.public.players[actor].color;
  const oppColor = obs.public.players[opp].color;

  // Scan publicLog chronologically once to extract:
  // - reshuffles
  // - per-gen research buy completion for opp (and actor)
  // - unknown discards from hand (Sell Patents, Mars University) for opp and actor
  // - publicly drawn cards for opp (Experimental Forest, Acquired Space Agency)
  // - publicly discarded cards ('Discarded X cards ...', 'revealed and discarded ...')
  // - fizzled preludes
  let curGen = 1;
  let lastReshuffleIdx = -1;
  let lastReshuffleGen = 0;

  const oppResearchBuyByGen = new Map<number, {logIdx: number; count: number}>();
  const actorResearchBuyByGen = new Map<number, {logIdx: number; count: number}>();
  const oppUnknownDiscardIndices: Array<number> = [];
  const actorUnknownDiscardIndices: Array<number> = [];
  const oppPublicDraws: Array<{card: string; logIdx: number}> = [];
  const publicDiscards: Array<{card: string; logIdx: number; generation: number}> = [];
  const fizzledPreludes: Array<{card: string; seat: Seat}> = [];
  const log = obs.public.publicLog;

  for (let idx = 0; idx < log.length; idx++) {
    const entry = log[idx];
    const msg = entry.message;

    if (msg === 'Generation ${0}') {
      const g = Number(entry.data[0]?.value);
      if (Number.isFinite(g)) {
        curGen = g;
      }
      continue;
    }

    if (msg === 'The project discard pile has been shuffled to form a new deck.') {
      lastReshuffleIdx = idx;
      lastReshuffleGen = curGen;
      continue;
    }

    if (msg === '${0} ${1} ${2} card(s)' && entry.data[1]?.value === 'bought') {
      const who = entry.data[0]?.value;
      const count = Number(entry.data[2]?.value);
      if (who === oppColor && curGen >= 2 && !oppResearchBuyByGen.has(curGen)) {
        oppResearchBuyByGen.set(curGen, {logIdx: idx, count});
      } else if (who === actorColor && curGen >= 2 && !actorResearchBuyByGen.has(curGen)) {
        actorResearchBuyByGen.set(curGen, {logIdx: idx, count});
      }
      continue;
    }

    if (msg === '${0} sold ${1} patents') {
      const who = entry.data[0]?.value;
      const n = Number(entry.data[1]?.value);
      if (n > 0) {
        if (who === oppColor) {
          oppUnknownDiscardIndices.push(idx);
        } else if (who === actorColor) {
          actorUnknownDiscardIndices.push(idx);
        }
      }
      continue;
    }

    if (msg === '${0} is using their ${1} effect to draw a card by discarding a card.') {
      const who = entry.data[0]?.value;
      if (who === oppColor) {
        oppUnknownDiscardIndices.push(idx);
      } else if (who === actorColor) {
        actorUnknownDiscardIndices.push(idx);
      }
      continue;
    }

    if (msg === '${0} played ${1}' && entry.data[1]?.value === 'Sponsored Academies') {
      const who = entry.data[0]?.value;
      if (who === oppColor) {
        oppUnknownDiscardIndices.push(idx);
      } else if (who === actorColor) {
        actorUnknownDiscardIndices.push(idx);
      }
      continue;
    }

    if (msg === '${0} drew ${1}' && entry.data[0]?.value === oppColor) {
      for (const c of extractCardNames(entry.data[1]?.value)) {
        if (pools.projectSet.has(c)) {
          oppPublicDraws.push({card: c, logIdx: idx});
        }
      }
      continue;
    }

    if (msg === 'Discarded ${0} cards ${1}') {
      for (const c of extractCardNames(entry.data[1]?.value)) {
        if (pools.projectSet.has(c)) {
          publicDiscards.push({card: c, logIdx: idx, generation: curGen});
        }
      }
      continue;
    }

    if (msg === '${0} revealed and discarded ${1}') {
      for (const c of extractCardNames(entry.data[1]?.value)) {
        if (pools.projectSet.has(c)) {
          publicDiscards.push({card: c, logIdx: idx, generation: curGen});
        }
      }
      continue;
    }

    if (msg === '${0} fizzled. ${1} gains 15 M€.') {
      const card = String(entry.data[0]?.value ?? '');
      const who = entry.data[1]?.value;
      if (pools.preludeSet.has(card) && (who === actorColor || who === oppColor)) {
        fizzledPreludes.push({card, seat: who === oppColor ? opp : actor});
      }
    }
  }

  // ---- Tableaus ----
  const tableauProjects = new Set<string>();
  const tableauCorps = new Set<string>();
  const tableauPreludes = new Set<string>();
  let oppTableauCorpsCount = 0;
  let oppTableauPreludesCount = 0;

  for (const s of [0, 1] as const) {
    for (const tc of obs.public.players[s].tableau) {
      if (pools.projectSet.has(tc.name)) {
        tableauProjects.add(tc.name);
      } else if (pools.corpSet.has(tc.name)) {
        tableauCorps.add(tc.name);
        if (s === opp) oppTableauCorpsCount += 1;
      } else if (pools.preludeSet.has(tc.name)) {
        tableauPreludes.add(tc.name);
        if (s === opp) oppTableauPreludesCount += 1;
      }
    }
  }

  // ---- Corporations & Preludes ----
  const actorInitialBuy = obs.history.initialBuy;
  const oppInitialBuyComplete =
    oppTableauCorpsCount > 0 ||
    obs.public.otherDeckSizes.corporations.discardPile > (actorInitialBuy !== null ? 1 : 0);
  const opponentHasHiddenPickedCorp = oppInitialBuyComplete && oppTableauCorpsCount === 0;

  const knownCorps = new Set<string>(tableauCorps);
  for (const c of obs.private.dealtCorporationCards) knownCorps.add(c);
  if (obs.private.pickedCorporationCard !== null) knownCorps.add(obs.private.pickedCorporationCard);
  if (actorInitialBuy !== null) {
    for (const c of actorInitialBuy.corporationsOffered) knownCorps.add(c);
  }
  const unseenCorps = pools.corpNames.filter((c) => !knownCorps.has(c));

  let oppFizzledPreludes = 0;
  const knownPreludes = new Set<string>(tableauPreludes);
  for (const p of obs.private.preludeCardsInHand) knownPreludes.add(p);
  if (actorInitialBuy === null) {
    for (const p of obs.private.dealtPreludeCards) knownPreludes.add(p);
  } else {
    for (const p of actorInitialBuy.preludesOffered) knownPreludes.add(p);
  }
  for (const sel of obs.history.cardSelections) {
    for (const p of sel.offered) {
      if (pools.preludeSet.has(p)) {
        knownPreludes.add(p);
      }
    }
  }
  for (const fp of fizzledPreludes) {
    knownPreludes.add(fp.card);
    if (fp.seat === opp) {
      oppFizzledPreludes += 1;
    }
  }
  const unseenPreludes = pools.preludeNames.filter((p) => !knownPreludes.has(p));
  const opponentPreludeHandSize = oppInitialBuyComplete
    ? Math.max(0, 2 - (oppTableauPreludesCount + oppFizzledPreludes))
    : 0;

  // ---- Actor-held project cards ----
  const actorHeldProjects = new Set<string>();
  for (const h of obs.private.cardsInHand) actorHeldProjects.add(h.name);
  if (actorInitialBuy === null) {
    for (const c of obs.private.dealtProjectCards) actorHeldProjects.add(c);
  }
  for (const c of obs.private.currentDraftPacket) actorHeldProjects.add(c);
  for (const c of obs.private.draftedCards) actorHeldProjects.add(c);

  const isLocatedElsewhere = (c: string): boolean => tableauProjects.has(c) || actorHeldProjects.has(c);

  // ---- Group drafts by generation ----
  const draftsByGen = new Map<number, Array<DraftHistoryEntry>>();
  for (const d of obs.history.drafts) {
    const arr = draftsByGen.get(d.generation) ?? [];
    arr.push(d);
    draftsByGen.set(d.generation, arr);
  }
  for (const arr of draftsByGen.values()) {
    arr.sort((a, b) => a.draftRound - b.draftRound);
  }

  const researchOfferByGen = new Map<number, ReadonlyArray<string>>();
  for (const rb of obs.history.researchBuys) {
    researchOfferByGen.set(rb.generation, rb.offered);
  }
  if (obs.public.phase === Phase.RESEARCH && obs.public.generation >= 2 && obs.private.draftedCards.length > 0) {
    if (!researchOfferByGen.has(obs.public.generation)) {
      researchOfferByGen.set(obs.public.generation, obs.private.draftedCards);
    }
  }

  // ---- Active draft / research constraints on opponent ----
  let opponentDraftHandSize = 0;
  let opponentDraftedSize = 0;
  let opponentResearchOfferSize = 0;
  const knownOpponentDraftHandSet = new Set<string>();
  const knownOpponentDraftedSet = new Set<string>();
  const knownOpponentResearchSet = new Set<string>();
  // Cards actor picked/passed in current incomplete draft that aren't in opp's current draftHand/drafted yet
  // (e.g. actor is seat 0 and picked first in the current draft round; d.passed is sitting in seat 0's draftHand)
  const actorWaitingToPass = new Set<string>();

  if (obs.public.phase === Phase.DRAFTING) {
    const genDrafts = draftsByGen.get(obs.public.generation) ?? [];
    let curRound: number;
    if (obs.private.currentDraftPacket.length > 0) {
      curRound = 5 - obs.private.currentDraftPacket.length;
    } else {
      curRound = genDrafts.length;
      const latest = genDrafts[genDrafts.length - 1];
      if (latest !== undefined) {
        for (const c of latest.passed) {
          actorWaitingToPass.add(c);
        }
      }
    }
    opponentDraftHandSize = Math.max(0, 5 - curRound);
    opponentDraftedSize = Math.max(0, curRound - 1);

    if (curRound === 2 && genDrafts[0] !== undefined) {
      for (const c of genDrafts[0].passed) {
        knownOpponentDraftHandSet.add(c);
      }
    } else if (curRound === 3 && genDrafts[0] !== undefined && genDrafts[1] !== undefined) {
      for (const c of genDrafts[1].passed) {
        knownOpponentDraftHandSet.add(c);
      }
      const r3Packet = obs.private.currentDraftPacket.length > 0
        ? obs.private.currentDraftPacket
        : (genDrafts[2]?.packet ?? []);
      if (r3Packet.length > 0) {
        for (const c of subtractCards(genDrafts[0].passed, r3Packet)) {
          knownOpponentDraftedSet.add(c);
        }
      }
    }
  } else if (obs.public.phase === Phase.RESEARCH && obs.public.generation >= 2) {
    const oppBuy = oppResearchBuyByGen.get(obs.public.generation);
    if (oppBuy === undefined) {
      const genDrafts = draftsByGen.get(obs.public.generation) ?? [];
      const d1 = genDrafts[0];
      const d2 = genDrafts[1];
      const d3 = genDrafts[2];
      const actorOffer = researchOfferByGen.get(obs.public.generation) ?? [];
      opponentResearchOfferSize = actorOffer.length > 0 ? actorOffer.length : 4;
      if (d1 !== undefined && d2 !== undefined && d3 !== undefined) {
        for (const c of subtractCards(d1.passed, d3.packet)) {
          knownOpponentResearchSet.add(c);
        }
        if (actorOffer.length > 0) {
          const aR4 = subtractCards(actorOffer, [...d1.taken, ...d2.taken, ...d3.taken]);
          for (const c of subtractCards(d2.passed, aR4)) {
            knownOpponentResearchSet.add(c);
          }
        }
        for (const c of d3.passed) {
          knownOpponentResearchSet.add(c);
        }
      }
    }
  }

  const isInOppActiveZone = (c: string): boolean =>
    knownOpponentDraftHandSet.has(c) ||
    knownOpponentDraftedSet.has(c) ||
    knownOpponentResearchSet.has(c) ||
    actorWaitingToPass.has(c);

  // ---- Track opponent known hand, possible hand, and known discard pile ----
  const knownOppHandSet = new Set<string>();
  const possibleOppHandSet = new Set<string>();
  const knownDiscardSet = new Set<string>();

  // Helper: is an event in `generation` definitely after the last reshuffle?
  const isReshuffleBeforeGenResearch = (gen: number, buyLogIdx?: number): boolean => {
    if (lastReshuffleIdx === -1) return true;
    if (buyLogIdx !== undefined) return buyLogIdx > lastReshuffleIdx;
    return gen > lastReshuffleGen;
  };

  // Initial buy discards by actor (Gen 1, occurs before Gen 2).
  if (actorInitialBuy !== null) {
    // Check if any reshuffle occurred AFTER initial buy; if lastReshuffleIdx === -1, definitely still in discard.
    // Note: Can a reshuffle occur during Gen 1 before anyone plays? No, only 20 dealt cards.
    if (lastReshuffleIdx === -1) {
      for (const c of actorInitialBuy.projectsDiscarded) {
        if (!isLocatedElsewhere(c) && !isInOppActiveZone(c)) {
          knownDiscardSet.add(c);
        }
      }
    }
  }

  // Actor research buys discards
  for (const rb of obs.history.researchBuys) {
    const rbLog = actorResearchBuyByGen.get(rb.generation);
    const afterReshuffle = lastReshuffleIdx === -1
      ? true
      : (rbLog !== undefined ? rbLog.logIdx > lastReshuffleIdx : rb.generation > lastReshuffleGen);
    if (afterReshuffle) {
      for (const c of rb.discarded) {
        if (!isLocatedElsewhere(c) && !isInOppActiveZone(c)) {
          knownDiscardSet.add(c);
        }
      }
    }
  }

  const classifyOppHandSince = (eventIdx: number): 'knownOppHand' | 'possibleOppHand' | 'unseen' => {
    if (eventIdx < lastReshuffleIdx) {
      const discardedBeforeReshuffle = oppUnknownDiscardIndices.some(
        (i) => i > eventIdx && i < lastReshuffleIdx,
      );
      if (discardedBeforeReshuffle) {
        return 'unseen';
      }
      const discardedAfterReshuffle = oppUnknownDiscardIndices.some((i) => i > lastReshuffleIdx);
      return discardedAfterReshuffle ? 'possibleOppHand' : 'knownOppHand';
    }
    const discardedAfterEvent = oppUnknownDiscardIndices.some((i) => i > eventIdx);
    return discardedAfterEvent ? 'possibleOppHand' : 'knownOppHand';
  };

  // Completed draft generations: classify the 3 cards opp held in Gen g Research
  for (const [gen, genDrafts] of draftsByGen.entries()) {
    const oppBuy = oppResearchBuyByGen.get(gen);
    if (oppBuy === undefined) {
      continue;
    }
    const d1 = genDrafts[0];
    const d2 = genDrafts[1];
    const d3 = genDrafts[2];
    if (d1 === undefined || d2 === undefined || d3 === undefined) {
      continue;
    }
    const known3: Array<string> = [...subtractCards(d1.passed, d3.packet), ...d3.passed];
    const actorOffer = researchOfferByGen.get(gen);
    if (actorOffer !== undefined && actorOffer.length > 0) {
      const aR4 = subtractCards(actorOffer, [...d1.taken, ...d2.taken, ...d3.taken]);
      known3.push(...subtractCards(d2.passed, aR4));
    } else {
      // Edge fallback: both cards of d2.passed if actor offer wasn't found
      known3.push(...d2.passed);
    }

    const noReshuffleSince = oppBuy.logIdx > lastReshuffleIdx;

    if (oppBuy.count === 0) {
      if (noReshuffleSince) {
        for (const c of known3) {
          if (!isLocatedElsewhere(c) && !isInOppActiveZone(c)) {
            knownDiscardSet.add(c);
          }
        }
      }
    } else if (oppBuy.count >= 4) {
      const status = classifyOppHandSince(oppBuy.logIdx);
      for (const c of known3) {
        if (isLocatedElsewhere(c) || isInOppActiveZone(c)) continue;
        if (status === 'knownOppHand') {
          knownOppHandSet.add(c);
        } else if (status === 'possibleOppHand') {
          possibleOppHandSet.add(c);
        }
      }
    } else {
      // 1 <= oppBuy.count <= 3: if no reshuffle since, any unplayed card is either in oppHand or discardPile;
      // if a reshuffle happened since, discardPile was shuffled into drawPile -> return to unseen pool.
      if (noReshuffleSince) {
        for (const c of known3) {
          if (!isLocatedElsewhere(c) && !isInOppActiveZone(c)) {
            possibleOppHandSet.add(c);
          }
        }
      }
    }
  }

  // Opponent publicly drawn cards ('${opp} drew ${1}')
  for (const draw of oppPublicDraws) {
    const c = draw.card;
    if (isLocatedElsewhere(c) || isInOppActiveZone(c)) continue;
    const status = classifyOppHandSince(draw.logIdx);
    if (status === 'knownOppHand') {
      knownOppHandSet.add(c);
    } else if (status === 'possibleOppHand') {
      possibleOppHandSet.add(c);
    }
  }

  // Actor cardSelections unselected (e.g. Inventors' Guild, Business Network, Business Contacts, Invention Contest)
  for (const sel of obs.history.cardSelections) {
    const afterReshuffle =
      lastReshuffleIdx === -1 ||
      sel.generation > lastReshuffleGen ||
      (sel.generation === lastReshuffleGen &&
        (actorResearchBuyByGen.get(sel.generation)?.logIdx ?? -1) > lastReshuffleIdx &&
        sel.phase === Phase.ACTION);
    if (afterReshuffle) {
      for (const c of sel.unselected) {
        if (pools.projectSet.has(c) && !isLocatedElsewhere(c) && !isInOppActiveZone(c)) {
          knownDiscardSet.add(c);
        }
      }
    }
  }

  // Publicly revealed discards ('Discarded ...', 'revealed and discarded ...')
  for (const pd of publicDiscards) {
    if (pd.logIdx > lastReshuffleIdx) {
      if (!isLocatedElsewhere(pd.card) && !isInOppActiveZone(pd.card)) {
        knownDiscardSet.add(pd.card);
      }
    }
  }

  // Cards actor previously bought/took/selected into own hand, now missing from both actorHand and tableau
  // -> actor discarded them via Sell Patents or Mars University!
  const allActorAcquired = new Set<string>();
  if (actorInitialBuy !== null) {
    for (const c of actorInitialBuy.projectsBought) allActorAcquired.add(c);
  }
  for (const d of obs.history.drafts) {
    // Only consider drafts whose Research phase finished (not still in DRAFTING or incomplete)
    if (d.generation < obs.public.generation || (obs.public.phase !== Phase.DRAFTING && obs.public.phase !== Phase.RESEARCH)) {
      for (const c of d.taken) allActorAcquired.add(c);
    }
  }
  for (const rb of obs.history.researchBuys) {
    for (const c of rb.bought) allActorAcquired.add(c);
  }
  for (const sel of obs.history.cardSelections) {
    for (const c of sel.selected) {
      if (pools.projectSet.has(c)) allActorAcquired.add(c);
    }
  }
  const actorHasUnknownDiscardBeforeReshuffle =
    lastReshuffleIdx !== -1 && actorUnknownDiscardIndices.some((i) => i < lastReshuffleIdx);
  if (!actorHasUnknownDiscardBeforeReshuffle && actorUnknownDiscardIndices.length > 0) {
    for (const c of allActorAcquired) {
      if (!isLocatedElsewhere(c) && !isInOppActiveZone(c)) {
        knownDiscardSet.add(c);
      }
    }
  }

  // Note: If a draft round for a generation completed drafting and actor finished research, any d1/d2/d3
  // cards NOT taken by actor or placed in opp's researchKnown3 are accounted above; ensure no overlap.
  for (const c of knownOppHandSet) {
    possibleOppHandSet.delete(c);
    knownDiscardSet.delete(c);
  }
  for (const c of possibleOppHandSet) {
    knownDiscardSet.delete(c);
  }

  const isExcludedFromUnseenProjects = (c: string): boolean =>
    isLocatedElsewhere(c) ||
    isInOppActiveZone(c) ||
    knownOppHandSet.has(c) ||
    possibleOppHandSet.has(c) ||
    knownDiscardSet.has(c);

  // Sanity guard for initialGen when reshuffle check didn't trigger yet
  void isReshuffleBeforeGenResearch;

  const unseenProjects = pools.projectNames.filter((c) => !isExcludedFromUnseenProjects(c));

  return {
    actor,
    generation: obs.public.generation,
    phase: obs.public.phase,
    deckSize: obs.public.deckSize,
    discardPileSize: obs.public.discardPileSize,
    opponentHandSize: obs.public.players[opp].cardsInHandCount,
    opponentPreludeHandSize,
    opponentInitialBuyComplete: oppInitialBuyComplete,
    opponentHasHiddenPickedCorp,
    opponentDraftHandSize,
    opponentDraftedSize,
    opponentResearchOfferSize,
    projects: unseenProjects,
    preludes: unseenPreludes,
    corporations: unseenCorps,
    knownOpponentHandProjects: Array.from(knownOppHandSet).sort(compareStrings),
    possibleOpponentHandProjects: Array.from(possibleOppHandSet).sort(compareStrings),
    knownOpponentDraftHandProjects: Array.from(knownOpponentDraftHandSet).sort(compareStrings),
    knownOpponentDraftedProjects: Array.from(knownOpponentDraftedSet).sort(compareStrings),
    knownOpponentResearchProjects: Array.from(knownOpponentResearchSet).sort(compareStrings),
    knownProjectDiscardPile: Array.from(knownDiscardSet).sort(compareStrings),
  };
}

function assertUniqueAndTrack(
  cards: ReadonlyArray<string>,
  seen: Set<string>,
  label: string,
): void {
  for (const c of cards) {
    if (seen.has(c)) {
      throw new Error(`Invalid sampled world: duplicate card "${c}" in ${label}`);
    }
    seen.add(c);
  }
}

function instantiateProject(name: string): IProjectCard {
  const card = newProjectCard(name);
  if (card === undefined) {
    throw new Error(`Unknown project card "${name}" in sampled world`);
  }
  return card;
}

function instantiatePrelude(name: string): IPreludeCard {
  const card = newPrelude(name);
  if (card === undefined) {
    throw new Error(`Unknown prelude card "${name}" in sampled world`);
  }
  return card;
}

function instantiateCorp(name: string): ICorporationCard {
  const card = newCorporationCard(name);
  if (card === undefined) {
    throw new Error(`Unknown corporation card "${name}" in sampled world`);
  }
  return card;
}

export function applyDeterminization(
  engineGame: IGame,
  obs: ActorObservation,
  actor: Seat,
  world: SampledWorld,
): void {
  const pool = computeUnseenPool(obs);
  const opp: Seat = actor === 0 ? 1 : 0;

  const unseenProjSet = new Set(pool.projects);
  const unseenPrelSet = new Set(pool.preludes);
  const unseenCorpSet = new Set(pool.corporations);
  const possOppSet = new Set(pool.possibleOpponentHandProjects);

  const usedProjects = new Set<string>();
  const usedPreludes = new Set<string>();
  const usedCorps = new Set<string>();

  // 1. Validate opponentHand
  if (!Array.isArray(world.opponentHand) || world.opponentHand.length !== pool.opponentHandSize) {
    throw new Error(
      `Invalid sampled world: opponentHand length ${world.opponentHand?.length ?? 'missing'} does not match expected opponentHandSize ${pool.opponentHandSize}`,
    );
  }
  assertUniqueAndTrack(world.opponentHand, usedProjects, 'opponentHand');
  const oppHandSet = new Set(world.opponentHand);
  for (const required of pool.knownOpponentHandProjects) {
    if (!oppHandSet.has(required)) {
      throw new Error(
        `Invalid sampled world: known opponent hand card "${required}" is missing from opponentHand`,
      );
    }
  }
  const knownOppHandSet = new Set(pool.knownOpponentHandProjects);
  for (const c of world.opponentHand) {
    if (!knownOppHandSet.has(c) && !possOppSet.has(c) && !unseenProjSet.has(c)) {
      throw new Error(
        `Invalid sampled world: opponentHand card "${c}" is outside actor unseen pool / known opponent holdings`,
      );
    }
  }

  // 2. Validate projectDeck
  if (!Array.isArray(world.projectDeck) || world.projectDeck.length !== pool.deckSize) {
    throw new Error(
      `Invalid sampled world: projectDeck length ${world.projectDeck?.length ?? 'missing'} does not match expected deckSize ${pool.deckSize}`,
    );
  }
  assertUniqueAndTrack(world.projectDeck, usedProjects, 'projectDeck');
  for (const c of world.projectDeck) {
    if (!unseenProjSet.has(c)) {
      throw new Error(
        `Invalid sampled world: projectDeck card "${c}" is outside actor unseen pool`,
      );
    }
  }

  // 3. Validate setup dealt offers if opponent initialBuy not yet complete
  if (!pool.opponentInitialBuyComplete) {
    if (!Array.isArray(world.opponentDealtCorporations) || world.opponentDealtCorporations.length !== constants.CORPORATION_CARDS_DEALT_PER_PLAYER) {
      throw new Error(
        `Invalid sampled world: opponentDealtCorporations must have length ${constants.CORPORATION_CARDS_DEALT_PER_PLAYER}`,
      );
    }
    assertUniqueAndTrack(world.opponentDealtCorporations, usedCorps, 'opponentDealtCorporations');
    for (const c of world.opponentDealtCorporations) {
      if (!unseenCorpSet.has(c)) {
        throw new Error(`Invalid sampled world: opponentDealtCorporations card "${c}" is outside unseen pool`);
      }
    }

    if (!Array.isArray(world.opponentDealtPreludes) || world.opponentDealtPreludes.length !== constants.PRELUDE_CARDS_DEALT_PER_PLAYER) {
      throw new Error(
        `Invalid sampled world: opponentDealtPreludes must have length ${constants.PRELUDE_CARDS_DEALT_PER_PLAYER}`,
      );
    }
    assertUniqueAndTrack(world.opponentDealtPreludes, usedPreludes, 'opponentDealtPreludes');
    for (const c of world.opponentDealtPreludes) {
      if (!unseenPrelSet.has(c)) {
        throw new Error(`Invalid sampled world: opponentDealtPreludes card "${c}" is outside unseen pool`);
      }
    }

    if (!Array.isArray(world.opponentDealtProjects) || world.opponentDealtProjects.length !== 10) {
      throw new Error('Invalid sampled world: opponentDealtProjects must have length 10');
    }
    assertUniqueAndTrack(world.opponentDealtProjects, usedProjects, 'opponentDealtProjects');
    for (const c of world.opponentDealtProjects) {
      if (!unseenProjSet.has(c)) {
        throw new Error(`Invalid sampled world: opponentDealtProjects card "${c}" is outside unseen pool`);
      }
    }
  } else if (pool.opponentHasHiddenPickedCorp) {
    if (typeof world.opponentPickedCorporation !== 'string' || !unseenCorpSet.has(world.opponentPickedCorporation)) {
      throw new Error(
        'Invalid sampled world: opponentPickedCorporation must be a valid unseen corporation when opponent submitted initial selection first',
      );
    }
    usedCorps.add(world.opponentPickedCorporation);
  }

  // 4. Validate opponentPreludes
  if (pool.opponentPreludeHandSize > 0) {
    if (!Array.isArray(world.opponentPreludes) || world.opponentPreludes.length !== pool.opponentPreludeHandSize) {
      throw new Error(
        `Invalid sampled world: opponentPreludes length ${world.opponentPreludes?.length ?? 'missing'} does not match expected opponentPreludeHandSize ${pool.opponentPreludeHandSize}`,
      );
    }
    assertUniqueAndTrack(world.opponentPreludes, usedPreludes, 'opponentPreludes');
    for (const c of world.opponentPreludes) {
      if (!unseenPrelSet.has(c)) {
        throw new Error(`Invalid sampled world: opponentPreludes card "${c}" is outside unseen pool`);
      }
    }
  }

  // 5. Validate opponentDraftHand & opponentDrafted
  if (pool.opponentDraftHandSize > 0 || pool.opponentDraftedSize > 0) {
    if (!Array.isArray(world.opponentDraftHand) || world.opponentDraftHand.length !== pool.opponentDraftHandSize) {
      throw new Error(
        `Invalid sampled world: opponentDraftHand length ${world.opponentDraftHand?.length ?? 'missing'} does not match expected ${pool.opponentDraftHandSize}`,
      );
    }
    assertUniqueAndTrack(world.opponentDraftHand, usedProjects, 'opponentDraftHand');
    const draftHandSet = new Set(world.opponentDraftHand);
    const knownDraftHandSet = new Set(pool.knownOpponentDraftHandProjects);
    for (const forced of pool.knownOpponentDraftHandProjects) {
      if (!draftHandSet.has(forced)) {
        throw new Error(
          `Invalid sampled world: forced draft hand card "${forced}" passed by actor is missing from opponentDraftHand`,
        );
      }
    }
    for (const c of world.opponentDraftHand) {
      if (!knownDraftHandSet.has(c) && !unseenProjSet.has(c)) {
        throw new Error(`Invalid sampled world: opponentDraftHand card "${c}" is outside unseen pool / forced passed cards`);
      }
    }

    const oppDrafted = world.opponentDrafted ?? [];
    if (oppDrafted.length !== pool.opponentDraftedSize) {
      throw new Error(
        `Invalid sampled world: opponentDrafted length ${oppDrafted.length} does not match expected ${pool.opponentDraftedSize}`,
      );
    }
    assertUniqueAndTrack(oppDrafted, usedProjects, 'opponentDrafted');
    const draftedSet = new Set(oppDrafted);
    const knownDraftedSet = new Set(pool.knownOpponentDraftedProjects);
    for (const forced of pool.knownOpponentDraftedProjects) {
      if (!draftedSet.has(forced)) {
        throw new Error(
          `Invalid sampled world: forced drafted card "${forced}" from actor draft history is missing from opponentDrafted`,
        );
      }
    }
    for (const c of oppDrafted) {
      if (!knownDraftedSet.has(c) && !unseenProjSet.has(c)) {
        throw new Error(`Invalid sampled world: opponentDrafted card "${c}" is outside unseen pool / forced drafted cards`);
      }
    }
  }

  // 6. Validate opponentResearchOffer
  if (pool.opponentResearchOfferSize > 0) {
    if (!Array.isArray(world.opponentResearchOffer) || world.opponentResearchOffer.length !== pool.opponentResearchOfferSize) {
      throw new Error(
        `Invalid sampled world: opponentResearchOffer length ${world.opponentResearchOffer?.length ?? 'missing'} does not match expected ${pool.opponentResearchOfferSize}`,
      );
    }
    assertUniqueAndTrack(world.opponentResearchOffer, usedProjects, 'opponentResearchOffer');
    const resSet = new Set(world.opponentResearchOffer);
    const knownResSet = new Set(pool.knownOpponentResearchProjects);
    for (const forced of pool.knownOpponentResearchProjects) {
      if (!resSet.has(forced)) {
        throw new Error(
          `Invalid sampled world: forced draft card "${forced}" is missing from opponentResearchOffer`,
        );
      }
    }
    for (const c of world.opponentResearchOffer) {
      if (!knownResSet.has(c) && !unseenProjSet.has(c)) {
        throw new Error(`Invalid sampled world: opponentResearchOffer card "${c}" is outside unseen pool / forced drafted cards`);
      }
    }
  }

  // 7. Compute/validate projectDiscardPile
  const mustBeDiscardedPoss = pool.possibleOpponentHandProjects.filter((c) => !oppHandSet.has(c));
  const minDiscardNeeded = pool.knownProjectDiscardPile.length + mustBeDiscardedPoss.length;
  if (minDiscardNeeded > pool.discardPileSize) {
    throw new Error(
      `Invalid sampled world: ${minDiscardNeeded} known/unkept cards must reside in discardPile, exceeding discardPileSize ${pool.discardPileSize}`,
    );
  }

  let discardPileNames: Array<string>;
  if (world.projectDiscardPile !== undefined) {
    if (world.projectDiscardPile.length !== pool.discardPileSize) {
      throw new Error(
        `Invalid sampled world: projectDiscardPile length ${world.projectDiscardPile.length} does not match expected discardPileSize ${pool.discardPileSize}`,
      );
    }
    assertUniqueAndTrack(world.projectDiscardPile, usedProjects, 'projectDiscardPile');
    const discSet = new Set(world.projectDiscardPile);
    const knownDiscSet = new Set(pool.knownProjectDiscardPile);
    const mustPossSet = new Set(mustBeDiscardedPoss);
    for (const forced of pool.knownProjectDiscardPile) {
      if (!discSet.has(forced)) {
        throw new Error(`Invalid sampled world: known discarded card "${forced}" missing from projectDiscardPile`);
      }
    }
    for (const poss of mustBeDiscardedPoss) {
      if (!discSet.has(poss)) {
        throw new Error(`Invalid sampled world: unkept passed/drawn card "${poss}" missing from projectDiscardPile`);
      }
    }
    for (const c of world.projectDiscardPile) {
      if (!knownDiscSet.has(c) && !mustPossSet.has(c) && !unseenProjSet.has(c)) {
        throw new Error(`Invalid sampled world: projectDiscardPile card "${c}" is outside unseen pool / known discards`);
      }
    }
    discardPileNames = world.projectDiscardPile.slice();
  } else {
    const remainingUnseen = pool.projects.filter((c) => !usedProjects.has(c));
    const fillNeeded = pool.discardPileSize - minDiscardNeeded;
    if (remainingUnseen.length < fillNeeded) {
      throw new Error(
        `Invalid sampled world: remaining unseen pool (${remainingUnseen.length}) cannot fill discardPile (${fillNeeded})`,
      );
    }
    discardPileNames = [
      ...pool.knownProjectDiscardPile,
      ...mustBeDiscardedPoss,
      ...remainingUnseen.slice(0, fillNeeded),
    ];
  }

  // ---- Write validated world strictly in-place into clone's engine state ----
  const oppPlayer = engineGame.players[opp];

  oppPlayer.cardsInHand.splice(0, oppPlayer.cardsInHand.length, ...world.opponentHand.map(instantiateProject));
  engineGame.projectDeck.drawPile.splice(0, engineGame.projectDeck.drawPile.length, ...world.projectDeck.map(instantiateProject));
  engineGame.projectDeck.discardPile.splice(0, engineGame.projectDeck.discardPile.length, ...discardPileNames.map(instantiateProject));

  // Corporations & preludes setup/decks
  const actorDiscardedCorps: Array<string> =
    obs.history.initialBuy !== null ? obs.history.initialBuy.corporationsDiscarded.slice() : [];
  const actorDiscardedPreludes: Array<string> =
    obs.history.initialBuy !== null ? obs.history.initialBuy.preludesDiscarded.slice() : [];
  const staticPools = getStaticCardPools(obs.public.map);
  for (const sel of obs.history.cardSelections) {
    for (const p of sel.unselected) {
      if (staticPools.preludeSet.has(p)) {
        actorDiscardedPreludes.push(p);
      }
    }
  }

  const writeCorporationAndPreludeDecks = (updatePreludes: boolean): void => {
    const remCorps = pool.corporations.filter((c) => !usedCorps.has(c));
    const corpDrawCount = obs.public.otherDeckSizes.corporations.drawPile;
    const corpDiscardCount = obs.public.otherDeckSizes.corporations.discardPile;
    const corpDiscardPool = [...actorDiscardedCorps, ...remCorps.slice(corpDrawCount)];
    engineGame.corporationDeck.drawPile.splice(
      0,
      engineGame.corporationDeck.drawPile.length,
      ...remCorps.slice(0, corpDrawCount).map(instantiateCorp),
    );
    engineGame.corporationDeck.discardPile.splice(
      0,
      engineGame.corporationDeck.discardPile.length,
      ...corpDiscardPool.slice(0, corpDiscardCount).map(instantiateCorp),
    );

    if (updatePreludes && obs.public.otherDeckSizes.preludes !== null) {
      const remPrels = pool.preludes.filter((c) => !usedPreludes.has(c));
      const prelDrawCount = obs.public.otherDeckSizes.preludes.drawPile;
      const prelDiscardCount = obs.public.otherDeckSizes.preludes.discardPile;
      const prelDiscardPool = [...actorDiscardedPreludes, ...remPrels.slice(prelDrawCount)];
      engineGame.preludeDeck.drawPile.splice(
        0,
        engineGame.preludeDeck.drawPile.length,
        ...remPrels.slice(0, prelDrawCount).map(instantiatePrelude),
      );
      engineGame.preludeDeck.discardPile.splice(
        0,
        engineGame.preludeDeck.discardPile.length,
        ...prelDiscardPool.slice(0, prelDiscardCount).map(instantiatePrelude),
      );
    }
  };

  if (!pool.opponentInitialBuyComplete) {
    oppPlayer.dealtCorporationCards.splice(
      0,
      oppPlayer.dealtCorporationCards.length,
      ...world.opponentDealtCorporations!.map(instantiateCorp),
    );
    oppPlayer.dealtPreludeCards.splice(
      0,
      oppPlayer.dealtPreludeCards.length,
      ...world.opponentDealtPreludes!.map(instantiatePrelude),
    );
    oppPlayer.dealtProjectCards.splice(
      0,
      oppPlayer.dealtProjectCards.length,
      ...world.opponentDealtProjects!.map(instantiateProject),
    );
    writeCorporationAndPreludeDecks(true);
  } else {
    if (pool.opponentHasHiddenPickedCorp && world.opponentPickedCorporation !== undefined) {
      oppPlayer.pickedCorporationCard = instantiateCorp(world.opponentPickedCorporation);
      writeCorporationAndPreludeDecks(false);
    }

    if (pool.opponentPreludeHandSize > 0 && world.opponentPreludes !== undefined) {
      oppPlayer.preludeCardsInHand.splice(
        0,
        oppPlayer.preludeCardsInHand.length,
        ...world.opponentPreludes.map(instantiatePrelude),
      );
    }

    if (obs.public.otherDeckSizes.preludes !== null && pool.opponentPreludeHandSize > 0) {
      const remPrels = pool.preludes.filter((c) => !usedPreludes.has(c));
      const prelDrawCount = obs.public.otherDeckSizes.preludes.drawPile;
      const prelDiscardCount = obs.public.otherDeckSizes.preludes.discardPile;
      const discardPool = [...actorDiscardedPreludes, ...remPrels.slice(prelDrawCount)];
      if (remPrels.length >= prelDrawCount && discardPool.length >= prelDiscardCount) {
        engineGame.preludeDeck.drawPile.splice(
          0,
          engineGame.preludeDeck.drawPile.length,
          ...remPrels.slice(0, prelDrawCount).map(instantiatePrelude),
        );
        engineGame.preludeDeck.discardPile.splice(
          0,
          engineGame.preludeDeck.discardPile.length,
          ...discardPool.slice(0, prelDiscardCount).map(instantiatePrelude),
        );
      }
    }
  }

  // Drafting: handle both opp still needing to draft AND opp (seat 0) having already picked on engine while actor === 1 is pending
  if (engineGame.phase === Phase.DRAFTING && (pool.opponentDraftHandSize > 0 || pool.opponentDraftedSize > 0)) {
    const newDraftHand = (world.opponentDraftHand ?? []).map(instantiateProject);
    const newDrafted = (world.opponentDrafted ?? []).map(instantiateProject);
    if (oppPlayer.needsToDraft === false) {
      oppPlayer.needsToDraft = true;
      oppPlayer.draftHand.splice(0, oppPlayer.draftHand.length, ...newDraftHand);
      oppPlayer.draftedCards.splice(0, oppPlayer.draftedCards.length, ...newDrafted);
      newStandardDraft(engineGame).restoreDraft();
    } else {
      oppPlayer.draftHand.splice(0, oppPlayer.draftHand.length, ...newDraftHand);
      oppPlayer.draftedCards.splice(0, oppPlayer.draftedCards.length, ...newDrafted);
    }
  }

  // Gen >= 2 Research: update opponent's 4-card SelectCard offer in-place
  if (engineGame.phase === Phase.RESEARCH && engineGame.generation >= 2 && pool.opponentResearchOfferSize > 0) {
    const waiting = oppPlayer.getWaitingFor();
    if (waiting !== undefined && waiting.type === 'card') {
      const cardsArr = (waiting as SelectCard<IProjectCard>).cards as Array<IProjectCard>;
      cardsArr.splice(0, cardsArr.length, ...world.opponentResearchOffer!.map(instantiateProject));
    }
  }
}
