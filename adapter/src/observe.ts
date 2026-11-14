import {
  Color,
  IGame,
  Phase,
  SelectCard,
  Server,
} from './engine';
import {
  ActorHistory,
  ActorObservation,
  AwardObservation,
  HandCardObservation,
  MilestoneObservation,
  PublicLogEntryObservation,
  PublicPlayerObservation,
  Seat,
  SpaceObservation,
  SupportedMap,
  TableauCardObservation,
} from './types';

function compareStrings(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function writeCanonicalJson(value: unknown, parts: Array<string>): void {
  if (value === null) {
    parts.push('null');
    return;
  }
  const t = typeof value;
  if (t === 'string') {
    parts.push(JSON.stringify(value));
    return;
  }
  if (t === 'boolean') {
    parts.push(value ? 'true' : 'false');
    return;
  }
  if (t === 'number') {
    if (!Number.isFinite(value as number)) {
      throw new Error(`Non-finite number in ActorObservation: ${String(value)}`);
    }
    parts.push(String(value));
    return;
  }
  if (Array.isArray(value)) {
    parts.push('[');
    for (let i = 0; i < value.length; i++) {
      if (i > 0) {
        parts.push(',');
      }
      writeCanonicalJson(value[i], parts);
    }
    parts.push(']');
    return;
  }
  if (t === 'object') {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record);
    let sorted = true;
    for (let i = 1; i < keys.length; i++) {
      if (keys[i - 1] > keys[i]) {
        sorted = false;
        break;
      }
    }
    if (!sorted) {
      keys.sort(compareStrings);
    }
    parts.push('{');
    let first = true;
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      const prop = record[key];
      if (prop === undefined) {
        continue;
      }
      if (!first) {
        parts.push(',');
      }
      first = false;
      parts.push(JSON.stringify(key));
      parts.push(':');
      writeCanonicalJson(prop, parts);
    }
    parts.push('}');
    return;
  }
  throw new Error(`Unsupported non-JSON value type "${t}" in ActorObservation`);
}

export function serializeObservation(observation: ActorObservation): string {
  const parts: Array<string> = [];
  writeCanonicalJson(observation, parts);
  return parts.join('');
}

function cloneHistory(history: ActorHistory): ActorHistory {
  return {
    initialBuy:
      history.initialBuy === null
        ? null
        : {
            corporationsOffered: history.initialBuy.corporationsOffered.slice(),
            corporationChosen: history.initialBuy.corporationChosen,
            corporationsDiscarded: history.initialBuy.corporationsDiscarded.slice(),
            preludesOffered: history.initialBuy.preludesOffered.slice(),
            preludesChosen: history.initialBuy.preludesChosen.slice(),
            preludesDiscarded: history.initialBuy.preludesDiscarded.slice(),
            projectsOffered: history.initialBuy.projectsOffered.slice(),
            projectsBought: history.initialBuy.projectsBought.slice(),
            projectsDiscarded: history.initialBuy.projectsDiscarded.slice(),
          },
    drafts: history.drafts.map((entry) => ({
      generation: entry.generation,
      draftRound: entry.draftRound,
      packet: entry.packet.slice(),
      taken: entry.taken.slice(),
      passed: entry.passed.slice(),
    })),
    researchBuys: history.researchBuys.map((entry) => ({
      generation: entry.generation,
      offered: entry.offered.slice(),
      bought: entry.bought.slice(),
      discarded: entry.discarded.slice(),
    })),
    cardSelections: history.cardSelections.map((entry) => ({
      generation: entry.generation,
      phase: entry.phase,
      offered: entry.offered.slice(),
      selected: entry.selected.slice(),
      unselected: entry.unselected.slice(),
    })),
  };
}

export function buildActorObservation(
  engineGame: IGame,
  map: SupportedMap,
  seat: Seat,
  history: ActorHistory,
): ActorObservation {
  if (seat !== 0 && seat !== 1) {
    throw new Error(`Invalid seat ${String(seat)}: expected 0 or 1`);
  }

  const actorPlayer = engineGame.players[seat];
  const model = Server.getPlayerModel(actorPlayer);

  const colorToSeatMap = new Map<Color, Seat>([
    [engineGame.players[0].color, 0],
    [engineGame.players[1].color, 1],
  ]);

  const colorToSeat = (color: Color): Seat => {
    const resolved = colorToSeatMap.get(color);
    if (resolved === undefined) {
      throw new Error(`Unknown player color "${color}" in game observation`);
    }
    return resolved;
  };

  const firstPlayer = colorToSeat(model.players[0].color);
  const activeEntry = model.players.find((p) => p.isActive);
  const activePlayer = activeEntry !== undefined
    ? colorToSeat(activeEntry.color)
    : colorToSeat(engineGame.activePlayer.color);

  const passedSet = new Set<Seat>(model.game.passedPlayers.map(colorToSeat));
  const passedPlayers = Array.from(passedSet).sort((a, b) => a - b);

  const spaces: Array<SpaceObservation> = model.game.spaces.map((s) => ({
    id: s.id,
    x: s.x,
    y: s.y,
    spaceType: s.spaceType,
    bonus: s.bonus.slice(),
    tileType: s.tileType ?? null,
    owner: s.color !== undefined ? (colorToSeatMap.get(s.color) ?? null) : null,
    highlight: s.highlight ?? null,
    rotated: s.rotated === true,
  }));

  const milestones: Array<MilestoneObservation> = model.game.milestones.map((m) => ({
    name: m.name,
    claimedBy: m.color !== undefined ? colorToSeat(m.color) : null,
    scores: m.scores
      .map((sc) => ({
        seat: colorToSeat(sc.color),
        score: sc.score,
        claimable: sc.claimable,
      }))
      .sort((a, b) => a.seat - b.seat),
  }));

  const awards: Array<AwardObservation> = model.game.awards.map((a) => ({
    name: a.name,
    fundedBy: a.color !== undefined ? colorToSeat(a.color) : null,
    scores: a.scores
      .map((sc) => ({
        seat: colorToSeat(sc.color),
        score: sc.score,
      }))
      .sort((a, b) => a.seat - b.seat),
  }));

  const buildPublicPlayer = (targetSeat: Seat): PublicPlayerObservation => {
    const p = model.players.find((candidate) => colorToSeat(candidate.color) === targetSeat);
    if (p === undefined) {
      throw new Error(`Missing player model for seat ${targetSeat}`);
    }
    const tableau: Array<TableauCardObservation> = p.tableau.map((card) => ({
      name: card.name,
      resources: card.resources ?? 0,
      isDisabled: card.isDisabled === true,
    }));
    return {
      seat: targetSeat,
      color: p.color,
      terraformRating: p.terraformRating,
      megacredits: p.megacredits,
      steel: p.steel,
      titanium: p.titanium,
      plants: p.plants,
      energy: p.energy,
      heat: p.heat,
      megacreditProduction: p.megacreditProduction,
      steelProduction: p.steelProduction,
      titaniumProduction: p.titaniumProduction,
      plantProduction: p.plantProduction,
      energyProduction: p.energyProduction,
      heatProduction: p.heatProduction,
      steelValue: p.steelValue,
      titaniumValue: p.titaniumValue,
      cardCost: p.cardCost,
      cardDiscount: p.cardDiscount,
      cardsInHandCount: p.cardsInHandNbr,
      citiesCount: p.citiesCount,
      noTagsCount: p.noTagsCount,
      tags: {...p.tags},
      tableau,
      usedActions: p.actionsThisGeneration.slice().sort(compareStrings),
      actionsTakenThisRound: p.actionsTakenThisRound,
      actionsTakenThisGame: p.actionsTakenThisGame,
      availableBlueCardActionCount: p.availableBlueCardActionCount,
      lastCardPlayed: p.lastCardPlayed ?? null,
      isActive: p.isActive,
      passed: passedSet.has(targetSeat),
      protectedResources: {...p.protectedResources},
      protectedProduction: {...p.protectedProduction},
    };
  };

  const publicLog: Array<PublicLogEntryObservation> = engineGame.gameLog
    .filter((entry) => entry.playerId === undefined)
    .map((entry) => ({
      type: entry.type ?? 0,
      message: entry.message,
      data: entry.data.map((datum) => ({
        type: datum.type ?? null,
        value: Array.isArray(datum.value) ? datum.value.slice() : datum.value,
      })),
    }));

  const cardsInHand: Array<HandCardObservation> = model.cardsInHand
    .map((c) => ({
      name: c.name,
      calculatedCost: c.calculatedCost ?? 0,
    }))
    .sort((a, b) => compareStrings(a.name, b.name));

  let currentDraftPacket: Array<string> = [];
  if (
    (engineGame.phase === Phase.DRAFTING || engineGame.phase === Phase.INITIALDRAFTING) &&
    actorPlayer.needsToDraft === true
  ) {
    const waiting = actorPlayer.getWaitingFor();
    if (waiting !== undefined && waiting.type === 'card') {
      currentDraftPacket = (waiting as SelectCard).cards.map((c) => c.name).sort(compareStrings);
    }
  }

  return {
    actor: seat,
    public: {
      map,
      generation: model.game.generation,
      phase: model.game.phase,
      firstPlayer,
      activePlayer,
      passedPlayers,
      globalParameters: {
        temperature: model.game.temperature,
        oxygenLevel: model.game.oxygenLevel,
        oceans: model.game.oceans,
        isTerraformed: model.game.isTerraformed,
      },
      deckSize: model.game.deckSize,
      discardPileSize: model.game.discardPileSize,
      otherDeckSizes: {
        corporations: {
          drawPile: model.game.otherDeckSizes.corporations.drawPile,
          discardPile: model.game.otherDeckSizes.corporations.discardPile,
        },
        preludes: model.game.otherDeckSizes.preludes !== undefined
          ? {
              drawPile: model.game.otherDeckSizes.preludes.drawPile,
              discardPile: model.game.otherDeckSizes.preludes.discardPile,
            }
          : null,
      },
      spaces,
      milestones,
      awards,
      players: [buildPublicPlayer(0), buildPublicPlayer(1)],
      publicLog,
    },
    private: {
      cardsInHand,
      dealtCorporationCards: model.dealtCorporationCards.map((c) => c.name).sort(compareStrings),
      dealtPreludeCards: model.dealtPreludeCards.map((c) => c.name).sort(compareStrings),
      dealtProjectCards: model.dealtProjectCards.map((c) => c.name).sort(compareStrings),
      pickedCorporationCard: model.pickedCorporationCard[0]?.name ?? null,
      preludeCardsInHand: model.preludeCardsInHand.map((c) => c.name).sort(compareStrings),
      draftedCards: model.draftedCards.map((c) => c.name).sort(compareStrings),
      currentDraftPacket,
    },
    history: cloneHistory(history),
  };
}
