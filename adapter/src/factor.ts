import {
  AndOptions,
  CardName,
  CardNames,
  EngineMessage,
  ICard,
  ICorporationCard,
  IPlayer,
  InputResponse,
  OrOptions,
  PaymentOptions,
  PlayerInput,
  SelectAmount,
  SelectCard,
  SelectInitialCards,
  SelectPayment,
  SelectPlayer,
  SelectProductionToLose,
  SelectProjectCardToPlay,
  SelectResource,
  SelectResources,
  SelectSpace,
  SelectStandardProjectToPlay,
  SpaceId,
  Units,
  UnitsUtil,
  constants,
} from './engine';
import {
  PaymentCandidate,
  ProjectPlayCandidate,
  enumerateCanonicalPayments,
  getPlayableProjectCandidates,
} from './payments';
import {Candidate, Choice, Seat} from './types';

function compareStrings(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

export function formatEngineTitle(title: string | EngineMessage | undefined): string {
  if (title === undefined) {
    return '';
  }
  if (typeof title === 'string') {
    return title;
  }
  if (typeof title === 'object' && typeof title.message === 'string') {
    return title.message.replace(/\$\{(\d+)\}/g, (_, idxStr: string) => {
      const entry = title.data?.[Number(idxStr)];
      return entry !== undefined ? String(entry.value) : '';
    });
  }
  return '';
}

function getEnabledCards(sc: SelectCard<ICard>): Array<ICard> {
  const enabled = sc.cards.filter((_, idx) => sc.config.enabled?.[idx] !== false);
  enabled.sort((a, b) => compareStrings(a.name, b.name));
  return enabled;
}

function getPaymentOptionsForSelectPayment(sp: SelectPayment, player: IPlayer): Partial<PaymentOptions> {
  return {
    heat: player.canUseHeatAsMegaCredits,
    lunaTradeFederationTitanium: player.canUseTitaniumAsMegacredits,
    ...sp.paymentOptions,
  };
}

function getProductionMinimums(): Record<string, number> {
  return (
    constants.PRODUCTION_MINIMUMS ?? {
      megacredits: -5,
      steel: 0,
      titanium: 0,
      plants: 0,
      energy: 0,
      heat: 0,
    }
  );
}

export function hasLegalContinuation(input: PlayerInput, player: IPlayer): boolean {
  switch (input.type) {
    case 'option':
      return true;
    case 'space':
      return (input as SelectSpace).spaces.length > 0;
    case 'player':
      return (input as SelectPlayer).players.length > 0;
    case 'amount': {
      const sa = input as SelectAmount;
      return sa.max >= sa.min;
    }
    case 'resource':
      return (input as SelectResource).include.length > 0;
    case 'resources':
      return (input as SelectResources).count >= 0;
    case 'productionToLose': {
      const spl = input as SelectProductionToLose;
      const minimums = getProductionMinimums();
      const current = spl.player.production.asUnits();
      let reducible = 0;
      for (const k of UnitsUtil.keys) {
        reducible += Math.max(0, current[k] - (minimums[k] ?? 0));
      }
      return reducible >= spl.unitsToLose;
    }
    case 'payment': {
      const sp = input as SelectPayment;
      const opts = getPaymentOptionsForSelectPayment(sp, player);
      return enumerateCanonicalPayments(sp.amount, opts, sp.reserveUnits, player).length > 0;
    }
    case 'card': {
      const sc = input as SelectCard<ICard>;
      return getEnabledCards(sc).length >= sc.config.min;
    }
    case 'projectCard': {
      const pc = input as SelectProjectCardToPlay | SelectStandardProjectToPlay;
      return getPlayableProjectCandidates(pc, player).length > 0;
    }
    case 'or': {
      const orInput = input as OrOptions;
      return orInput.options.some((opt) => hasLegalContinuation(opt, player));
    }
    case 'and': {
      const andInput = input as AndOptions;
      return andInput.options.every((opt) => hasLegalContinuation(opt, player));
    }
    case 'initialCards':
      return true;
    default:
      return false;
  }
}

type PromptFrame =
  | {
      kind: 'or';
      input: OrOptions;
      legalIndices: Array<number>;
      chosenOrigIndex?: number;
      child?: PromptFrame;
    }
  | {
      kind: 'and';
      input: AndOptions;
      stepIndex: number;
      responses: Array<InputResponse>;
      child: PromptFrame;
    }
  | {
      kind: 'initialCards';
      input: SelectInitialCards;
      stepIndex: number;
      responses: Array<InputResponse>;
      chosenCorp?: ICorporationCard;
      child: PromptFrame;
    }
  | {
      kind: 'single_card';
      input: SelectCard<ICard>;
      orderedCards: ReadonlyArray<ICard>;
    }
  | {
      kind: 'multi_card';
      input: SelectCard<ICard>;
      orderedCards: ReadonlyArray<ICard>;
      picked: Array<CardName>;
      min: number;
      max: number;
    }
  | {
      kind: 'project_card';
      input: SelectProjectCardToPlay | SelectStandardProjectToPlay;
      candidates: ReadonlyArray<ProjectPlayCandidate>;
      chosen?: ProjectPlayCandidate;
    }
  | {
      kind: 'payment';
      input: SelectPayment;
      payments: ReadonlyArray<PaymentCandidate>;
    }
  | {
      kind: 'space';
      input: SelectSpace;
      spaces: ReadonlyArray<{id: SpaceId}>;
    }
  | {
      kind: 'player';
      input: SelectPlayer;
      players: ReadonlyArray<IPlayer>;
    }
  | {
      kind: 'amount';
      input: SelectAmount;
    }
  | {
      kind: 'resource';
      input: SelectResource;
    }
  | {
      kind: 'resources';
      input: SelectResources;
      units: Units;
      remaining: number;
    }
  | {
      kind: 'productionToLose';
      input: SelectProductionToLose;
      lost: Units;
      remaining: number;
      minimums: Record<string, number>;
    }
  | {
      kind: 'option';
      input: PlayerInput;
    };

function buildFrame(input: PlayerInput, player: IPlayer, maxCardOverride?: number): PromptFrame {
  switch (input.type) {
    case 'or': {
      const orInput = input as OrOptions;
      const legalIndices: Array<number> = [];
      for (let i = 0; i < orInput.options.length; i++) {
        if (hasLegalContinuation(orInput.options[i], player)) {
          legalIndices.push(i);
        }
      }
      if (legalIndices.length === 0) {
        throw new Error(`OrOptions "${formatEngineTitle(orInput.title)}" has no legal branches`);
      }
      return {kind: 'or', input: orInput, legalIndices};
    }

    case 'and': {
      const andInput = input as AndOptions;
      if (andInput.options.length === 0) {
        throw new Error('AndOptions has 0 options');
      }
      return {
        kind: 'and',
        input: andInput,
        stepIndex: 0,
        responses: [],
        child: buildFrame(andInput.options[0], player),
      };
    }

    case 'initialCards': {
      const init = input as SelectInitialCards;
      if (init.options.length === 0) {
        throw new Error('SelectInitialCards has 0 options');
      }
      return {
        kind: 'initialCards',
        input: init,
        stepIndex: 0,
        responses: [],
        child: buildFrame(init.options[0], player),
      };
    }

    case 'card': {
      const sc = input as SelectCard<ICard>;
      const orderedCards = getEnabledCards(sc);
      const maxCap = maxCardOverride !== undefined ? Math.min(sc.config.max, maxCardOverride) : sc.config.max;
      const effectiveMax = Math.min(maxCap, orderedCards.length);
      const effectiveMin = Math.min(sc.config.min, effectiveMax);
      if (orderedCards.length < sc.config.min) {
        throw new Error(`SelectCard requires at least ${sc.config.min} cards, only ${orderedCards.length} enabled`);
      }
      if (effectiveMin === 1 && effectiveMax === 1) {
        return {kind: 'single_card', input: sc, orderedCards};
      }
      return {
        kind: 'multi_card',
        input: sc,
        orderedCards,
        picked: [],
        min: effectiveMin,
        max: effectiveMax,
      };
    }

    case 'projectCard': {
      const pc = input as SelectProjectCardToPlay | SelectStandardProjectToPlay;
      const candidates = getPlayableProjectCandidates(pc, player);
      if (candidates.length === 0) {
        throw new Error('SelectCardToPlay has no playable/affordable cards');
      }
      return {kind: 'project_card', input: pc, candidates};
    }

    case 'payment': {
      const sp = input as SelectPayment;
      const opts = getPaymentOptionsForSelectPayment(sp, player);
      const payments = enumerateCanonicalPayments(sp.amount, opts, sp.reserveUnits, player);
      if (payments.length === 0) {
        throw new Error(`Cannot afford SelectPayment of ${sp.amount} M€`);
      }
      return {kind: 'payment', input: sp, payments};
    }

    case 'space': {
      const ss = input as SelectSpace;
      if (ss.spaces.length === 0) {
        throw new Error('SelectSpace has no available spaces');
      }
      return {kind: 'space', input: ss, spaces: ss.spaces};
    }

    case 'player': {
      const sp = input as SelectPlayer;
      if (sp.players.length === 0) {
        throw new Error('SelectPlayer has no available players');
      }
      return {kind: 'player', input: sp, players: sp.players};
    }

    case 'amount': {
      const sa = input as SelectAmount;
      if (sa.max < sa.min) {
        throw new Error(`SelectAmount invalid range [${sa.min}, ${sa.max}]`);
      }
      return {kind: 'amount', input: sa};
    }

    case 'resource': {
      const sr = input as SelectResource;
      if (sr.include.length === 0) {
        throw new Error('SelectResource has no included resources');
      }
      return {kind: 'resource', input: sr};
    }

    case 'resources': {
      const srs = input as SelectResources;
      if (srs.count <= 0) {
        throw new Error(`SelectResources unexpected count ${srs.count}`);
      }
      return {
        kind: 'resources',
        input: srs,
        units: {...UnitsUtil.EMPTY},
        remaining: srs.count,
      };
    }

    case 'productionToLose': {
      const spl = input as SelectProductionToLose;
      if (spl.unitsToLose <= 0) {
        throw new Error(`SelectProductionToLose unexpected unitsToLose ${spl.unitsToLose}`);
      }
      return {
        kind: 'productionToLose',
        input: spl,
        lost: {...UnitsUtil.EMPTY},
        remaining: spl.unitsToLose,
        minimums: getProductionMinimums(),
      };
    }

    case 'option':
      return {kind: 'option', input};

    default:
      throw new Error(`Unsupported PlayerInput type in training format: "${input.type}"`);
  }
}

export class DecisionStepper {
  public readonly seat: Seat;
  public readonly rootInput: PlayerInput;
  public readonly phase: string;
  public readonly generation: number;
  public readonly candidateIndices: Array<number> = [];
  public readonly candidateIds: Array<string> = [];
  private readonly player: IPlayer;
  private readonly rootFrame: PromptFrame;

  constructor(seat: Seat, rootInput: PlayerInput, player: IPlayer, phase: string, generation: number) {
    this.seat = seat;
    this.rootInput = rootInput;
    this.player = player;
    this.phase = phase;
    this.generation = generation;
    this.rootFrame = buildFrame(rootInput, player);
  }

  public currentChoice(): Choice {
    return this.choiceForFrame(this.rootFrame);
  }

  public choose(candidateIndex: number): {done: false} | {done: true; response: InputResponse} {
    const choice = this.currentChoice();
    if (!Number.isInteger(candidateIndex) || candidateIndex < 0 || candidateIndex >= choice.candidates.length) {
      throw new RangeError(
        `Candidate index ${candidateIndex} out of bounds [0, ${choice.candidates.length - 1}] for seat ${this.seat} (${choice.kind})`,
      );
    }
    this.candidateIndices.push(candidateIndex);
    this.candidateIds.push(choice.candidates[candidateIndex].id);
    const response = this.applyToFrame(this.rootFrame, candidateIndex);
    if (response === undefined) {
      return {done: false};
    }
    return {done: true, response};
  }

  private choiceForFrame(frame: PromptFrame): Choice {
    switch (frame.kind) {
      case 'or': {
        if (frame.child !== undefined) {
          return this.choiceForFrame(frame.child);
        }
        const candidates: Array<Candidate> = frame.legalIndices.map((origIdx) => {
          const opt = frame.input.options[origIdx];
          const label = formatEngineTitle(opt.title) || opt.buttonLabel || opt.type;
          return {id: `branch:${origIdx}`, label};
        });
        return {
          seat: this.seat,
          kind: 'branch',
          prompt: formatEngineTitle(frame.input.title) || 'Select one option',
          candidates,
        };
      }

      case 'and':
      case 'initialCards':
        return this.choiceForFrame(frame.child);

      case 'single_card':
        return {
          seat: this.seat,
          kind: 'card',
          prompt: formatEngineTitle(frame.input.title) || 'Select a card',
          candidates: frame.orderedCards.map((c) => ({id: c.name, label: c.name})),
        };

      case 'multi_card': {
        const pickedSet = new Set(frame.picked);
        const candidates: Array<Candidate> = [];
        if (frame.picked.length < frame.max) {
          for (const c of frame.orderedCards) {
            if (!pickedSet.has(c.name)) {
              candidates.push({id: c.name, label: c.name});
            }
          }
        }
        if (frame.picked.length >= frame.min) {
          candidates.push({
            id: 'done',
            label: `Done (${frame.picked.length} selected)`,
          });
        }
        return {
          seat: this.seat,
          kind: 'multi_card',
          prompt: formatEngineTitle(frame.input.title) || 'Select cards',
          candidates,
        };
      }

      case 'project_card': {
        if (frame.chosen !== undefined) {
          return {
            seat: this.seat,
            kind: 'payment',
            prompt: `Select payment for ${frame.chosen.card.name}`,
            candidates: frame.chosen.payments.map((p) => ({id: p.id, label: p.label})),
          };
        }
        return {
          seat: this.seat,
          kind: 'project_card',
          prompt: formatEngineTitle(frame.input.title) || 'Play project card',
          candidates: frame.candidates.map((c) => ({id: c.card.name, label: c.card.name})),
        };
      }

      case 'payment':
        return {
          seat: this.seat,
          kind: 'payment',
          prompt: formatEngineTitle(frame.input.title) || `Select payment (${frame.input.amount} M€)`,
          candidates: frame.payments.map((p) => ({id: p.id, label: p.label})),
        };

      case 'space':
        return {
          seat: this.seat,
          kind: 'space',
          prompt: formatEngineTitle(frame.input.title) || 'Select space',
          candidates: frame.spaces.map((s) => ({id: s.id, label: s.id})),
        };

      case 'player':
        return {
          seat: this.seat,
          kind: 'player',
          prompt: formatEngineTitle(frame.input.title) || 'Select player',
          candidates: frame.players.map((p) => ({id: p.color, label: `${p.name} (${p.color})`})),
        };

      case 'amount': {
        const candidates: Array<Candidate> = [];
        for (let n = frame.input.min; n <= frame.input.max; n++) {
          candidates.push({id: String(n), label: String(n)});
        }
        return {
          seat: this.seat,
          kind: 'amount',
          prompt: formatEngineTitle(frame.input.title) || 'Select amount',
          candidates,
        };
      }

      case 'resource':
        return {
          seat: this.seat,
          kind: 'resource',
          prompt: formatEngineTitle(frame.input.title) || 'Select resource',
          candidates: frame.input.include.map((res) => ({id: res, label: res})),
        };

      case 'resources':
        return {
          seat: this.seat,
          kind: 'resource',
          prompt: `${formatEngineTitle(frame.input.title) || 'Gain standard resources'} (${frame.remaining} remaining)`,
          candidates: UnitsUtil.keys.map((k) => ({id: k, label: k})),
        };

      case 'productionToLose': {
        const current = frame.input.player.production.asUnits();
        const reducible = UnitsUtil.keys.filter(
          (k) => current[k] - frame.lost[k] > (frame.minimums[k] ?? 0),
        );
        return {
          seat: this.seat,
          kind: 'resource',
          prompt: `${formatEngineTitle(frame.input.title) || 'Lose production'} (${frame.remaining} remaining)`,
          candidates: reducible.map((k) => ({id: k, label: k})),
        };
      }

      case 'option':
        return {
          seat: this.seat,
          kind: 'branch',
          prompt: formatEngineTitle(frame.input.title) || frame.input.buttonLabel || 'Confirm',
          candidates: [{id: 'confirm', label: frame.input.buttonLabel || formatEngineTitle(frame.input.title) || 'Confirm'}],
        };
    }
  }

  private applyToFrame(frame: PromptFrame, candidateIndex: number): InputResponse | undefined {
    switch (frame.kind) {
      case 'or': {
        if (frame.child !== undefined) {
          const childResp = this.applyToFrame(frame.child, candidateIndex);
          if (childResp === undefined) {
            return undefined;
          }
          return {type: 'or', index: frame.chosenOrigIndex!, response: childResp};
        }
        const origIdx = frame.legalIndices[candidateIndex];
        const chosenOpt = frame.input.options[origIdx];
        if (chosenOpt.type === 'option') {
          return {type: 'or', index: origIdx, response: {type: 'option'}};
        }
        frame.chosenOrigIndex = origIdx;
        frame.child = buildFrame(chosenOpt, this.player);
        return undefined;
      }

      case 'and': {
        const childResp = this.applyToFrame(frame.child, candidateIndex);
        if (childResp === undefined) {
          return undefined;
        }
        frame.responses.push(childResp);
        frame.stepIndex += 1;
        if (frame.stepIndex >= frame.input.options.length) {
          return {type: 'and', responses: frame.responses};
        }
        frame.child = buildFrame(frame.input.options[frame.stepIndex], this.player);
        return undefined;
      }

      case 'initialCards': {
        const childResp = this.applyToFrame(frame.child, candidateIndex);
        if (childResp === undefined) {
          return undefined;
        }
        const completedOption = frame.input.options[frame.stepIndex];
        if (completedOption === frame.input.inputs.corp && childResp.type === 'card') {
          const corpSelect = completedOption as SelectCard<ICorporationCard>;
          frame.chosenCorp = corpSelect.cards.find((c) => c.name === childResp.cards[0]);
        }
        frame.responses.push(childResp);
        frame.stepIndex += 1;
        if (frame.stepIndex >= frame.input.options.length) {
          return {type: 'initialCards', responses: frame.responses};
        }
        const nextOption = frame.input.options[frame.stepIndex];
        let maxOverride: number | undefined;
        if (nextOption === frame.input.inputs.project && frame.chosenCorp !== undefined) {
          if (frame.chosenCorp.name !== CardNames.BEGINNER_CORPORATION) {
            const cardCost = frame.chosenCorp.cardCost ?? this.player.cardCost;
            maxOverride = Math.floor(frame.chosenCorp.startingMegaCredits / cardCost);
          }
        }
        frame.child = buildFrame(nextOption, this.player, maxOverride);
        return undefined;
      }

      case 'single_card': {
        const card = frame.orderedCards[candidateIndex];
        return {type: 'card', cards: [card.name]};
      }

      case 'multi_card': {
        const pickedSet = new Set(frame.picked);
        const remainingCards: Array<ICard> = [];
        if (frame.picked.length < frame.max) {
          for (const c of frame.orderedCards) {
            if (!pickedSet.has(c.name)) {
              remainingCards.push(c);
            }
          }
        }
        if (candidateIndex < remainingCards.length) {
          frame.picked.push(remainingCards[candidateIndex].name);
          return undefined;
        }
        // 'done' candidate selected
        return {type: 'card', cards: frame.picked.slice().sort(compareStrings)};
      }

      case 'project_card': {
        if (frame.chosen !== undefined) {
          const payment = frame.chosen.payments[candidateIndex].payment;
          return {type: 'projectCard', card: frame.chosen.card.name, payment};
        }
        const chosen = frame.candidates[candidateIndex];
        if (chosen.payments.length === 1) {
          return {type: 'projectCard', card: chosen.card.name, payment: chosen.payments[0].payment};
        }
        frame.chosen = chosen;
        return undefined;
      }

      case 'payment':
        return {type: 'payment', payment: frame.payments[candidateIndex].payment};

      case 'space':
        return {type: 'space', spaceId: frame.spaces[candidateIndex].id};

      case 'player':
        return {type: 'player', player: frame.players[candidateIndex].color};

      case 'amount':
        return {type: 'amount', amount: frame.input.min + candidateIndex};

      case 'resource':
        return {type: 'resource', resource: frame.input.include[candidateIndex]};

      case 'resources': {
        const key = UnitsUtil.keys[candidateIndex];
        frame.units[key] += 1;
        frame.remaining -= 1;
        if (frame.remaining <= 0) {
          return {type: 'resources', units: frame.units};
        }
        return undefined;
      }

      case 'productionToLose': {
        const current = frame.input.player.production.asUnits();
        const reducible = UnitsUtil.keys.filter(
          (k) => current[k] - frame.lost[k] > (frame.minimums[k] ?? 0),
        );
        const key = reducible[candidateIndex];
        frame.lost[key] += 1;
        frame.remaining -= 1;
        if (frame.remaining <= 0) {
          return {type: 'productionToLose', units: frame.lost};
        }
        return undefined;
      }

      case 'option':
        return {type: 'option'};
    }
  }
}
