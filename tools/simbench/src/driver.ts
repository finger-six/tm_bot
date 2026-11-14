import {
  AndOptions,
  BoardNameValue,
  CardName,
  Color,
  DEFAULT_PAYMENT_VALUES,
  ICard,
  ICorporationCard,
  IGame,
  IPlayer,
  IProjectCard,
  IStandardProjectCard,
  InputResponse,
  OrOptions,
  Payment,
  PaymentOptions,
  PaymentUtil,
  Phase,
  PlayerId,
  PlayerInput,
  SPENDABLE_RESOURCES,
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
  SpendableResource,
  Units,
  UnitsUtil,
  constants,
} from './engine';
import {createExplicitTwoPlayerGame} from './config';
import {SeededPrng} from './prng';

export interface RecordedAction {
  step: number;
  playerColor: Color;
  playerId: PlayerId;
  phase: string;
  generation: number;
  waitingRequiredColors: Array<Color>;
  inputType: string;
  response: InputResponse;
}

export interface GameRunResult {
  boardName: BoardNameValue;
  seed: number;
  driverSeed: number;
  completed: boolean;
  phase: string;
  generations: number;
  decisions: number;
  invalidActions: number;
  deadlocks: number;
  hitStepCap: boolean;
  unexpectedInputTypes: Array<string>;
  scores: Record<string, number>;
  tie: boolean;
  wallTimeMs: number;
  error?: string;
  trace?: Array<RecordedAction>;
}

const DEFAULT_MAX_STEPS = 20_000;
const MAX_RETRIES_PER_DECISION = 16;

function isStandardProjectCard(card: ICard): card is IStandardProjectCard {
  return typeof (card as IStandardProjectCard).canAct === 'function' &&
    typeof (card as IStandardProjectCard).canPayWith === 'function';
}

function getAvailableResourceAmount(
  resource: SpendableResource,
  player: IPlayer,
  reserveUnits: Readonly<Units> | undefined,
  cardName?: string,
): number {
  switch (resource) {
  case 'megacredits':
    return Math.max(0, player.megaCredits - (reserveUnits?.megacredits ?? 0));
  case 'steel':
    return Math.max(0, player.steel - (reserveUnits?.steel ?? 0));
  case 'titanium':
    return Math.max(0, player.titanium - (reserveUnits?.titanium ?? 0));
  case 'plants':
    return Math.max(0, player.plants - (reserveUnits?.plants ?? 0));
  case 'heat':
    return Math.max(0, player.availableHeat() - (reserveUnits?.heat ?? 0));
  case 'microbes': {
    const avail = player.getSpendable('microbes');
    if (cardName === CardName.SOIL_ENRICHMENT && avail > 0) {
      const cardsWithMicrobe = player.getCardsWithResources('microbe' as never);
      if (cardsWithMicrobe.length === 1) {
        return Math.max(0, avail - 1);
      }
    }
    return avail;
  }
  case 'floaters': {
    const avail = player.getSpendable('floaters');
    if (cardName === CardName.STRATOSPHERIC_BIRDS && avail > 0) {
      const cardsWithFloater = player.getCardsWithResources('floater' as never);
      if (cardsWithFloater.length === 1) {
        return Math.max(0, avail - 1);
      }
    }
    return avail;
  }
  default:
    return player.getSpendable(resource);
  }
}

function getUnitMultiplier(
  resource: SpendableResource,
  player: IPlayer,
  options: Partial<PaymentOptions>,
): number {
  switch (resource) {
  case 'megacredits':
    return 1;
  case 'steel':
    return options.steel ? player.getSteelValue() : 0;
  case 'titanium':
    if (options.titanium) {
      return player.getTitaniumValue();
    }
    if (options.lunaTradeFederationTitanium || player.canUseTitaniumAsMegacredits) {
      return Math.max(0, player.getTitaniumValue() - 1);
    }
    return 0;
  case 'heat':
    return (options.heat || player.canUseHeatAsMegaCredits) ? 1 : 0;
  case 'plants':
    return options.plants ? DEFAULT_PAYMENT_VALUES.plants : 0;
  default:
    return options[resource] ? DEFAULT_PAYMENT_VALUES[resource] : 0;
  }
}

/**
 * Constructs a valid random Payment covering at least `requiredCost` M€ value
 * using the player's available non-reserved resources, or returns undefined if unaffordable.
 */
function buildRandomPayment(
  requiredCost: number,
  options: Partial<PaymentOptions>,
  reserveUnits: Readonly<Units> | undefined,
  player: IPlayer,
  prng: SeededPrng,
  cardName?: string,
): Payment | undefined {
  const payment: Record<SpendableResource, number> = {...PaymentUtil.EMPTY};
  if (requiredCost <= 0) {
    return payment;
  }

  const nonMcEntries: Array<{resource: SpendableResource; avail: number; value: number}> = [];
  for (const resource of SPENDABLE_RESOURCES) {
    if (resource === 'megacredits') {
      continue;
    }
    const value = getUnitMultiplier(resource, player, options);
    if (value <= 0) {
      continue;
    }
    const avail = getAvailableResourceAmount(resource, player, reserveUnits, cardName);
    if (avail > 0) {
      nonMcEntries.push({resource, avail, value});
    }
  }

  const mcAvail = getAvailableResourceAmount('megacredits', player, reserveUnits, cardName);
  const maxPossible = mcAvail + nonMcEntries.reduce((acc, e) => acc + e.avail * e.value, 0);
  if (maxPossible < requiredCost) {
    return undefined;
  }

  const ordered = prng.shuffled(nonMcEntries);
  // Suffix sums of max value remaining after index i (including megacredits).
  const suffixMax = new Array<number>(ordered.length + 1).fill(mcAvail);
  for (let i = ordered.length - 1; i >= 0; i--) {
    suffixMax[i] = suffixMax[i + 1] + ordered[i].avail * ordered[i].value;
  }

  let remaining = requiredCost;
  for (let i = 0; i < ordered.length; i++) {
    if (remaining <= 0) {
      break;
    }
    const entry = ordered[i];
    const laterMax = suffixMax[i + 1];
    const minNeeded = Math.ceil(Math.max(0, remaining - laterMax) / entry.value);
    const maxUseful = Math.min(entry.avail, Math.ceil(remaining / entry.value));
    const low = Math.min(minNeeded, maxUseful);
    const high = Math.max(low, maxUseful);
    const chosen = prng.nextIntRange(low, high);
    payment[entry.resource] = chosen;
    remaining = Math.max(0, remaining - chosen * entry.value);
  }

  payment.megacredits = Math.min(mcAvail, remaining);
  return payment;
}

function getPlayableProjectCandidates(
  input: SelectProjectCardToPlay | SelectStandardProjectToPlay,
  player: IPlayer,
  prng: SeededPrng,
): Array<{card: IProjectCard | IStandardProjectCard; payment: Payment}> {
  const candidates: Array<{card: IProjectCard | IStandardProjectCard; payment: Payment}> = [];
  const enabled = input.enabled;

  for (let i = 0; i < input.cards.length; i++) {
    if (enabled?.[i] === false) {
      continue;
    }
    const card = input.cards[i];
    const details = input.extras.get(card.name);

    if (isStandardProjectCard(card)) {
      if (details?.overriddenCost === undefined && !card.canAct(player)) {
        continue;
      }
      const canPayWith = card.canPayWith(player);
      const paymentOptions: Partial<PaymentOptions> = {
        heat: player.canUseHeatAsMegaCredits,
        steel: canPayWith.steel,
        titanium: canPayWith.titanium,
        lunaTradeFederationTitanium: player.canUseTitaniumAsMegacredits,
        seeds: canPayWith.seeds,
        auroraiData: player.tableau.has(CardName.AURORAI as never),
        spireScience: player.tableau.has(CardName.SPIRE as never),
        kuiperAsteroids: canPayWith.kuiperAsteroids ? player.tableau.has(CardName.KUIPER_COOPERATIVE as never) : false,
      };
      const requiredCost = details?.overriddenCost ?? card.getAdjustedCost(player);
      const payment = buildRandomPayment(requiredCost, paymentOptions, details?.reserveUnits, player, prng, card.name);
      if (payment !== undefined) {
        candidates.push({card, payment});
      }
    } else {
      const afford = player.affordOptionsForCard(card);
      const payment = buildRandomPayment(
        afford.cost,
        afford,
        afford.reserveUnits ?? details?.reserveUnits,
        player,
        prng,
        card.name,
      );
      if (payment !== undefined) {
        candidates.push({card, payment});
      }
    }
  }

  return candidates;
}

function hasAnyLegalChoice(input: PlayerInput, player: IPlayer): boolean {
  switch (input.type) {
  case 'option':
  case 'amount':
  case 'payment':
  case 'productionToLose':
  case 'resource':
  case 'resources':
  case 'initialCards':
    return true;
  case 'space':
    return (input as SelectSpace).spaces.length > 0;
  case 'player':
    return (input as SelectPlayer).players.length > 0;
  case 'card': {
    const sc = input as SelectCard<ICard>;
    const enabledCount = sc.cards.filter((_, idx) => sc.config.enabled?.[idx] !== false).length;
    return enabledCount >= sc.config.min;
  }
  case 'projectCard': {
    const pc = input as SelectProjectCardToPlay | SelectStandardProjectToPlay;
    // Quick check without mutating PRNG using a dummy PRNG
    return getPlayableProjectCandidates(pc, player, new SeededPrng(1)).length > 0;
  }
  case 'or': {
    const orInput = input as OrOptions;
    return orInput.options.some((opt) => hasAnyLegalChoice(opt, player));
  }
  case 'and': {
    const andInput = input as AndOptions;
    return andInput.options.every((opt) => hasAnyLegalChoice(opt, player));
  }
  default:
    return true;
  }
}

function buildRandomResponse(
  input: PlayerInput,
  player: IPlayer,
  prng: SeededPrng,
  unexpectedInputTypes: Set<string>,
  failedOrIndices?: Set<number>,
): InputResponse {
  switch (input.type) {
  case 'option':
    return {type: 'option'};

  case 'or': {
    const orInput = input as OrOptions;
    let legalIndices: Array<number> = [];
    for (let i = 0; i < orInput.options.length; i++) {
      if (failedOrIndices?.has(i)) {
        continue;
      }
      if (hasAnyLegalChoice(orInput.options[i], player)) {
        legalIndices.push(i);
      }
    }
    if (legalIndices.length === 0) {
      // Fallback: try any non-failed index so engine validation reports any issue
      for (let i = 0; i < orInput.options.length; i++) {
        if (!failedOrIndices?.has(i)) {
          legalIndices.push(i);
        }
      }
    }
    if (legalIndices.length === 0) {
      throw new Error('OrOptions has no remaining untried options');
    }
    const idx = prng.pick(legalIndices);
    const response = buildRandomResponse(orInput.options[idx], player, prng, unexpectedInputTypes);
    return {type: 'or', index: idx, response};
  }

  case 'and': {
    const andInput = input as AndOptions;
    return {
      type: 'and',
      responses: andInput.options.map((opt) => buildRandomResponse(opt, player, prng, unexpectedInputTypes)),
    };
  }

  case 'initialCards': {
    const init = input as SelectInitialCards;
    let chosenCorp: ICorporationCard | undefined;
    const responses: Array<InputResponse> = [];

    for (const opt of init.options) {
      if (opt === init.inputs.corp) {
        const corpSelect = opt as SelectCard<ICorporationCard>;
        const enabled = corpSelect.cards.filter((_, idx) => corpSelect.config.enabled?.[idx] !== false);
        chosenCorp = prng.pick(enabled);
        responses.push({type: 'card', cards: [chosenCorp.name]});
      } else if (opt === init.inputs.project) {
        const projSelect = opt as SelectCard<IProjectCard>;
        const enabled = projSelect.cards.filter((_, idx) => projSelect.config.enabled?.[idx] !== false);
        const cardCost = chosenCorp?.cardCost ?? player.cardCost;
        const maxAffordable = chosenCorp ?
          Math.min(projSelect.config.max, enabled.length, Math.floor(chosenCorp.startingMegaCredits / cardCost)) :
          Math.min(projSelect.config.max, enabled.length);
        const minCards = Math.min(projSelect.config.min, maxAffordable);
        const count = prng.nextIntRange(minCards, maxAffordable);
        const picked = prng.sample(enabled, count);
        responses.push({type: 'card', cards: picked.map((c) => c.name)});
      } else {
        responses.push(buildRandomResponse(opt, player, prng, unexpectedInputTypes));
      }
    }
    return {type: 'initialCards', responses};
  }

  case 'card': {
    const sc = input as SelectCard<ICard>;
    const enabled = sc.cards.filter((_, idx) => sc.config.enabled?.[idx] !== false);
    const maxCount = Math.min(sc.config.max, enabled.length);
    const minCount = Math.min(sc.config.min, maxCount);
    const count = prng.nextIntRange(minCount, maxCount);
    const picked = prng.sample(enabled, count);
    return {type: 'card', cards: picked.map((c) => c.name)};
  }

  case 'projectCard': {
    const pc = input as SelectProjectCardToPlay | SelectStandardProjectToPlay;
    const candidates = getPlayableProjectCandidates(pc, player, prng);
    if (candidates.length === 0) {
      throw new Error('No playable/affordable project cards in SelectCardToPlay');
    }
    const picked = prng.pick(candidates);
    return {type: 'projectCard', card: picked.card.name, payment: picked.payment};
  }

  case 'payment': {
    const sp = input as SelectPayment;
    const paymentOptions: Partial<PaymentOptions> = {
      heat: player.canUseHeatAsMegaCredits,
      lunaTradeFederationTitanium: player.canUseTitaniumAsMegacredits,
      ...sp.paymentOptions,
    };
    const payment = buildRandomPayment(sp.amount, paymentOptions, sp.reserveUnits, player, prng);
    if (payment === undefined) {
      throw new Error(`Cannot construct SelectPayment of ${sp.amount}`);
    }
    return {type: 'payment', payment};
  }

  case 'space': {
    const ss = input as SelectSpace;
    const space = prng.pick(ss.spaces);
    return {type: 'space', spaceId: space.id};
  }

  case 'amount': {
    const sa = input as SelectAmount;
    const amount = prng.nextIntRange(sa.min, sa.max);
    return {type: 'amount', amount};
  }

  case 'player': {
    const sp = input as SelectPlayer;
    const target = prng.pick(sp.players);
    return {type: 'player', player: target.color};
  }

  case 'productionToLose': {
    const spl = input as SelectProductionToLose;
    const minimums = constants.PRODUCTION_MINIMUMS ?? {
      megacredits: -5,
      steel: 0,
      titanium: 0,
      plants: 0,
      energy: 0,
      heat: 0,
    };
    const current = spl.player.production.asUnits();
    const lost: Units = {...UnitsUtil.EMPTY};
    const keys: Array<keyof Units> = ['megacredits', 'steel', 'titanium', 'plants', 'energy', 'heat'];
    let remaining = spl.unitsToLose;
    while (remaining > 0) {
      const reducible = keys.filter((k) => current[k] - lost[k] > (minimums[k] ?? 0));
      if (reducible.length === 0) {
        throw new Error(`Cannot lose ${spl.unitsToLose} production`);
      }
      const key = prng.pick(reducible);
      lost[key] += 1;
      remaining -= 1;
    }
    return {type: 'productionToLose', units: lost};
  }

  case 'resource': {
    const sr = input as SelectResource;
    const resource = prng.pick(sr.include);
    return {type: 'resource', resource};
  }

  case 'resources': {
    const srs = input as SelectResources;
    const units: Units = {...UnitsUtil.EMPTY};
    for (let i = 0; i < srs.count; i++) {
      const k = prng.pick(UnitsUtil.keys);
      units[k] += 1;
    }
    return {type: 'resources', units};
  }

  default: {
    unexpectedInputTypes.add(input.type);
    throw new Error(`Unexpected input type: ${input.type}`);
  }
  }
}

export function getRequiredWaitingPlayers(game: IGame): Array<IPlayer> {
  return game.players.filter((p) => {
    const waiting = p.getWaitingFor();
    return waiting !== undefined && !waiting.optional;
  });
}

export function collectPlayerScores(game: IGame): {scores: Record<string, number>; tie: boolean} {
  const scores: Record<string, number> = {};
  const totals: Array<number> = [];
  for (const player of game.players) {
    const vp = player.getVictoryPoints().total;
    scores[player.color] = vp;
    totals.push(vp);
  }
  const maxVp = Math.max(...totals);
  const winners = totals.filter((v) => v === maxVp);
  return {scores, tie: winners.length > 1};
}

export interface RunRandomGameOptions {
  boardName: BoardNameValue;
  seed: number;
  driverSeed?: number;
  maxSteps?: number;
  recordTrace?: boolean;
}

export function runRandomGame(opts: RunRandomGameOptions): GameRunResult {
  const startNs = process.hrtime.bigint();
  const driverSeed = opts.driverSeed ?? (opts.seed ^ 0x9e3779b9) >>> 0;
  const maxSteps = opts.maxSteps ?? DEFAULT_MAX_STEPS;
  const prng = new SeededPrng(driverSeed);

  const {game} = createExplicitTwoPlayerGame({
    boardName: opts.boardName,
    seed: opts.seed,
  });

  let decisions = 0;
  let invalidActions = 0;
  let deadlocks = 0;
  let hitStepCap = false;
  const unexpectedInputTypes = new Set<string>();
  const trace: Array<RecordedAction> | undefined = opts.recordTrace ? [] : undefined;
  let fatalError: string | undefined;

  while (game.phase !== Phase.END) {
    if (decisions >= maxSteps) {
      hitStepCap = true;
      break;
    }

    const waitingPlayers = getRequiredWaitingPlayers(game);
    if (waitingPlayers.length === 0) {
      deadlocks += 1;
      fatalError = `Deadlock at phase=${game.phase}, generation=${game.generation}, decisions=${decisions}`;
      break;
    }

    const player = prng.pick(waitingPlayers);
    const waitingFor = player.getWaitingFor();
    if (!waitingFor) {
      deadlocks += 1;
      break;
    }

    const failedOrIndices = new Set<number>();
    let succeeded = false;
    let lastErr: unknown;

    for (let attempt = 0; attempt < MAX_RETRIES_PER_DECISION; attempt++) {
      let response: InputResponse;
      try {
        response = buildRandomResponse(waitingFor, player, prng, unexpectedInputTypes, failedOrIndices);
      } catch (err) {
        invalidActions += 1;
        lastErr = err;
        break;
      }

      const phaseBefore = game.phase;
      const genBefore = game.generation;
      try {
        player.process(response);
        decisions += 1;
        succeeded = true;
        if (trace) {
          trace.push({
            step: decisions,
            playerColor: player.color,
            playerId: player.id,
            phase: phaseBefore,
            generation: genBefore,
            waitingRequiredColors: waitingPlayers.map((p) => p.color),
            inputType: waitingFor.type,
            response,
          });
        }
        break;
      } catch (err) {
        invalidActions += 1;
        lastErr = err;
        if (response.type === 'or') {
          failedOrIndices.add(response.index);
        }
      }
    }

    if (!succeeded) {
      fatalError = `Failed after retries at phase=${game.phase}, gen=${game.generation}, player=${player.color}, input=${waitingFor.type}: ${lastErr instanceof Error ? lastErr.message : String(lastErr)}`;
      break;
    }
  }

  const endNs = process.hrtime.bigint();
  const wallTimeMs = Number(endNs - startNs) / 1e6;
  const completed = game.phase === Phase.END && fatalError === undefined;
  const {scores, tie} = completed ? collectPlayerScores(game) : {scores: {}, tie: false};

  return {
    boardName: opts.boardName,
    seed: opts.seed,
    driverSeed,
    completed,
    phase: game.phase,
    generations: game.generation,
    decisions,
    invalidActions,
    deadlocks,
    hitStepCap,
    unexpectedInputTypes: Array.from(unexpectedInputTypes),
    scores,
    tie,
    wallTimeMs,
    ...(fatalError ? {error: fatalError} : {}),
    ...(trace ? {trace} : {}),
  };
}
