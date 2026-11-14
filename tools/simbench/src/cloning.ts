import fs from 'node:fs';
import path from 'node:path';
import {TARGET_MAPS, createExplicitTwoPlayerGame} from './config';
import {
  DEFAULT_MAX_STEPS,
  RecordedAction,
  buildRandomResponse,
  collectPlayerScores,
  getRequiredWaitingPlayers,
  pickNextWaitingPlayer,
  stepDecision,
} from './driver';
import {
  BoardNameValue,
  Game,
  GameId,
  IGame,
  IPlayer,
  InputResponse,
  PaymentUtil,
  Phase,
  PlayerInput,
  SelectAmount,
  SelectPlayer,
  Server,
  SpectatorId,
  UnitsUtil,
} from './engine';
import {findFirstDiff, normalizeSerializedGame} from './normalize';
import {SeededPrng} from './prng';

export type DecisionPointCategory =
  | 'initial_selection'
  | 'drafting_before_pick'
  | 'drafting_after_one_pick'
  | 'research_buy'
  | 'prelude_root'
  | 'prelude_effect'
  | 'action_root'
  | 'nested_payment'
  | 'nested_space'
  | 'nested_player_attack'
  | 'nested_deferred_action'
  | 'final_greenery';

const TARGET_CATEGORIES: ReadonlyArray<DecisionPointCategory> = [
  'initial_selection',
  'drafting_before_pick',
  'drafting_after_one_pick',
  'research_buy',
  'prelude_root',
  'prelude_effect',
  'action_root',
  'nested_payment',
  'nested_space',
  'nested_player_attack',
  'nested_deferred_action',
  'final_greenery',
];

interface PromptSignatureEntry {
  color: string;
  optional: boolean;
  model: Record<string, unknown>;
}

interface InvalidResponseAttempt {
  label: string;
  response: InputResponse;
}

interface AtomicityResult {
  passed: boolean;
  threwError: boolean;
  errorMessage?: string;
  stateUnchanged: boolean;
  stateDiff: string | null;
  promptUnchanged: boolean;
  promptDiff: string | null;
  testedLabel: string;
}

interface DirectCloneTestResult {
  category: DecisionPointCategory;
  boardName: BoardNameValue;
  seed: number;
  stepIndex: number;
  generation: number;
  phase: string;
  activeInputType: string;
  activeInputTitle: string;
  // (a) prompt equivalence
  promptEquivalent: boolean;
  promptDiff: string | null;
  // (b) continuation equivalence
  continuationPassed: boolean;
  continuationSameNextActionAccepted: boolean;
  continuationScoresMatch: boolean;
  continuationOrigScores: Record<string, number>;
  continuationCloneScores: Record<string, number>;
  continuationFinalStateDiff: string | null;
  continuationError?: string;
  // (c) isolation
  isolationPassed: boolean;
  isolationCloneStepsAdvanced: number;
  isolationOrigStateDiff: string | null;
  isolationOrigPromptDiff: string | null;
  // (d) atomicity on original
  atomicity: AtomicityResult;
  // overall direct clone verdict at this point (a && b && c && d)
  directClonePassed: boolean;
}

interface FallbackCloneTestResult {
  category: DecisionPointCategory;
  boardName: BoardNameValue;
  seed: number;
  stepIndex: number;
  checkpointStepIndex: number;
  suffixLength: number;
  fallbackCloneMs: number;
  promptEquivalent: boolean;
  promptDiff: string | null;
  continuationPassed: boolean;
  continuationScoresMatch: boolean;
  continuationFinalStateDiff: string | null;
  continuationError?: string;
  isolationPassed: boolean;
  isolationCloneStepsAdvanced: number;
  isolationOrigStateDiff: string | null;
  fallbackPassed: boolean;
}

interface SuffixSweepEntry {
  suffixLength: number;
  medianMs: number;
  p95Ms: number;
  maxMs: number;
  samples: number;
}

function getTitleText(title: unknown): string {
  if (typeof title === 'string') {
    return title;
  }
  if (title && typeof title === 'object' && 'message' in title) {
    return String((title as {message: unknown}).message);
  }
  return String(title ?? '');
}

function extractPromptSignature(game: IGame): Array<PromptSignatureEntry> {
  const entries: Array<PromptSignatureEntry> = [];
  for (const player of game.players) {
    const wf = player.getWaitingFor();
    if (wf !== undefined) {
      const model = Server.getWaitingFor(player, wf) ?? {};
      entries.push({
        color: player.color,
        optional: Boolean(wf.optional),
        model,
      });
    }
  }
  return entries;
}

function isFinalGreeneryState(game: IGame): boolean {
  return game.phase === Phase.PRODUCTION && game.gameIsOver() && game.isDoneWithFinalProduction();
}

function isActionRootPrompt(game: IGame, wf: PlayerInput): boolean {
  if (game.phase !== Phase.ACTION || wf.type !== 'or') {
    return false;
  }
  const title = getTitleText(wf.title);
  return (
    title === 'Take your first action' ||
    title === 'Take your next action' ||
    title === 'Select one option'
  );
}

function isPreludeRootPrompt(game: IGame, wf: PlayerInput): boolean {
  return (
    game.phase === Phase.PRELUDES &&
    wf.type === 'card' &&
    getTitleText(wf.title) === 'Select prelude card to play'
  );
}

/**
 * Root boundaries where Game.deserialize(game.serialize()) is expected to restore cleanly.
 */
function isSafeDirectRestoreBoundary(game: IGame, waitingRequired: ReadonlyArray<IPlayer>): boolean {
  if (waitingRequired.length === 0) {
    return false;
  }
  if (game.deferredActions.length > 0) {
    return false;
  }
  const wf = waitingRequired[0].getWaitingFor();
  if (!wf) {
    return false;
  }
  if (game.generation === 1 && game.phase === Phase.RESEARCH && wf.type === 'initialCards') {
    return true;
  }
  if (game.phase === Phase.DRAFTING) {
    return true;
  }
  if (isPreludeRootPrompt(game, wf)) {
    return true;
  }
  if (isActionRootPrompt(game, wf)) {
    return true;
  }
  if (isFinalGreeneryState(game) && wf.type === 'or') {
    return true;
  }
  return false;
}

function classifyDecisionPoint(
  game: IGame,
  waitingRequired: ReadonlyArray<IPlayer>,
  wf: PlayerInput,
): Array<DecisionPointCategory> {
  const cats: Array<DecisionPointCategory> = [];
  const title = getTitleText(wf.title);

  if (game.generation === 1 && game.phase === Phase.RESEARCH && wf.type === 'initialCards') {
    cats.push('initial_selection');
  }
  if (game.phase === Phase.DRAFTING) {
    if (waitingRequired.length === 2) {
      cats.push('drafting_before_pick');
    } else if (waitingRequired.length === 1) {
      cats.push('drafting_after_one_pick');
    }
  }
  if (game.generation >= 2 && game.phase === Phase.RESEARCH && wf.type === 'card') {
    cats.push('research_buy');
  }
  if (isPreludeRootPrompt(game, wf)) {
    cats.push('prelude_root');
  } else if (game.phase === Phase.PRELUDES) {
    cats.push('prelude_effect');
  }
  if (isActionRootPrompt(game, wf)) {
    cats.push('action_root');
  }
  if (wf.type === 'payment') {
    cats.push('nested_payment');
  }
  if (wf.type === 'space' && !isFinalGreeneryState(game)) {
    cats.push('nested_space');
  }
  if (wf.type === 'player') {
    cats.push('nested_player_attack');
  }
  if (
    game.phase === Phase.ACTION &&
    !isActionRootPrompt(game, wf) &&
    wf.type !== 'payment' &&
    wf.type !== 'space' &&
    wf.type !== 'player'
  ) {
    cats.push('nested_deferred_action');
  } else if (
    game.phase !== Phase.ACTION &&
    game.phase !== Phase.PRELUDES &&
    game.phase !== Phase.DRAFTING &&
    game.phase !== Phase.RESEARCH &&
    !isFinalGreeneryState(game) &&
    wf.type !== 'payment' &&
    wf.type !== 'space' &&
    wf.type !== 'player'
  ) {
    cats.push('nested_deferred_action');
  }
  if (isFinalGreeneryState(game) && (wf.type === 'or' || wf.type === 'space' || title.includes('final greenery'))) {
    cats.push('final_greenery');
  }
  return cats;
}

function buildInvalidResponse(input: PlayerInput): InvalidResponseAttempt {
  switch (input.type) {
  case 'initialCards':
    return {
      label: 'initialCards with nonexistent corporation card',
      response: {
        type: 'initialCards',
        responses: [{type: 'card', cards: ['__INVALID_CORP_CARD__']}],
      },
    };
  case 'card':
    return {
      label: 'card not offered',
      response: {type: 'card', cards: ['__INVALID_CARD_NAME__']},
    };
  case 'projectCard':
    return {
      label: 'projectCard not offered',
      response: {
        type: 'projectCard',
        card: '__INVALID_PROJECT_CARD__',
        payment: {...PaymentUtil.EMPTY},
      },
    };
  case 'space':
    return {
      label: 'spaceId not in available spaces',
      response: {type: 'space', spaceId: '__INVALID_SPACE_ID__'},
    };
  case 'player': {
    const sp = input as SelectPlayer;
    const offered = new Set(sp.players.map((p) => p.color));
    const invalidColor = (['green', 'yellow', 'purple', 'orange', 'red', 'blue'] as const).find(
      (c) => !offered.has(c),
    ) ?? 'neutral';
    return {
      label: `player color "${invalidColor}" not offered`,
      response: {type: 'player', player: invalidColor},
    };
  }
  case 'amount': {
    const sa = input as SelectAmount;
    return {
      label: `amount=${sa.max + 1000} out of range [${sa.min}, ${sa.max}]`,
      response: {type: 'amount', amount: sa.max + 1000},
    };
  }
  case 'payment':
    return {
      label: 'payment with negative megacredits (-1)',
      response: {type: 'payment', payment: {...PaymentUtil.EMPTY, megacredits: -1}},
    };
  case 'productionToLose':
    return {
      label: 'productionToLose exceeding unitsToLose (999 megacredits)',
      response: {
        type: 'productionToLose',
        units: {...UnitsUtil.EMPTY, megacredits: 999},
      },
    };
  case 'resource':
    return {
      label: 'resource name not in include list',
      response: {type: 'resource', resource: '__invalid__' as never},
    };
  case 'resources':
    return {
      label: 'resources count exceeding required (999)',
      response: {
        type: 'resources',
        units: {...UnitsUtil.EMPTY, megacredits: 999},
      },
    };
  case 'or':
    return {
      label: 'or index=999 out of bounds',
      response: {type: 'or', index: 999, response: {type: 'option'}},
    };
  case 'and':
    return {
      label: 'and with empty responses',
      response: {type: 'and', responses: []},
    };
  default:
    return {
      label: 'mismatched response type for input',
      response: {type: 'amount', amount: -999},
    };
  }
}

interface CapturedLocation {
  category: DecisionPointCategory;
  boardName: BoardNameValue;
  seed: number;
  driverSeed: number;
  stepIndex: number;
}

function scanMapForCategories(boardName: BoardNameValue, maxSeeds = 250): Map<DecisionPointCategory, CapturedLocation> {
  const found = new Map<DecisionPointCategory, CapturedLocation>();
  const baseSeed = 3001;

  for (let offset = 0; offset < maxSeeds && found.size < TARGET_CATEGORIES.length; offset++) {
    const seed = baseSeed + offset;
    const driverSeed = (seed ^ 0x9e3779b9) >>> 0;
    const prng = new SeededPrng(driverSeed);
    const {game} = createExplicitTwoPlayerGame({boardName, seed});
    const unexpectedInputTypes = new Set<string>();

    let stepIndex = 0;
    while (game.phase !== Phase.END && stepIndex < DEFAULT_MAX_STEPS) {
      const waitingPlayers = getRequiredWaitingPlayers(game);
      if (waitingPlayers.length === 0) {
        break;
      }
      const player = pickNextWaitingPlayer(waitingPlayers, prng, 'canonical');
      const wf = player.getWaitingFor();
      if (!wf) {
        break;
      }

      const cats = classifyDecisionPoint(game, waitingPlayers, wf);
      for (const cat of cats) {
        if (!found.has(cat)) {
          found.set(cat, {
            category: cat,
            boardName,
            seed,
            driverSeed,
            stepIndex,
          });
        }
      }
      if (found.size === TARGET_CATEGORIES.length) {
        break;
      }

      const res = stepDecision(player, wf, prng, unexpectedInputTypes);
      if (!res.succeeded) {
        break;
      }
      stepIndex += 1;
    }
  }

  return found;
}

interface ReplayedAtPoint {
  game: IGame;
  player: IPlayer;
  waitingFor: PlayerInput;
  prngAtPoint: SeededPrng;
  checkpointStepIndex: number;
  checkpointSnapshotJson: string;
  suffixActions: Array<RecordedAction>;
}

function replayToPoint(
  loc: CapturedLocation,
  gameTag: string,
): ReplayedAtPoint {
  const prng = new SeededPrng(loc.driverSeed);
  const {game} = createExplicitTwoPlayerGame({
    boardName: loc.boardName,
    seed: loc.seed,
    gameId: `g-${gameTag}-${loc.seed}` as GameId,
    spectatorId: `s-${gameTag}-${loc.seed}` as SpectatorId,
  });
  const unexpectedInputTypes = new Set<string>();

  let checkpointStepIndex = 0;
  let checkpointSnapshotJson = JSON.stringify(game.serialize());
  let suffixActions: Array<RecordedAction> = [];

  for (let step = 0; step < loc.stepIndex; step++) {
    const waitingPlayers = getRequiredWaitingPlayers(game);
    if (waitingPlayers.length === 0) {
      throw new Error(`Unexpected deadlock replaying ${loc.category} at step=${step}`);
    }
    if (isSafeDirectRestoreBoundary(game, waitingPlayers)) {
      checkpointStepIndex = step;
      checkpointSnapshotJson = JSON.stringify(game.serialize());
      suffixActions = [];
    }

    const player = pickNextWaitingPlayer(waitingPlayers, prng, 'canonical');
    const wf = player.getWaitingFor();
    if (!wf) {
      throw new Error(`Missing waitingFor replaying ${loc.category} at step=${step}`);
    }

    const phaseBefore = game.phase;
    const genBefore = game.generation;
    const res = stepDecision(player, wf, prng, unexpectedInputTypes);
    if (!res.succeeded || !res.response) {
      throw new Error(`Step failed replaying ${loc.category} at step=${step}: ${res.error}`);
    }
    suffixActions.push({
      step: step + 1,
      playerColor: player.color,
      playerId: player.id,
      phase: phaseBefore,
      generation: genBefore,
      waitingRequiredColors: waitingPlayers.map((p) => p.color),
      inputType: wf.type,
      response: res.response,
    });
  }

  const waitingPlayers = getRequiredWaitingPlayers(game);
  if (waitingPlayers.length === 0) {
    throw new Error(`No waiting player at target ${loc.category} step=${loc.stepIndex}`);
  }
  if (isSafeDirectRestoreBoundary(game, waitingPlayers)) {
    checkpointStepIndex = loc.stepIndex;
    checkpointSnapshotJson = JSON.stringify(game.serialize());
    suffixActions = [];
  }

  const player = pickNextWaitingPlayer(waitingPlayers, prng, 'canonical');
  const waitingFor = player.getWaitingFor();
  if (!waitingFor) {
    throw new Error(`Missing waitingFor at target ${loc.category}`);
  }

  return {
    game,
    player,
    waitingFor,
    prngAtPoint: prng,
    checkpointStepIndex,
    checkpointSnapshotJson,
    suffixActions,
  };
}

function advanceDecisions(
  game: IGame,
  prng: SeededPrng,
  limit: number,
  recordActions = false,
): {
  stepsAdvanced: number;
  completed: boolean;
  actions: Array<{playerColor: string; response: InputResponse}>;
  error?: string;
} {
  const unexpectedInputTypes = new Set<string>();
  const actions: Array<{playerColor: string; response: InputResponse}> = [];
  let stepsAdvanced = 0;

  while (game.phase !== Phase.END && stepsAdvanced < limit) {
    const waitingPlayers = getRequiredWaitingPlayers(game);
    if (waitingPlayers.length === 0) {
      return {
        stepsAdvanced,
        completed: false,
        actions,
        error: `Deadlock at phase=${game.phase}, gen=${game.generation}`,
      };
    }
    const player = pickNextWaitingPlayer(waitingPlayers, prng, 'canonical');
    const wf = player.getWaitingFor();
    if (!wf) {
      return {
        stepsAdvanced,
        completed: false,
        actions,
        error: `Missing waitingFor at phase=${game.phase}`,
      };
    }
    const res = stepDecision(player, wf, prng, unexpectedInputTypes);
    if (!res.succeeded || !res.response) {
      return {
        stepsAdvanced,
        completed: false,
        actions,
        error: `Step failed at phase=${game.phase}, gen=${game.generation}, input=${wf.type}: ${res.error}`,
      };
    }
    if (recordActions) {
      actions.push({playerColor: player.color, response: res.response});
    }
    stepsAdvanced += 1;
  }

  return {
    stepsAdvanced,
    completed: game.phase === Phase.END,
    actions,
  };
}

function testAtomicity(loc: CapturedLocation): AtomicityResult {
  const replayed = replayToPoint(loc, 'atom');
  const normBefore = JSON.stringify(normalizeSerializedGame(replayed.game.serialize()));
  const promptBefore = extractPromptSignature(replayed.game);
  const invalidAttempt = buildInvalidResponse(replayed.waitingFor);

  let threwError = false;
  let errorMessage: string | undefined;
  try {
    replayed.player.process(invalidAttempt.response);
  } catch (err) {
    threwError = true;
    errorMessage = err instanceof Error ? err.message : String(err);
  }

  const normAfter = JSON.stringify(normalizeSerializedGame(replayed.game.serialize()));
  const promptAfter = extractPromptSignature(replayed.game);
  const stateDiff = normBefore === normAfter ?
    null :
    findFirstDiff(JSON.parse(normBefore), JSON.parse(normAfter));
  const promptDiff = findFirstDiff(promptBefore, promptAfter);

  return {
    passed: threwError && stateDiff === null && promptDiff === null,
    threwError,
    errorMessage,
    stateUnchanged: stateDiff === null,
    stateDiff,
    promptUnchanged: promptDiff === null,
    promptDiff,
    testedLabel: invalidAttempt.label,
  };
}

function testDirectCloneAtPoint(loc: CapturedLocation): {
  directResult: DirectCloneTestResult;
  replayedPoint: ReplayedAtPoint;
  origFutureActions: Array<{playerColor: string; response: InputResponse}>;
  origFinalNorm: unknown;
  origFinalScores: Record<string, number>;
} {
  // Atomicity check on isolated replay
  const atomicity = testAtomicity(loc);

  // Replay fresh to exact point for (a), (b), (c)
  const replayed = replayToPoint(loc, 'direct');
  const orig = replayed.game;
  const origNormBeforeObj = normalizeSerializedGame(orig.serialize());
  const origNormBeforeJson = JSON.stringify(origNormBeforeObj);
  const origPrompt = extractPromptSignature(orig);

  // Deep-copy serialized snapshot & deserialize (direct clone)
  const deepCopy = JSON.parse(JSON.stringify(orig.serialize())) as Record<string, unknown>;
  const directCloneForC = Game.deserialize(JSON.parse(JSON.stringify(deepCopy)));
  const directCloneForB = Game.deserialize(deepCopy);

  // (a) Prompt equivalence
  const clonePrompt = extractPromptSignature(directCloneForC);
  const promptDiff = findFirstDiff(origPrompt, clonePrompt);
  const promptEquivalent = promptDiff === null;

  // (c) Isolation: advance directCloneForC 20 decisions, verify orig is unchanged
  const isoAdv = advanceDecisions(directCloneForC, replayed.prngAtPoint.clone(), 20, false);
  const origNormAfterCJson = JSON.stringify(normalizeSerializedGame(orig.serialize()));
  const origStateDiff = origNormBeforeJson === origNormAfterCJson ?
    null :
    findFirstDiff(origNormBeforeObj, JSON.parse(origNormAfterCJson));
  const origPromptAfterC = extractPromptSignature(orig);
  const origPromptDiff = findFirstDiff(origPrompt, origPromptAfterC);
  const isolationPassed = origStateDiff === null && origPromptDiff === null;

  // (b) Continuation: advance orig to Phase.END recording actions, then advance directCloneForB
  const origRun = advanceDecisions(orig, replayed.prngAtPoint.clone(), DEFAULT_MAX_STEPS, true);
  const origFinalNorm = normalizeSerializedGame(orig.serialize());
  const origFinalScores = origRun.completed ? collectPlayerScores(orig).scores : {};

  let continuationSameNextActionAccepted = false;
  let continuationError: string | undefined;

  if (origRun.actions.length > 0) {
    // Check if the first action taken by orig is even accepted by a throwaway clone,
    // and also run directCloneForB with the exact same PRNG state & policy.
    const probeClone = Game.deserialize(JSON.parse(origNormBeforeJson === '' ? '{}' : JSON.stringify(
      JSON.parse(JSON.stringify(replayToPoint(loc, 'probe').game.serialize())),
    )));
    try {
      const firstAct = origRun.actions[0];
      const probePlayer = probeClone.players.find((p) => p.color === firstAct.playerColor);
      if (probePlayer && probePlayer.getWaitingFor() !== undefined) {
        probePlayer.process(firstAct.response);
        continuationSameNextActionAccepted = true;
      }
    } catch (err) {
      continuationSameNextActionAccepted = false;
      continuationError = err instanceof Error ? err.message : String(err);
    }
  }

  const cloneRun = advanceDecisions(directCloneForB, replayed.prngAtPoint.clone(), DEFAULT_MAX_STEPS, false);
  const cloneFinalNorm = normalizeSerializedGame(directCloneForB.serialize());
  const cloneFinalScores = cloneRun.completed ? collectPlayerScores(directCloneForB).scores : {};
  const scoresMatch =
    origRun.completed &&
    cloneRun.completed &&
    JSON.stringify(origFinalScores) === JSON.stringify(cloneFinalScores);
  const finalStateDiff = findFirstDiff(origFinalNorm, cloneFinalNorm);
  if (!cloneRun.completed && cloneRun.error) {
    continuationError = continuationError ? `${continuationError}; clone run: ${cloneRun.error}` : cloneRun.error;
  }

  const continuationPassed =
    origRun.completed &&
    cloneRun.completed &&
    continuationSameNextActionAccepted &&
    scoresMatch &&
    finalStateDiff === null;

  const directClonePassed = promptEquivalent && continuationPassed && isolationPassed && atomicity.passed;

  const directResult: DirectCloneTestResult = {
    category: loc.category,
    boardName: loc.boardName,
    seed: loc.seed,
    stepIndex: loc.stepIndex,
    generation: orig.generation, // note: updated below before orig advances? Wait: let's read before advancing!
    phase: replayed.game.phase,
    activeInputType: replayed.waitingFor.type,
    activeInputTitle: getTitleText(replayed.waitingFor.title),
    promptEquivalent,
    promptDiff,
    continuationPassed,
    continuationSameNextActionAccepted,
    continuationScoresMatch: scoresMatch,
    continuationOrigScores: origFinalScores,
    continuationCloneScores: cloneFinalScores,
    continuationFinalStateDiff: finalStateDiff,
    ...(continuationError ? {continuationError} : {}),
    isolationPassed,
    isolationCloneStepsAdvanced: isoAdv.stepsAdvanced,
    isolationOrigStateDiff: origStateDiff,
    isolationOrigPromptDiff: origPromptDiff,
    atomicity,
    directClonePassed,
  };

  return {
    directResult,
    replayedPoint: replayed,
    origFutureActions: origRun.actions,
    origFinalNorm,
    origFinalScores,
  };
}

function buildFallbackClone(checkpointSnapshotJson: string, suffixActions: ReadonlyArray<RecordedAction>): {
  game: IGame;
  cloneMs: number;
} {
  const startNs = process.hrtime.bigint();
  const cloned = Game.deserialize(JSON.parse(checkpointSnapshotJson) as Record<string, unknown>);
  const byColor = new Map(cloned.players.map((p) => [p.color, p]));
  for (const act of suffixActions) {
    const p = byColor.get(act.playerColor);
    if (!p) {
      throw new Error(`Missing player ${act.playerColor} in fallback replay`);
    }
    p.process(act.response);
  }
  const endNs = process.hrtime.bigint();
  return {
    game: cloned,
    cloneMs: Number(endNs - startNs) / 1e6,
  };
}

function testFallbackCloneAtPoint(
  loc: CapturedLocation,
  origFinalNorm: unknown,
  origFinalScores: Record<string, number>,
): FallbackCloneTestResult {
  const replayed = replayToPoint(loc, 'fb-orig');
  const orig = replayed.game;
  const origNormBeforeObj = normalizeSerializedGame(orig.serialize());
  const origNormBeforeJson = JSON.stringify(origNormBeforeObj);
  const origPrompt = extractPromptSignature(orig);

  const fbB = buildFallbackClone(replayed.checkpointSnapshotJson, replayed.suffixActions);
  const fbC = buildFallbackClone(replayed.checkpointSnapshotJson, replayed.suffixActions);

  // (a) Prompt equivalence
  const fbPrompt = extractPromptSignature(fbB.game);
  const promptDiff = findFirstDiff(origPrompt, fbPrompt);
  const promptEquivalent = promptDiff === null;

  // (c) Isolation: advance fbC 20 decisions, verify orig unchanged
  const isoAdv = advanceDecisions(fbC.game, replayed.prngAtPoint.clone(), 20, false);
  const origNormAfterJson = JSON.stringify(normalizeSerializedGame(orig.serialize()));
  const origStateDiff = origNormBeforeJson === origNormAfterJson ?
    null :
    findFirstDiff(origNormBeforeObj, JSON.parse(origNormAfterJson));
  const isolationPassed = origStateDiff === null;

  // (b) Continuation: advance fbB to Phase.END with cloned PRNG, compare against origFinalNorm & origFinalScores
  const fbRun = advanceDecisions(fbB.game, replayed.prngAtPoint.clone(), DEFAULT_MAX_STEPS, false);
  const fbFinalNorm = normalizeSerializedGame(fbB.game.serialize());
  const fbFinalScores = fbRun.completed ? collectPlayerScores(fbB.game).scores : {};
  const scoresMatch = fbRun.completed && JSON.stringify(origFinalScores) === JSON.stringify(fbFinalScores);
  const finalStateDiff = findFirstDiff(origFinalNorm, fbFinalNorm);
  const continuationPassed = fbRun.completed && scoresMatch && finalStateDiff === null;

  return {
    category: loc.category,
    boardName: loc.boardName,
    seed: loc.seed,
    stepIndex: loc.stepIndex,
    checkpointStepIndex: replayed.checkpointStepIndex,
    suffixLength: replayed.suffixActions.length,
    fallbackCloneMs: Number(fbB.cloneMs.toFixed(3)),
    promptEquivalent,
    promptDiff,
    continuationPassed,
    continuationScoresMatch: scoresMatch,
    continuationFinalStateDiff: finalStateDiff,
    ...(fbRun.error ? {continuationError: fbRun.error} : {}),
    isolationPassed,
    isolationCloneStepsAdvanced: isoAdv.stepsAdvanced,
    isolationOrigStateDiff: origStateDiff,
    fallbackPassed: promptEquivalent && continuationPassed && isolationPassed,
  };
}

function percentile(sorted: ReadonlyArray<number>, p: number): number {
  if (sorted.length === 0) {
    return 0;
  }
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[idx];
}

function measureFallbackCostVsSuffixLength(): Array<SuffixSweepEntry> {
  const lengths = [0, 1, 2, 5, 10, 20, 40];
  const entries: Array<SuffixSweepEntry> = [];
  const seed = 3101;
  const driverSeed = (seed ^ 0x9e3779b9) >>> 0;
  const prng = new SeededPrng(driverSeed);
  const {game} = createExplicitTwoPlayerGame({boardName: TARGET_MAPS[0], seed});
  const unexpected = new Set<string>();

  // Advance to a mid-game ACTION root boundary around step 120
  let checkpointJson = '';
  const recorded: Array<RecordedAction> = [];
  let step = 0;
  while (game.phase !== Phase.END && step < 300) {
    const waiting = getRequiredWaitingPlayers(game);
    if (waiting.length === 0) {
      break;
    }
    const p = pickNextWaitingPlayer(waiting, prng, 'canonical');
    const wf = p.getWaitingFor()!;
    if (!checkpointJson && step >= 120 && isSafeDirectRestoreBoundary(game, waiting)) {
      checkpointJson = JSON.stringify(game.serialize());
    }
    const res = stepDecision(p, wf, prng, unexpected);
    if (!res.succeeded || !res.response) {
      break;
    }
    if (checkpointJson) {
      recorded.push({
        step: step + 1,
        playerColor: p.color,
        playerId: p.id,
        phase: game.phase,
        generation: game.generation,
        waitingRequiredColors: waiting.map((w) => w.color),
        inputType: wf.type,
        response: res.response,
      });
      if (recorded.length >= 45) {
        break;
      }
    }
    step += 1;
  }

  for (const len of lengths) {
    const suffix = recorded.slice(0, len);
    const times: Array<number> = [];
    // Warm-up 2 calls
    buildFallbackClone(checkpointJson, suffix);
    buildFallbackClone(checkpointJson, suffix);
    const samples = 25;
    for (let i = 0; i < samples; i++) {
      const {cloneMs} = buildFallbackClone(checkpointJson, suffix);
      times.push(cloneMs);
    }
    times.sort((a, b) => a - b);
    entries.push({
      suffixLength: len,
      medianMs: Number(percentile(times, 0.5).toFixed(3)),
      p95Ms: Number(percentile(times, 0.95).toFixed(3)),
      maxMs: Number(times[times.length - 1].toFixed(3)),
      samples,
    });
  }

  return entries;
}

function updateSummaryFile(resultsDir: string, patch: Record<string, unknown>): void {
  const summaryPath = path.join(resultsDir, 'summary.json');
  let existing: Record<string, unknown> = {};
  if (fs.existsSync(summaryPath)) {
    try {
      existing = JSON.parse(fs.readFileSync(summaryPath, 'utf8')) as Record<string, unknown>;
    } catch {
      existing = {};
    }
  }
  fs.writeFileSync(summaryPath, JSON.stringify({...existing, ...patch}, null, 2) + '\n', 'utf8');
}

function main(): void {
  const resultsDir = path.resolve(__dirname, '../results');
  fs.mkdirSync(resultsDir, {recursive: true});

  console.log('=== Stage 3: Cloning (Direct Game.deserialize(deepCopy(serialize())) + Fallback Checkpoint+Replay) ===');

  const directResults: Array<DirectCloneTestResult> = [];
  const fallbackResults: Array<FallbackCloneTestResult> = [];
  const missingByMap: Record<string, Array<DecisionPointCategory>> = {};

  for (const boardName of TARGET_MAPS) {
    const locations = scanMapForCategories(boardName, 250);
    const missing = TARGET_CATEGORIES.filter((c) => !locations.has(c));
    missingByMap[boardName] = missing;

    console.log(`\n[${boardName}] Found ${locations.size}/${TARGET_CATEGORIES.length} decision point categories` +
      (missing.length > 0 ? ` (missing: ${missing.join(', ')})` : ''));

    for (const cat of TARGET_CATEGORIES) {
      const loc = locations.get(cat);
      if (!loc) {
        continue;
      }
      // Capture generation/phase before advancing
      const preSnap = replayToPoint(loc, 'pre');
      const genAtPoint = preSnap.game.generation;
      const phaseAtPoint = preSnap.game.phase;

      const {directResult, origFinalNorm, origFinalScores} = testDirectCloneAtPoint(loc);
      directResult.generation = genAtPoint;
      directResult.phase = phaseAtPoint;
      directResults.push(directResult);

      console.log(
        `  ${cat.padEnd(23)} seed=${loc.seed} step=${String(loc.stepIndex).padStart(3)} ` +
          `(${phaseAtPoint}/gen${genAtPoint}/${directResult.activeInputType}): ` +
          `a_prompt=${directResult.promptEquivalent ? 'PASS' : 'FAIL'}, ` +
          `b_cont=${directResult.continuationPassed ? 'PASS' : 'FAIL'}, ` +
          `c_iso=${directResult.isolationPassed ? 'PASS' : 'FAIL'}, ` +
          `d_atom=${directResult.atomicity.passed ? 'PASS' : 'FAIL'} ` +
          `=> DIRECT=${directResult.directClonePassed ? 'PASS' : 'FAIL'}`,
      );

      if (!directResult.promptEquivalent || !directResult.continuationPassed || !directResult.isolationPassed) {
        const fb = testFallbackCloneAtPoint(loc, origFinalNorm, origFinalScores);
        fallbackResults.push(fb);
        console.log(
          `    -> FALLBACK (ckpt@${fb.checkpointStepIndex}, suffix=${fb.suffixLength}, ${fb.fallbackCloneMs}ms): ` +
            `a_prompt=${fb.promptEquivalent ? 'PASS' : 'FAIL'}, ` +
            `b_cont=${fb.continuationPassed ? 'PASS' : 'FAIL'}, ` +
            `c_iso=${fb.isolationPassed ? 'PASS' : 'FAIL'} ` +
            `=> FALLBACK=${fb.fallbackPassed ? 'PASS' : 'FAIL'}`,
        );
      }
    }
  }

  const suffixSweep = measureFallbackCostVsSuffixLength();
  console.log('\nFallback Clone Cost vs Suffix Length (Tharsis mid-game ACTION root):');
  for (const s of suffixSweep) {
    console.log(`  suffix=${String(s.suffixLength).padStart(2)}: median=${s.medianMs}ms, p95=${s.p95Ms}ms, max=${s.maxMs}ms`);
  }

  const categorySummary = TARGET_CATEGORIES.map((cat) => {
    const rows = directResults.filter((r) => r.category === cat);
    const fbRows = fallbackResults.filter((r) => r.category === cat);
    return {
      category: cat,
      mapsTested: rows.length,
      directPromptPass: rows.filter((r) => r.promptEquivalent).length,
      directContinuationPass: rows.filter((r) => r.continuationPassed).length,
      directIsolationPass: rows.filter((r) => r.isolationPassed).length,
      atomicityPass: rows.filter((r) => r.atomicity.passed).length,
      directAllPass: rows.filter((r) => r.directClonePassed).length,
      fallbackTested: fbRows.length,
      fallbackPass: fbRows.filter((r) => r.fallbackPassed).length,
    };
  });

  console.log('\n=== Stage 3 Summary by Decision Point Category ===');
  for (const s of categorySummary) {
    console.log(
      `  ${s.category.padEnd(23)} tested=${s.mapsTested}/4 | ` +
        `direct(a=${s.directPromptPass}, b=${s.directContinuationPass}, c=${s.directIsolationPass}, d_atom=${s.atomicityPass}) | ` +
        `fallback(${s.fallbackPass}/${s.fallbackTested})`,
    );
  }

  const payload = {
    generatedAt: new Date().toISOString(),
    nodeVersion: process.version,
    simultaneousOrder: 'canonical',
    missingByMap,
    categorySummary,
    directResults,
    fallbackResults,
    fallbackSuffixSweep: suffixSweep,
  };

  fs.writeFileSync(
    path.join(resultsDir, 'cloning.json'),
    JSON.stringify(payload, null, 2) + '\n',
    'utf8',
  );

  updateSummaryFile(resultsDir, {
    stage3_cloning: {
      generatedAt: payload.generatedAt,
      nodeVersion: process.version,
      simultaneousOrder: 'canonical',
      missingByMap,
      categorySummary,
      fallbackSuffixSweep: suffixSweep,
      atomicityFailures: directResults
        .filter((r) => !r.atomicity.passed)
        .map((r) => ({
          category: r.category,
          boardName: r.boardName,
          seed: r.seed,
          testedLabel: r.atomicity.testedLabel,
          threwError: r.atomicity.threwError,
          errorMessage: r.atomicity.errorMessage,
          stateDiff: r.atomicity.stateDiff,
          promptDiff: r.atomicity.promptDiff,
        })),
      directFailureSamples: directResults
        .filter((r) => !r.directClonePassed)
        .slice(0, 20)
        .map((r) => ({
          category: r.category,
          boardName: r.boardName,
          seed: r.seed,
          phase: r.phase,
          activeInputType: r.activeInputType,
          promptDiff: r.promptDiff,
          continuationError: r.continuationError,
          continuationFinalStateDiff: r.continuationFinalStateDiff,
        })),
    },
  });
}

main();
