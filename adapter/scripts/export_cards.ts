import fs from 'node:fs';
import path from 'node:path';

const ENGINE_BUILD_ROOT = path.resolve(__dirname, '../../../tools/simbench/engine/build/src');

function engineRequire<T>(relPath: string): T {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require(path.join(ENGINE_BUILD_ROOT, relPath)) as T;
}

interface RawCard {
  name: string;
  type: string;
  cost?: number;
  tags?: ReadonlyArray<string>;
  startingMegaCredits?: number;
  cardCost?: number;
  victoryPoints?: number | 'special' | Record<string, unknown>;
  tr?: {tr?: number; temperature?: number; oxygen?: number; oceans?: number};
  behavior?: Record<string, unknown>;
  cardDiscount?: unknown;
  resourceType?: string;
  requirements?: ReadonlyArray<unknown>;
  tilesBuilt?: ReadonlyArray<number>;
  action?: unknown;
  canAct?: unknown;
}

const {globalInitialize} = engineRequire<{globalInitialize: () => void}>('server/globalInitialize.js');
const {ALL_MODULE_MANIFESTS} = engineRequire<{
  ALL_MODULE_MANIFESTS: ReadonlyArray<{
    module: string;
    projectCards: unknown;
    corporationCards: unknown;
    preludeCards: unknown;
    standardProjects: unknown;
  }>;
}>('server/cards/AllManifests.js');
const {CardManifest} = engineRequire<{
  CardManifest: {
    values<T>(manifest: unknown): Array<{Factory: new () => T}>;
  };
}>('server/cards/ModuleManifest.js');

const UNIT_KEYS = ['megacredits', 'steel', 'titanium', 'plants', 'energy', 'heat'] as const;
const TRAINING_MODULES = new Set(['base', 'corpera', 'prelude']);

function emptyUnits(): Record<(typeof UNIT_KEYS)[number], number> {
  return {megacredits: 0, steel: 0, titanium: 0, plants: 0, energy: 0, heat: 0};
}

function extractCardRecord(moduleName: string, card: RawCard): Record<string, unknown> {
  const behavior = (card.behavior ?? {}) as Record<string, unknown>;

  const production = emptyUnits();
  let hasDynamicProduction = false;
  const rawProd = behavior.production as Record<string, unknown> | undefined;
  if (rawProd && typeof rawProd === 'object') {
    for (const k of UNIT_KEYS) {
      const v = rawProd[k];
      if (typeof v === 'number') {
        production[k] = v;
      } else if (v !== undefined) {
        hasDynamicProduction = true;
      }
    }
  }

  const stock = emptyUnits();
  const rawStock = behavior.stock as Record<string, unknown> | undefined;
  if (rawStock && typeof rawStock === 'object') {
    for (const k of UNIT_KEYS) {
      const v = rawStock[k];
      if (typeof v === 'number') {
        stock[k] = v;
      }
    }
  }

  const globalObj = (behavior.global ?? {}) as Record<string, unknown>;
  const tempSteps = typeof globalObj.temperature === 'number' ? Math.max(0, globalObj.temperature) : 0;
  const oxygenSteps = typeof globalObj.oxygen === 'number' ? Math.max(0, globalObj.oxygen) : 0;
  const oceanRaw = behavior.ocean as Record<string, unknown> | undefined;
  const oceans = oceanRaw !== undefined ? (typeof oceanRaw.count === 'number' ? oceanRaw.count : 1) : 0;
  const greenery = behavior.greenery !== undefined ? 1 : 0;
  const city = behavior.city !== undefined ? 1 : 0;

  let tr = 0;
  if (typeof behavior.tr === 'number') {
    tr += behavior.tr;
  }
  if (card.tr && typeof card.tr === 'object') {
    if (typeof card.tr.tr === 'number') tr += card.tr.tr;
    if (typeof card.tr.temperature === 'number') tr += card.tr.temperature;
    if (typeof card.tr.oxygen === 'number') tr += card.tr.oxygen;
    if (typeof card.tr.oceans === 'number') tr += card.tr.oceans;
  } else {
    tr += tempSteps + oxygenSteps + oceans + greenery;
  }

  let drawCard = 0;
  const rawDraw = behavior.drawCard;
  if (typeof rawDraw === 'number') {
    drawCard = rawDraw;
  } else if (rawDraw && typeof rawDraw === 'object') {
    const drawObj = rawDraw as Record<string, unknown>;
    if (typeof drawObj.keep === 'number') {
      drawCard = drawObj.keep;
    } else if (typeof drawObj.count === 'number') {
      drawCard = drawObj.count;
    }
  }

  let staticVp = 0;
  let hasDynamicVp = false;
  let vpPerResource = 0;
  if (typeof card.victoryPoints === 'number') {
    staticVp = card.victoryPoints;
  } else if (card.victoryPoints !== undefined) {
    hasDynamicVp = true;
    if (typeof card.victoryPoints === 'object' && card.victoryPoints !== null) {
      const vpObj = card.victoryPoints as Record<string, unknown>;
      if (vpObj.resourcesHere !== undefined) {
        const each = typeof vpObj.each === 'number' ? vpObj.each : 1;
        const per = typeof vpObj.per === 'number' ? vpObj.per : 1;
        vpPerResource = each / per;
      }
    }
  }

  let discountAmount = 0;
  if (card.cardDiscount !== undefined) {
    const discounts = Array.isArray(card.cardDiscount) ? card.cardDiscount : [card.cardDiscount];
    for (const d of discounts) {
      if (d && typeof d === 'object' && typeof (d as Record<string, unknown>).amount === 'number') {
        discountAmount += (d as Record<string, unknown>).amount as number;
      }
    }
  }

  const hasAction =
    typeof card.action === 'function' ||
    typeof card.canAct === 'function' ||
    (behavior !== undefined && Object.keys(behavior).length > 0 && card.type === 'active' && card.action !== undefined);

  return {
    name: card.name,
    module: moduleName,
    cardType: card.type,
    cost: typeof card.cost === 'number' ? card.cost : 0,
    tags: Array.isArray(card.tags) ? [...card.tags] : [],
    startingMegaCredits: typeof card.startingMegaCredits === 'number' ? card.startingMegaCredits : 0,
    cardCost: typeof card.cardCost === 'number' ? card.cardCost : null,
    staticVp,
    hasDynamicVp,
    vpPerResource,
    production,
    hasDynamicProduction,
    stock,
    tr,
    temperatureSteps: tempSteps,
    oxygenSteps,
    oceans,
    greenery,
    city,
    drawCard,
    hasAction,
    resourceType: card.resourceType ?? null,
    discountAmount,
    tilesBuilt: Array.isArray(card.tilesBuilt) ? [...card.tilesBuilt] : [],
    requirementsCount: Array.isArray(card.requirements) ? card.requirements.length : 0,
    requirements: Array.isArray(card.requirements) ? card.requirements : [],
  };
}

export function exportCards(outPath: string): Array<Record<string, unknown>> {
  globalInitialize();
  const records: Array<Record<string, unknown>> = [];
  const seen = new Set<string>();

  for (const manifest of ALL_MODULE_MANIFESTS) {
    if (!TRAINING_MODULES.has(manifest.module)) {
      continue;
    }
    const decks = [
      manifest.projectCards,
      manifest.corporationCards,
      manifest.preludeCards,
      manifest.standardProjects,
    ];
    for (const deck of decks) {
      for (const entry of CardManifest.values<RawCard>(deck)) {
        const card = new entry.Factory();
        if (card.type === 'proxy' || seen.has(card.name)) {
          continue;
        }
        seen.add(card.name);
        records.push(extractCardRecord(manifest.module, card));
      }
    }
  }

  records.sort((a, b) => String(a.name).localeCompare(String(b.name)));
  const resolvedOut = path.resolve(outPath);
  fs.mkdirSync(path.dirname(resolvedOut), {recursive: true});
  fs.writeFileSync(resolvedOut, JSON.stringify({version: 1, cards: records}, null, 2) + '\n', 'utf8');
  return records;
}

if (require.main === module) {
  const args = process.argv.slice(2);
  let outPath = path.resolve(__dirname, '../../../data/cards.json');
  const outIdx = args.indexOf('--out');
  if (outIdx !== -1 && args[outIdx + 1]) {
    outPath = args[outIdx + 1];
  }
  const records = exportCards(outPath);
  process.stdout.write(`Exported ${records.length} cards to ${path.resolve(outPath)}\n`);
}
