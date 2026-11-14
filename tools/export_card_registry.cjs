#!/usr/bin/env node
/**
 * Export the pinned engine's CardName registry to a JSON file under data/.
 *
 * Prerequisites (inside tools/simbench/engine):
 *   npm ci --no-audit --no-fund && npm run make:json && npm run build:server
 *
 * Regenerate with:
 *   node tools/export_card_registry.cjs [--out data/card_registry.json]
 */

'use strict';

const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.resolve(__dirname, '..');
const cardNamePath = path.join(
  repoRoot,
  'tools',
  'simbench',
  'engine',
  'build',
  'src',
  'common',
  'cards',
  'CardName.js',
);
const allManifestsPath = path.join(
  repoRoot,
  'tools',
  'simbench',
  'engine',
  'build',
  'src',
  'server',
  'cards',
  'AllManifests.js',
);

function parseArgs(argv) {
  let outPath = path.join(repoRoot, 'data', 'card_registry.json');
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--out' && argv[i + 1]) {
      outPath = path.resolve(argv[i + 1]);
      i += 1;
    }
  }
  return {outPath};
}

function main() {
  const {outPath} = parseArgs(process.argv.slice(2));

  if (!fs.existsSync(cardNamePath)) {
    throw new Error(
      `Missing built engine CardName.js at ${cardNamePath}. ` +
        'Run in tools/simbench/engine: npm ci --no-audit --no-fund && npm run make:json && npm run build:server',
    );
  }

  const {CardName} = require(cardNamePath);
  const cards = Array.from(new Set(Object.values(CardName))).sort();

  const cardModules = {};
  const cardKinds = {};
  if (fs.existsSync(allManifestsPath)) {
    const {ALL_MODULE_MANIFESTS} = require(allManifestsPath);
    for (const manifest of ALL_MODULE_MANIFESTS) {
      const mod = manifest.module;
      for (const [field, kind] of [
        ['projectCards', 'project'],
        ['corporationCards', 'corporation'],
        ['preludeCards', 'prelude'],
        ['ceoCards', 'ceo'],
      ]) {
        const entries = manifest[field] || {};
        for (const name of Object.keys(entries)) {
          if (!cardModules[name]) {
            cardModules[name] = mod;
          }
          if (!cardKinds[name]) {
            cardKinds[name] = kind;
          }
        }
      }
    }
  }

  const payload = {
    source: 'tools/simbench/engine/build/src/common/cards/CardName.js',
    count: cards.length,
    cards,
    card_modules: cardModules,
    card_kinds: cardKinds,
  };

  fs.mkdirSync(path.dirname(outPath), {recursive: true});
  fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  console.log(`Wrote ${cards.length} card names to ${outPath}`);
}

main();
