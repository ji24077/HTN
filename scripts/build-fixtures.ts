/**
 * Build the pinned artifacts for the inference workload.
 *
 *   node scripts/build-fixtures.ts <mnist-images> <mnist-labels> <model.onnx> [count]
 *
 * Everything a host runs is content-addressed: the model and the inputs are identified
 * by their SHA-256, so a host can verify it received exactly what the job specified and
 * every host demonstrably ran the same thing.
 */
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'

const [imagesPath, labelsPath, modelPath, countArg] = process.argv.slice(2)
if (!imagesPath || !labelsPath || !modelPath) {
  console.error('usage: node scripts/build-fixtures.ts <images-idx> <labels-idx> <model.onnx> [count]')
  process.exit(1)
}
const count = Number(countArg ?? 1000)

const images = readFileSync(imagesPath)
const labels = readFileSync(labelsPath)

// IDX format: 4-byte magic, then big-endian dimensions.
const imageCount = images.readUInt32BE(4)
const rows = images.readUInt32BE(8)
const cols = images.readUInt32BE(12)
const labelCount = labels.readUInt32BE(4)
if (rows !== 28 || cols !== 28) throw new Error(`expected 28x28, got ${rows}x${cols}`)
if (imageCount !== labelCount) throw new Error(`${imageCount} images but ${labelCount} labels`)
if (count > imageCount) throw new Error(`asked for ${count} but only ${imageCount} available`)

/**
 * One self-contained file: a small header, then pixels, then the true labels.
 *
 * Labels travel with the inputs so a host can report accuracy without a second artifact,
 * and so "did every host agree" and "was it right" are answerable from one download.
 */
const PIXELS = 28 * 28
const out = Buffer.alloc(8 + count * PIXELS + count)
out.write('DWPI', 0, 'ascii')
out.writeUInt32BE(count, 4)
images.copy(out, 8, 16, 16 + count * PIXELS)
labels.copy(out, 8 + count * PIXELS, 8, 8 + count)

const model = readFileSync(modelPath)
const sha = (b: Buffer): string => createHash('sha256').update(b).digest('hex')

mkdirSync('fixtures', { recursive: true })
const modelHash = sha(model)
const inputsHash = sha(out)
writeFileSync(join('fixtures', modelHash), model)
writeFileSync(join('fixtures', inputsHash), out)

const manifest = {
  name: 'mnist-handwritten-digits',
  model: { hash: modelHash, bytes: model.length, source: 'onnx/models mnist-8', inputName: 'Input3', outputName: 'Plus214_Output_0' },
  inputs: { hash: inputsHash, bytes: out.length, count, shape: [1, 1, 28, 28], note: 'MNIST test split, pixels then labels' },
  preprocessing: 'v1: uint8 pixel / 255',
  builtAt: new Date().toISOString(),
}
writeFileSync(join('fixtures', 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n')

console.log(`\n  model   ${modelHash}  ${(model.length / 1024).toFixed(0)} KB`)
console.log(`  inputs  ${inputsHash}  ${(out.length / 1024).toFixed(0)} KB  (${count} digits)`)
console.log(`\n  written to fixtures/, manifest in fixtures/manifest.json\n`)
