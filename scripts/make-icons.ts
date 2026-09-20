/**
 * Draw the app icon, and write it in the three formats the platforms want.
 *
 *   node scripts/make-icons.ts
 *
 * Generated rather than committed as a binary blob, and generated here rather than with
 * an image tool, so that the icon has a source anyone can read and change — and so the
 * build needs nothing installed beyond what is already in the tree. The outputs *are*
 * committed, because packaging must not depend on this having been run.
 *
 * The drawing is a network: one filled node with others around it, joined by lines.
 * Supersampled 4x and box-filtered down, which is enough anti-aliasing for an icon and
 * far less code than a real rasteriser.
 */
import { deflateSync } from 'node:zlib'
import { writeFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'

const OUT = 'assets'

// --------------------------------------------------------------- drawing

type RGBA = [number, number, number, number]

const BACKDROP_TOP: RGBA = [16, 132, 165, 255]
const BACKDROP_BOTTOM: RGBA = [9, 78, 102, 255]
const NODE: RGBA = [255, 255, 255, 255]
const LINK: RGBA = [255, 255, 255, 140]

/** Satellites, as angle in turns and distance as a fraction of the canvas. */
const SATELLITES = [0.08, 0.25, 0.42, 0.58, 0.75, 0.92].map(turn => ({
  x: 0.5 + 0.295 * Math.cos(turn * Math.PI * 2),
  y: 0.5 + 0.295 * Math.sin(turn * Math.PI * 2),
}))

function blend(dst: Uint8Array, i: number, [r, g, b, a]: RGBA): void {
  const alpha = a / 255
  dst[i] = Math.round(dst[i]! * (1 - alpha) + r * alpha)
  dst[i + 1] = Math.round(dst[i + 1]! * (1 - alpha) + g * alpha)
  dst[i + 2] = Math.round(dst[i + 2]! * (1 - alpha) + b * alpha)
  dst[i + 3] = Math.max(dst[i + 3]!, Math.round(255 * alpha))
}

/** Distance from a point to a line segment, for drawing the links as capsules. */
function distanceToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax
  const dy = by - ay
  const lengthSq = dx * dx + dy * dy
  const t = lengthSq === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSq))
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy))
}

function draw(size: number): Uint8Array {
  const px = new Uint8Array(size * size * 4)
  const radius = size * 0.225          // corner radius of the rounded square
  const centre = size * 0.5
  const nodeR = size * 0.105
  const satR = size * 0.072
  const linkW = size * 0.030

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const i = (y * size + x) * 4

      // Rounded square: inside when the point is within `radius` of the inset rectangle.
      const dx = Math.max(radius - x, 0, x - (size - radius))
      const dy = Math.max(radius - y, 0, y - (size - radius))
      if (Math.hypot(dx, dy) > radius) continue

      const t = y / size
      blend(px, i, [
        Math.round(BACKDROP_TOP[0] + (BACKDROP_BOTTOM[0] - BACKDROP_TOP[0]) * t),
        Math.round(BACKDROP_TOP[1] + (BACKDROP_BOTTOM[1] - BACKDROP_TOP[1]) * t),
        Math.round(BACKDROP_TOP[2] + (BACKDROP_BOTTOM[2] - BACKDROP_TOP[2]) * t),
        255,
      ])

      for (const s of SATELLITES) {
        if (distanceToSegment(x, y, centre, centre, s.x * size, s.y * size) <= linkW / 2) {
          blend(px, i, LINK)
          break
        }
      }
      for (const s of SATELLITES) {
        if (Math.hypot(x - s.x * size, y - s.y * size) <= satR) { blend(px, i, NODE); break }
      }
      if (Math.hypot(x - centre, y - centre) <= nodeR) blend(px, i, NODE)
    }
  }
  return px
}

/** Render at 4x and box-filter down — cheap, adequate anti-aliasing. */
function render(size: number): Uint8Array {
  const scale = 4
  const big = draw(size * scale)
  const out = new Uint8Array(size * size * 4)
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const acc = [0, 0, 0, 0]
      for (let sy = 0; sy < scale; sy += 1) {
        for (let sx = 0; sx < scale; sx += 1) {
          const j = (((y * scale + sy) * size * scale) + (x * scale + sx)) * 4
          for (let c = 0; c < 4; c += 1) acc[c]! += big[j + c]!
        }
      }
      const i = (y * size + x) * 4
      for (let c = 0; c < 4; c += 1) out[i + c] = Math.round(acc[c]! / (scale * scale))
    }
  }
  return out
}

// ------------------------------------------------------------------ PNG

const CRC_TABLE = (() => {
  const table = new Uint32Array(256)
  for (let n = 0; n < 256; n += 1) {
    let c = n
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[n] = c >>> 0
  }
  return table
})()

function crc32(bytes: Buffer): number {
  let c = 0xffffffff
  for (const byte of bytes) c = CRC_TABLE[(c ^ byte) & 0xff]! ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

function chunk(type: string, data: Buffer): Buffer {
  const length = Buffer.alloc(4)
  length.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(body))
  return Buffer.concat([length, body, crc])
}

function png(size: number): Buffer {
  const pixels = render(size)
  // One filter byte (0 = none) per scanline, then the raw RGBA.
  const raw = Buffer.alloc(size * (size * 4 + 1))
  for (let y = 0; y < size; y += 1) {
    raw[y * (size * 4 + 1)] = 0
    Buffer.from(pixels.buffer, y * size * 4, size * 4).copy(raw, y * (size * 4 + 1) + 1)
  }
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(size, 0)
  ihdr.writeUInt32BE(size, 4)
  ihdr[8] = 8    // bit depth
  ihdr[9] = 6    // colour type: RGBA
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ])
}

// ------------------------------------------------------------------ ICO

/**
 * Windows .ico, carrying PNGs rather than BMPs.
 *
 * PNG-compressed entries have been supported since Vista and keep the file small; the
 * alternative is a BMP with an upside-down bitmap and a separate AND mask.
 */
function ico(sizes: number[]): Buffer {
  const images = sizes.map(s => ({ size: s, data: png(s) }))
  const header = Buffer.alloc(6)
  header.writeUInt16LE(0, 0)          // reserved
  header.writeUInt16LE(1, 2)          // type: icon
  header.writeUInt16LE(images.length, 4)

  let offset = 6 + images.length * 16
  const entries: Buffer[] = []
  for (const image of images) {
    const e = Buffer.alloc(16)
    // 256 is written as 0: the field is one byte and 256 does not fit.
    e[0] = image.size >= 256 ? 0 : image.size
    e[1] = image.size >= 256 ? 0 : image.size
    e[2] = 0                          // palette size
    e[3] = 0                          // reserved
    e.writeUInt16LE(1, 4)             // colour planes
    e.writeUInt16LE(32, 6)            // bits per pixel
    e.writeUInt32BE(0, 8)
    e.writeUInt32LE(image.data.length, 8)
    e.writeUInt32LE(offset, 12)
    entries.push(e)
    offset += image.data.length
  }
  return Buffer.concat([header, ...entries, ...images.map(i => i.data)])
}

// ----------------------------------------------------------------- ICNS

/** Apple .icns: a magic, a total length, then one typed chunk per size. */
function icns(entries: { type: string; size: number }[]): Buffer {
  const chunks = entries.map(e => {
    const data = png(e.size)
    const header = Buffer.alloc(8)
    header.write(e.type, 0, 'ascii')
    header.writeUInt32BE(data.length + 8, 4)
    return Buffer.concat([header, data])
  })
  const body = Buffer.concat(chunks)
  const head = Buffer.alloc(8)
  head.write('icns', 0, 'ascii')
  head.writeUInt32BE(body.length + 8, 4)
  return Buffer.concat([head, body])
}

// ---------------------------------------------------------------- write

mkdirSync(OUT, { recursive: true })

writeFileSync(join(OUT, 'icon.png'), png(512))
writeFileSync(join(OUT, 'icon.ico'), ico([16, 24, 32, 48, 64, 128, 256]))
writeFileSync(join(OUT, 'icon.icns'), icns([
  { type: 'icp4', size: 16 },
  { type: 'icp5', size: 32 },
  { type: 'ic07', size: 128 },
  { type: 'ic08', size: 256 },
  { type: 'ic09', size: 512 },
  { type: 'ic10', size: 1024 },
]))

console.log(`\n  assets/icon.png   512x512`)
console.log(`  assets/icon.ico   16–256, for the Windows executable and its taskbar entry`)
console.log(`  assets/icon.icns  16–1024, for the macOS bundle\n`)
