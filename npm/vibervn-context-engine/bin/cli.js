#!/usr/bin/env node

const { execFileSync } = require('child_process')
const { existsSync } = require('fs')
const path = require('path')

const PLATFORMS = {
  'linux-x64': 'vibervn-context-engine-linux-x64',
  'linux-arm64': 'vibervn-context-engine-linux-arm64',
  'darwin-arm64': 'vibervn-context-engine-darwin-arm64',
  'win32-x64': 'vibervn-context-engine-win32-x64',
}

const platformKey = `${process.platform}-${process.arch}`
const packageName = PLATFORMS[platformKey]

if (!packageName) {
  console.error(
    `Unsupported platform: ${process.platform} ${process.arch}\n` +
    `Supported platforms: ${Object.keys(PLATFORMS).join(', ')}`
  )
  process.exit(1)
}

let binPath
try {
  const binName = process.platform === 'win32' ? 'context-engine-rs.exe' : 'context-engine-rs'
  binPath = path.join(
    path.dirname(require.resolve(`${packageName}/package.json`)),
    'bin',
    binName
  )
} catch {
  console.error(
    `Could not find the binary package "${packageName}".\n` +
    `This usually means it was not installed (e.g., --no-optional was used) or your platform is unsupported.\n` +
    `Try reinstalling: npm install -g vibervn-context-engine`
  )
  process.exit(1)
}

if (!existsSync(binPath)) {
  console.error(
    `Binary not found at "${binPath}".\n` +
    `The platform package "${packageName}" is installed but the binary is missing.\n` +
    `Try reinstalling: npm install -g vibervn-context-engine`
  )
  process.exit(1)
}

try {
  execFileSync(binPath, process.argv.slice(2), {
    stdio: 'inherit',
    env: process.env,
  })
  process.exit(0)
} catch (e) {
  if (e.signal) {
    process.kill(process.pid, e.signal)
  }
  process.exit(e.status ?? 1)
}
