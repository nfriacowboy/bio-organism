import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useParams, useSearchParams, Link } from 'react-router-dom'
import { listarSteps } from '../api'
import type { Step } from '../types'

// ─── Canvas drawing ───────────────────────────────────────────────────────────

const CANVAS_LOGICAL = 600

function drawFrame(
  ctx: CanvasRenderingContext2D,
  steps: Step[],
  currentIndex: number,
  scale: number
) {
  const W = CANVAS_LOGICAL * scale
  const H = CANVAS_LOGICAL * scale

  ctx.save()
  ctx.scale(scale, scale)

  // Background
  ctx.fillStyle = '#2a2a2a'
  ctx.fillRect(0, 0, CANVAS_LOGICAL, CANVAS_LOGICAL)

  if (steps.length === 0) {
    ctx.restore()
    return
  }

  const current = steps[currentIndex]

  // Trail dots (all steps up to current)
  for (let i = 0; i <= currentIndex; i++) {
    const s = steps[i]
    const alpha = 0.15 + 0.5 * (i / Math.max(currentIndex, 1))
    ctx.beginPath()
    ctx.arc(s.pos_x, s.pos_y, 3, 0, Math.PI * 2)
    ctx.fillStyle = `rgba(200, 200, 200, ${alpha})`
    ctx.fill()
  }

  // Flash when comeu
  if (current.comeu) {
    ctx.beginPath()
    ctx.arc(current.pos_x, current.pos_y, 18, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(255, 230, 50, 0.35)'
    ctx.fill()
  }

  // Organism body
  ctx.beginPath()
  ctx.arc(current.pos_x, current.pos_y, 8, 0, Math.PI * 2)
  ctx.fillStyle = '#3fb950'
  ctx.fill()
  ctx.strokeStyle = '#fff'
  ctx.lineWidth = 1.5
  ctx.stroke()

  // Direction indicator line (angulo in radians)
  const ang = current.angulo
  const lineLen = 14
  ctx.beginPath()
  ctx.moveTo(current.pos_x, current.pos_y)
  ctx.lineTo(
    current.pos_x + Math.cos(ang) * lineLen,
    current.pos_y + Math.sin(ang) * lineLen
  )
  ctx.strokeStyle = '#fff'
  ctx.lineWidth = 2
  ctx.stroke()

  // Energy bar (top-left corner)
  const barX = 12
  const barY = 12
  const barW = 120
  const barH = 14
  const ratio =
    current.energia_max > 0
      ? Math.max(0, Math.min(1, current.energia_actual / current.energia_max))
      : 0

  // Background
  ctx.fillStyle = '#111'
  ctx.fillRect(barX, barY, barW, barH)

  // Fill: interpolate red → green
  const r = Math.round(255 * (1 - ratio))
  const g = Math.round(200 * ratio)
  ctx.fillStyle = `rgb(${r}, ${g}, 40)`
  ctx.fillRect(barX, barY, barW * ratio, barH)

  // Border
  ctx.strokeStyle = '#555'
  ctx.lineWidth = 1
  ctx.strokeRect(barX, barY, barW, barH)

  // Label
  ctx.fillStyle = '#fff'
  ctx.font = '10px monospace'
  ctx.fillText(
    `E: ${current.energia_actual.toFixed(1)}/${current.energia_max.toFixed(1)}`,
    barX + 4,
    barY + barH - 3
  )

  ctx.restore()

  // Suppress unused variable warning by referencing scale via W/H
  void W
  void H
}

// ─── Info panel ───────────────────────────────────────────────────────────────

function InfoPanel({ step }: { step: Step | null }) {
  if (!step) return null
  const rows: [string, string][] = [
    ['Step', String(step.step)],
    ['Episodio', String(step.episodio)],
    ['Accao', String(step.accao)],
    ['Modo', step.modo],
    ['Energia', `${step.energia_actual.toFixed(1)} / ${step.energia_max.toFixed(1)}`],
    ['Recompensa', step.recompensa.toFixed(4)],
    ['Novidade', step.novidade.toFixed(4)],
    ['Pos', `(${step.pos_x.toFixed(1)}, ${step.pos_y.toFixed(1)})`],
    ['Angulo', step.angulo.toFixed(3)],
    ['Comeu', step.comeu ? 'Sim' : 'Nao'],
    ['Fase', step.fase_mundo],
  ]
  return (
    <div
      style={{
        backgroundColor: '#1a1a1a',
        border: '1px solid #2a2a2a',
        borderRadius: '8px',
        padding: '14px 18px',
        fontSize: '13px',
        minWidth: '200px',
      }}
    >
      <div
        style={{
          fontWeight: 700,
          marginBottom: '10px',
          color: '#ccc',
          fontSize: '12px',
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
        }}
      >
        Step Info
      </div>
      {rows.map(([label, value]) => (
        <div
          key={label}
          style={{ display: 'flex', justifyContent: 'space-between', gap: '16px', marginBottom: '4px' }}
        >
          <span style={{ color: '#888' }}>{label}</span>
          <span
            style={{
              color: label === 'Comeu' && step.comeu ? '#3fb950' : '#e0e0e0',
              fontFamily: 'monospace',
            }}
          >
            {value}
          </span>
        </div>
      ))}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function Replay() {
  const { id } = useParams<{ id: string }>()
  const [searchParams] = useSearchParams()
  const treinoId = Number(id)

  const initialEpisodio = Number(searchParams.get('episodio') ?? '1') || 1

  const [episodio, setEpisodio] = useState(initialEpisodio)
  const [episodioInput, setEpisodioInput] = useState(String(initialEpisodio))
  const [steps, setSteps] = useState<Step[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const animRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const scaleRef = useRef(1)

  // ── Load steps for episode ──
  useEffect(() => {
    async function load() {
      setLoading(true)
      setError(null)
      setPlaying(false)
      setCurrentIndex(0)
      if (animRef.current) clearTimeout(animRef.current)
      try {
        const data = await listarSteps(treinoId, episodio, 5000)
        // Sort by step ascending
        data.sort((a, b) => a.step - b.step)
        setSteps(data)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Erro ao carregar steps')
        setSteps([])
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [treinoId, episodio])

  // ── Canvas resize observer ──
  const updateScale = useCallback(() => {
    if (!containerRef.current || !canvasRef.current) return
    const containerW = containerRef.current.clientWidth
    const newScale = Math.max(0.3, Math.min(2, containerW / CANVAS_LOGICAL))
    scaleRef.current = newScale
    canvasRef.current.width = CANVAS_LOGICAL * newScale
    canvasRef.current.height = CANVAS_LOGICAL * newScale
  }, [])

  useEffect(() => {
    updateScale()
    const obs = new ResizeObserver(updateScale)
    if (containerRef.current) obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [updateScale])

  // ── Draw on every state change ──
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    drawFrame(ctx, steps, currentIndex, scaleRef.current)
  }, [steps, currentIndex])

  // ── Animation loop ──
  useEffect(() => {
    if (!playing) return
    if (currentIndex >= steps.length - 1) {
      setPlaying(false)
      return
    }
    animRef.current = setTimeout(() => {
      setCurrentIndex((i) => i + 1)
    }, 50)
    return () => {
      if (animRef.current) clearTimeout(animRef.current)
    }
  }, [playing, currentIndex, steps.length])

  function handlePlayPause() {
    if (currentIndex >= steps.length - 1) {
      setCurrentIndex(0)
      setPlaying(true)
    } else {
      setPlaying((p) => !p)
    }
  }

  function handleReset() {
    setPlaying(false)
    if (animRef.current) clearTimeout(animRef.current)
    setCurrentIndex(0)
  }

  function handleEpisodioCommit() {
    const n = parseInt(episodioInput, 10)
    if (!isNaN(n) && n > 0) setEpisodio(n)
  }

  function handleEpisodioKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter') handleEpisodioCommit()
  }

  const currentStep = steps[currentIndex] ?? null
  const progressPct = steps.length > 1 ? (currentIndex / (steps.length - 1)) * 100 : 0

  return (
    <div>
      {/* Breadcrumb */}
      <div style={{ color: '#888', fontSize: '13px', marginBottom: '12px' }}>
        <Link to="/" style={{ color: '#58a6ff' }}>
          Dashboard
        </Link>{' '}
        /{' '}
        <Link to={`/treinos/${treinoId}`} style={{ color: '#58a6ff' }}>
          Treino #{treinoId}
        </Link>{' '}
        / Replay
      </div>

      <h1 style={{ marginTop: 0, fontSize: '22px', fontWeight: 700, marginBottom: '20px' }}>
        Replay — Treino #{treinoId}
      </h1>

      {/* Controls row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          marginBottom: '16px',
          flexWrap: 'wrap',
        }}
      >
        {/* Episodio selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <button
            onClick={() => {
              const n = Math.max(1, episodio - 1)
              setEpisodio(n)
              setEpisodioInput(String(n))
            }}
            style={btnStyle}
          >
            ‹
          </button>
          <input
            type="number"
            min={1}
            value={episodioInput}
            onChange={(e) => setEpisodioInput(e.target.value)}
            onBlur={handleEpisodioCommit}
            onKeyDown={handleEpisodioKey}
            style={{ width: '72px', textAlign: 'center', ...inputStyle }}
          />
          <button
            onClick={() => {
              const n = episodio + 1
              setEpisodio(n)
              setEpisodioInput(String(n))
            }}
            style={btnStyle}
          >
            ›
          </button>
          <span style={{ color: '#888', fontSize: '13px' }}>Episodio</span>
        </div>

        <div style={{ width: '1px', height: '28px', backgroundColor: '#333' }} />

        {/* Playback controls */}
        <button
          onClick={handleReset}
          disabled={loading}
          style={{ ...btnStyle, minWidth: '36px' }}
          title="Reset"
        >
          &#9632;
        </button>
        <button
          onClick={handlePlayPause}
          disabled={loading || steps.length === 0}
          style={{
            ...btnStyle,
            minWidth: '80px',
            backgroundColor: playing ? '#3a1a1a' : '#1a3a2a',
            color: playing ? '#f85149' : '#3fb950',
            borderColor: playing ? '#f85149' : '#3fb950',
          }}
        >
          {loading ? 'A carregar…' : playing ? 'Pausar' : currentIndex >= steps.length - 1 ? 'Reiniciar' : 'Reproduzir'}
        </button>

        {/* Step counter */}
        <span style={{ color: '#aaa', fontSize: '13px', fontFamily: 'monospace' }}>
          {steps.length > 0
            ? `${currentIndex + 1} / ${steps.length} steps`
            : loading
              ? '…'
              : 'Sem dados'}
        </span>
      </div>

      {/* Progress bar */}
      {steps.length > 0 && (
        <div
          style={{
            height: '4px',
            backgroundColor: '#2a2a2a',
            borderRadius: '2px',
            marginBottom: '16px',
            cursor: 'pointer',
          }}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            const pct = (e.clientX - rect.left) / rect.width
            setCurrentIndex(Math.round(pct * (steps.length - 1)))
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${progressPct}%`,
              backgroundColor: '#58a6ff',
              borderRadius: '2px',
              transition: 'width 0.04s linear',
            }}
          />
        </div>
      )}

      {error && (
        <div
          style={{
            backgroundColor: '#3a1a1a',
            border: '1px solid #f85149',
            borderRadius: '6px',
            padding: '12px',
            color: '#f85149',
            marginBottom: '12px',
            fontSize: '14px',
          }}
        >
          {error}
        </div>
      )}

      {/* Main content: canvas + info panel */}
      <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div
          ref={containerRef}
          style={{
            flex: '1 1 400px',
            maxWidth: '620px',
            borderRadius: '8px',
            overflow: 'hidden',
            border: '1px solid #2a2a2a',
            backgroundColor: '#2a2a2a',
          }}
        >
          <canvas
            ref={canvasRef}
            style={{ display: 'block', width: '100%', imageRendering: 'pixelated' }}
          />
        </div>

        <InfoPanel step={currentStep} />
      </div>
    </div>
  )
}

// ─── Shared styles ────────────────────────────────────────────────────────────

const btnStyle: React.CSSProperties = {
  backgroundColor: '#2a2a2a',
  color: '#e0e0e0',
  border: '1px solid #444',
  borderRadius: '6px',
  padding: '6px 12px',
  fontSize: '14px',
  cursor: 'pointer',
}

const inputStyle: React.CSSProperties = {
  backgroundColor: '#1e1e1e',
  border: '1px solid #444',
  borderRadius: '6px',
  color: '#e0e0e0',
  padding: '6px 8px',
  fontSize: '14px',
}
