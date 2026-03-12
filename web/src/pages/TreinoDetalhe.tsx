import React, { useEffect, useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { obterTreino, listarModelos, listarSteps } from '../api'
import type { Treino, Modelo, Step } from '../types'

type EpisodioSummary = {
  episodio: number
  totalSteps: number
  totalComeu: number
}

function buildEpisodioSummaries(steps: Step[]): EpisodioSummary[] {
  const map = new Map<number, EpisodioSummary>()
  for (const s of steps) {
    const existing = map.get(s.episodio)
    if (existing) {
      existing.totalSteps += 1
      if (s.comeu) existing.totalComeu += 1
    } else {
      map.set(s.episodio, {
        episodio: s.episodio,
        totalSteps: 1,
        totalComeu: s.comeu ? 1 : 0,
      })
    }
  }
  return Array.from(map.values()).sort((a, b) => b.episodio - a.episodio)
}

function Section({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <div
      style={{
        marginBottom: '28px',
        backgroundColor: '#1a1a1a',
        border: '1px solid #2a2a2a',
        borderRadius: '8px',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          padding: '12px 18px',
          borderBottom: '1px solid #2a2a2a',
          fontWeight: 700,
          fontSize: '14px',
          color: '#ccc',
          backgroundColor: '#141414',
        }}
      >
        {title}
      </div>
      <div style={{ padding: '16px 18px' }}>{children}</div>
    </div>
  )
}

function KV({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', gap: '16px', marginBottom: '8px', fontSize: '14px' }}>
      <span style={{ color: '#888', minWidth: '160px', flexShrink: 0 }}>{label}</span>
      <span style={{ color: '#e0e0e0', wordBreak: 'break-all' }}>{value ?? '—'}</span>
    </div>
  )
}

const ESTADO_COLORS: Record<string, { bg: string; color: string }> = {
  pendente: { bg: '#3a3a3a', color: '#aaa' },
  a_correr: { bg: '#1a3a5c', color: '#58a6ff' },
  concluido: { bg: '#1a3a2a', color: '#3fb950' },
  erro: { bg: '#3a1a1a', color: '#f85149' },
}

function EstadoBadge({ estado }: { estado: string }) {
  const c = ESTADO_COLORS[estado] ?? ESTADO_COLORS['pendente']
  return (
    <span
      style={{
        backgroundColor: c.bg,
        color: c.color,
        borderRadius: '12px',
        padding: '2px 10px',
        fontSize: '12px',
        fontWeight: 600,
      }}
    >
      {estado}
    </span>
  )
}

export default function TreinoDetalhe() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const treinoId = Number(id)

  const [treino, setTreino] = useState<Treino | null>(null)
  const [modelos, setModelos] = useState<Modelo[]>([])
  const [episodios, setEpisodios] = useState<EpisodioSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function carregar() {
      setLoading(true)
      setError(null)
      try {
        const [t, m, steps] = await Promise.all([
          obterTreino(treinoId),
          listarModelos(treinoId),
          listarSteps(treinoId, undefined, 500),
        ])
        setTreino(t)
        setModelos(m)
        setEpisodios(buildEpisodioSummaries(steps))
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Erro ao carregar treino')
      } finally {
        setLoading(false)
      }
    }
    void carregar()
  }, [treinoId])

  if (loading) {
    return (
      <div style={{ color: '#888', padding: '40px 0', textAlign: 'center' }}>
        A carregar…
      </div>
    )
  }

  if (error || !treino) {
    return (
      <div
        style={{
          backgroundColor: '#3a1a1a',
          border: '1px solid #f85149',
          borderRadius: '6px',
          padding: '16px',
          color: '#f85149',
        }}
      >
        {error ?? 'Treino não encontrado'}
      </div>
    )
  }

  const params = treino.parametros as Record<string, unknown> | null
  const metricas = treino.metricas_finais as Record<string, unknown> | null
  const last10 = episodios.slice(0, 10)

  return (
    <div style={{ maxWidth: '860px' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '28px',
        }}
      >
        <div>
          <div style={{ color: '#888', fontSize: '13px', marginBottom: '4px' }}>
            <Link to="/" style={{ color: '#58a6ff' }}>
              Dashboard
            </Link>{' '}
            / Treino #{treino.id}
          </div>
          <h1 style={{ margin: 0, fontSize: '22px', fontWeight: 700 }}>
            Treino #{treino.id}
            <span style={{ marginLeft: '12px' }}>
              <EstadoBadge estado={treino.estado} />
            </span>
          </h1>
        </div>
        {treino.estado === 'concluido' && (
          <button
            onClick={() => navigate(`/treinos/${treino.id}/replay`)}
            style={{
              backgroundColor: '#1a3a2a',
              color: '#3fb950',
              border: '1px solid #3fb950',
              borderRadius: '6px',
              padding: '8px 18px',
              fontSize: '14px',
              fontWeight: 600,
            }}
          >
            Ver Replay
          </button>
        )}
      </div>

      <Section title="Informacao Geral">
        <KV label="ID" value={treino.id} />
        <KV
          label="Data"
          value={
            treino.data_human || new Date(treino.criado_em).toLocaleString('pt-PT')
          }
        />
        <KV label="Nota" value={treino.nota} />
        <KV label="Conclusao" value={treino.conclusao} />
        <KV label="MLflow Run ID" value={treino.mlflow_run_id} />
      </Section>

      {params && Object.keys(params).length > 0 && (
        <Section title="Parametros">
          {Object.entries(params).map(([k, v]) => (
            <KV key={k} label={k} value={String(v)} />
          ))}
        </Section>
      )}

      {metricas && Object.keys(metricas).length > 0 && (
        <Section title="Metricas Finais">
          {Object.entries(metricas).map(([k, v]) => (
            <KV
              key={k}
              label={k}
              value={typeof v === 'number' ? v.toFixed(4) : String(v)}
            />
          ))}
        </Section>
      )}

      {modelos.length > 0 && (
        <Section title={`Modelos Guardados (${modelos.length})`}>
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Episodio</th>
                <th>Caminho</th>
                <th>Criado Em</th>
              </tr>
            </thead>
            <tbody>
              {modelos.map((m) => (
                <tr key={m.id}>
                  <td style={{ color: '#aaa', fontSize: '12px' }}>#{m.id}</td>
                  <td>{m.episodio}</td>
                  <td
                    style={{
                      fontSize: '12px',
                      color: '#888',
                      maxWidth: '320px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {m.caminho}
                  </td>
                  <td style={{ fontSize: '12px', color: '#666' }}>
                    {new Date(m.criado_em).toLocaleString('pt-PT')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      )}

      {last10.length > 0 && (
        <Section title="Ultimos 10 Episodios (resumo)">
          <table>
            <thead>
              <tr>
                <th>Episodio</th>
                <th>Steps</th>
                <th>Comeu</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {last10.map((ep) => (
                <tr key={ep.episodio}>
                  <td>Ep. {ep.episodio}</td>
                  <td>{ep.totalSteps}</td>
                  <td style={{ color: ep.totalComeu > 0 ? '#3fb950' : '#666' }}>
                    {ep.totalComeu}
                  </td>
                  <td>
                    <Link
                      to={`/treinos/${treino.id}/replay?episodio=${ep.episodio}`}
                      style={{ color: '#58a6ff', fontSize: '13px' }}
                    >
                      Replay
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      )}
    </div>
  )
}
