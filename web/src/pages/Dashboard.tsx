import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listarTreinos } from '../api'
import type { Treino } from '../types'

type EstadoBadgeProps = { estado: Treino['estado'] }

const ESTADO_COLORS: Record<Treino['estado'], { bg: string; color: string }> = {
  pendente: { bg: '#3a3a3a', color: '#aaa' },
  a_correr: { bg: '#1a3a5c', color: '#58a6ff' },
  concluido: { bg: '#1a3a2a', color: '#3fb950' },
  erro: { bg: '#3a1a1a', color: '#f85149' },
}

function EstadoBadge({ estado }: EstadoBadgeProps) {
  const c = ESTADO_COLORS[estado] ?? ESTADO_COLORS.pendente
  return (
    <span
      style={{
        backgroundColor: c.bg,
        color: c.color,
        borderRadius: '12px',
        padding: '2px 10px',
        fontSize: '12px',
        fontWeight: 600,
        display: 'inline-block',
        whiteSpace: 'nowrap',
      }}
    >
      {estado}
    </span>
  )
}

function metricaComida(treino: Treino): string {
  if (!treino.metricas_finais) return '—'
  const mf = treino.metricas_finais as Record<string, unknown>
  // Try various possible keys for food-related metric
  for (const key of ['media_comida', 'avg_food', 'comida_media', 'total_comida']) {
    if (key in mf && mf[key] !== undefined) {
      const v = mf[key]
      return typeof v === 'number' ? v.toFixed(2) : String(v)
    }
  }
  // Return first numeric metric as fallback
  for (const v of Object.values(mf)) {
    if (typeof v === 'number') return v.toFixed(2)
  }
  return '—'
}

export default function Dashboard() {
  const [treinos, setTreinos] = useState<Treino[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function carregar() {
    setLoading(true)
    setError(null)
    try {
      const data = await listarTreinos()
      setTreinos(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Erro ao carregar treinos')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void carregar()
  }, [])

  return (
    <div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '24px',
        }}
      >
        <h1 style={{ margin: 0, fontSize: '22px', fontWeight: 700 }}>Treinos</h1>
        <div style={{ display: 'flex', gap: '12px' }}>
          <button
            onClick={() => void carregar()}
            disabled={loading}
            style={{
              backgroundColor: '#2a2a2a',
              color: '#e0e0e0',
              border: '1px solid #444',
              borderRadius: '6px',
              padding: '7px 16px',
              fontSize: '14px',
            }}
          >
            {loading ? 'A carregar…' : 'Atualizar'}
          </button>
          <Link
            to="/treinos/novo"
            style={{
              backgroundColor: '#238636',
              color: '#fff',
              borderRadius: '6px',
              padding: '7px 16px',
              fontSize: '14px',
              fontWeight: 600,
              display: 'inline-block',
            }}
          >
            + Novo Treino
          </Link>
        </div>
      </div>

      {error && (
        <div
          style={{
            backgroundColor: '#3a1a1a',
            border: '1px solid #f85149',
            borderRadius: '6px',
            padding: '12px 16px',
            color: '#f85149',
            marginBottom: '16px',
          }}
        >
          {error}
        </div>
      )}

      {!loading && treinos.length === 0 && !error && (
        <div
          style={{
            textAlign: 'center',
            padding: '60px 20px',
            color: '#666',
          }}
        >
          <p style={{ fontSize: '16px' }}>Nenhum treino encontrado.</p>
          <Link to="/treinos/novo" style={{ color: '#58a6ff' }}>
            Criar o primeiro treino
          </Link>
        </div>
      )}

      {treinos.length > 0 && (
        <div
          style={{
            backgroundColor: '#1a1a1a',
            border: '1px solid #2a2a2a',
            borderRadius: '8px',
            overflow: 'hidden',
          }}
        >
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Data</th>
                <th>Nota</th>
                <th>Estado</th>
                <th>Métrica</th>
                <th>MLflow</th>
                <th>Ações</th>
              </tr>
            </thead>
            <tbody>
              {treinos.map((t) => (
                <tr key={t.id}>
                  <td style={{ color: '#aaa', fontSize: '13px' }}>#{t.id}</td>
                  <td style={{ fontSize: '13px', whiteSpace: 'nowrap' }}>
                    {t.data_human || new Date(t.criado_em).toLocaleString('pt-PT')}
                  </td>
                  <td style={{ color: t.nota ? '#e0e0e0' : '#555', maxWidth: '220px' }}>
                    <span
                      style={{
                        display: 'block',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {t.nota ?? '—'}
                    </span>
                  </td>
                  <td>
                    <EstadoBadge estado={t.estado} />
                  </td>
                  <td style={{ fontSize: '13px', color: '#aaa' }}>{metricaComida(t)}</td>
                  <td style={{ fontSize: '12px', color: '#666', maxWidth: '140px' }}>
                    <span
                      style={{
                        display: 'block',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {t.mlflow_run_id ?? '—'}
                    </span>
                  </td>
                  <td>
                    <Link
                      to={`/treinos/${t.id}`}
                      style={{
                        color: '#58a6ff',
                        fontSize: '13px',
                        marginRight: '12px',
                      }}
                    >
                      Detalhes
                    </Link>
                    {t.estado === 'concluido' && (
                      <Link
                        to={`/treinos/${t.id}/replay`}
                        style={{ color: '#3fb950', fontSize: '13px' }}
                      >
                        Replay
                      </Link>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
