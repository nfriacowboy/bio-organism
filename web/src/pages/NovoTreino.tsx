import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { criarTreino } from '../api'

const labelStyle: React.CSSProperties = {
  display: 'block',
  marginBottom: '6px',
  fontSize: '13px',
  fontWeight: 600,
  color: '#ccc',
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  fontSize: '14px',
  backgroundColor: '#1e1e1e',
  border: '1px solid #444',
  borderRadius: '6px',
  color: '#e0e0e0',
}

const fieldStyle: React.CSSProperties = {
  marginBottom: '20px',
}

export default function NovoTreino() {
  const navigate = useNavigate()
  const [nota, setNota] = useState('')
  const [nAmostras, setNAmostras] = useState(5000)
  const [epochs, setEpochs] = useState(30)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      const treino = await criarTreino({
        nota: nota.trim() || undefined,
        parametros: {
          n_amostras: nAmostras,
          epochs,
        },
      })
      navigate(`/treinos/${treino.id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro ao criar treino')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div style={{ maxWidth: '520px' }}>
      <h1 style={{ marginTop: 0, fontSize: '22px', fontWeight: 700, marginBottom: '28px' }}>
        Novo Treino
      </h1>

      {error && (
        <div
          style={{
            backgroundColor: '#3a1a1a',
            border: '1px solid #f85149',
            borderRadius: '6px',
            padding: '12px 16px',
            color: '#f85149',
            marginBottom: '20px',
            fontSize: '14px',
          }}
        >
          {error}
        </div>
      )}

      <form onSubmit={(e) => void handleSubmit(e)}>
        <div style={fieldStyle}>
          <label htmlFor="nota" style={labelStyle}>
            Nota (opcional)
          </label>
          <input
            id="nota"
            type="text"
            value={nota}
            onChange={(e) => setNota(e.target.value)}
            placeholder="Descreva este treino…"
            style={inputStyle}
          />
        </div>

        <div style={fieldStyle}>
          <label htmlFor="n_amostras" style={labelStyle}>
            N.º de Amostras
          </label>
          <input
            id="n_amostras"
            type="number"
            min={1}
            value={nAmostras}
            onChange={(e) => setNAmostras(Number(e.target.value))}
            style={inputStyle}
          />
          <span style={{ fontSize: '12px', color: '#666', marginTop: '4px', display: 'block' }}>
            Número total de amostras de treino
          </span>
        </div>

        <div style={fieldStyle}>
          <label htmlFor="epochs" style={labelStyle}>
            Epochs
          </label>
          <input
            id="epochs"
            type="number"
            min={1}
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
            style={inputStyle}
          />
          <span style={{ fontSize: '12px', color: '#666', marginTop: '4px', display: 'block' }}>
            Número de epochs de treino
          </span>
        </div>

        <div style={{ display: 'flex', gap: '12px', marginTop: '8px' }}>
          <button
            type="submit"
            disabled={submitting}
            style={{
              backgroundColor: submitting ? '#1a3a2a' : '#238636',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              padding: '9px 20px',
              fontSize: '14px',
              fontWeight: 600,
              opacity: submitting ? 0.7 : 1,
            }}
          >
            {submitting ? 'A criar…' : 'Criar Treino'}
          </button>
          <button
            type="button"
            onClick={() => navigate('/')}
            style={{
              backgroundColor: '#2a2a2a',
              color: '#e0e0e0',
              border: '1px solid #444',
              borderRadius: '6px',
              padding: '9px 20px',
              fontSize: '14px',
            }}
          >
            Cancelar
          </button>
        </div>
      </form>
    </div>
  )
}
