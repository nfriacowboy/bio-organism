import type { Treino, Inferencia, Step, Modelo } from './types'

const BASE = '/api'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`HTTP ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

export function listarTreinos(): Promise<Treino[]> {
  return request<Treino[]>('/treinos')
}

export function criarTreino(data: {
  nota?: string
  parametros?: Record<string, unknown>
}): Promise<Treino> {
  return request<Treino>('/treinos', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export function obterTreino(id: number): Promise<Treino> {
  return request<Treino>(`/treinos/${id}`)
}

export function atualizarTreino(
  id: number,
  data: Partial<Treino>
): Promise<Treino> {
  return request<Treino>(`/treinos/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  })
}

export function listarModelos(treinoId: number): Promise<Modelo[]> {
  return request<Modelo[]>(`/treinos/${treinoId}/modelos`)
}

export function listarSteps(
  treinoId: number,
  episodio?: number,
  limit?: number
): Promise<Step[]> {
  const params = new URLSearchParams()
  if (episodio !== undefined) params.set('episodio', String(episodio))
  if (limit !== undefined) params.set('limit', String(limit))
  const qs = params.toString() ? `?${params.toString()}` : ''
  return request<Step[]>(`/treinos/${treinoId}/steps${qs}`)
}

export function listarInferencias(): Promise<Inferencia[]> {
  return request<Inferencia[]>('/inferencias')
}

export function criarInferencia(data: {
  treino_id: number
  seed_mundo: number
  fase_mundo?: string
  nota?: string
}): Promise<Inferencia> {
  return request<Inferencia>('/inferencias', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export function obterInferencia(infId: number): Promise<Inferencia> {
  return request<Inferencia>(`/inferencias/${infId}`)
}
