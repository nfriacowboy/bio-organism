export interface Treino {
  id: number
  data_human: string
  nota: string | null
  conclusao: string | null
  mlflow_run_id: string | null
  estado: 'pendente' | 'a_correr' | 'concluido' | 'erro'
  parametros: Record<string, unknown> | null
  metricas_finais: Record<string, unknown> | null
  criado_em: string
}

export interface Inferencia {
  id: number
  treino_id: number
  seed_mundo: number
  fase_mundo: string | null
  nota: string | null
  estado: string
  criado_em: string
  resultado: Record<string, unknown> | null
}

export interface Step {
  id: number
  treino_id: number
  episodio: number
  step: number
  timestamp_ms: number
  pos_x: number
  pos_y: number
  angulo: number
  energia_actual: number
  energia_max: number
  visao: number[]
  accao: number
  modo: string
  novidade: number
  recompensa: number
  bias_explorar: number
  bias_exploitar: number
  comeu: boolean
  fase_mundo: string
}

export interface Modelo {
  id: number
  treino_id: number
  episodio: number
  caminho: string
  criado_em: string
}
