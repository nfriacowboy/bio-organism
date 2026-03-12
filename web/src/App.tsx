import React from 'react'
import { Routes, Route, Link, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import NovoTreino from './pages/NovoTreino'
import TreinoDetalhe from './pages/TreinoDetalhe'
import Replay from './pages/Replay'

const navStyle: React.CSSProperties = {
  backgroundColor: '#1a1a1a',
  borderBottom: '1px solid #2a2a2a',
  padding: '0 24px',
  display: 'flex',
  alignItems: 'center',
  gap: '24px',
  height: '52px',
}

const logoStyle: React.CSSProperties = {
  fontWeight: 700,
  fontSize: '18px',
  color: '#e0e0e0',
  letterSpacing: '0.02em',
  marginRight: '16px',
  textDecoration: 'none',
}

const activeLinkStyle: React.CSSProperties = {
  color: '#58a6ff',
  fontWeight: 600,
}

const linkStyle: React.CSSProperties = {
  color: '#aaa',
  fontSize: '14px',
}

export default function App() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <nav style={navStyle}>
        <Link to="/" style={logoStyle}>
          SNN Organismo
        </Link>
        <NavLink
          to="/"
          end
          style={({ isActive }) => (isActive ? { ...linkStyle, ...activeLinkStyle } : linkStyle)}
        >
          Dashboard
        </NavLink>
        <NavLink
          to="/treinos/novo"
          style={({ isActive }) => (isActive ? { ...linkStyle, ...activeLinkStyle } : linkStyle)}
        >
          Novo Treino
        </NavLink>
      </nav>
      <main style={{ flex: 1, padding: '24px' }}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/treinos/novo" element={<NovoTreino />} />
          <Route path="/treinos/:id" element={<TreinoDetalhe />} />
          <Route path="/treinos/:id/replay" element={<Replay />} />
        </Routes>
      </main>
    </div>
  )
}
