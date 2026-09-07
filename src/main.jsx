import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

function SubjectCard({ item }) {
  const isNotes = item.kind === 'notes'
  return (
    <article className={isNotes ? 'subject-card notes-card' : 'subject-card'}>
      <div className="subject-title">
        <span>{isNotes ? '📝' : '📘'}</span>
        <strong>{item.subject || (isNotes ? 'Notas personales' : 'General')}</strong>
      </div>

      {item.tasks && (
        <div className="field">
          <span className="label">{isNotes ? 'Contenido' : 'Tareas'}</span>
          <p>{item.tasks}</p>
        </div>
      )}

      {!isNotes && item.planning && (
        <div className="field">
          <span className="label">Planificación</span>
          <p>{item.planning}</p>
        </div>
      )}
    </article>
  )
}

function Login({ onLogin, loading, error }) {
  const [username, setUsername] = useState(() => localStorage.getItem('sallenet-user') || '')
  const [password, setPassword] = useState('')

  function submit(event) {
    event.preventDefault()
    localStorage.setItem('sallenet-user', username)
    onLogin({ username, password })
  }

  return (
    <main className="login-page">
      <section className="login-card">
        <div className="logo">✓</div>
        <div className="eyebrow">La agenda de tu hijo</div>
        <h1>Mi Agenda Sallenet</h1>
        <p className="intro">Accede con las mismas credenciales que utilizas en Sallenet.</p>

        <form onSubmit={submit}>
          <label>
            Usuario
            <input
              name="username"
              autoComplete="username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              required
            />
          </label>

          <label>
            Contraseña
            <input
              name="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
            />
          </label>

          {error && <div className="error">{error}</div>}

          <button className="primary" disabled={loading}>
            {loading ? 'Consultando Sallenet…' : 'Ver agenda'}
          </button>
        </form>

        <p className="privacy-note">
          La contraseña no se guarda en el servidor. Puedes dejar que Safari, Chrome o tu gestor de contraseñas la recuerde en este dispositivo.
        </p>
      </section>
    </main>
  )
}

function Agenda({ data, view, onView, reload, loading, logout }) {
  const days = data?.days || []

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">Agenda escolar</div>
          <h1>Mi Agenda Sallenet</h1>
        </div>
        <button className="ghost" onClick={logout}>Salir</button>
      </header>

      <div className="tabs" role="tablist">
        <button className={view === 'day' ? 'active' : ''} onClick={() => onView('day')}>Hoy</button>
        <button className={view === 'week' ? 'active' : ''} onClick={() => onView('week')}>Semana</button>
      </div>

      {data?.warning && <div className="notice">{data.warning}</div>}

      <section className="agenda">
        {!days.length && !loading && (
          <div className="empty">
            <strong>No he encontrado datos de agenda.</strong>
            <span>Puede que no haya contenido para este periodo o que Sallenet haya cambiado la vista.</span>
          </div>
        )}

        {days.map((day, index) => (
          <section className="day" key={day.date || index}>
            <h2>{day.date || 'Día'}</h2>
            <div className="cards">
              {(day.items || []).map((item, i) => <SubjectCard item={item} key={i} />)}
            </div>
          </section>
        ))}
      </section>

      <button className="refresh" disabled={loading} onClick={reload}>
        {loading ? 'Actualizando…' : '↻ Actualizar agenda'}
      </button>

      <p className="privacy-note">La consulta se hace bajo demanda y la respuesta no se almacena en el servidor.</p>
    </main>
  )
}

function App() {
  const [credentials, setCredentials] = useState(null)
  const [view, setView] = useState('week')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function loadAgenda(creds = credentials, selectedView = view) {
    if (!creds) return false
    setLoading(true)
    setError('')

    try {
      const response = await fetch('/api/agenda', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...creds, view: selectedView })
      })

      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.error || 'No se pudo consultar Sallenet')
      setData(body)
      return true
    } catch (err) {
      setError(err.message || 'Error inesperado')
      return false
    } finally {
      setLoading(false)
    }
  }

  async function login(creds) {
    setCredentials(creds)
    const ok = await loadAgenda(creds, view)
    if (!ok) setCredentials(null)
  }

  function changeView(nextView) {
    setView(nextView)
    loadAgenda(credentials, nextView)
  }

  if (!credentials || !data) return <Login onLogin={login} loading={loading} error={error} />

  return (
    <Agenda
      data={data}
      view={view}
      onView={changeView}
      reload={() => loadAgenda()}
      loading={loading}
      logout={() => {
        setCredentials(null)
        setData(null)
        setError('')
      }}
    />
  )
}

createRoot(document.getElementById('root')).render(<App />)

if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js'))
}
