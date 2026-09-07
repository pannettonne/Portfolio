import { useEffect, useMemo, useState } from 'react'
import { clearCredentials, loadCredentials, saveCredentials } from './secureStore.js'

const CACHE_KEY = 'agenda-sallenet:last-agenda'

function normalizeDate(value) {
  const match = String(value || '').match(/(\d{1,2})\/(\d{1,2})(?:\/(\d{2,4}))?/)
  if (!match) return null
  const [, d, m, y] = match
  const now = new Date()
  let year = y ? Number(y) : now.getFullYear()
  if (year < 100) year += 2000
  return `${String(d).padStart(2, '0')}/${String(m).padStart(2, '0')}/${year}`
}

function todayMadrid() {
  const parts = new Intl.DateTimeFormat('es-ES', {
    timeZone: 'Europe/Madrid', day: '2-digit', month: '2-digit', year: 'numeric'
  }).formatToParts(new Date())
  const get = (type) => parts.find((p) => p.type === type)?.value
  return `${get('day')}/${get('month')}/${get('year')}`
}

function SafeHtml({ html }) {
  if (!html) return <span className="muted">—</span>
  return <div className="rich-text" dangerouslySetInnerHTML={{ __html: html }} />
}

function DayCard({ day }) {
  return (
    <section className="day-card">
      <div className="day-heading">
        <span>{day.label || day.date}</span>
        <span className="date-chip">{day.date}</span>
      </div>
      <div className="subjects">
        {day.subjects.map((subject, index) => (
          <article className="subject-card" key={`${subject.subject}-${index}`}>
            <h3>{subject.subject || 'General'}</h3>
            {subject.tasks && (
              <div className="field">
                <div className="field-label">Tareas</div>
                <SafeHtml html={subject.tasks} />
              </div>
            )}
            {subject.planning && (
              <div className="field">
                <div className="field-label">Planificación</div>
                <SafeHtml html={subject.planning} />
              </div>
            )}
            {subject.notes && (
              <div className="field personal-notes">
                <div className="field-label">Notas personales</div>
                <SafeHtml html={subject.notes} />
              </div>
            )}
          </article>
        ))}
      </div>
    </section>
  )
}

export default function App() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [remember, setRemember] = useState(true)
  const [agenda, setAgenda] = useState(() => {
    try { return JSON.parse(localStorage.getItem(CACHE_KEY)) } catch { return null }
  })
  const [view, setView] = useState('today')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [credentialsLoaded, setCredentialsLoaded] = useState(false)

  useEffect(() => {
    loadCredentials().then((saved) => {
      if (saved) {
        setUsername(saved.username || '')
        setPassword(saved.password || '')
        setRemember(true)
      }
    }).finally(() => setCredentialsLoaded(true))
  }, [])

  const today = todayMadrid()
  const todayDay = useMemo(
    () => agenda?.days?.find((day) => normalizeDate(day.date) === today),
    [agenda, today],
  )

  async function refreshAgenda(event) {
    event?.preventDefault()
    setError('')
    setLoading(true)
    try {
      const response = await fetch('/api/agenda', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        body: JSON.stringify({ username: username.trim(), password }),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(data.message || 'No se pudo consultar Sallenet.')

      const savedAgenda = { ...data, cachedAt: new Date().toISOString() }
      setAgenda(savedAgenda)
      localStorage.setItem(CACHE_KEY, JSON.stringify(savedAgenda))

      if (remember) await saveCredentials(username.trim(), password)
      else await clearCredentials()
    } catch (err) {
      setError(err.message || 'Ha ocurrido un error inesperado.')
    } finally {
      setLoading(false)
    }
  }

  async function logout() {
    await clearCredentials()
    localStorage.removeItem(CACHE_KEY)
    setAgenda(null)
    setUsername('')
    setPassword('')
    setError('')
  }

  const visibleDays = view === 'today'
    ? (todayDay ? [todayDay] : [])
    : (agenda?.days || [])

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">Sallenet</div>
          <h1>Agenda escolar</h1>
        </div>
        {agenda && <button className="ghost-button" onClick={logout}>Salir</button>}
      </header>

      <section className="hero">
        <div className="hero-mark">A</div>
        <div>
          <strong>La agenda de tu hijo, sin correos.</strong>
          <p>Los datos se consultan con tu propia cuenta y las credenciales no se guardan en el servidor.</p>
        </div>
      </section>

      <form className="login-card" onSubmit={refreshAgenda}>
        <label>
          Usuario de Sallenet
          <input
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Tu usuario"
            required
          />
        </label>
        <label>
          Contraseña
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Tu contraseña"
            required
          />
        </label>
        <label className="remember-row">
          <input
            type="checkbox"
            checked={remember}
            onChange={(e) => setRemember(e.target.checked)}
          />
          <span>Recordarme en este dispositivo</span>
        </label>
        <button className="primary-button" disabled={loading || !credentialsLoaded}>
          {loading ? 'Consultando Sallenet…' : agenda ? 'Actualizar agenda' : 'Ver agenda'}
        </button>
        <p className="privacy-note">Al recordar la cuenta, la PWA cifra las credenciales en el almacenamiento local del dispositivo mediante Web Crypto.</p>
      </form>

      {error && <div className="error-box">{error}</div>}

      {agenda && (
        <section className="agenda-section">
          <div className="agenda-toolbar">
            <div className="tabs" role="tablist">
              <button className={view === 'today' ? 'active' : ''} onClick={() => setView('today')}>Hoy</button>
              <button className={view === 'week' ? 'active' : ''} onClick={() => setView('week')}>Semana</button>
            </div>
            <span className="updated">Actualizado {new Date(agenda.cachedAt).toLocaleString('es-ES')}</span>
          </div>

          {view === 'today' && !todayDay ? (
            <div className="empty-card">No aparece información para hoy ({today}). Puedes revisar la semana completa.</div>
          ) : visibleDays.length ? (
            visibleDays.map((day, index) => <DayCard day={day} key={`${day.date}-${index}`} />)
          ) : (
            <div className="empty-card">Sallenet no devolvió tareas para esta semana.</div>
          )}
        </section>
      )}
    </main>
  )
}
