import * as cheerio from 'cheerio'
import sanitizeHtml from 'sanitize-html'
import setCookieParser from 'set-cookie-parser'

const BASE = 'https://lasallesanrafael.sallenet.org'
const LOGIN_URL = `${BASE}/login/index.php`
const PARENTS_URL = `${BASE}/mod/sallenet/modulos/padres/`

class HttpError extends Error {
  constructor(status, code, message, details) {
    super(message)
    this.status = status
    this.code = code
    this.details = details
  }
}

class Session {
  constructor() {
    this.cookies = new Map()
  }

  cookieHeader() {
    return [...this.cookies.entries()].map(([name, value]) => `${name}=${value}`).join('; ')
  }

  absorbCookies(headers) {
    let raw = []
    if (typeof headers.getSetCookie === 'function') raw = headers.getSetCookie()
    if (!raw?.length) {
      const combined = headers.get('set-cookie')
      if (combined) raw = setCookieParser.splitCookiesString(combined)
    }
    for (const parsed of setCookieParser.parse(raw, { map: false })) {
      if (!parsed?.name) continue
      if (parsed.expires && parsed.expires.getTime() < Date.now()) this.cookies.delete(parsed.name)
      else this.cookies.set(parsed.name, parsed.value || '')
    }
  }

  async request(url, options = {}) {
    let currentUrl = url
    let method = options.method || 'GET'
    let body = options.body
    const baseHeaders = { ...(options.headers || {}) }

    for (let redirectCount = 0; redirectCount < 8; redirectCount += 1) {
      const headers = new Headers(baseHeaders)
      headers.set('user-agent', 'AgendaPWA/0.1 (+private Sallenet client)')
      headers.set('accept-language', 'es-ES,es;q=0.9,en;q=0.7')
      const cookie = this.cookieHeader()
      if (cookie) headers.set('cookie', cookie)

      const response = await fetch(currentUrl, {
        method,
        body,
        headers,
        redirect: 'manual',
        signal: AbortSignal.timeout(20000),
      })
      this.absorbCookies(response.headers)

      if (![301, 302, 303, 307, 308].includes(response.status)) return response
      const location = response.headers.get('location')
      if (!location) return response

      currentUrl = new URL(location, currentUrl).toString()
      if (response.status === 303 || ((response.status === 301 || response.status === 302) && method === 'POST')) {
        method = 'GET'
        body = undefined
        delete baseHeaders['content-type']
        delete baseHeaders['Content-Type']
      }
    }
    throw new HttpError(502, 'TOO_MANY_REDIRECTS', 'Sallenet ha devuelto demasiadas redirecciones.')
  }
}

function cleanText(value) {
  return String(value || '').replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim()
}

function cleanLabel(value) {
  return cleanText(value)
    .replace(/^asignatura:\s*/i, '')
    .replace(/^NCA\s*\d[ºo]\s*EP\)\s*/i, '')
    .replace(/^\d[ºo][A-Z]?\)\s*/i, '')
}

function cleanFragment(html) {
  return sanitizeHtml(String(html || ''), {
    allowedTags: ['br', 'p', 'strong', 'b', 'em', 'i', 'u', 'ul', 'ol', 'li', 'a', 'span'],
    allowedAttributes: { a: ['href', 'target', 'rel'] },
    allowedSchemes: ['http', 'https', 'mailto'],
    transformTags: {
      a: (tagName, attribs) => ({
        tagName,
        attribs: { ...attribs, target: '_blank', rel: 'noopener noreferrer' },
      }),
    },
  }).trim()
}

function directChildren($, element, tag) {
  return $(element).children(tag).toArray()
}

function parseDateHeader($, tr) {
  const ths = directChildren($, tr, 'th')
  const tds = directChildren($, tr, 'td')
  if (ths.length !== 1 || tds.length !== 0) return null
  const text = cleanText($(ths[0]).text())
  return text.match(/\b([0-3]?\d\/[01]?\d(?:\/\d{2,4})?)\b/i)?.[1] || null
}

function dayLabel(dateText) {
  const match = String(dateText).match(/(\d{1,2})\/(\d{1,2})(?:\/(\d{2,4}))?/)
  if (!match) return dateText
  let year = Number(match[3] || new Date().getFullYear())
  if (year < 100) year += 2000
  const date = new Date(Date.UTC(year, Number(match[2]) - 1, Number(match[1])))
  const weekday = new Intl.DateTimeFormat('es-ES', { weekday: 'long', timeZone: 'UTC' }).format(date)
  return weekday.charAt(0).toUpperCase() + weekday.slice(1)
}

function parseAgendaTable(tableHtml) {
  const $ = cheerio.load(tableHtml)
  const table = $('table').first()
  if (!table.length) return []

  const days = []
  let currentDay = null
  let vertical = null
  let matrixHeaders = null
  let taskCols = null
  let planCols = null
  let noteCols = null

  const ensureDay = () => {
    if (!currentDay) {
      currentDay = { date: 'Día', label: 'Día', subjects: [] }
      days.push(currentDay)
    }
  }

  const pushVertical = () => {
    if (!vertical || !currentDay) {
      vertical = null
      return
    }
    if (vertical.subject || vertical.tasks || vertical.planning || vertical.notes) currentDay.subjects.push(vertical)
    vertical = null
  }

  const flushMatrix = () => {
    if (!matrixHeaders || !currentDay) {
      matrixHeaders = taskCols = planCols = noteCols = null
      return
    }
    const count = matrixHeaders.length
    for (let i = 0; i < count; i += 1) {
      const subject = {
        subject: cleanLabel(matrixHeaders[i]),
        tasks: taskCols?.[i] || '',
        planning: planCols?.[i] || '',
        notes: noteCols?.[i] || '',
      }
      if (subject.subject || subject.tasks || subject.planning || subject.notes) currentDay.subjects.push(subject)
    }
    matrixHeaders = taskCols = planCols = noteCols = null
  }

  table.find('tr').each((_, tr) => {
    const date = parseDateHeader($, tr)
    if (date) {
      flushMatrix()
      pushVertical()
      currentDay = { date, label: dayLabel(date), subjects: [] }
      days.push(currentDay)
      return
    }

    ensureDay()
    const classes = ($(tr).attr('class') || '').split(/\s+/).filter(Boolean)
    const ths = directChildren($, tr, 'th')
    const tds = directChildren($, tr, 'td')

    if (classes.includes('alert') || classes.includes('alert-info')) {
      flushMatrix()
      pushVertical()
      matrixHeaders = ths.map((th) => cleanText($(th).text()))
      taskCols = planCols = noteCols = null
      return
    }

    const labelRaw = ths.length ? cleanText($(ths[0]).text()) : ''
    const label = cleanLabel(labelRaw).toLowerCase()
    const cells = tds.map((td) => cleanFragment($(td).html() || ''))

    if (matrixHeaders && ths.length) {
      if (label === 'tareas') {
        taskCols = cells
        return
      }
      if (label === 'planificación' || label === 'planificacion') {
        planCols = cells
        return
      }
      if (label === 'notas personales' || label === 'notas' || label === 'observaciones') {
        noteCols = cells
        return
      }
      flushMatrix()
    }

    if (ths.length) {
      if (['tareas', 'planificación', 'planificacion', 'notas personales', 'notas', 'observaciones'].includes(label)) {
        if (!vertical) vertical = { subject: 'General', tasks: '', planning: '', notes: '' }
        const value = cells[0] || ''
        if (label === 'tareas') vertical.tasks = [vertical.tasks, value].filter(Boolean).join('<br>')
        else if (label === 'planificación' || label === 'planificacion') vertical.planning = [vertical.planning, value].filter(Boolean).join('<br>')
        else vertical.notes = [vertical.notes, value].filter(Boolean).join('<br>')
        return
      }

      pushVertical()
      vertical = {
        subject: cleanLabel(labelRaw),
        tasks: cells[0] || '',
        planning: cells[1] || '',
        notes: cells[2] || '',
      }
      return
    }

    if (tds.length && vertical) {
      const value = cells[0] || ''
      vertical.tasks = [vertical.tasks, value].filter(Boolean).join('<br>')
    }
  })

  flushMatrix()
  pushVertical()
  return days.filter((day) => day.subjects.length)
}

function resolveElementUrl(element, baseUrl) {
  const candidates = [element.attr('href'), element.attr('data-href'), element.attr('data-url'), element.attr('onclick')]
    .filter(Boolean)

  for (const candidate of candidates) {
    const trimmed = candidate.trim()
    if (/^https?:\/\//i.test(trimmed) || trimmed.startsWith('/')) {
      try { return new URL(trimmed, baseUrl).toString() } catch { /* continue */ }
    }
    const urls = [...trimmed.matchAll(/['"]([^'"]*(?:\.php|\/mod\/|\/modulos\/)[^'"]*)['"]/gi)]
    for (const match of urls) {
      try { return new URL(match[1], baseUrl).toString() } catch { /* continue */ }
    }
  }
  return null
}

function findUrlByText(html, text, baseUrl) {
  const $ = cheerio.load(html)
  const wanted = text.toLocaleLowerCase('es')
  let found = null
  $('a,button').each((_, el) => {
    if (found) return
    if (cleanText($(el).text()).toLocaleLowerCase('es').includes(wanted)) {
      found = resolveElementUrl($(el), baseUrl)
    }
  })
  return found
}

function findAgendaTable(html) {
  const $ = cheerio.load(html)
  const table = $('#div_padres table.table.table-bordered').first().length
    ? $('#div_padres table.table.table-bordered').first()
    : $('table.table.table-bordered').first()
  return table.length ? $.html(table) : null
}

async function login(session, username, password) {
  const loginPage = await session.request(LOGIN_URL)
  const html = await loginPage.text()
  const $ = cheerio.load(html)
  const form = $('form').filter((_, formEl) => $(formEl).find('input[name="username"]').length).first()
  if (!form.length) throw new HttpError(502, 'LOGIN_FORM_NOT_FOUND', 'No encuentro el formulario de acceso de Sallenet.')

  const payload = new URLSearchParams()
  form.find('input[type="hidden"]').each((_, input) => {
    const name = $(input).attr('name')
    if (name) payload.set(name, $(input).attr('value') || '')
  })
  payload.set('username', username)
  payload.set('password', password)
  const action = new URL(form.attr('action') || LOGIN_URL, LOGIN_URL).toString()

  const result = await session.request(action, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: payload.toString(),
  })
  const resultHtml = await result.text()
  const finalUrl = result.url || action
  const parsed = cheerio.load(resultHtml)

  if (parsed('input[name="username"]').length || /login\/index\.php/i.test(finalUrl)) {
    throw new HttpError(401, 'INVALID_CREDENTIALS', 'Usuario o contraseña incorrectos, o Sallenet ha rechazado el acceso.')
  }
  if (/admin\/tool\/policy/i.test(finalUrl) || cleanText(parsed('body').text()).includes('aceptar nuestras políticas')) {
    throw new HttpError(409, 'POLICY_REQUIRED', 'Sallenet te pide aceptar sus políticas. Entra una vez en Sallenet desde el navegador, acéptalas y vuelve a intentarlo.')
  }
}

async function loadWeeklyAgenda(session) {
  const parentsResponse = await session.request(PARENTS_URL)
  const parentsHtml = await parentsResponse.text()
  const workUrl = findUrlByText(parentsHtml, 'Trabajo', PARENTS_URL)

  if (!workUrl) {
    const existingTable = findAgendaTable(parentsHtml)
    if (existingTable) return parseAgendaTable(existingTable)
    throw new HttpError(502, 'WORK_LINK_NOT_FOUND', 'He iniciado sesión, pero no he podido localizar el apartado Trabajo de Sallenet. Necesito ajustar cómo carga ese botón.')
  }

  const workResponse = await session.request(workUrl, { headers: { referer: PARENTS_URL } })
  let workHtml = await workResponse.text()
  let workBase = workResponse.url || workUrl

  const weeklyUrl = findUrlByText(workHtml, 'Vista semanal', workBase)
  if (weeklyUrl) {
    const weeklyResponse = await session.request(weeklyUrl, { headers: { referer: workBase } })
    workHtml = await weeklyResponse.text()
    workBase = weeklyResponse.url || weeklyUrl
  }

  const tableHtml = findAgendaTable(workHtml)
  if (!tableHtml) {
    throw new HttpError(502, 'AGENDA_TABLE_NOT_FOUND', 'He llegado al apartado Trabajo, pero no encuentro la tabla de la agenda. Puede que Sallenet la cargue por JavaScript y tengamos que replicar esa petición.')
  }

  const days = parseAgendaTable(tableHtml)
  if (!days.length) {
    throw new HttpError(502, 'AGENDA_PARSE_EMPTY', 'He encontrado la tabla de Sallenet, pero no he podido interpretar su contenido.')
  }
  return days
}

export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store, max-age=0')
  res.setHeader('Pragma', 'no-cache')

  if (req.method !== 'POST') return res.status(405).json({ code: 'METHOD_NOT_ALLOWED', message: 'Usa POST.' })

  const origin = req.headers.origin
  const host = req.headers.host
  if (origin && host) {
    try {
      if (new URL(origin).host !== host) return res.status(403).json({ code: 'BAD_ORIGIN', message: 'Origen no permitido.' })
    } catch {
      return res.status(403).json({ code: 'BAD_ORIGIN', message: 'Origen no permitido.' })
    }
  }

  const username = String(req.body?.username || '').trim()
  const password = String(req.body?.password || '')
  if (!username || !password || username.length > 200 || password.length > 500) {
    return res.status(400).json({ code: 'MISSING_CREDENTIALS', message: 'Introduce usuario y contraseña.' })
  }

  try {
    const session = new Session()
    await login(session, username, password)
    const days = await loadWeeklyAgenda(session)
    return res.status(200).json({ days, source: 'Sallenet' })
  } catch (error) {
    if (error instanceof HttpError) {
      return res.status(error.status).json({ code: error.code, message: error.message })
    }
    const timeout = error?.name === 'TimeoutError' || error?.name === 'AbortError'
    return res.status(timeout ? 504 : 502).json({
      code: timeout ? 'SALLENET_TIMEOUT' : 'SALLENET_ERROR',
      message: timeout ? 'Sallenet está tardando demasiado en responder.' : 'No he podido completar la consulta a Sallenet.',
    })
  }
}
