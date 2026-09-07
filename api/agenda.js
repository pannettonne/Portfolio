import * as cheerio from 'cheerio'

const BASE = 'https://lasallesanrafael.sallenet.org'
const LOGIN_URL = BASE + '/login/index.php'
const PARENTS_URL = BASE + '/mod/sallenet/modulos/padres/'

class CookieJar {
  constructor() {
    this.cookies = new Map()
  }

  absorb(headers) {
    let values = []
    if (typeof headers.getSetCookie === 'function') {
      values = headers.getSetCookie()
    } else {
      const raw = headers.get('set-cookie')
      if (raw) values = raw.split(/,(?=\s*[^;,]+=)/)
    }

    for (const value of values) {
      const pair = value.split(';', 1)[0]
      const idx = pair.indexOf('=')
      if (idx > 0) {
        this.cookies.set(pair.slice(0, idx).trim(), pair.slice(idx + 1).trim())
      }
    }
  }

  header() {
    return [...this.cookies.entries()].map(([k, v]) => `${k}=${v}`).join('; ')
  }
}

async function fetchWithSession(url, options, jar, depth = 0) {
  if (depth > 8) throw new Error('Demasiadas redirecciones')

  const headers = new Headers(options?.headers || {})
  const cookieHeader = jar.header()
  if (cookieHeader) headers.set('Cookie', cookieHeader)
  headers.set('User-Agent', 'Mozilla/5.0 AgendaSallenetPWA/0.1')

  const response = await fetch(url, {
    ...(options || {}),
    headers,
    redirect: 'manual'
  })

  jar.absorb(response.headers)

  if ([301, 302, 303, 307, 308].includes(response.status)) {
    const location = response.headers.get('location')
    if (!location) return response

    const nextUrl = new URL(location, url).toString()
    const switchToGet =
      response.status === 303 ||
      ((response.status === 301 || response.status === 302) && options?.method === 'POST')

    return fetchWithSession(
      nextUrl,
      switchToGet ? { method: 'GET' } : options,
      jar,
      depth + 1
    )
  }

  return response
}

function normalize(value = '') {
  return value.replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim()
}

function absoluteHref(href, base = BASE) {
  if (!href || href.startsWith('javascript:') || href === '#') return null
  try {
    return new URL(href, base).toString()
  } catch {
    return null
  }
}

function findLinkByText(html, wantedText) {
  const $ = cheerio.load(html)
  const wanted = wantedText.toLowerCase()
  let result = null

  $('a').each((_, el) => {
    if (result) return
    const label = normalize($(el).text()).toLowerCase()
    if (label.includes(wanted)) result = absoluteHref($(el).attr('href'))
  })

  return result
}

function directChildren($, row, tag) {
  return $(row).children(tag).toArray()
}

function textOf($, el) {
  return normalize($(el).text())
}

function cleanSubject(value) {
  return normalize(value)
    .replace(/^asignatura:\s*/i, '')
    .replace(/^NCA\s*\d+[ºo]?\s*EP\)\s*/i, '')
    .replace(/^\d+[ºo]?[A-Z]?\)\s*/i, '')
}

function parseAgenda(html) {
  const $ = cheerio.load(html)
  const table = $('#div_padres table.table-bordered').first().length
    ? $('#div_padres table.table-bordered').first()
    : $('table.table-bordered').first()

  if (!table.length) return []

  const days = []
  let day = null
  let current = null
  let matrixHeaders = null
  let matrixTasks = null
  let matrixPlanning = null

  const ensureDay = () => {
    if (!day) {
      day = { date: 'Agenda', items: [] }
      days.push(day)
    }
    return day
  }

  const pushCurrent = () => {
    if (!current) return
    if (current.subject || current.tasks || current.planning) {
      ensureDay().items.push(current)
    }
    current = null
  }

  const flushMatrix = () => {
    if (!matrixHeaders?.length) return

    matrixHeaders.forEach((subject, index) => {
      const item = {
        subject: cleanSubject(subject) || 'General',
        tasks: matrixTasks?.[index] || '',
        planning: matrixPlanning?.[index] || '',
        kind: 'subject'
      }
      if (item.subject || item.tasks || item.planning) ensureDay().items.push(item)
    })

    matrixHeaders = null
    matrixTasks = null
    matrixPlanning = null
  }

  table.find('tr').each((_, row) => {
    const ths = directChildren($, row, 'th')
    const tds = directChildren($, row, 'td')
    const rowClass = ($(row).attr('class') || '').toLowerCase()

    if (ths.length === 1 && tds.length === 0) {
      const label = textOf($, ths[0])
      const match = label.match(/\b([0-3]?\d\/[01]?\d(?:\/\d{2,4})?)\b/)
      if (match) {
        flushMatrix()
        pushCurrent()
        day = { date: match[1], items: [] }
        days.push(day)
        return
      }
    }

    ensureDay()

    if (rowClass.includes('alert') && rowClass.includes('alert-info') && ths.length) {
      flushMatrix()
      pushCurrent()
      matrixHeaders = ths.map(th => textOf($, th)).filter(Boolean)
      return
    }

    if (matrixHeaders && ths.length) {
      const label = textOf($, ths[0]).toLowerCase()
      const cells = tds.map(td => textOf($, td))

      if (label === 'tareas') {
        matrixTasks = cells
        return
      }
      if (label === 'planificación' || label === 'planificacion') {
        matrixPlanning = cells
        return
      }

      flushMatrix()
    }

    if (!ths.length) {
      if (current && tds.length) {
        const extra = tds.map(td => textOf($, td)).filter(Boolean).join(' · ')
        if (extra) current.tasks = [current.tasks, extra].filter(Boolean).join('\n')
      }
      return
    }

    const rawLabel = textOf($, ths[0])
    const label = rawLabel.toLowerCase()
    const cells = tds.map(td => textOf($, td)).filter(Boolean)

    if (['notas personales', 'notas', 'observaciones'].includes(label)) {
      pushCurrent()
      const content = cells.join('\n')
      if (content) {
        day.items.push({
          subject: 'Notas personales',
          tasks: content,
          planning: '',
          kind: 'notes'
        })
      }
      return
    }

    if (label === 'tareas' || label === 'planificación' || label === 'planificacion') {
      if (!current) {
        current = { subject: 'General', tasks: '', planning: '', kind: 'subject' }
      }
      const content = cells.join('\n')
      if (label === 'tareas') {
        current.tasks = [current.tasks, content].filter(Boolean).join('\n')
      } else {
        current.planning = [current.planning, content].filter(Boolean).join('\n')
      }
      return
    }

    pushCurrent()
    current = {
      subject: cleanSubject(rawLabel),
      tasks: cells[0] || '',
      planning: cells[1] || '',
      kind: 'subject'
    }
  })

  flushMatrix()
  pushCurrent()

  return days.filter(entry => entry.items.length)
}

function diagnosticLinks(html) {
  const $ = cheerio.load(html)
  return $('a')
    .map((_, el) => ({
      text: normalize($(el).text()).slice(0, 80),
      href: absoluteHref($(el).attr('href'))
    }))
    .get()
    .filter(item => item.text && item.href)
    .slice(0, 25)
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST')
    return res.status(405).json({ error: 'Usa POST para consultar la agenda.' })
  }

  const { username, password, view = 'week' } = req.body || {}

  if (!username || !password) {
    return res.status(400).json({ error: 'Faltan usuario o contraseña.' })
  }

  if (!['day', 'week'].includes(view)) {
    return res.status(400).json({ error: 'Vista no válida.' })
  }

  try {
    const jar = new CookieJar()

    const loginPage = await fetchWithSession(LOGIN_URL, { method: 'GET' }, jar)
    const loginHtml = await loginPage.text()
    const $login = cheerio.load(loginHtml)
    const logintoken = $login('input[name="logintoken"]').attr('value') || ''

    const form = new URLSearchParams()
    form.set('username', username)
    form.set('password', password)
    if (logintoken) form.set('logintoken', logintoken)

    const loginResponse = await fetchWithSession(
      LOGIN_URL,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: form.toString()
      },
      jar
    )

    const afterLogin = await loginResponse.text()
    const $after = cheerio.load(afterLogin)
    const loginError =
      $after('#loginerrormessage').text() ||
      $after('.loginerrors').text()

    if (loginError || ($after('#username').length && $after('#password').length)) {
      return res.status(401).json({
        error: normalize(loginError) || 'Usuario o contraseña incorrectos.'
      })
    }

    if (/debe aceptar nuestras políticas/i.test(afterLogin)) {
      return res.status(409).json({
        error: 'Sallenet solicita aceptar sus políticas. Entra una vez en Sallenet, acéptalas y vuelve a intentarlo.'
      })
    }

    const parentsResponse = await fetchWithSession(PARENTS_URL, { method: 'GET' }, jar)
    const parentsHtml = await parentsResponse.text()

    const workUrl = findLinkByText(parentsHtml, 'Trabajo')
    if (!workUrl) {
      return res.status(502).json({
        error: 'He entrado en Sallenet, pero no encuentro el enlace “Trabajo”.',
        stage: 'parents',
        links: diagnosticLinks(parentsHtml)
      })
    }

    const workResponse = await fetchWithSession(workUrl, { method: 'GET' }, jar)
    let workHtml = await workResponse.text()

    const targetText = view === 'week' ? 'Vista semanal' : 'Vista diaria'
    const targetUrl = findLinkByText(workHtml, targetText)
    let warning = null

    if (targetUrl) {
      const viewResponse = await fetchWithSession(targetUrl, { method: 'GET' }, jar)
      workHtml = await viewResponse.text()
    } else {
      warning =
        'Sallenet no expone el cambio de vista como enlace HTTP. Muestro la vista que devuelve por defecto.'
    }

    const days = parseAgenda(workHtml)

    if (!days.length) {
      return res.status(502).json({
        error: 'He llegado a “Trabajo”, pero no he podido interpretar la tabla de agenda.',
        stage: 'work',
        warning,
        links: diagnosticLinks(workHtml)
      })
    }

    res.setHeader('Cache-Control', 'no-store')
    return res.status(200).json({
      ok: true,
      view,
      days,
      warning,
      fetchedAt: new Date().toISOString()
    })
  } catch (error) {
    return res.status(500).json({
      error: 'Error consultando Sallenet.',
      detail: error instanceof Error ? error.message : String(error)
    })
  }
}
