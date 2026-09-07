const DB_NAME = 'agenda-sallenet-secure'
const DB_VERSION = 1
const KEY_STORE = 'keys'
const DATA_STORE = 'data'
const CREDENTIAL_KEY = 'credentials-key'
const CREDENTIAL_DATA = 'credentials'

function openDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)
    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(KEY_STORE)) db.createObjectStore(KEY_STORE)
      if (!db.objectStoreNames.contains(DATA_STORE)) db.createObjectStore(DATA_STORE)
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

function idbGet(db, storeName, key) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly')
    const req = tx.objectStore(storeName).get(key)
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

function idbPut(db, storeName, value, key) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite')
    tx.objectStore(storeName).put(value, key)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

function idbDelete(db, storeName, key) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite')
    tx.objectStore(storeName).delete(key)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

async function getOrCreateKey(db) {
  let key = await idbGet(db, KEY_STORE, CREDENTIAL_KEY)
  if (!key) {
    key = await crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt'])
    await idbPut(db, KEY_STORE, key, CREDENTIAL_KEY)
  }
  return key
}

export async function saveCredentials(username, password) {
  const db = await openDb()
  const key = await getOrCreateKey(db)
  const iv = crypto.getRandomValues(new Uint8Array(12))
  const data = new TextEncoder().encode(JSON.stringify({ username, password }))
  const cipher = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, data)
  await idbPut(db, DATA_STORE, { iv: Array.from(iv), cipher }, CREDENTIAL_DATA)
  db.close()
}

export async function loadCredentials() {
  const db = await openDb()
  const record = await idbGet(db, DATA_STORE, CREDENTIAL_DATA)
  const key = await idbGet(db, KEY_STORE, CREDENTIAL_KEY)
  if (!record || !key) {
    db.close()
    return null
  }
  try {
    const plain = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: new Uint8Array(record.iv) },
      key,
      record.cipher,
    )
    db.close()
    return JSON.parse(new TextDecoder().decode(plain))
  } catch {
    db.close()
    return null
  }
}

export async function clearCredentials() {
  const db = await openDb()
  await idbDelete(db, DATA_STORE, CREDENTIAL_DATA)
  await idbDelete(db, KEY_STORE, CREDENTIAL_KEY)
  db.close()
}
