# Agenda Sallenet PWA

PWA para consultar la agenda escolar de Sallenet con la cuenta individual de cada familia.

## Principios

- Sin base de datos de usuarios.
- Las credenciales solo se envían al endpoint `/api/agenda` durante la consulta y no se persisten en servidor.
- Si el usuario marca “Recordarme”, la PWA cifra las credenciales con Web Crypto y las conserva únicamente en IndexedDB del dispositivo.
- La última agenda se almacena localmente para poder seguir viéndola si no hay conexión.
- Las notas personales se mantienen: cada familia accede con sus propias credenciales.

## Desarrollo

```bash
npm install
npm run dev
```

Para probar `/api/agenda` localmente conviene usar Vercel CLI (`vercel dev`), ya que Vite por sí solo no ejecuta las funciones serverless de `/api`.

## Despliegue

El repositorio está preparado para Vercel: el frontend se compila con Vite y `api/agenda.js` se publica como función serverless.

## Estado del conector Sallenet

La primera implementación evita Selenium e intenta reproducir por HTTP el flujo de Moodle/Sallenet:

1. GET del formulario de login.
2. POST del formulario conservando cookies de sesión.
3. GET del módulo de Padres.
4. Localización del enlace `Trabajo`.
5. Localización de `Vista semanal` si está disponible como URL.
6. Parseo de la tabla semanal a JSON.

Si `Trabajo` o `Vista semanal` se cargan exclusivamente por JavaScript/AJAX, el endpoint devolverá un error explícito para que podamos adaptar esa petición concreta sin volver a Selenium.
