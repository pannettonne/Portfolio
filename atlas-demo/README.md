# ATLAS · Healthcare Network Intelligence

**Demo ejecutiva con datos 100 % ficticios.** Preparada para presentar el concepto de IPA vivo y planificación asistencial. No es una aplicación corporativa ni contiene información real de ASISA.

## Inicio de sesión de demostración

- Usuario: \`admin\`
- Contraseña: \`AtlasDemo2026!\`

**Importante:** la autenticación de esta prueba es una pantalla ilustrativa implementada en el navegador. La contraseña está incluida en el código fuente. **No es un control de acceso seguro**. No incorporar datos reales, datos de salud, proveedores confidenciales o documentación corporativa. Antes de permitir acceso a datos de empresa, sustituir por SSO/OIDC corporativo y sesiones verificadas en servidor, y restringir el acceso al despliegue desde Vercel.

## Recorrido comercial recomendado (10 minutos)

1. **Command Center**: indicadores, España en 3D, capas geográficas y radar estratégico.
2. **Inteligencia territorial**: seleccionar Sevilla o Málaga para explorar demanda, cartera, población, renta y necesidades detectadas.
3. **Red asistencial 3D**: alternar columnas de actividad, conexiones y concentración. Volar hasta una ciudad.
4. **Proveedores 360°**: abrir Hospital Guadalquivir o Instituto Diagnóstico Nervión; mostrar grupo empresarial, cartera, contrato, actividad, espera y dependencia.
5. **Cobertura asistencial**: comparar Sevilla y Madrid; explicar diferencia entre cercanía y cobertura real.
6. **Scenario Lab**: Sevilla, retirar centro, aumentar capacidad, variar tarifas y crecimiento; ver antes/después e incorporar escenario al IPA.
7. **Radar estratégico**: ver reglas sin IA generativa y acceder al análisis territorial.
8. **IPA vivo**: actuaciones, estados, trazabilidad y exportación CSV.

## Tecnología y costes

- HTML, CSS y JavaScript estáticos, sin compilación ni backend para la demo.
- MapLibre GL JS para cartografía 3D; deck.gl para ColumnLayer, ArcLayer, ScatterplotLayer y HeatmapLayer.
- Mapa base público de CARTO Dark Matter **solo para la demostración**. Para uso corporativo, sustituir por cartografía propia de España en PMTiles servida desde infraestructura autorizada y con atribución OSM.
- Datos geográficos y actividad asistencial sintéticos, definidos localmente en \`app.js\`.
- No utiliza IA, DuckDB ni PostGIS en esta prueba. La integración sería una fase posterior.
- Si un navegador bloquea WebGL o el servicio cartográfico externo, aparece una vista SVG alternativa, para que la presentación no quede en blanco.

## Despliegue en Vercel

En una importación independiente, seleccionar este directorio (\`atlas-demo\`) como Root Directory y framework "Other". No requiere comandos de instalación ni compilación. \`index.html\`, \`style.css\` y \`app.js\` se sirven como recursos estáticos.

Para obtener una URL de preview desde el proyecto Portfolio conectado a Vercel, la rama \`atlas-enterprise-demo\` incorpora también un \`vercel.json\` en la raíz que especifica \`outputDirectory: atlas-demo\` y desactiva la compilación. **No fusionar ese archivo en la rama principal del portfolio**, ya que modificaría la configuración de su despliegue.

## Evolución a producto corporativo

- Backend SQL Server + DuckDB Spatial y GeoParquet, contratos, profesionales, centros y grupos empresariales con modelo maestro único.
- Datos de INE/CN Hospitales con fechas y metodologías visibles.
- PMTiles de España + motor de rutas (OSRM/Valhalla) para tiempos realistas.
- SSO, permisos por ámbito territorial, auditoría, versionado y control de datos sensibles.
- Escenarios calculados con matrices de accesibilidad y restricciones de capacidad reales, no los coeficientes sintéticos del prototipo.
- Persistencia del IPA, flujo de aprobaciones y exportación de resultados a presentaciones.

## Alcance y limitaciones

Los números de cobertura, renta, población, actividad y costes han sido inventados. Los nombres de centros y grupos de la prueba son ficticios. Los puntos usan coordenadas aproximadas de ciudades reales, sin indicar centros sanitarios concretos. El motor de simulación de la demo **no debe utilizarse para decisiones asistenciales o presupuestarias reales**.

## Actualización de la presentación (diseño corporativo y mapas)

- Tema visual azul/blanco inspirado en la identidad pública de ASISA, expresamente **no oficial**.
- Cartografía principal: OpenFreeMap Liberty, con CARTO Positron y MapLibre Demo como alternativas automáticas. Si fallan los servicios externos, mapa vectorial simplificado en memoria.
- MapLibre y deck.gl se cargan de forma independiente desde dos CDN alternativos. Si deck.gl no carga, se mantienen los puntos de MapLibre, la navegación y el mapa base.
- Visualizaciones 3D con columnas de actividad, constelación sintética de ubicaciones, rutas y mapa de calor.
- El visor tiene botones **Cambiar mapa** y **Estado técnico**. Si ves el mapa en blanco, abre Estado técnico e indica el estado de WebGL, librerías, proveedor y último error.
- Para una futura versión sin servicios cartográficos externos, utilizar España en PMTiles alojada en un servidor que admita solicitudes HTTP Range y librerías empaquetadas localmente; la demo actual sigue necesitando acceso a los CDN para renderizar el 3D.
