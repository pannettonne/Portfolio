(function(){
'use strict';
var P=[
{id:1,name:'Hospital Metropolitano Norte',group:'Grupo Salud Norte',type:'Hospital',city:'Madrid',region:'Madrid',lat:40.492,lng:-3.695,specialties:38,acts:185400,cost:26.8,occupancy:87,quality:94,waiting:9,contract:'2027-05-31',trend:8,dependency:23,code:'MAD-001',services:['Urgencias 24 h','Cirugía','Radiodiagnóstico','Cardiología','Traumatología','Oncología']},
{id:2,name:'Centro Médico Castellana',group:'Grupo Salud Norte',type:'Centro médico',city:'Madrid',region:'Madrid',lat:40.466,lng:-3.69,specialties:29,acts:103250,cost:10.7,occupancy:81,quality:91,waiting:7,contract:'2027-11-30',trend:11,dependency:14,code:'MAD-002',services:['Consultas externas','Diagnóstico','Laboratorio','Dermatología']},
{id:3,name:'Hospital Costa Mediterránea',group:'Mediterránea Salud',type:'Hospital',city:'Barcelona',region:'Cataluña',lat:41.403,lng:2.176,specialties:36,acts:169300,cost:22.3,occupancy:91,quality:88,waiting:13,contract:'2027-03-31',trend:13,dependency:29,code:'CAT-001',services:['Urgencias','Hospitalización','Cirugía','Diagnóstico avanzado']},
{id:4,name:'Clínica Turia',group:'Grupo Levante',type:'Clínica',city:'Valencia',region:'C. Valenciana',lat:39.474,lng:-.366,specialties:23,acts:79300,cost:8.5,occupancy:83,quality:92,waiting:8,contract:'2028-02-28',trend:7,dependency:18,code:'VAL-001',services:['Consultas','Radiodiagnóstico','Traumatología','Cardiología']},
{id:5,name:'Hospital Guadalquivir',group:'Andalucía Médica',type:'Hospital',city:'Sevilla',region:'Andalucía',lat:37.392,lng:-5.976,specialties:34,acts:128600,cost:17.4,occupancy:92,quality:89,waiting:16,contract:'2027-01-31',trend:16,dependency:36,code:'AND-001',services:['Hospitalización','Urgencias','Cirugía','Diagnóstico']},
{id:6,name:'Clínica Costa del Sol',group:'Andalucía Médica',type:'Clínica',city:'Málaga',region:'Andalucía',lat:36.724,lng:-4.421,specialties:26,acts:88400,cost:9.6,occupancy:86,quality:93,waiting:10,contract:'2027-09-30',trend:9,dependency:21,code:'AND-002',services:['Consultas','Cirugía ambulatoria','Diagnóstico']},
{id:7,name:'Hospital Atlántico',group:'Atlántico Salud',type:'Hospital',city:'A Coruña',region:'Galicia',lat:43.359,lng:-8.41,specialties:30,acts:108500,cost:12.4,occupancy:77,quality:92,waiting:7,contract:'2028-04-30',trend:5,dependency:19,code:'GAL-001',services:['Urgencias','Cirugía','Laboratorio','Cardiología']},
{id:8,name:'Instituto Diagnóstico Nervión',group:'Nervión',type:'Diagnóstico',city:'Bilbao',region:'País Vasco',lat:43.268,lng:-2.935,specialties:11,acts:145700,cost:6.3,occupancy:94,quality:85,waiting:19,contract:'2027-02-28',trend:18,dependency:41,code:'VAS-001',services:['RM','TAC','Radiología','Ecografía']},
{id:9,name:'Centro Médico Ebro',group:'Grupo Ebro',type:'Centro médico',city:'Zaragoza',region:'Aragón',lat:41.649,lng:-.887,specialties:19,acts:56300,cost:4.9,occupancy:75,quality:95,waiting:5,contract:'2029-01-31',trend:4,dependency:11,code:'ARA-001',services:['Consultas','Laboratorio','Traumatología']},
{id:10,name:'Hospital Norte Cantábrico',group:'Cantábrico Salud',type:'Hospital',city:'Santander',region:'Cantabria',lat:43.462,lng:-3.804,specialties:27,acts:72800,cost:8.8,occupancy:79,quality:92,waiting:9,contract:'2028-07-31',trend:6,dependency:16,code:'CAN-001',services:['Hospitalización','Urgencias','Cirugía']},
{id:11,name:'Centro Valladolid Salud',group:'Grupo Meseta',type:'Centro médico',city:'Valladolid',region:'Castilla y León',lat:41.652,lng:-4.725,specialties:21,acts:61500,cost:4.8,occupancy:82,quality:90,waiting:11,contract:'2027-12-31',trend:8,dependency:15,code:'CYL-001',services:['Consultas','Diagnóstico','Cardiología']},
{id:12,name:'Hospital Vega del Segura',group:'Grupo Levante',type:'Hospital',city:'Murcia',region:'Murcia',lat:37.986,lng:-1.13,specialties:31,acts:92600,cost:11.2,occupancy:88,quality:87,waiting:15,contract:'2027-06-30',trend:12,dependency:27,code:'MUR-001',services:['Hospitalización','Urgencias','Cirugía']},
{id:13,name:'Clínica Sierra Nevada',group:'Andalucía Médica',type:'Clínica',city:'Granada',region:'Andalucía',lat:37.176,lng:-3.596,specialties:22,acts:55200,cost:5.4,occupancy:72,quality:95,waiting:6,contract:'2028-12-31',trend:8,dependency:12,code:'AND-003',services:['Consultas','Dermatología','Diagnóstico']},
{id:14,name:'Hospital del Principado',group:'Cantábrico Salud',type:'Hospital',city:'Oviedo',region:'Asturias',lat:43.362,lng:-5.845,specialties:29,acts:88600,cost:10.1,occupancy:84,quality:93,waiting:10,contract:'2028-08-31',trend:6,dependency:20,code:'AST-001',services:['Cirugía','Urgencias','Traumatología']}
];
var T=[
{name:'Madrid',pop:7100000,insured:870000,penetration:12.3,elder:19.1,income:26500,coverage:91,access:18,growth:8.4,uncovered:37000,lon:-3.70,lat:40.42,needs:['Cardiología en el corredor norte','Capacidad diagnóstica en el sur']},
{name:'Barcelona',pop:5800000,insured:615000,penetration:10.6,elder:20.7,income:24200,coverage:88,access:21,growth:5.8,uncovered:51500,lon:2.17,lat:41.38,needs:['Refuerzo de pruebas diagnósticas','Disponibilidad de traumatología']},
{name:'Valencia',pop:2700000,insured:294000,penetration:10.9,elder:21.5,income:19300,coverage:83,access:26,growth:10.5,uncovered:42000,lon:-.38,lat:39.47,needs:['Cobertura de dermatología periurbana','Aumentar acceso a imagen']},
{name:'Sevilla',pop:1960000,insured:326000,penetration:16.6,elder:17.5,income:17500,coverage:79,access:31,growth:12.7,uncovered:68400,lon:-5.98,lat:37.39,needs:['Diversificar dependencia hospitalaria','Reducir espera de cirugía']},
{name:'Málaga',pop:1780000,insured:207000,penetration:11.6,elder:18.9,income:18500,coverage:82,access:29,growth:15.4,uncovered:33200,lon:-4.42,lat:36.72,needs:['Capacidad ante crecimiento estacional','Diagnóstico en área oriental']},
{name:'Bilbao',pop:1150000,insured:188000,penetration:16.3,elder:24.5,income:25100,coverage:86,access:23,growth:3.8,uncovered:24400,lon:-2.93,lat:43.26,needs:['Alternativa al diagnóstico concentrado','Acceso de mayores']},
{name:'Zaragoza',pop:980000,insured:154000,penetration:15.7,elder:21.4,income:21300,coverage:89,access:20,growth:4.8,uncovered:15500,lon:-.88,lat:41.65,needs:['Seguimiento de capacidad ambulatoria']},
{name:'A Coruña',pop:1120000,insured:121000,penetration:10.8,elder:26.2,income:20400,coverage:85,access:27,growth:2.7,uncovered:23100,lon:-8.41,lat:43.36,needs:['Acceso rural','Especialidades de alta complejidad']}
];
var scenarios={provider:0,capacity:18,tariff:0,growth:6,selected:'Sevilla'};
var state={view:'dashboard',region:'Todas',specialty:'Todas',layer:'columns',selected:1,selectedCity:'Madrid',period:'2026',sort:'acts',search:'',map:null,overlay:null,mapToken:0,plan:[{title:'Revisar capacidad diagnóstica en Sevilla',territory:'Sevilla',status:'En análisis',owner:'Dirección de proveedores',priority:'Alta'},{title:'Evaluar alternativas en Bilbao',territory:'Bilbao',status:'Pendiente',owner:'Planificación',priority:'Media'},{title:'Actualización de cartera de servicios en Valencia',territory:'Valencia',status:'En curso',owner:'Red territorial',priority:'Media'}]};
var nav=[['dashboard','◈','Visión global'],['territory','◎','Inteligencia territorial'],['network','⌁','Mapa de la red 3D'],['providers','▤','Proveedores 360°'],['coverage','◉','Cobertura asistencial'],['scenarios','◇','Simulación'],['insights','✦','Radar asistencial'],['ipa','▣','Plan asistencial · IPA']];
var names={dashboard:'Visión global',territory:'Inteligencia territorial',network:'Mapa de la red 3D',providers:'Proveedores 360°',coverage:'Cobertura asistencial',scenarios:'Simulación',insights:'Radar asistencial',ipa:'Plan asistencial · IPA'};
function $ (id){return document.getElementById(id)}function euro(n){return n.toLocaleString('es-ES',{maximumFractionDigits:1})+' M€'}function num(n){return Math.round(n).toLocaleString('es-ES')}function pct(n){return n.toFixed(1).replace('.',',')+' %'}function pill(t,c){return '<span class="pill '+(c||'')+'">'+t+'</span>'}function fmt(n){return n.toLocaleString('es-ES')}function range(v,min,max){return Math.max(min,Math.min(max,v))}
function toast(s){var el=$('toast');if(!el)return;el.textContent=s;el.style.display='block';clearTimeout(window.__toast);window.__toast=setTimeout(function(){el.style.display='none'},3200)}
function kpi(l,v,f,color){return '<div class="kpi"><div class="kpi-label">'+l+'</div><div class="kpi-value">'+v+'</div><div class="kpi-foot '+(color||'')+'">'+f+'</div></div>'}
function head(label,title,subtitle,actions){return '<div class="page-head"><div><div class="eyebrow">'+label+'</div><h1>'+title+'</h1><p>'+subtitle+'</p></div><div class="actions">'+(actions||'')+'</div></div>'}
function card(title,sub,body,cls){return '<section class="card '+(cls||'')+'"><div class="card-pad"><div class="card-head"><h3>'+title+'</h3><small>'+sub+'</small></div>'+body+'</div></section>'}
function insight(icon,title,desc,note){return '<div class="insight"><div class="insight-icon">'+icon+'</div><div><b>'+title+'</b><p>'+desc+'</p>'+(note?'<small>'+note+'</small>':'')+'</div></div>'}
function metric(k,v){return '<div class="metric-pair"><span>'+k+'</span><strong>'+v+'</strong></div>'}
function barChart(values,labels){var max=Math.max.apply(null,values)*1.08;return '<div class="bar-chart">'+values.map(function(v,i){return '<div class="bar" title="'+labels[i]+': '+v+'" data-label="'+labels[i]+'" style="height:'+Math.max(5,v/max*100)+'%"></div>'}).join('')+'</div>'}
function mapHTML(id,tall){return '<div class="map-wrap '+(tall?'tall':'')+'"><img class="map-preloader" src="./iberia-preview.svg" alt="Mapa de España y Portugal"/><div class="map" id="'+id+'"></div><div class="map-label"><b>ATLAS · MAPA ASISTENCIAL</b><span>14 centros destacados · Datos ficticios</span></div><div class="map-legend">● Ubicaciones simuladas<br/>◢ Columnas: actividad anual<br/>⌁ Arcos: relaciones ilustrativas</div><button type="button" class="map-style-toggle" title="Cambiar el proveedor de mapa base" data-basemap-cycle="1">◫ Cambiar mapa</button><details class="map-diagnostics-native"><summary>ⓘ Estado técnico</summary><div class="map-diagnostics-static"><b>Diagnóstico de ATLAS</b><span class="map-diagnostics-status">Si no aparece el mapa detallado, esta vista muestra España localmente.</span><button type="button" class="map-retry">↻ Reintentar mapa</button><span class="map-diagnostics-hint">El mapa local no necesita servicios de cartografía externos.</span></div></details><div class="map-health" role="status">Preparando cartografía…</div></div>'}
function fallback(el){el.innerHTML='<div class="map-fallback"><svg viewBox="0 0 500 320"><defs><radialGradient id="gl"><stop stop-color="#5be9d1" stop-opacity=".4"/><stop offset="1" stop-color="#5be9d1" stop-opacity="0"/></radialGradient></defs><path fill="#143a4a" stroke="#6addca" stroke-width="2" d="M92 72L170 56 225 63 275 52 329 66 388 86 423 107 400 144 372 151 359 192 332 226 280 249 244 274 196 251 149 236 112 206 88 162 100 125Z"/><g fill="url(#gl)"><circle cx="249" cy="146" r="58"/><circle cx="374" cy="116" r="44"/><circle cx="320" cy="195" r="35"/><circle cx="161" cy="210" r="42"/></g><g fill="#a9ffee" stroke="#6af1cb"><circle cx="249" cy="146" r="5"/><circle cx="374" cy="116" r="5"/><circle cx="320" cy="195" r="5"/><circle cx="161" cy="210" r="5"/><circle cx="209" cy="84" r="4"/></g><g stroke="#60d8d1" fill="none" stroke-dasharray="4 5"><path d="M249 146Q315 73 374 116M249 146Q280 145 320 195M249 146Q164 122 161 210M249 146Q221 110 209 84"/></g><text x="249" y="133" fill="#e8fff9" font-size="11" text-anchor="middle">Madrid</text><text x="374" y="104" fill="#e8fff9" font-size="11" text-anchor="middle">Barcelona</text><text x="320" y="183" fill="#e8fff9" font-size="11" text-anchor="middle">Valencia</text><text x="161" y="198" fill="#e8fff9" font-size="11" text-anchor="middle">Sevilla</text><text x="250" y="301" fill="#9bc8d5" font-size="11" text-anchor="middle">Vista esquemática de respaldo · mapa en vivo no disponible</text></svg></div>'}
function renderNav(){$('nav').innerHTML=nav.map(function(x){return '<button class="nav-item '+(state.view===x[0]?'active':'')+'" data-go="'+x[0]+'"><span class="nav-icon">'+x[1]+'</span>'+x[2]+'</button>'}).join('');$('section-name').textContent=names[state.view]}
function go(v){state.view=v;render();document.querySelector('.sidebar').classList.remove('open');window.scrollTo({top:0,behavior:'smooth'})}
function render(){renderNav();if(state.map){try{state.map.remove()}catch(e){}state.map=null;state.overlay=null}var views={dashboard:dashboard,territory:territory,network:network,providers:providers,coverage:coverage,scenarios:scenario,insights:insights,ipa:ipa};$('page').innerHTML=views[state.view]();document.querySelectorAll('[data-go]').forEach(function(el){el.addEventListener('click',function(){go(el.dataset.go)})});if(['dashboard','territory','network','coverage','scenarios'].indexOf(state.view)>-1){setTimeout(function(){initMap(state.view==='dashboard'?'map-dash':state.view==='territory'?'map-ter':state.view==='network'?'map-net':state.view==='coverage'?'map-cov':'map-sim')},30)}bindPage()}
function dashboard(){return head('STRATEGIC OVERVIEW','Nuestra red asistencial, de un vistazo.','Visión integral de cobertura, actividad, proveedores y necesidades de planificación.', '<button class="outline" data-go="ipa">Ver IPA territorial →</button>')+
'<div class="grid kpis">'+kpi('Asegurados en territorios piloto','2,78 M','↑ 8,2 % interanual')+kpi('Proveedores simulados',num(3586),'18 centros destacados')+kpi('Actividad anual','9,24 M','↑ 11,3 % interanual')+kpi('Cobertura objetivo','87,4 %','8 áreas de análisis','dim')+'</div>'+
'<div class="grid layout-70">'+
'<section class="card"><div class="card-pad"><div class="card-head"><h3>Digital Twin · Red asistencial</h3>'+pill('WEBGL · INTERACTIVO','green')+'</div><div class="tools"><button class="tab active" data-layer="columns">◢ Actividad 3D</button><button class="tab" data-layer="arcs">⌁ Conexiones</button><button class="tab" data-layer="heat">◉ Concentración</button><button class="tab" data-layer="points">• Centros</button></div></div>'+mapHTML('map-dash')+'</section>'+
card('Radar estratégico','Alertas de demostración',insight('⚑','Capacidad diagnóstica · Bilbao','Alta concentración de actividad y capacidad al 94 %.','Revisar capacidad alternativa')+insight('↗','Crecimiento de cartera · Málaga','La cartera aumenta y exige revisar cobertura estacional.','Comparar oferta en Scenario Lab')+insight('◇','Dependencia hospitalaria · Sevilla','Riesgo de concentración en un mismo grupo empresarial.','Simular pérdida del principal concierto')+ '<button class="outline" data-go="insights" style="width:100%;margin-top:7px">Explorar el radar →</button>')+'</div>'+
'<div class="grid layout-60" style="margin-top:17px">'+card('Actividad asistencial','Evolución anual simulada',barChart([68,72,77,82,74,88,91,87,102,99,110,119],['E','F','M','A','M','J','J','A','S','O','N','D'])+'<div class="mini" style="margin-top:22px">Índice mensual ilustrativo · Base enero = 68</div>')+
card('Prioridades de planificación','Situación de los territorios',T.slice(0,5).map(function(t){return '<div class="metric-pair"><span>'+t.name+'</span><div style="width:44%"><div class="row"><strong>'+t.coverage+' %</strong><small class="'+(t.coverage<85?'warn':'good')+'">'+(t.coverage<85?'Revisar':'En objetivo')+'</small></div><div class="progress" style="margin-top:6px"><div style="width:'+t.coverage+'%"></div></div></div></div>'}).join(''))+'</div>'}
function territory(){var t=T.filter(function(x){return x.name===state.selectedCity})[0]||T[0];return head('TERRITORIAL INTELLIGENCE','El territorio explica la necesidad.','Combina cartera asegurada, demografía, renta, acceso y crecimiento.','<select id="city-select">'+T.map(function(x){return '<option '+(x.name===t.name?'selected':'')+'>'+x.name+'</option>'}).join('')+'</select>')+
'<div class="grid kpis">'+kpi('Población general',fmt(t.pop),'Territorio simulado','dim')+kpi('Cartera asegurada',fmt(t.insured),'Penetración: '+pct(t.penetration))+kpi('Renta media',fmt(t.income)+' €','Indicador ficticio','dim')+kpi('Mayores de 65 años',pct(t.elder),'Demanda futura de referencia','dim')+'</div>'+
'<div class="grid layout-70"><section class="card"><div class="card-pad"><div class="card-head"><h3>Atlas de demanda y oferta</h3>'+pill('SEGMENTACIÓN TERRITORIAL','blue')+'</div><div class="tools"><button class="tab active" data-layer="heat">◉ Densidad</button><button class="tab" data-layer="columns">◢ Actividad</button><button class="tab" data-layer="points">• Proveedores</button></div></div>'+mapHTML('map-ter',true)+'</section>'+
card('Ficha territorial',t.name+' · 2026',metric('Cobertura estimada',t.coverage+' %')+metric('Acceso medio',t.access+' min')+metric('Crecimiento de cartera','+'+pct(t.growth))+metric('Población potencialmente descubierta',num(t.uncovered))+metric('Penetración comercial',pct(t.penetration))+'<h4 class="section-title" style="margin-top:23px;font-size:14px">Necesidades identificadas</h4>'+t.needs.map(function(n){return insight('◉',n,'Hipótesis para análisis y contraste territorial.','Datos simulados')}).join(''))+'</div>'+
'<div class="grid layout-60" style="margin-top:16px">'+card('Pirámide poblacional','Ejemplo de distribución por edad',barChart([15,19,23,27,29,24,22,19,14,10],['0-9','10','20','30','40','50','60','70','80','90+'])+'<p class="mini" style="margin-top:24px">Distribución hipotética; sustituible por INE y cartera interna.</p>')+
card('Segmentación de cartera','Mix de productos simulado',metric('Salud individual','43 %')+metric('Salud colectivo','31 %')+metric('Mutualismo','18 %')+metric('Otros productos','8 %')+'<div class="divider"></div>'+insight('↗','Crecimiento previsto', 'Un escenario de crecimiento de '+pct(t.growth)+' obliga a revisar la capacidad contratada.'))+'</div>'}
function network(){return head('MAPA ASISTENCIAL 3D','Digital Twin asistencial.','Explora la red en 3D, cruza capas y navega hasta cualquier proveedor.', '<button class="outline" data-go="providers">Abrir censo 360° →</button>')+
'<div class="grid kpis">'+kpi('Nodos georreferenciados','3.586','Muestra sintética')+kpi('Grupos empresariales','248','Relaciones consolidadas','dim')+kpi('Especialidades','42','Modelo normalizado','dim')+kpi('Flujos analizados','86 K','Simulación anual','dim')+'</div>'+
'<section class="card"><div class="card-pad"><div class="card-head"><h3>Visor geoespacial nacional</h3>'+pill('MAPLIBRE + DECK.GL','green')+'</div><div class="tools"><button class="tab active" data-layer="columns">◢ Columnas 3D</button><button class="tab" data-layer="arcs">⌁ Arcos asistenciales</button><button class="tab" data-layer="heat">◉ Concentración</button><button class="tab" data-layer="points">• Localizaciones</button><select id="map-fly"><option value="">Volar a una ciudad...</option>'+T.map(function(t){return '<option value="'+t.name+'">'+t.name+'</option>'}).join('')+'</select><button class="outline" id="map-reset">Vista nacional</button></div></div>'+mapHTML('map-net',true)+'</section>'+
'<div class="grid layout-60" style="margin-top:16px">'+card('Capas del Digital Twin','Capacidad analítica',insight('◈','Cartera y población','Centros cruzados con concentración de asegurados.')+insight('◎','Accesibilidad y cobertura','Áreas de influencia por especialidad y territorio.')+insight('⌁','Relaciones contractuales','Grupos, centros, profesionales y flujos de derivación.'))+
card('Explorador de especialidades','Catálogo de referencia', ['Cardiología','Traumatología','Radiodiagnóstico','Dermatología','Oftalmología','Ginecología','Pediatría'].map(function(s,i){return metric(s,num(55+i*23)+' profesionales')}).join(''))+'</div>'}
function providers(){var list=P.filter(function(p){return (state.region==='Todas'||p.region===state.region)&&(state.search===''||(p.name+' '+p.city+' '+p.group+' '+p.code).toLowerCase().indexOf(state.search.toLowerCase())>=0)});list.sort(function(a,b){return b[state.sort]-a[state.sort]});return head('PROVIDER INTELLIGENCE','Proveedores 360°.','Una única visión de centros, grupos, servicios, contratos, actividad y riesgo.','<button class="outline" id="export-providers">↓ Exportar muestra CSV</button>')+
'<div class="grid kpis">'+kpi('Centros destacados',num(P.length),'Muestra navegable de la demo')+kpi('Grupos empresariales',new Set(P.map(function(p){return p.group})).size,'Identificados')+kpi('Actividad de la muestra',num(P.reduce(function(a,p){return a+p.acts},0)),'Actos anuales')+kpi('Coste de la muestra',euro(P.reduce(function(a,p){return a+p.cost},0)),'Importes inventados','dim')+'</div>'+
'<section class="card"><div class="card-pad"><div class="card-head"><h3>Censo de proveedores</h3>'+pill('FICHA 360°','green')+'</div><div class="tools"><input id="provider-search" class="field" style="flex:1;min-width:180px" placeholder="Buscar por nombre, grupo o localidad..." value="'+state.search.replace(/"/g,'&quot;')+'"/><select id="provider-region"><option>Todas</option>'+Array.from(new Set(P.map(function(p){return p.region}))).map(function(r){return '<option '+(state.region===r?'selected':'')+'>'+r+'</option>'}).join('')+'</select><select id="provider-sort"><option value="acts" '+(state.sort==='acts'?'selected':'')+'>Mayor actividad</option><option value="cost" '+(state.sort==='cost'?'selected':'')+'>Mayor coste</option><option value="dependency" '+(state.sort==='dependency'?'selected':'')+'>Mayor dependencia</option></select></div><div class="table-scroll"><table class="data-table"><thead><tr><th>Proveedor</th><th>Localización</th><th>Especialidades</th><th>Actividad</th><th>Coste anual</th><th>Ocupación</th><th>Dependencia</th><th>Ficha</th></tr></thead><tbody>'+list.map(function(p){return '<tr data-provider="'+p.id+'"><td><div class="provider-cell"><div class="provider-icon">✚</div><div><b>'+p.name+'</b><small>'+p.group+'</small></div></div></td><td>'+p.city+'</td><td>'+p.specialties+'</td><td>'+num(p.acts)+'</td><td>'+euro(p.cost)+'</td><td><span class="'+(p.occupancy>90?'warn':'good')+'">'+p.occupancy+' %</span></td><td><span class="'+(p.dependency>30?'risk':'soft')+'">'+p.dependency+' %</span></td><td>↗</td></tr>'}).join('')+'</tbody></table></div><p class="mini">'+list.length+' resultados. Selecciona un proveedor para abrir la ficha completa.</p></div></section><div id="provider-detail" style="margin-top:17px">'+detail(P.find(function(p){return p.id===state.selected})||P[0])+'</div>'}
function detail(p){return '<div class="grid layout-60">'+card('Ficha de proveedor',p.code+' · '+p.type,
'<div class="detail-hero"><div class="detail-icon">✚</div><div><h2>'+p.name+'</h2><p>'+p.group+' · '+p.city+'</p><div class="tag-row" style="margin-top:9px">'+pill('CONCERTADO','green')+pill(p.type,'blue')+'</div></div></div>'+
'<div class="grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:15px">'+kpi('Especialidades',p.specialties,'Cartera activa')+kpi('Actividad',num(p.acts),'Actos/año')+kpi('Coste',euro(p.cost),'Año simulado')+'</div>'+
'<h4 class="section-title" style="font-size:14px">Cartera de servicios</h4><div class="tag-row">'+p.services.map(function(s){return pill(s)}).join('')+'</div><div class="divider"></div><h4 class="section-title" style="font-size:14px">Relación contractual</h4>'+
metric('Grupo empresarial',p.group)+metric('Vencimiento del contrato',p.contract)+metric('Dependencia territorial',p.dependency+' %')+metric('Variación interanual','+'+p.trend+' %'))+
card('Desempeño, calidad y riesgo','Scorecard descriptivo',
metric('Ocupación',p.occupancy+' %')+'<div class="progress"><div style="width:'+p.occupancy+'%"></div></div>'+
metric('Indicador de calidad',p.quality+' / 100')+'<div class="progress"><div style="width:'+p.quality+'%"></div></div>'+
metric('Espera media',p.waiting+' días')+metric('Posición geográfica',p.lat.toFixed(3)+', '+p.lng.toFixed(3))+
'<div class="divider"></div>'+insight('◇','Dependencia del proveedor','Este centro representa un '+p.dependency+' % de la actividad ficticia en su ámbito. Revisar alternativas si supera el umbral interno.')+
'<button class="primary" data-go="scenarios" style="width:100%;margin-top:8px">Simular cambios en la red →</button>')+'</div>'}
function coverage(){var t=T.find(function(x){return x.name===state.selectedCity})||T[0];return head('ACCESS & CAPACITY','Cobertura asistencial real.','De la proximidad geográfica a la accesibilidad efectiva y capacidad disponible.', '<select id="city-select">'+T.map(function(x){return '<option '+(x.name===t.name?'selected':'')+'>'+x.name+'</option>'}).join('')+'</select>')+
'<div class="grid kpis">'+kpi('Cobertura de referencia',t.coverage+' %','Población dentro del objetivo','dim')+kpi('Tiempo medio',t.access+' min','Desplazamiento ilustrativo','dim')+kpi('Fuera del objetivo',num(t.uncovered),'Asegurados simulados','warn')+kpi('Centros en el área',P.filter(function(p){return p.city===t.name}).length,'Muestra de la demo','dim')+'</div>'+
'<div class="grid layout-70"><section class="card"><div class="card-pad"><div class="card-head"><h3>Mapa de accesibilidad</h3>'+pill('MODELO ILUSTRATIVO','amber')+'</div><div class="tools"><select id="specialty"><option>Todas las especialidades</option><option>Cardiología</option><option>Traumatología</option><option>Radiodiagnóstico</option><option>Dermatología</option></select><button class="tab active" data-layer="heat">◉ Concentración</button><button class="tab" data-layer="points">• Centros</button></div></div>'+mapHTML('map-cov',true)+'</section>'+
card('Cobertura por tiempo','Distribución sintética', '<div class="metric-pair"><span>Hasta 15 minutos</span><strong class="good">51 %</strong></div><div class="progress"><div style="width:51%"></div></div><div class="metric-pair"><span>Entre 15 y 30 minutos</span><strong class="warn">33 %</strong></div><div class="progress"><div style="width:33%;background:#e1b870"></div></div><div class="metric-pair"><span>Más de 30 minutos</span><strong class="risk">16 %</strong></div><div class="progress"><div style="width:16%;background:#ef8988"></div></div><div class="divider"></div>'+insight('⚠','Cobertura no equivale a proximidad','Para producción, combinaríamos tiempos por carretera, agendas, horarios y capacidad real.')+'<button class="outline" data-go="scenarios" style="width:100%;margin-top:15px">Evaluar nuevo proveedor →</button>')+'</div>'+
'<div style="margin-top:16px">'+card('Brechas detectadas','Motor de reglas · resultados ilustrativos',T.filter(function(x){return x.coverage<87}).map(function(x){return insight('◎',x.name+' · '+x.coverage+' % de cobertura','Tiempo medio de acceso: '+x.access+' minutos. Necesidades a contrastar: '+x.needs[0]+'.',fmt(x.uncovered)+' asegurados potencialmente afectados')}).join(''))+'</div>'}
function results(){var t=T.find(function(x){return x.name===scenarios.selected})||T[3];var centre=Math.round(t.coverage+scenarios.capacity*.24+(scenarios.provider===1?5:scenarios.provider===-1?-12:0)-scenarios.growth*.17);var cov=range(centre,35,99);var cost=(scenarios.provider===1?1.4:scenarios.provider===-1?-1.7:0)+scenarios.capacity*.042+scenarios.tariff*.1;return {t:t,cov:cov,cost:cost,people:Math.round(t.insured*(cov-t.coverage)/100)}}
function scenario(){var r=results();return head('STRATEGIC SIMULATION','Scenario Lab.','Prueba decisiones antes de ejecutarlas. Compara cobertura, capacidad y coste.', '<button class="outline" id="reset-sim">↺ Restablecer</button>')+
'<div class="grid kpis">'+kpi('Territorio',r.t.name,'Escenario seleccionado','dim')+kpi('Cobertura actual',r.t.coverage+' %','Valor sintético','dim')+kpi('Cobertura proyectada',r.cov+' %',(r.cov>=r.t.coverage?'+':'')+(r.cov-r.t.coverage)+' puntos')+kpi('Impacto económico',(r.cost>=0?'+':'')+euro(r.cost),'Variación anual estimada','dim')+'</div>'+
'<div class="grid layout-60"><section class="card"><div class="card-pad"><div class="card-head"><h3>Configurar escenario</h3>'+pill('WHAT-IF','green')+'</div>'+
'<label class="mini">Territorio</label><select id="sim-city" style="width:100%;margin:9px 0 18px">'+T.map(function(t){return '<option '+(scenarios.selected===t.name?'selected':'')+'>'+t.name+'</option>'}).join('')+'</select>'+
'<div class="mini">Cambio en la red</div><div class="tools"><button class="tab '+(scenarios.provider===0?'active':'')+'" data-provider-change="0">Sin cambios</button><button class="tab '+(scenarios.provider===1?'active':'')+'" data-provider-change="1">+ Nuevo centro</button><button class="tab '+(scenarios.provider===-1?'active':'')+'" data-provider-change="-1">− Retirar centro</button></div>'+
'<div class="range-line"><span>Capacidad asistencial adicional</span><strong id="capacity-value">'+scenarios.capacity+' %</strong></div><input type="range" id="capacity-range" min="-30" max="45" step="3" value="'+scenarios.capacity+'"/>'+
'<div class="range-line"><span>Variación de tarifas</span><strong id="tariff-value">'+scenarios.tariff+' %</strong></div><input type="range" id="tariff-range" min="-15" max="25" step="1" value="'+scenarios.tariff+'"/>'+
'<div class="range-line"><span>Crecimiento previsto de cartera</span><strong id="growth-value">'+scenarios.growth+' %</strong></div><input type="range" id="growth-range" min="0" max="30" step="1" value="'+scenarios.growth+'"/>'+
'<div class="divider"></div><p class="mini">Modelo de demostración con coeficientes ficticios. No representa una estimación sanitaria ni económica real.</p></div></section>'+
'<section class="card"><div class="card-pad"><div class="card-head"><h3>Comparativa de escenarios</h3>'+pill('ACTUAL VS. PROPUESTO','green')+'</div><div class="scenario-grid"><div class="scenario-box"><div class="mini">SITUACIÓN ACTUAL</div><div class="number">'+r.t.coverage+' %</div><div class="mini">Cobertura de referencia</div></div><div class="scenario-box after"><div class="mini">ESCENARIO PROPUESTO</div><div class="number good">'+r.cov+' %</div><div class="mini">Cobertura modelizada</div></div></div>'+
'<div class="divider"></div>'+metric('Asegurados con acceso adicional', (r.people>=0?'+':'')+num(r.people))+metric('Impacto económico anual',(r.cost>=0?'+':'')+euro(r.cost))+metric('Variación cobertura',(r.cov-r.t.coverage>=0?'+':'')+(r.cov-r.t.coverage)+' pp')+
'<div class="divider"></div>'+insight('◈','Hipótesis de trabajo','El resultado usa reglas simplificadas. En producción se sustituiría por matrices de acceso, capacidad acreditada y escenarios económicos auditables.')+
'<button class="primary" id="save-scenario" style="width:100%;margin-top:16px">Añadir escenario al IPA →</button></div></section></div>'+
'<div style="margin-top:16px"><section class="card"><div class="card-pad"><div class="card-head"><h3>Impacto geográfico</h3>'+pill('DRILL-DOWN TERRITORIAL','blue')+'</div></div>'+mapHTML('map-sim')+'</section></div>'}
function insights(){var i=[{level:'Alta',city:'Sevilla',title:'Concentración en un grupo hospitalario',desc:'Dependencia asistencial elevada en especialidades de alta utilización. Analizar proveedores alternativos.',value:'36 %',metric:'Concentración ilustrativa'}, {level:'Alta',city:'Bilbao',title:'Presión en diagnóstico por imagen',desc:'Alta utilización en el principal instituto diagnóstico y espera superior al umbral de referencia.',value:'94 %',metric:'Capacidad utilizada'}, {level:'Media',city:'Málaga',title:'Crecimiento acelerado de la cartera',desc:'Se recomienda contrastar el aumento de asegurados con la capacidad asistencial y estacionalidad.',value:'+15,4 %',metric:'Variación de cartera'}, {level:'Media',city:'Valencia',title:'Oportunidad en periferia urbana',desc:'Territorios con crecimiento y tiempos de acceso superiores al objetivo ilustrativo.',value:'26 min',metric:'Tiempo medio de acceso'}, {level:'Seguimiento',city:'Madrid',title:'Expansión del corredor norte',desc:'Estudiar aumento de capacidad cardiológica en nuevas áreas residenciales.',value:'37 K',metric:'Asegurados fuera de objetivo'}];return head('EARLY SIGNALS','Radar estratégico.','Alertas analíticas configurables, transparentes y explicables. Sin IA generativa.','<button class="outline" data-go="ipa">Ver planes activos →</button>')+
'<div class="grid kpis">'+kpi('Señales activas','12','Datos simulados','dim')+kpi('Prioridad alta','2','Requiere validación','warn')+kpi('Oportunidades','5','Zonas identificadas')+kpi('Planes abiertos',state.plan.length,'Actuaciones registradas')+'</div>'+
'<div class="grid layout-70">'+card('Señales de planificación','Reglas analíticas predefinidas',i.map(function(x){return '<div class="insight"><div class="insight-icon">'+(x.level==='Alta'?'⚑':'◎')+'</div><div style="flex:1"><div class="row">'+pill(x.level,x.level==='Alta'?'red':x.level==='Media'?'amber':'blue')+'<span class="mini">'+x.city+'</span></div><h4 style="font-size:13px;margin:12px 0 7px">'+x.title+'</h4><p>'+x.desc+'</p><div class="row" style="margin-top:11px"><small>'+x.metric+': <b>'+x.value+'</b></small><button class="ghost" data-investigate="'+x.city+'">Investigar →</button></div></div></div>'}).join(''))+
card('Motor de reglas','Explicable y auditable',insight('1','Cobertura','Alertar cuando la cobertura cae por debajo del umbral configurado.')+insight('2','Capacidad','Detectar ocupación excesiva, espera o presión de la demanda.')+insight('3','Dependencia','Medir concentración de actividad por proveedor o grupo.')+insight('4','Crecimiento','Cruzar evolución de cartera y capacidad disponible.')+
'<div class="divider"></div><p class="text-block">Cada señal conserva fecha, versión de la regla y datos utilizados. Las alertas no sustituyen la validación del responsable territorial.</p>')+'</div>'}
function ipa(){return head('LIVING ASSISTANCE PLAN','El IPA ya no es un documento.','De la detección de necesidades a la decisión, aprobación y seguimiento.','<button class="outline" id="export-ipa">↓ Exportar resumen</button>')+
'<div class="grid kpis">'+kpi('Actuaciones del plan',state.plan.length,'Ejemplos demostrativos')+kpi('En análisis',state.plan.filter(function(x){return x.status==='En análisis'}).length,'Pendientes de evaluación','dim')+kpi('En curso',state.plan.filter(function(x){return x.status==='En curso'}).length,'Seguimiento de ejecución')+kpi('Territorios cubiertos',new Set(state.plan.map(function(x){return x.territory})).size,'Fichas dinámicas','dim')+'</div>'+
'<div class="grid layout-60">'+card('Roadmap del IPA','Planificación 2026–2027','<div class="steps">'+state.plan.map(function(x,i){return '<div class="step"><div class="row"><div>'+pill('ACTUACIÓN '+String(i+1).padStart(2,'0'),'blue')+'<h4>'+x.title+'</h4></div>'+pill(x.status,x.status==='En curso'?'green':x.status==='Pendiente'?'amber':'blue')+'</div><p>'+x.territory+' · '+x.owner+' · Prioridad '+x.priority+'</p></div>'}).join('')+'</div>'+
'<button class="primary" id="new-ipa" style="margin-top:17px;width:100%">+ Nueva actuación</button>')+
card('Ciclo de decisión','Trazabilidad desde el diagnóstico','<div class="timeline"><div class="timeline-item"><b>Diagnóstico territorial</b><small>Caracterizar cartera, demanda y cobertura</small></div><div class="timeline-item"><b>Necesidad documentada</b><small>Evidencias, impacto y objetivos medibles</small></div><div class="timeline-item"><b>Escenarios evaluados</b><small>Comparar alternativas de la red asistencial</small></div><div class="timeline-item"><b>Decisión y aprobación</b><small>Responsable, presupuesto y fecha objetivo</small></div><div class="timeline-item"><b>Seguimiento del impacto</b><small>Comprobar resultados observados frente a hipótesis</small></div></div>')+'</div>'+
'<div style="margin-top:16px">'+card('El IPA como aplicación','Visión ejecutiva',insight('▣','Un único origen de información','El mapa, los proveedores, los análisis y los escenarios comparten el mismo modelo de datos.')+insight('◇','Versionado de decisiones','Cada actuación conserva el escenario y los indicadores en el momento de su aprobación.')+insight('↗','Informe cuando lo necesites','La exportación es una salida del sistema, no la finalidad de ATLAS.'))+'</div>'}
function bindPage(){
  document.querySelectorAll('.map-retry').forEach(function(button){
    button.onclick=function(){
      var mapId=state.view==='dashboard'?'map-dash':state.view==='territory'?'map-ter':state.view==='network'?'map-net':state.view==='coverage'?'map-cov':'map-sim';
      if(state.map){try{state.map.remove()}catch(e){}state.map=null;state.overlay=null}
      var pre=document.querySelector('.map-preloader');if(pre)pre.style.opacity='1';
      var el=$(mapId);if(el){el.innerHTML='';initMap(mapId)}
    }
  });document.querySelectorAll('[data-basemap-cycle]').forEach(function(el){el.onclick=function(){if(state.map){atlasManualStyleIndex=(atlasStyleIndex+1)%(atlasStyleSources.length+1);if(state.map.__atlasSetStyle)state.map.__atlasSetStyle(atlasManualStyleIndex);else atlasLoadStyle(state.map,atlasManualStyleIndex);toast('Cambiando cartografía…')}else toast('El mapa aún está iniciándose')}});document.querySelectorAll('[data-map-diagnostic]').forEach(function(el){el.onclick=function(){var pane=document.querySelector('.map-diagnostics-pane');if(!pane)return;if(pane.style.display==='block'){pane.style.display='none';return}var m=state.map;var loaded=false,tiles=false;try{loaded=!!m&&m.isStyleLoaded();tiles=!!m&&m.areTilesLoaded()}catch(e){}pane.textContent='WebGL: '+(window.maplibregl&&maplibregl.supported()?'Sí':'No')+'\nMapLibre: '+(window.maplibregl?'Cargado':'No')+'\ndeck.gl: '+(window.deck?'Cargado':'No')+'\nCartografía: '+(atlasStyleSources[atlasStyleIndex]?atlasStyleSources[atlasStyleIndex].name:'Local')+'\nEstilo: '+(loaded?'Listo':'Cargando')+'\nTeselas: '+(tiles?'Listas':'Pendientes')+'\nÚltimo aviso: '+atlasLastMapError;pane.style.display='block'}});document.querySelectorAll('[data-layer]').forEach(function(el){el.addEventListener('click',function(){state.layer=el.dataset.layer;document.querySelectorAll('[data-layer]').forEach(function(e){e.classList.toggle('active',e===el)});updateLayers()})});var s=$('city-select');if(s)s.onchange=function(){state.selectedCity=s.value;render()};var mf=$('map-fly');if(mf)mf.onchange=function(){var t=T.find(function(x){return x.name===mf.value});if(t&&state.map)state.map.flyTo({center:[t.lon,t.lat],zoom:10,pitch:64,bearing:28,duration:1800})};var reset=$('map-reset');if(reset)reset.onclick=function(){if(state.map)state.map.flyTo({center:[-3.8,40.25],zoom:5.1,pitch:54,bearing:-8,duration:1500})};var q=$('provider-search');if(q)q.oninput=function(){state.search=q.value;refreshProviders()};var re=$('provider-region');if(re)re.onchange=function(){state.region=re.value;render()};var so=$('provider-sort');if(so)so.onchange=function(){state.sort=so.value;render()};document.querySelectorAll('[data-provider]').forEach(function(el){el.onclick=function(){state.selected=Number(el.dataset.provider);$('provider-detail').innerHTML=detail(P.find(function(p){return p.id===state.selected}));$('provider-detail').scrollIntoView({behavior:'smooth',block:'center'});$('provider-detail').querySelectorAll('[data-go]').forEach(function(a){a.onclick=function(){go(a.dataset.go)}})}});var ex=$('export-providers');if(ex)ex.onclick=exportProviders;var exp=$('export-ipa');if(exp)exp.onclick=exportIPA;var sc=$('sim-city');if(sc)sc.onchange=function(){scenarios.selected=sc.value;render()};document.querySelectorAll('[data-provider-change]').forEach(function(el){el.onclick=function(){scenarios.provider=Number(el.dataset.providerChange);render()}});[['capacity-range','capacity'],['tariff-range','tariff'],['growth-range','growth']].forEach(function(pair){var el=$(pair[0]);if(el){el.oninput=function(){var label=$(pair[1]+'-value');if(label)label.textContent=el.value+' %'};el.onchange=function(){scenarios[pair[1]]=Number(el.value);render()}}});var rs=$('reset-sim');if(rs)rs.onclick=function(){scenarios={provider:0,capacity:18,tariff:0,growth:6,selected:'Sevilla'};render()};var save=$('save-scenario');if(save)save.onclick=function(){var r=results();state.plan.push({title:'Escenario de red: cobertura '+r.cov+' %',territory:r.t.name,status:'En análisis',owner:'Planificación asistencial',priority:'Alta'});toast('Escenario incorporado al IPA');go('ipa')};document.querySelectorAll('[data-investigate]').forEach(function(el){el.onclick=function(){state.selectedCity=el.dataset.investigate;go('territory')}});var add=$('new-ipa');if(add)add.onclick=function(){var title=prompt('Título de la nueva actuación (demo)');if(title&&title.trim()){state.plan.push({title:title.trim().slice(0,120),territory:state.selectedCity,status:'Pendiente',owner:'Planificación asistencial',priority:'Media'});render();toast('Actuación creada en memoria para esta sesión')}}}
function refreshProviders(){var s=state.search;state.search=s;var p=$('page');var start=p.querySelector('.table-scroll');if(!start)return;var data=P.filter(function(x){return (state.region==='Todas'||x.region===state.region)&&(x.name+' '+x.city+' '+x.group+' '+x.code).toLowerCase().includes(s.toLowerCase())});var tbody=start.querySelector('tbody');tbody.innerHTML=data.map(function(x){return '<tr data-provider="'+x.id+'"><td><div class="provider-cell"><div class="provider-icon">✚</div><div><b>'+x.name+'</b><small>'+x.group+'</small></div></div></td><td>'+x.city+'</td><td>'+x.specialties+'</td><td>'+num(x.acts)+'</td><td>'+euro(x.cost)+'</td><td>'+x.occupancy+' %</td><td>'+x.dependency+' %</td><td>↗</td></tr>'}).join('');tbody.querySelectorAll('[data-provider]').forEach(function(el){el.onclick=function(){state.selected=Number(el.dataset.provider);$('provider-detail').innerHTML=detail(P.find(function(x){return x.id===state.selected}));$('provider-detail').scrollIntoView({behavior:'smooth'});$('provider-detail').querySelectorAll('[data-go]').forEach(function(a){a.onclick=function(){go(a.dataset.go)}})}})}
function download(name,s,type){var b=new Blob([s],{type:type||'text/plain;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=name;a.click();setTimeout(function(){URL.revokeObjectURL(a.href)},1000)}
function exportProviders(){var cols=['code','name','group','type','city','region','specialties','acts','cost','occupancy','quality','waiting','contract'];download('atlas-proveedores-demo.csv','\uFEFF'+cols.join(';')+'\n'+P.map(function(x){return cols.map(function(k){return '"'+String(x[k]).replace(/"/g,'""')+'"'}).join(';')}).join('\n'),'text/csv;charset=utf-8')}
function exportIPA(){download('atlas-ipa-demo.csv','\uFEFFActuación;Territorio;Estado;Responsable;Prioridad\n'+state.plan.map(function(x){return [x.title,x.territory,x.status,x.owner,x.priority].join(';')}).join('\n'),'text/csv;charset=utf-8')}

/* ATLAS Geospatial Engine v2:
   MapLibre initializes without waiting for deck.gl, with 3 cartography providers
   and a local inline geographic style as a last resort. */
var atlasMaplibrePromise=null,atlasDeckPromise=null;
var atlasStyleIndex=0,atlasManualStyleIndex=0,atlasLastMapError='Ninguno';
var atlasStyleSources=[
  {name:'OpenFreeMap · Liberty',url:'https://tiles.openfreemap.org/styles/liberty'},
  {name:'CARTO · Positron',url:'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'},
  {name:'MapLibre · World',url:'https://demotiles.maplibre.org/style.json'}
];
/* Same-origin cartography: Vercel proxies OpenFreeMap instead of the browser calling it. */
var atlasProxyRoot=window.location.origin+'/mapdata';
var atlasProxyStyle={
  version:8,name:'ATLAS · ASISA corporate map',
  glyphs:atlasProxyRoot+'/fonts/{fontstack}/{range}.pbf',
  sources:{
    openmaptiles:{type:'vector',tiles:[atlasProxyRoot+'/planet/latest/{z}/{x}/{y}.pbf'],minzoom:0,maxzoom:14},
    relief:{type:'raster',tiles:[atlasProxyRoot+'/natural_earth/ne2sr/{z}/{x}/{y}.png'],tileSize:256,maxzoom:6}
  },
  layers:[
    {id:'atlas-proxy-background',type:'background',paint:{'background-color':'#dceef7'}},
    {id:'atlas-proxy-relief',type:'raster',source:'relief',maxzoom:7,paint:{'raster-opacity':.42}},
    {id:'atlas-proxy-land',type:'fill',source:'openmaptiles','source-layer':'landcover',paint:{'fill-color':'#edf5ec','fill-opacity':.76}},
    {id:'atlas-proxy-parks',type:'fill',source:'openmaptiles','source-layer':'park',paint:{'fill-color':'#d9eddd'}},
    {id:'atlas-proxy-urban',type:'fill',source:'openmaptiles','source-layer':'landuse',filter:['in',['get','class'],['literal',['residential','commercial','industrial']]],paint:{'fill-color':'#e6eaf1','fill-opacity':.58}},
    {id:'atlas-proxy-water',type:'fill',source:'openmaptiles','source-layer':'water',paint:{'fill-color':'#b3d8ef'}},
    {id:'atlas-proxy-waterway',type:'line',source:'openmaptiles','source-layer':'waterway',paint:{'line-color':'#93c9e6','line-width':['interpolate',['linear'],['zoom'],4,.25,13,2]}},
    {id:'atlas-proxy-boundaries',type:'line',source:'openmaptiles','source-layer':'boundary',filter:['<=',['get','admin_level'],4],paint:{'line-color':'#8eaac0','line-dasharray':[3,2],'line-width':1}},
    {id:'atlas-proxy-highways',type:'line',source:'openmaptiles','source-layer':'transportation',filter:['in',['get','class'],['literal',['motorway','trunk','primary']]],paint:{'line-color':'#ecbc7c','line-width':['interpolate',['linear'],['zoom'],5,.45,13,5]}},
    {id:'atlas-proxy-road',type:'line',source:'openmaptiles','source-layer':'transportation',filter:['in',['get','class'],['literal',['secondary','tertiary','minor','residential']]],minzoom:8,paint:{'line-color':'#f8faff','line-width':['interpolate',['linear'],['zoom'],8,.4,14,3]}},
    {id:'atlas-proxy-buildings',type:'fill-extrusion',source:'openmaptiles','source-layer':'building',minzoom:13,paint:{'fill-extrusion-color':'#8fb9d5','fill-extrusion-height':['coalesce',['to-number',['get','render_height']],['to-number',['get','height']],9],'fill-extrusion-base':['coalesce',['to-number',['get','render_min_height']],0],'fill-extrusion-opacity':.62}},
    {id:'atlas-proxy-city-labels',type:'symbol',source:'openmaptiles','source-layer':'place',filter:['in',['get','class'],['literal',['city','town','village']]],layout:{'text-field':['coalesce',['get','name:es'],['get','name']],'text-font':['Open Sans Regular'],'text-size':['interpolate',['linear'],['zoom'],4,10,9,15,13,17],'text-max-width':8},paint:{'text-color':'#244b6b','text-halo-color':'#f5fbfe','text-halo-width':1.6}}
  ]
};
atlasStyleSources.unshift({name:'ATLAS · cartografía corporativa',url:atlasProxyStyle});
var atlasLocalCoast=[
[-9.29,43.13],[-9,43.36],[-8.4,43.4],[-7.2,43.73],[-6.1,43.61],[-5.3,43.55],[-4.4,43.47],[-3.4,43.48],[-2.1,43.36],[-1.78,43.38],[-1.4,43.08],[-.7,42.84],[.6,42.72],[1.4,42.62],[1.9,42.45],[2.48,42.43],[3.2,42.35],[3.32,41.9],[2.84,41.67],[2.3,41.45],[.95,40.79],[.18,40.03],[-.34,39.46],[-.22,38.75],[-.69,37.97],[-1.41,37.42],[-1.91,36.99],[-2.45,36.73],[-3.52,36.72],[-4.45,36.71],[-5.35,36.16],[-5.61,36.03],[-6.1,36.22],[-6.39,36.81],[-6.95,37.16],[-7.43,37.19],[-7.52,37.55],[-7.16,38.1],[-7.04,38.87],[-7.03,39.67],[-6.85,40.03],[-6.92,40.35],[-6.78,41.04],[-6.59,41.97],[-7.18,41.87],[-7.42,41.82],[-8.16,41.82],[-8.75,41.9],[-8.88,42.25],[-9.15,42.72]
];
var atlasLocalStyle={
  version:8,
  name:'ATLAS Offline Outline',
  glyphs:undefined,
  sources:{'atlas-outline':{type:'geojson',data:{type:'FeatureCollection',features:[
    {type:'Feature',properties:{name:'Península ibérica · contorno ilustrativo'},geometry:{type:'Polygon',coordinates:[atlasLocalCoast.concat([atlasLocalCoast[0]])]}},
    {type:'Feature',properties:{name:'Baleares · contorno ilustrativo'},geometry:{type:'Polygon',coordinates:[[[2.4,39.5],[2.8,39.35],[3.47,39.72],[3.27,39.9],[2.8,39.85],[2.4,39.5]]]}}
  ]}}},
  layers:[
    {id:'atlas-bg',type:'background',paint:{'background-color':'#e8f3fa'}},
    {id:'atlas-land',type:'fill',source:'atlas-outline',paint:{'fill-color':'#d1e4f1','fill-outline-color':'#87acc8'}},
    {id:'atlas-land-line',type:'line',source:'atlas-outline',paint:{'line-color':'#79abc8','line-width':1.5}}
  ]
};
atlasLocalStyle.sources['atlas-outline'].data='./iberia.geojson';
function atlasLoadScript(urls,globalName){
  if(window[globalName])return Promise.resolve();
  return new Promise(function(resolve,reject){
    var index=0;
    function next(){
      if(window[globalName]){resolve();return}
      if(index>=urls.length){reject(new Error('CDN: no se pudo cargar '+globalName));return}
      var src=urls[index++],script=document.createElement('script'),finished=false;
      script.src=src;script.async=true;
      var timeout=setTimeout(function(){if(finished)return;finished=true;script.remove();next()},9500);
      script.onload=function(){if(finished)return;finished=true;clearTimeout(timeout);if(window[globalName])resolve();else next()};
      script.onerror=function(){if(finished)return;finished=true;clearTimeout(timeout);next()};
      document.head.appendChild(script);
    }next();
  });
}
function atlasLoadMaplibre(){
  if(!atlasMaplibrePromise){
    var css=document.createElement('link');css.rel='stylesheet';
    css.href='https://cdn.jsdelivr.net/npm/maplibre-gl@4.7.1/dist/maplibre-gl.css';
    document.head.appendChild(css);
    atlasMaplibrePromise=atlasLoadScript([
      'https://cdn.jsdelivr.net/npm/maplibre-gl@4.7.1/dist/maplibre-gl.js',
      'https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js'
    ],'maplibregl');
  }
  return atlasMaplibrePromise;
}
function atlasLoadDeck(){
  if(!atlasDeckPromise){
    atlasDeckPromise=atlasLoadScript([
      'https://cdn.jsdelivr.net/npm/deck.gl@9.1.5/dist.min.js',
      'https://unpkg.com/deck.gl@9.1.5/dist.min.js'
    ],'deck');
  }return atlasDeckPromise;
}
function atlasStatus(label,warning){
  var e=document.querySelector('.map-health');
  if(e){e.textContent=label;e.classList.toggle('error',!!warning)}
}
function atlasMapErrorDetail(map,label){
  atlasLastMapError=label;
  var e=document.querySelector('.map-health');
  if(!e)return;
  e.title='Cartografía: '+(atlasStyleSources[atlasStyleIndex]&&atlasStyleSources[atlasStyleIndex].name||'local')+
    ' · MapLibre: '+(!!window.maplibregl)+' · deck.gl: '+(!!window.deck)+' · '+label;
}
function atlasMapPoints(){
  var seed=12051;
  function rand(){seed=(1664525*seed+1013904223)>>>0;return seed/4294967296}
  var points=[];
  P.forEach(function(p){
    var n=Math.min(105,Math.round(p.acts/2200));
    for(var i=0;i<n;i++){
      var r=Math.sqrt(rand())*.24,a=rand()*Math.PI*2;
      points.push({position:[p.lng+Math.cos(a)*r,p.lat+Math.sin(a)*r*.7],weight:1+p.acts/100000,name:p.city});
    }
  });return points;
}
var atlasSyntheticPoints=atlasMapPoints();
function atlasGeoJSON(){
  return {type:'FeatureCollection',features:P.map(function(p){return {type:'Feature',properties:{id:p.id,name:p.name,acts:p.acts,city:p.city,occupancy:p.occupancy},geometry:{type:'Point',coordinates:[p.lng,p.lat]}}})};
}
function atlasNativeLayers(map){
  if(map.getSource('atlas-native'))return;
  map.addSource('atlas-native',{type:'geojson',data:atlasGeoJSON()});
  map.addLayer({id:'atlas-native-glow',type:'circle',source:'atlas-native',paint:{'circle-radius':['interpolate',['linear'],['zoom'],4,13,9,32,13,65],'circle-color':'#169ed2','circle-opacity':.16,'circle-blur':.7}});
  map.addLayer({id:'atlas-native-points',type:'circle',source:'atlas-native',paint:{'circle-radius':['interpolate',['linear'],['zoom'],4,4,9,9,13,13],'circle-color':'#1387be','circle-stroke-color':'#ffffff','circle-stroke-width':2,'circle-opacity':.92}});
  map.on('click','atlas-native-points',function(e){var feature=e.features&&e.features[0];if(!feature)return;var p=P.find(function(v){return v.id===Number(feature.properties.id)});if(p){state.selected=p.id;go('providers');setTimeout(function(){var target=$('provider-detail');if(target)target.scrollIntoView({behavior:'smooth',block:'start'})},80)}});
  map.on('mouseenter','atlas-native-points',function(){map.getCanvas().style.cursor='pointer'});
  map.on('mouseleave','atlas-native-points',function(){map.getCanvas().style.cursor=''});
}
function atlasDecorateMap(map){
  var layer=map.getStyle().layers.find(function(l){return l.type==='symbol'&&l.layout&&l.layout['text-field']});
  var before=layer&&layer.id;
  // Color only our optional extruded buildings. Standard base style stays legible.
  var s=map.getStyle();
  if(s.sources&&s.sources.openmaptiles&&s.layers.some(function(l){return l['source-layer']==='building'})){
    try{map.addLayer({id:'atlas-buildings',source:'openmaptiles','source-layer':'building',type:'fill-extrusion',minzoom:13,paint:{'fill-extrusion-color':'#78b6d8','fill-extrusion-height':['coalesce',['get','render_height'],['get','height'],12],'fill-extrusion-base':['coalesce',['get','render_min_height'],0],'fill-extrusion-opacity':.55}},before)}catch(e){console.info('Building extrusions not available for this style')}
  }
  atlasNativeLayers(map);
}
function atlasDeckLayers(){
  if(!window.deck)return [];
  var d=window.deck;
  var main=P.map(function(p){return {position:[p.lng,p.lat],name:p.name,n:p.acts,original:p}});
  var city=T.map(function(t){return {position:[t.lon,t.lat],name:t.name,n:t.insured}});
  if(state.layer==='columns')return [
    new d.ScatterplotLayer({id:'atlas-background-constellation',data:atlasSyntheticPoints,getPosition:function(x){return x.position},getRadius:2500,radiusMinPixels:1.5,getFillColor:[23,143,195,110],pickable:false}),
    new d.ArcLayer({id:'atlas-background-routes',data:P.filter(function(x){return x.city!=='Madrid'}),getSourcePosition:[-3.7038,40.4168],getTargetPosition:function(x){return [x.lng,x.lat]},getSourceColor:[44,129,191,95],getTargetColor:[236,143,75,130],getWidth:1.8,widthMinPixels:1,pickable:false}),
    new d.ColumnLayer({id:'atlas-3d-columns',data:main,radius:8500,diskResolution:8,extruded:true,elevationScale:1,pickable:true,autoHighlight:true,getPosition:function(x){return x.position},getElevation:function(x){return 3000+x.n*.22},getFillColor:function(x){return x.original.occupancy>90?[235,140,67,210]:[25,127,194,210]},getLineColor:[255,255,255,210]}),
    new d.ScatterplotLayer({id:'atlas-city-nodes',data:city,getPosition:function(x){return x.position},getRadius:4500,getFillColor:[20,132,180,175],getLineColor:[255,255,255],stroked:true,pickable:true})
  ];
  if(state.layer==='arcs')return [
    new d.ArcLayer({id:'atlas-flow-arcs',data:P.filter(function(p){return p.city!=='Madrid'}),getSourcePosition:[-3.7038,40.4168],getTargetPosition:function(p){return [p.lng,p.lat]},getSourceColor:[23,112,188,235],getTargetColor:[247,146,67,230],getWidth:4,widthMinPixels:2,pickable:true}),
    new d.ScatterplotLayer({id:'atlas-network-nodes',data:main,getPosition:function(x){return x.position},getRadius:9500,getFillColor:[23,112,188],pickable:true})
  ];
  if(state.layer==='heat')return [new d.HeatmapLayer({id:'atlas-heat',data:atlasSyntheticPoints,getPosition:function(x){return x.position},getWeight:function(x){return x.weight},radiusPixels:70,intensity:1.5,threshold:.05})];
  return [
    new d.ScatterplotLayer({id:'atlas-constellation',data:atlasSyntheticPoints,getPosition:function(x){return x.position},getRadius:3000,radiusMinPixels:2,getFillColor:[16,133,197,155],pickable:true}),
    new d.ScatterplotLayer({id:'atlas-main-points',data:main,getPosition:function(x){return x.position},getRadius:8000,getFillColor:[241,137,67,200],getLineColor:[255,255,255],stroked:true,pickable:true})
  ];
}
function atlasAddDeck(map){
  if(!map||state.map!==map||!window.deck||!window.deck.MapboxOverlay)return;
  try{
    if(!state.overlay){
      state.overlay=new deck.MapboxOverlay({
        interleaved:false,layers:atlasDeckLayers(),
        getTooltip:function(info){
          var p=info.object;
          if(!p)return null;
          return {text:(p.name||p.city||'Centro simulado')+(p.original?' · '+num(p.n)+' actos/año':''),style:{backgroundColor:'#114e84',color:'white',fontSize:'12px',borderRadius:'8px'}};
        },
        onClick:function(info){var p=info.object&&info.object.original;if(p){state.selected=p.id;go('providers')}}
      });
      map.addControl(state.overlay);
    }else state.overlay.setProps({layers:atlasDeckLayers()});
    // Base-map layers are retained until the 3D overlay is ready.
    if(map.getLayer('atlas-native-glow'))map.setPaintProperty('atlas-native-glow','circle-opacity',.04);
    if(map.getLayer('atlas-native-points'))map.setPaintProperty('atlas-native-points','circle-opacity',.36);
    atlasStatus(atlasStyleIndex>=atlasStyleSources.length?'3D activo · cartografía local':'3D activo · cartografía detallada',false);
  }catch(e){console.warn('ATLAS deck overlay:',e);atlasStatus('Mapa activo · capa 3D no disponible',true)}
}
function atlasAttachDeck(map){
  atlasLoadDeck().then(function(){if(map===state.map&&map.loaded())atlasAddDeck(map)})
    .catch(function(e){console.warn('ATLAS deck CDN:',e);if(map===state.map)atlasStatus('Mapa activo · CDN 3D no disponible',true)});
}
function atlasLoadStyle(map,index){
  if(map!==state.map)return;
  if(index>=atlasStyleSources.length){
    atlasStyleIndex=index;
    atlasStatus('Cartografía local · sin teselas externas',true);
    map.setStyle(atlasLocalStyle);return;
  }
  atlasStatus('Cargando '+atlasStyleSources[index].name+'…',false);
  atlasStyleIndex=index;
  map.setStyle(atlasStyleSources[index].url);
}
function initMap(id){
  var el=$(id);if(!el)return;
  var initialStatus=document.querySelector('.map-diagnostics-status');
  if(initialStatus)initialStatus.textContent='Inicializando MapLibre · '+(window.maplibregl?'Motor disponible':'Esperando el motor')+
    ' · WebGL: '+(window.maplibregl&&window.maplibregl.supported()?'Sí':'No');
  atlasStatus('Iniciando el mapa…',false);
  if(!window.maplibregl){
    el.innerHTML='<div class="map-error"><div><strong>Preparando el mapa interactivo de España…</strong><p>Cargando el motor cartográfico; no es necesario esperar para usar el resto de ATLAS.</p></div></div>';
    atlasLoadMaplibre().then(function(){if(el.isConnected&&$(id)===el)initMap(id)})
      .catch(function(e){console.warn('ATLAS MapLibre CDN:',e);if(el.isConnected){fallback(el);atlasStatus('Mapa esquemático · motor externo no disponible',true)}});
    return;
  }
  if(!maplibregl.supported()){atlasStatus('Sin WebGL · mostrando mapa de España local',true);var status=document.querySelector('.map-diagnostics-status');if(status)status.textContent='WebGL deshabilitado en el navegador';return}
  try{
    el.innerHTML='';
    var t=T.find(function(x){return x.name===state.selectedCity})||T[0];
    var focused=state.view==='territory'||state.view==='coverage';
    var map=new maplibregl.Map({
      container:el,
      style:atlasLocalStyle,
      center:focused?[t.lon,t.lat]:[-3.8,40.25],
      zoom:focused?8:5.15,pitch:focused?42:56,bearing:-8,
      antialias:true,attributionControl:true,
      maxPitch:75
    });
    state.map=map;state.overlay=null;
    atlasStyleIndex=atlasStyleSources.length;
    atlasStatus('Activando mapa local de España…',false);
    var attempted=atlasStyleSources.length,rendered=false,timer=null,tileFailures=0,resourceProbe=null;
    map.addControl(new maplibregl.NavigationControl({visualizePitch:true,showCompass:true}),'top-right');
    function failover(reason){
      if(map!==state.map||attempted>=atlasStyleSources.length)return;
      attempted++;rendered=false;tileFailures=0;
      clearTimeout(timer);clearTimeout(resourceProbe);
      atlasStatus('Cambiando de cartografía ('+reason+')…',true);
      atlasLoadStyle(map,attempted);
      startWatchdog();
    }
    map.on('style.load',function(){
      if(map!==state.map)return;
      var preview=document.querySelector('.map-preloader');if(preview)preview.style.opacity='0';
      var status=document.querySelector('.map-diagnostics-status');if(status)status.textContent='MapLibre iniciado · '+(window.deck?'deck.gl disponible':'deck.gl no disponible')+' · Cartografía: '+(atlasStyleSources[atlasStyleIndex]?atlasStyleSources[atlasStyleIndex].name:'Local');
      clearTimeout(timer);clearTimeout(resourceProbe);rendered=true;tileFailures=0;
      try{atlasDecorateMap(map)}catch(e){console.warn('Map layer setup:',e)}
      if(window.deck)atlasAddDeck(map);else atlasStatus('Mapa disponible · preparando 3D…',false);
      atlasMapErrorDetail(map,'El mapa base ha inicializado');
      if(attempted<atlasStyleSources.length){
        resourceProbe=setTimeout(function(){
          if(map!==state.map||!map.areTilesLoaded)return;
          if(!map.areTilesLoaded()){console.warn('ATLAS: basemap tiles timed out');failover('teselas')}
        },14000);
      }
    });
    map.on('error',function(e){
      if(map!==state.map)return;
      var error=e&&e.error;console.warn('ATLAS map resource error:',error||e);var status=document.querySelector('.map-diagnostics-status');if(status)status.textContent='Error cartográfico: '+String(error&&error.message||'Error desconocido');atlasLastMapError=String(error&&error.message||'Fallo de carga del recurso cartográfico');
      tileFailures++;
      if(!rendered&&tileFailures>=1)failover('estilo');
      else if(rendered&&tileFailures>=4)failover('recursos');
      else atlasMapErrorDetail(map,String(error&&error.message||'error de recurso'));
    });
    map.__atlasSetStyle=function(index){
      attempted=index;rendered=false;tileFailures=0;
      clearTimeout(resourceProbe);clearTimeout(timer);
      atlasLoadStyle(map,index);
      startWatchdog();
    };
    function startWatchdog(){
      clearTimeout(timer);
      timer=setTimeout(function(){
        if(map!==state.map||rendered)return;
        if(attempted<atlasStyleSources.length)failover('tiempo de espera');
        else{atlasStatus('Cartografía local activa',false)}
      },11000);
    }
    startWatchdog();
    atlasAttachDeck(map);
    // First paint is guaranteed to use our own Spanish geography and locally bundled WebGL.
    // Attempt the detailed basemap only after checking an actual Spanish vector tile
    // through Vercel. A blocked external service cannot blank the entire map.
    var controller=typeof AbortController!=='undefined'?new AbortController():null;
    var preflightTimeout=setTimeout(function(){if(controller)controller.abort()},8500);
    fetch('/mapdata/planet/latest/5/15/12.pbf',{signal:controller?controller.signal:undefined})
      .then(function(r){if(!r.ok)throw new Error('Tile HTTP '+r.status);return r.arrayBuffer()})
      .then(function(data){
        clearTimeout(preflightTimeout);
        if(data.byteLength<80)throw new Error('Empty vector tile');
        if(map===state.map)map.__atlasSetStyle(0);
      })
      .catch(function(e){
        clearTimeout(preflightTimeout);
        if(map===state.map){
          console.warn('External tiles unavailable, offline map remains visible:',e);
          atlasStatus('Mapa geográfico local · 3D activo',false);
          atlasLastMapError='OpenFreeMap a través de Vercel: '+e.message;
        }
      });
  }catch(e){console.error('ATLAS map initialization:',e);atlasStatus('Mapa de España local · motor 3D no disponible',true);var status=document.querySelector('.map-diagnostics-status');if(status)status.textContent='Error del motor: '+String(e&&e.message||e)}
}
function updateLayers(){
  if(!state.map)return;
  if(window.deck&&state.overlay){try{state.overlay.setProps({layers:atlasDeckLayers()});return}catch(e){console.warn('Updating deck:',e)}}
  // Native MapLibre always remains interactive while the overlay loads.
  var map=state.map;
  if(map.getLayer('atlas-native-glow'))map.setPaintProperty('atlas-native-glow','circle-opacity',state.layer==='heat'?.35:.17);
  if(map.getLayer('atlas-native-points'))map.setPaintProperty('atlas-native-points','circle-radius',state.layer==='points'?9:5);
}
function boot(){try{if(sessionStorage.getItem('atlas-demo')==='yes'){showApp()}}catch(e){console.warn('Session restore:',e)}var f=$('login-form');f.onsubmit=function(e){e.preventDefault();if($('username').value.trim().toLowerCase()==='admin'&&$('password').value==='AtlasDemo2026!'){try{sessionStorage.setItem('atlas-demo','yes')}catch(e){}showApp()}else $('login-error').textContent='Credenciales incorrectas. Consulta el README de la demo.'};$('logout').onclick=function(){sessionStorage.removeItem('atlas-demo');$('app').classList.add('hidden');$('login').classList.remove('hidden');if(state.map){state.map.remove();state.map=null}};$('mobile-menu').onclick=function(){document.querySelector('.sidebar').classList.toggle('open')}}
function showApp(){$('login').classList.add('hidden');$('app').classList.remove('hidden');try{render()}catch(e){console.error('ATLAS start failed:',e);$('page').innerHTML='<div class="card card-pad"><h2>Se ha producido un problema al iniciar ATLAS</h2><p>Recarga la página. Si persiste, abre la consola del navegador o comunica el error.</p></div>'}}
boot();
})();